import math
import inspect
from typing import Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from allamo.logging import logger
from allamo.model.attentions import attention_version

@dataclass
class AllamoTransformerConfig:
    block_size: int = 1024
    vocab_size: int = 32000
    rope_freq_base: int = 10000
    rope_freq_scale: float = 1.0
    n_layer: int = 12
    num_kv_heads: Union[None, int] = None
    head_size: Union[None, int] = None
    n_head: int = 12
    n_embd: int = 768
    intermediate_size: int = None
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears. False: a bit better and faster
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5
    sliding_window: int = None
    gradient_checkpointing: bool = False

class RMSNorm(torch.nn.Module):
    """RMSNorm normalizing function, introduced by Zhang and Sennrich (2019)"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)
    
    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)
        
class RotaryEmbedding(torch.nn.Module):
    
    def __init__(self, config: AllamoTransformerConfig):
        super().__init__()
        self.dim = config.head_size
        self.max_seq_len = config.block_size
        self.base = config.rope_freq_base if config.rope_freq_base is not None else 10000
        self.scale = config.rope_freq_scale if config.rope_freq_scale is not None else 1.0
        self._rope_init()
        
    # Define reset_parameters for FSDP initialization
    def reset_parameters(self):
        self._rope_init()

    def _rope_init(self):
        inv_freq = 1.0 / (self.scale * (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim)))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.build_rope_cache(self.max_seq_len)
        
    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        t = torch.arange(max_seq_len, device=self.inv_freq.device, dtype=torch.int64).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq).float()
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self,
        q: torch.Tensor,
        k: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
    ):
        # q,k: [bs, num_attention_heads, seq_len, head_size]
        dtype = q.dtype
        q = q.float()
        k = k.float()
        if input_pos is None:
            cos = self.cos_cached[None, None, :q.size(2), ...]
            sin = self.sin_cached[None, None, :q.size(2), ...]
        else:
            cos = self.cos_cached[input_pos].unsqueeze(1)
            sin = self.sin_cached[input_pos].unsqueeze(1)
        
        q_out = (q * cos) + (self.__rotate_half(q) * sin)
        k_out = (k * cos) + (self.__rotate_half(k) * sin)
        return q_out.to(dtype=dtype), k_out.to(dtype=dtype)
    
    def __rotate_half(self, x):
        # Rotates half the hidden dims of the input
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

class FeedForward(nn.Module):

    def __init__(self, config: AllamoTransformerConfig):
        super().__init__()
        if config.intermediate_size is None:
            config.intermediate_size = int(2 * (4 * config.n_embd) / 3)
            config.intermediate_size = config.multiple_of * ((config.intermediate_size + config.multiple_of - 1) // config.multiple_of)
        
        self.gate_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)
        self.up_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.act_fn  = nn.SiLU() # SwiGLU activation function
        self.dropout = nn.Dropout(config.dropout) if config.dropout != 0 else None
        self.gradient_checkpointing = config.gradient_checkpointing
        
    def init_weights(self, init_std: float):
        torch.nn.init.trunc_normal_(self.gate_proj.weight, mean=0.0, std=0.02)
        for module in (self.down_proj, self.up_proj):
            torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=init_std)

    def forward(self, x):
        if self.training and self.gradient_checkpointing:
            x = checkpoint(self.mlp, x, use_reentrant=False, preserve_rng_state=False)
        else:
            x = self.mlp(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x
        
    def mlp(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class Attention(nn.Module):

    def __init__(self, config: AllamoTransformerConfig):
        super().__init__()
        self.head_size = config.head_size
        self.num_heads = config.n_head
        self.num_kv_heads = config.num_kv_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.dropout = config.dropout
        self.sliding_window = config.sliding_window if attention_version.flash_attn_2_supports_window_size else None
        self.gradient_checkpointing = config.gradient_checkpointing
        
        assert self.num_key_value_groups * self.num_kv_heads == self.num_heads
        
        # key, query, value projections for all heads
        self.q_proj = nn.Linear(config.n_embd, self.num_heads * self.head_size, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, self.num_kv_heads * self.head_size, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, self.num_kv_heads * self.head_size, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(self.num_heads * self.head_size, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout) if attention_version.version == 0 and config.dropout != 0 else None
        self.proj_dropout = nn.Dropout(config.dropout) if config.dropout != 0 else None
        
        if attention_version.version == 0:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("temp_mask", torch.tril(torch.ones(1, 1, config.block_size, config.block_size)), persistent=False)
            
    def init_weights(self, init_std: float):
        for module in (self.q_proj, self.k_proj, self.v_proj):
            torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
        torch.nn.init.trunc_normal_(self.c_proj.weight, mean=0.0, std=init_std)
        
        if attention_version.version == 0:
            with torch.device(self.temp_mask.device):
                self.temp_mask = torch.tril(torch.ones(1, 1, self.temp_mask.shape[2], self.temp_mask.shape[3]))
        
    def forward(self,
                q_x: torch.Tensor,
                kv_x: torch.Tensor,
                rotary_emb: RotaryEmbedding,
                attn_mask: Optional[torch.Tensor] = None,
                input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # notation:
        # B  | batch
        # T  | time-step (sequence length)
        # C  | embeddings size
        # hs | head size
        # nh | number of heads
        B, T, C = q_x.size()
        
        if self.training and self.gradient_checkpointing:
            q, k, v = checkpoint(self.project_qkv, q_x, kv_x, use_reentrant=False, preserve_rng_state=False)
        else:
            q, k, v = self.project_qkv(q_x, kv_x)
        
        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.num_kv_heads, self.head_size).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.num_kv_heads, self.head_size).transpose(1, 2) # (B, nh, T, hs)
        
        q, k = rotary_emb(q, k, input_pos=input_pos)
        
        if self.num_key_value_groups > 1:
            k = self.repeat_kv(k, self.num_key_value_groups)
            v = self.repeat_kv(v, self.num_key_value_groups)
        
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if attention_version.version == 2:
            if attn_mask is not None:
                raise ValueError(f"Custom attention mask is not supported for FlashAttention2")
            # Flash Attention 2 requires (B, T, nh, hs)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            if self.sliding_window:            
                y = flash_attn_func(q, k, v, dropout_p=self.dropout if self.training else 0, causal=True, window_size=(self.sliding_window, self.sliding_window))
            else:
                y = flash_attn_func(q, k, v, dropout_p=self.dropout if self.training else 0, causal=True)
        elif attention_version.version == 1:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0,
                is_causal=attn_mask is None,
            )
            y = y.transpose(1, 2)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if attn_mask is None:
                att = att.masked_fill(self.temp_mask[:,:,:T,:T] == 0, float('-inf'))
            else:
                att = att.masked_fill(attn_mask.logical_not(), float('-inf'))
            att = F.softmax(att, dim=-1)
            if self.attn_dropout is not None:
                att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            y = y.transpose(1, 2)

        # output projection
        y = y.contiguous().view(B, T, self.num_heads * self.head_size) # re-assemble all head outputs side by side
        if self.training and self.gradient_checkpointing:
            y = checkpoint(self.c_proj, y, use_reentrant=False, preserve_rng_state=False)
        else:
            y = self.c_proj(y) # (B, T, nh * hs) -> (B, T, C)
        if self.proj_dropout is not None:
            y = self.proj_dropout(y)
        return y
        
    def project_qkv(self, q_x: torch.Tensor, kv_x: torch.Tensor):
        if kv_x is None:
            kv_x = q_x # self attention
        q = self.q_proj(q_x)
        k = self.k_proj(kv_x)
        v = self.v_proj(kv_x)
        return q, k, v
        
    def repeat_kv(self, x: torch.Tensor, num_key_value_groups: int) -> torch.Tensor:
        # (B, num_kv_heads, T, hs) -> (B, nh, T, hs)
        if num_key_value_groups == 1:
            return x
        B, num_kv_heads, T, hs = x.shape
        x = x[:, :, None, :, :].expand(B, num_kv_heads, num_key_value_groups, T, hs)
        return x.reshape(B, num_kv_heads * num_key_value_groups, T, hs)
        
class SelfAttentionBlock(nn.Module):

    def __init__(self, config: AllamoTransformerConfig):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.n_embd, eps=config.norm_eps)
    
    def init_weights(self, init_std: float):
        self.attention.init_weights(init_std)
        self.feed_forward.init_weights(init_std)
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()

    def forward(self,
        x: torch.Tensor,
        rotary_emb: RotaryEmbedding,
        attn_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        x = x + self.attention(self.attention_norm(x), None, rotary_emb, attn_mask=attn_mask, input_pos=input_pos)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x

class AllamoTransformer(nn.Module):

    def __init__(self, config: AllamoTransformerConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        if config.head_size is None:
            assert config.n_embd % config.n_head == 0
            config.head_size = config.n_embd // config.n_head
            logger.info(f"defaulting to head_size={config.head_size} (n_embd / n_head)")
        if config.num_kv_heads is None:
            config.num_kv_heads = config.n_head
        logger.info(f"AllamoTransformerConfig: {config}")
        
        self.init_attention()
        attention_version.log_version(self.config.sliding_window)
        
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.tok_drop = nn.Dropout(config.dropout) if config.dropout != 0 else None
        
        self.rotary_emb = RotaryEmbedding(config)
        
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layer):
            self.layers.append(SelfAttentionBlock(config))
        
        self.norm = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self.log_estimated_size()
        self.init_model_weights()
        
    def init_model_weights(self):
        with torch.device(self.rotary_emb.inv_freq.device):
            self.rotary_emb.reset_parameters()
        
        if self.tok_embeddings is not None:
            torch.nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=1.0)
            
        weight_init_std = self.calculate_weight_init_std(self.config.n_layer)
        for layer in self.layers:
            if layer is not None:
                layer.init_weights(weight_init_std)
        
        if self.norm is not None:
            self.norm.reset_parameters()
        
        if self.lm_head is not None:
            cutoff_factor = 3
            weight_init_std = self.config.n_embd ** -0.5
            lower = -cutoff_factor * weight_init_std
            upper = cutoff_factor * weight_init_std
            torch.nn.init.trunc_normal_(self.lm_head.weight, mean=0.0, std=weight_init_std, a=lower, b=upper)

    def init_attention(self):
        if attention_version.version == 2:
            try:
                from flash_attn import flash_attn_func
                attention_version.flash_attn_2_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)
            except ImportError:
                attention_version.disable_flash_attn_2()
                logger.warning("Flash Attention 2 is not available!")
                
    def calculate_weight_init_std(self, num_layers):
        return 0.02 / math.sqrt(2 * num_layers)

    def estimate_size(self):
        """
        Return the number of parameters and their size in the model.
        """
        params = 0
        bytes = 0
        for p in self.parameters():
            params += p.numel()
            bytes += p.numel() * p.element_size()
        for b in self.buffers():
            # don't count buffers as params
            bytes += b.numel() * b.element_size()
        return params, bytes
        
    def log_estimated_size(self):
        self.model_num_params, self.model_num_bytes = self.estimate_size()
        model_params = self.model_num_params / 1e6
        model_bytes = self.model_num_bytes / 1024**2
        logger.info(f"Model parameters: {model_params:.2f}M, Est. Size: {model_bytes:.3f}MB")
            
    def add_layer(self, new_layers=1):
        for _ in range(new_layers):
            layer = SelfAttentionBlock(self.config)
            self.layers.append(layer)
            self.config.n_layer += 1
            layer.init_weights(self.calculate_weight_init_std(self.config.n_layer))
            
    def freeze_params(self, module):
        for param in module.parameters():
            if param.requires_grad:
                param.requires_grad = False
            
    def token_embeddings(self, input_ids):
        _, t = input_ids.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        x = self.tok_embeddings(input_ids) # token embeddings of shape (b, t, n_embd)
        if self.tok_drop is not None:
            x = self.tok_drop(x)
        return x

    def apply_layers(self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, self.rotary_emb, attn_mask=attn_mask, input_pos=input_pos)
        return x

    def forward(self,
        input_ids: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        target_weights: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        ignore_index: Optional[int] = -100,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.FloatTensor]]]:
        if inputs_embeds is not None:
            b, t, n_embd = inputs_embeds.size()
            assert t <= self.config.block_size, f"Cannot forward embeddings of length {t}, block size is only {self.config.block_size}"
        else:
            inputs_embeds = self.token_embeddings(input_ids)
        
        if attn_mask is not None:
            if attn_mask.ndim == 3:
                attn_mask = attn_mask.unsqueeze(1) # (B, T, T) -> (B, 1, T, T)
            elif attn_mask.ndim != 4:
                raise ValueError(f"Unsupport attn_mask shape {attn_mask.shape}")
        
        hidden_states = self.apply_layers(inputs_embeds, attn_mask=attn_mask, input_pos=input_pos)
        final_embeddings = self.norm(hidden_states)
        if target_ids is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(final_embeddings)
            if target_weights is None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), ignore_index=ignore_index)
            else:
                loss = (target_weights.view(-1) * F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), reduction="none")).sum()
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(final_embeddings[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
        return logits, loss, hidden_states
        
    @torch.no_grad()
    def generate_embeddings(self, tokens):
        x = self.token_embeddings(tokens)
        x = self.apply_layers(x)
        x = self.norm(x)
        return x

    @torch.no_grad()
    def generate(self, tokens, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of tokens (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        if tokens.size(1) > self.config.block_size:
            logger.info(
                f"Input of {tokens.size(1)} tokens exceeds limit {self.config.block_size}. "
                f"Initial tokens will be dropped to fit."
            )
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            if tokens.size(1) > self.config.block_size:
                tokens = tokens[:, -self.config.block_size:]
            # forward the model to get the logits for the tokens
            logits = self(tokens)[0]
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            # append next token to the running sequence and continue
            tokens = torch.cat((tokens, next_token), dim=1)

        return tokens

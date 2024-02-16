"""
The full definition of the model is located in this file.
"""

import math
import inspect
import logging
from typing import Optional, Tuple, Union
from functools import reduce
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

_flash_attention_version = 1 if hasattr(torch.nn.functional, 'scaled_dot_product_attention') else 0
try:
    from flash_attn import flash_attn_func
    _flash_attention_version = 2
    _flash_attn_2_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)
except ImportError:
    _flash_attn_2_supports_window_size = False

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
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)
        
class RotaryEmbedding(torch.nn.Module):
    
    def __init__(self, dim, max_position_embeddings=2048, base=10000, scale=1.0):
        super().__init__()
        self.max_position_embeddings = max_position_embeddings
        inv_freq = 1.0 / (scale * (base ** (torch.arange(0, dim, 2).float() / dim)))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
    def _set_cos_sin_cache(self, dtype):
        t = torch.arange(self.max_position_embeddings, device=self.inv_freq.device, dtype=torch.int64).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)
        
    def forward(self, q, k, seq_len=None):
        # q,k: [bs, num_attention_heads, seq_len, head_size]
        if not hasattr(self, 'cos_cached'):
            self._set_cos_sin_cache(q.dtype)
            
        cos = self.cos_cached[:, :, :seq_len, ...].to(dtype=q.dtype)
        sin = self.sin_cached[:, :, :seq_len, ...].to(dtype=q.dtype)
        q_out = (q * cos) + (self.__rotate_half(q) * sin)
        k_out = (k * cos) + (self.__rotate_half(k) * sin)
        return q_out, k_out
    
    def __rotate_half(self, x):
        # Rotates half the hidden dims of the input
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

class FeedForward(nn.Module):

    def __init__(self, config: AllamoTransformerConfig):
        super().__init__()
        hidden_dim = int(2 * (4 * config.n_embd) / 3)
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        
        self.gate_proj = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.down_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.up_proj = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.act_fn  = nn.SiLU() # SwiGLU activation function
        self.dropout = nn.Dropout(config.dropout) if config.dropout != 0 else None
        self.gradient_checkpointing = config.gradient_checkpointing

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
        self.sliding_window = config.sliding_window if _flash_attn_2_supports_window_size else None
        self.gradient_checkpointing = config.gradient_checkpointing
        
        assert self.num_key_value_groups * self.num_kv_heads == self.num_heads
        
        # key, query, value projections for all heads
        self.q_proj = nn.Linear(config.n_embd, self.num_heads * self.head_size, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, self.num_kv_heads * self.head_size, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, self.num_kv_heads * self.head_size, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(self.num_heads * self.head_size, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout) if _flash_attention_version == 0 and config.dropout != 0 else None
        self.proj_dropout = nn.Dropout(config.dropout) if config.dropout != 0 else None
        
        if _flash_attention_version == 0:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(1, 1, config.block_size, config.block_size)))

    def forward(self, q_x: torch.Tensor, kv_x: torch.Tensor, rotary_emb: RotaryEmbedding):
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
        
        q, k = rotary_emb(q, k, T)
        
        if self.num_key_value_groups > 1:
            k = self.repeat_kv(k, self.num_key_value_groups)
            v = self.repeat_kv(v, self.num_key_value_groups)
        
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if _flash_attention_version == 2:
            # Flash Attention 2 requires (B, T, nh, hs)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            if self.sliding_window:            
                y = flash_attn_func(q, k, v, dropout_p=self.dropout if self.training else 0, causal=True, window_size=(self.sliding_window, self.sliding_window))
            else:
                y = flash_attn_func(q, k, v, dropout_p=self.dropout if self.training else 0, causal=True)
        elif _flash_attention_version == 1:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
            y = y.transpose(1, 2)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
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

    def forward(self, x: torch.Tensor, rotary_emb: RotaryEmbedding):
        x = x + self.attention(self.attention_norm(x), None, rotary_emb)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x

class AllamoTransformer(nn.Module):

    def __init__(self, config: AllamoTransformerConfig):
        super().__init__()
        self.logger = logging.getLogger('AllamoTransformer')
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        if config.head_size is None:
            assert config.n_embd % config.n_head == 0
            config.head_size = config.n_embd // config.n_head
            self.logger.info(f"defaulting to head_size={config.head_size} (n_embd / n_head)")
        if config.num_kv_heads is None:
            config.num_kv_heads = config.n_head
        self.logger.info(f"AllamoTransformerConfig: {config}")
        self.__log_flash_attention_version()

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.tok_drop = nn.Dropout(config.dropout) if config.dropout != 0 else None
        
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layer):
            self.layers.append(SelfAttentionBlock(config))
        
        self.norm = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self.rotary_emb = RotaryEmbedding(config.head_size, config.block_size*2, config.rope_freq_base, config.rope_freq_scale)

        # init all weights
        self.apply(self._init_weights)
        self._init_scaled_residual_projections(self)
        
        self.log_estimated_size()

    def __log_flash_attention_version(self):
        if _flash_attention_version == 2:
            self.logger.info("Using Flash Attention 2")
            if _flash_attn_2_supports_window_size and self.config.sliding_window:
                self.logger.info("Using sliding window")
        elif _flash_attention_version == 1:
            self.logger.info("Using scaled_dot_product_attention")
        elif _flash_attention_version == 0:
            self.logger.info("WARNING: using slow attention")
        else:
            raise Exception('Unsupported Flash Attention version!')

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
        self.logger.info(f"Model parameters: {model_params:.2f}M, Est. Size: {model_bytes:.3f}MB")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def _init_scaled_residual_projections(self, module):
        for pn, p in module.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.config.n_layer))
            
    def add_layer(self, new_layers=1):
        for _ in range(new_layers):
            layer = SelfAttentionBlock(self.config)
            layer.apply(self._init_weights)
            self._init_scaled_residual_projections(layer)
            self.layers.append(layer)
            self.config.n_layer += 1
            
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

    def apply_layers(self, x):
        for layer in self.layers:
            x = layer(x, self.rotary_emb)
        return x

    def forward(self, 
        input_ids: torch.Tensor, 
        labels: Optional[torch.Tensor] = None, 
        ignore_index: Optional[int] = -1,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.FloatTensor]]]:
        if inputs_embeds is not None:
            b, t, n_embd = inputs_embeds.size()
            assert t <= self.config.block_size, f"Cannot forward embeddings of length {t}, block size is only {self.config.block_size}"
        else:
            inputs_embeds = self.token_embeddings(input_ids)
        hidden_states = self.apply_layers(inputs_embeds)
        final_embeddings = self.norm(hidden_states)
        if labels is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(final_embeddings)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=ignore_index)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(final_embeddings[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
        return logits, loss, hidden_states

    def configure_optimizers(self, config, device_type):
        # start with all of the candidate parameters
        param_dict = {param_name: p for param_name, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {param_name: p for param_name, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        def is_weight_decay_forbidden(param_name):
            return param_name.endswith('.bias') or param_name.endswith('_norm.weight') or param_name == 'norm.weight'
        decay_params = [p for n, p in param_dict.items() if not is_weight_decay_forbidden(n)]
        nodecay_params = [p for n, p in param_dict.items() if is_weight_decay_forbidden(n)]
        optim_groups = [
            {'params': decay_params, 'weight_decay': config.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        self.logger.info(f"Decayed parameter tensors: {len(decay_params):,}, with {num_decay_params:,} parameters")
        self.logger.info(f"Non-decayed parameter tensors: {len(nodecay_params):,}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=config.learning_rate, betas=(config.beta1, config.beta2), **extra_args)
        self.logger.info(f"Using fused AdamW: {use_fused}")

        return optimizer
        
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
            self.logger.info(
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

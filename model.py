"""
The full definition of the model is located in this file.
"""

import math
import inspect
from typing import Optional, Tuple, Union
from functools import reduce
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class AllamoTransformerConfig:
    block_size: int = 1024
    vocab_size: int = 32000
    layers_multiplicator: int = 1
    n_layer: int = 12
    head_size: Union[None, int] = None
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears. False: a bit better and faster
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5
    flash_attention_version: int = 0

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
    
    def __init__(self, dim: int, max_seq_len=2048, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        t = torch.arange(max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        dtype = torch.get_default_dtype()
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)
        
    def forward(self, q, k):
        # q,k: [bs, num_attention_heads, seq_len, head_size]
        cos = self.cos_cached[:, :, :q.shape[-2], ...].to(dtype=q.dtype)
        sin = self.sin_cached[:, :, :q.shape[-2], ...].to(dtype=q.dtype)
        q_out = (q * cos) + (self.__rotate_half(q) * sin)
        k_out = (k * cos) + (self.__rotate_half(k) * sin)
        return q_out, k_out
    
    def __rotate_half(self, x):
        # Rotates half the hidden dims of the input
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

class Attention(nn.Module):

    def __init__(self, config: AllamoTransformerConfig):
        super().__init__()
        self.head_size = config.head_size
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash_attention_version = config.flash_attention_version
        
        # key, query, value projections for all heads
        self.q_proj = nn.Linear(self.n_embd, self.n_head * self.head_size, bias=config.bias)
        self.k_proj = nn.Linear(self.n_embd, self.n_head * self.head_size, bias=config.bias)
        self.v_proj = nn.Linear(self.n_embd, self.n_head * self.head_size, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(self.n_head * self.head_size, self.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout) if self.flash_attention_version == 0 and config.dropout != 0 else None
        self.proj_dropout = nn.Dropout(config.dropout) if config.dropout != 0 else None
        
        self.rotary_emb = RotaryEmbedding(config.head_size, config.block_size*2)

        if self.flash_attention_version == 0:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(1, 1, config.block_size, config.block_size)))

    def forward(self, q_x: torch.Tensor, kv_x: Optional[torch.Tensor] = None):
        # notation:
        # B  | batch
        # T  | time-step (sequence length)
        # C  | embeddings size
        # hs | head size
        # nh | number of heads
        B, T, C = q_x.size()
        
        if kv_x is None:
            kv_x = q_x # self attention
        
        q = self.q_proj(q_x).view(B, T, self.n_head, self.head_size).transpose(1, 2) # (B, nh, T, hs)
        k = self.k_proj(kv_x).view(B, T, self.n_head, self.head_size).transpose(1, 2) # (B, nh, T, hs)
        v = self.v_proj(kv_x).view(B, T, self.n_head, self.head_size).transpose(1, 2) # (B, nh, T, hs)
        
        q, k = self.rotary_emb(q, k)
        
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash_attention_version == 2:
            # Flash Attention 2 requires (B, T, nh, hs)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            y = flash_attn_func(q, k, v, dropout_p=self.dropout if self.training else 0, causal=True).transpose(1, 2)
        elif self.flash_attention_version == 1:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            if self.attn_dropout is not None:
                att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_size) # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y) # (B, T, nh * hs) -> (B, T, C)
        if self.proj_dropout is not None:
            y = self.proj_dropout(y)
        return y


class FeedForward(nn.Module):

    def __init__(self, config: AllamoTransformerConfig):
        super().__init__()
        dim = config.n_embd
        hidden_dim = int(2 * (4 * config.n_embd) / 3)
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=config.bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=config.bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=config.bias)
        self.act_fn  = nn.SiLU() # SwiGLU activation function
        self.dropout = nn.Dropout(config.dropout) if config.dropout != 0 else None

    def forward(self, x):
        x = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        if self.dropout is not None:
            x = self.dropout(x)
        return x
        
class SelfAttentionBlock(nn.Module):

    def __init__(self, config: AllamoTransformerConfig):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.n_embd, eps=config.norm_eps)

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x

class AllamoTransformer(nn.Module):

    def __init__(self, config: AllamoTransformerConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        config.flash_attention_version = self.__detect_flash_attention_version()
        self.config = config
        self.max_seq_len = config.block_size
        if config.head_size is None:
            assert config.n_embd % config.n_head == 0
            config.head_size = config.n_embd // config.n_head
            print(f"defaulting to head_size={config.head_size} (n_embd / n_head)")
        print(f"AllamoTransformerConfig: {config}")

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.tok_drop = nn.Dropout(config.dropout) if config.dropout != 0 else None
        
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layer):
            self.layers.append(SelfAttentionBlock(config))
        
        self.norm = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        model_params, model_bytes = self.estimate_size()
        model_params /= 1e6
        model_bytes /= 1024**2
        print(f"Model parameters: {model_params:.2f}M Est. Size: {model_bytes:.3f}MB")

    def __detect_flash_attention_version(self):
        try:
            from flash_attn import flash_attn_func
        except ImportError:
            print("Flash Attention 2 is not installed.")
        else:
            print("Using Flash Attention 2")
            return 2
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            print("Using Flash Attention 1")
            return 1
        else:
            print("WARNING: using slow attention")
            return 0

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

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def apply_layers(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def embeddings(self, input_ids, layers_multiplicator=None):
        b, t = input_ids.size()
        assert t <= self.max_seq_len, f"Cannot forward sequence of length {t}, block size is only {self.max_seq_len}"

        x = self.tok_embeddings(input_ids) # token embeddings of shape (b, t, n_embd)
        if self.tok_drop is not None:
            x = self.tok_drop(x)
            
        layers_multiplicator = layers_multiplicator if layers_multiplicator is not None and layers_multiplicator >= 1 else self.config.layers_multiplicator
        if layers_multiplicator > 1:
            for m in range(layers_multiplicator):
                x = self.apply_layers(x)
        else:
            x = self.apply_layers(x)
        
        x = self.norm(x)
        return x

    def forward(self, 
        input_ids: torch.Tensor, 
        labels: Optional[torch.Tensor] = None, 
        ignore_index: Optional[int] = -1,
        layers_multiplicator: Optional[int] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        final_embeddings = self.embeddings(input_ids, layers_multiplicator)
        if labels is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(final_embeddings)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=ignore_index)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(final_embeddings[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {param_name: p for param_name, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {param_name: p for param_name, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"Decayed parameter tensors: {len(decay_params):,}, with {num_decay_params:,} parameters")
        print(f"Non-decayed parameter tensors: {len(nodecay_params):,}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"Using fused AdamW: {use_fused}")

        return optimizer
        
    @torch.no_grad()
    def generate_embeddings(self, tokens, layers_multiplicator=None):
        return self.embeddings(tokens, layers_multiplicator=layers_multiplicator)

    @torch.no_grad()
    def generate(self, tokens, max_new_tokens, temperature=1.0, top_k=None, layers_multiplicator=None):
        """
        Take a conditioning sequence of tokens (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            if tokens.size(1) > self.max_seq_len:
                tokens = tokens[:, -self.max_seq_len:]
            # forward the model to get the logits for the tokens
            logits, _ = self(tokens, layers_multiplicator=layers_multiplicator)
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

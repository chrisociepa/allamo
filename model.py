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
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    head_size: Union[None, int] = None
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears. False: a bit better and faster
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

class RMSNorm(torch.nn.Module):
    """RMSNorm normalizing function, introduced by Zhang and Sennrich (2019)"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
        
class RotaryEmbedding(torch.nn.Module):
    
    def __init__(self, dim: int, max_position_embeddings=2048, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[None, None, :, :]
        self.sin_cached = emb.sin()[None, None, :, :]

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype, device=x.device),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype, device=x.device),
        )

def rotate_half(x):
    # Rotates half the hidden dims of the input
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    cos = cos[..., offset : q.shape[-2] + offset, :]
    sin = sin[..., offset : q.shape[-2] + offset, :]
    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out, k_out

class Attention(nn.Module):

    def __init__(self, config: AllamoTransformerConfig):
        super().__init__()
        self.head_size = config.head_size
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # Flash Attention is supported only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        
        # key, query, value projections for all heads
        self.q_proj = nn.Linear(self.n_embd, self.n_head * self.head_size, bias=config.bias)
        self.k_proj = nn.Linear(self.n_embd, self.n_head * self.head_size, bias=config.bias)
        self.v_proj = nn.Linear(self.n_embd, self.n_head * self.head_size, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(self.n_head * self.head_size, self.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout) if self.flash else None
        self.proj_dropout = nn.Dropout(config.dropout)
        
        self.rotary_emb = RotaryEmbedding(config.head_size, config.block_size*2)

        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(1, 1, config.block_size, config.block_size)))

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor]):
        # notation:
        # B  | batch
        # T  | time-step (sequence length)
        # C  | embeddings size
        # hs | head size
        # nh | number of heads
        B, T, C = x.size()
        
        if attention_mask is not None and attention_mask.size() != (B, T):
            raise ValueError(
                f"Attention mask should be of size {(B, T)}, but is {attention_mask.size()}"
            )

        q = self.q_proj(x).view(B, T, self.n_head, self.head_size).transpose(1, 2) # (B, nh, T, hs)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_size).transpose(1, 2) # (B, nh, T, hs)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_size).transpose(1, 2) # (B, nh, T, hs)
        
        cos, sin = self.rotary_emb(v, seq_len=k.shape[-2])
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        is_causal = attention_mask is None
        # FIXME: it is still now working as expected. I recommend not using attention_mask
        expanded_attention_mask = attention_mask[:, None, None, :].expand(B, 1, T, T).tril().to(torch.bool) if not is_causal else None

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=expanded_attention_mask, dropout_p=self.dropout, is_causal=is_causal)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill((self.bias[:,:,:T,:T] == 0 if is_causal else expanded_attention_mask[:,:,:T,:T]), float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_size) # re-assemble all head outputs side by side

        # output projection
        y = self.proj_dropout(self.c_proj(y)) # (B, T, nh * hs) -> (B, T, C)
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
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        x = self.dropout(x)
        return x
        
class TransformerBlock(nn.Module):

    def __init__(self, layer_id: int, config: AllamoTransformerConfig):
        super().__init__()
        self.layer_id = layer_id
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.n_embd, eps=config.norm_eps)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor]):
        x = x + self.attention(self.attention_norm(x), attention_mask)
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
            print(f"defaulting to head_size={config.head_size} (n_embd / n_head)")
            
        print(f"AllamoTransformerConfig: {config}")

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.tok_drop = nn.Dropout(config.dropout)
        
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layer):
            self.layers.append(TransformerBlock(layer_id, config))
        
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

    def embeddings(self, tokens, attention_mask):
        b, t = tokens.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        tok_emb = self.tok_embeddings(tokens) # token embeddings of shape (b, t, n_embd)
        
        x = self.tok_drop(tok_emb)
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.norm(x)
        return x

    def forward(self, 
        input_ids: torch.Tensor, 
        labels: Optional[torch.Tensor] = None, 
        attention_mask: Optional[torch.Tensor] = None, 
        ignore_index: Optional[int] = -1
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        final_embeddings = self.embeddings(input_ids, attention_mask)
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
    def generate_embeddings(self, tokens):
        return self.embeddings(tokens)

    @torch.no_grad()
    def generate(self, tokens, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of tokens (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            if tokens.size(1) > self.config.block_size:
                tokens = tokens[:, -self.config.block_size:]
            # forward the model to get the logits for the tokens
            logits, _ = self(tokens)
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

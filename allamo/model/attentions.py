import inspect
import itertools
import math
import torch
from functools import lru_cache
from torch.nn import functional as F
from typing import Optional

from allamo.configuration import AllamoConfiguration
from allamo.logging import logger

_flex_attn_impl_module = None

def _causal_mask_fn(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

def _make_sliding_window_fn(window_size: int):
    def mask_fn(b, h, q_idx, kv_idx):
        return (q_idx - kv_idx <= window_size) & (kv_idx - q_idx <= window_size)
    return mask_fn

def _make_diffusion_mask_fn(T: int, q_len: int, anchor_pos: torch.Tensor):
    """
    anchor_pos: (B, A) local ctx indices each noise block corresponds to.
    ctx_ok: noise block t can attend to ctx positions 0..anchor_pos[b,t] (inclusive).
    noise_ok: bidirectional within own block only.
    """
    def diffusion_mask(b, h, q_idx, kv_idx):
        t = q_idx // q_len # Which anchor block this query token belongs to

        # anchor_pos is in target_ids space, so +1 to get the corresponding input_ids index,
        # which is the ctx segment's local index space (kv_idx < T walks input_ids).
        ctx_ok   = (kv_idx < T) & (kv_idx <= anchor_pos[b, t] + 1)

        noise_start = T + t * q_len
        noise_end   = T + (t + 1) * q_len
        noise_ok = (kv_idx >= noise_start) & (kv_idx < noise_end)

        return ctx_ok | noise_ok

    return diffusion_mask

def _make_diffusion_mask_with_docs_fn(T: int, q_len: int,
                                      anchor_pos: torch.Tensor,
                                      input_pos: torch.Tensor,
                                      attn_mask: torch.Tensor):
    """
    Extends _make_diffusion_mask_fn with document isolation.

    anchor_pos: (B, A) local ctx indices each noise block corresponds to
    input_pos:  (B, T) absolute sequence positions of ctx tokens
    attn_mask:  (B, T) document ids aligned with ctx (input_ids space)

    ctx_ok: causal via absolute positions + same document as anchor.
    noise_ok: own block only + same document as anchor.
    """
    def diffusion_mask(b, h, q_idx, kv_idx):
        t = q_idx // q_len
        anchor_idx = anchor_pos[b, t] + 1 # local ctx index of this block's anchor
        anchor_abs = input_pos[b, anchor_idx] # absolute position of anchor
        anchor_doc = attn_mask[b, anchor_idx] # document id of anchor

        # ctx: absolute position must be <= anchor's + same document
        ctx_abs_ok = (kv_idx < T) & (input_pos[b, kv_idx] <= anchor_abs)
        ctx_doc_ok = attn_mask[b, kv_idx] == anchor_doc
        ctx_ok     = ctx_abs_ok & ctx_doc_ok

        # noise: own block only + same document as kv noise block's anchor
        noise_start  = T + t * q_len
        noise_end    = T + (t + 1) * q_len
        noise_pos_ok = (kv_idx >= noise_start) & (kv_idx < noise_end)
        kv_t         = (kv_idx - T) // q_len
        noise_doc_ok = attn_mask[b, anchor_pos[b, kv_t] + 1] == anchor_doc
        noise_ok     = noise_pos_ok & noise_doc_ok

        return ctx_ok | noise_ok

    return diffusion_mask

@lru_cache(maxsize=32)
def _create_block_mask_cached(mask, b, h, q_len, kv_len, device="cuda"):
    assert _flex_attn_impl_module is not None, "FlexAttention module not initialized"
    return _flex_attn_impl_module.create_block_mask(
        mask, b, h, q_len, kv_len, device=device, _compile=True
    )

@lru_cache(maxsize=32)
def _get_causal_mask_mod(sliding_window=None):
    if sliding_window is None:
        return _causal_mask_fn
    assert _flex_attn_impl_module is not None, "FlexAttention module not initialized"
    return _flex_attn_impl_module.and_masks(_causal_mask_fn, _make_sliding_window_fn(sliding_window))

class AttentionVersion(torch.nn.Module):
    """
    Versions:
    0 - eager
    1 - SDPA
    2 - FA2
    3 - FA3
    4 - xformers
    5 - FlexAttention
    """
    
    def __init__(self):
        super().__init__()
        self.attn_impl_module = None
        self.causal_mask = None
        self.enable_flex_attn() # FlexAttention is required to make DFlash working
        
    def configure(self, config: AllamoConfiguration):
        if config.attention_implementation:
            if config.attention_implementation == 'sdpa':
                self.enable_sdpa()
            elif config.attention_implementation == 'fa2':
                self.enable_flash_attn_2()
            elif config.attention_implementation == 'fa3':
                self.enable_flash_attn_3()
                if config.dropout > 0:
                    logger.warning("Flash Attention 3 does not support dropout")
            elif config.attention_implementation == 'xformers':
                self.enable_xformers()
            elif config.attention_implementation == 'flex':
                self.enable_flex_attn()
            elif config.attention_implementation == 'eager':
                self.force_eager()
        
    def force_eager(self):
        self.version = 'eager'
        self.flash_attn_supports_window_size = False

    def enable_sdpa(self):
        self.version = 'sdpa' if hasattr(torch.nn.functional, 'scaled_dot_product_attention') else 'eager'
        self.flash_attn_supports_window_size = False
    
    def enable_flash_attn_2(self):
        self.version = 'fa2'
        try:            
            import flash_attn
            self.flash_attn_supports_window_size = "window_size" in list(inspect.signature(flash_attn.flash_attn_func).parameters)
            self.attn_impl_module = flash_attn
        except ImportError:
            self.enable_sdpa()
            logger.warning("Flash Attention 2 is not available, falling back to scaled_dot_product_attention!")
        
    def enable_flash_attn_3(self):
        self.version = 'fa3'
        try:
            import flash_attn_interface
            self.flash_attn_supports_window_size = "window_size" in list(inspect.signature(flash_attn_interface.flash_attn_func).parameters)
            self.attn_impl_module = flash_attn_interface
        except ImportError:
            self.enable_sdpa()
            logger.warning("Flash Attention 3 is not available, falling back to scaled_dot_product_attention!")
        
    def enable_xformers(self):
        self.version = 'xformers'
        try:
            import xformers.ops as xops
            self.attn_impl_module = xops
        except ImportError:
            self.enable_sdpa()
            logger.warning("xformers is not available, falling back to scaled_dot_product_attention!")
        self.flash_attn_supports_window_size = False # TODO: check xops.fmha.attn_bias.LowerTriangularFromBottomRightLocalAttentionMask

    def enable_flex_attn(self):
        global _flex_attn_impl_module
        self.version = 'flex'
        try:
            import torch.nn.attention.flex_attention as flexatt
            self.attn_impl_module = flexatt
            _flex_attn_impl_module = flexatt

            compiled_flex_attention = torch.compile(flexatt.flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")
            @torch.compiler.disable(recursive=False)
            def compiled_flex_attention_fn(
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                block_mask: flexatt.BlockMask,
            ) -> torch.Tensor:
                return compiled_flex_attention(q, k, v, block_mask=block_mask)
            _flex_attn_impl_module.compiled_flex_attention_fn = compiled_flex_attention_fn
            _flex_attn_impl_module.compiled_create_block_mask = torch.compile(flexatt.create_block_mask, dynamic=False, mode="max-autotune-no-cudagraphs")

        except ImportError:
            self.enable_sdpa()
            logger.warning("FlexAttention is not available, falling back to scaled_dot_product_attention!")
        self.flash_attn_supports_window_size = False

    def extra_repr(self) -> str:
        return self.version
    
    def log_version(self, sliding_window):
        if self.version == 'eager':
            logger.info("WARNING: using slow eager attention")
        elif self.version == 'sdpa':
            logger.info("Using scaled_dot_product_attention")
        elif self.version == 'fa2':
            logger.info("Using Flash Attention 2")
            if self.flash_attn_supports_window_size and sliding_window:
                logger.info("Using sliding window")
        elif self.version == 'fa3':
            logger.info("Using Flash Attention 3")
            if self.flash_attn_supports_window_size and sliding_window:
                logger.info("Using sliding window")
        elif self.version == 'xformers':
            logger.info("Using xformers memory_efficient_attention")
        elif self.version == 'flex':
            logger.info("Using FlexAttention")
            if sliding_window:
                logger.info("Using sliding window")
        else:
            raise Exception('Unsupported attention version!')
    
    def apply(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor], seq_lens: Optional[torch.Tensor], dropout: float = 0.0, sliding_window: int = None) -> torch.Tensor:
        if self.version == 'eager':
            return self.eager(q, k, v, attn_mask, dropout)
        elif self.version == 'sdpa':
            return self.sdpa(q, k, v, attn_mask, dropout)
        elif self.version == 'fa2':
            return self.fa2(q, k, v, attn_mask, dropout, sliding_window)
        elif self.version == 'fa3':
            return self.fa3(q, k, v, attn_mask, sliding_window)
        elif self.version == 'xformers':
            return self.xformers(q, k, v, attn_mask, seq_lens, dropout)
        elif self.version == 'flex':
            return self.flex_attention(q, k, v, attn_mask, sliding_window)
        else:
            raise Exception('Unsupported attention version!')
    
    def eager(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor], dropout: float = 0.0) -> torch.Tensor:
        # eager implementation of attention
        seq_len = q.size(2)
        if self.causal_mask is None or self.causal_mask.shape[-1] < seq_len:
            with torch.device(q.device):
                self.causal_mask = torch.tril(torch.ones(1, 1, seq_len, seq_len))
        
        scale_factor = 1.0 / math.sqrt(q.size(-1))
        att = (q @ k.transpose(-2, -1)) * scale_factor
        if attn_mask is None:
            att = att.masked_fill(self.causal_mask[:,:,:seq_len,:seq_len] == 0, float('-inf'))
        else:
            att = att.masked_fill(attn_mask.logical_not(), float('-inf'))
        att = F.softmax(att, dim=-1)
        if dropout > 0:
            F.dropout(att, dropout, training=self.training, inplace=True)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        return y.transpose(1, 2)
    
    def sdpa(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor], dropout: float = 0.0) -> torch.Tensor:
        # efficient attention using SDPA: (B, nh, T, hs) -> (B, nh, T, hs)
        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout if self.training else 0,
            is_causal=attn_mask is None,
        )
        return y.transpose(1, 2)

    def fa2(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor], dropout: float = 0.0, sliding_window: int = None) -> torch.Tensor:
        # Flash Attention 2: (B, T, nh, hs) -> (B, T, nh, hs)
        if attn_mask is not None:
            raise ValueError(f"Custom attention mask is not supported for Flash Attention 2")
        dropout_p=dropout if self.training else 0
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        if sliding_window:
            y = self.attn_impl_module.flash_attn_func(q, k, v, dropout_p=dropout_p, causal=True, window_size=(sliding_window, sliding_window))
        else:
            y = self.attn_impl_module.flash_attn_func(q, k, v, dropout_p=dropout_p, causal=True)
        return y

    def fa3(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor], sliding_window: int = None) -> torch.Tensor:
        # Flash Attention 3: (B, T, nh, hs) -> (B, T, nh, hs)
        if attn_mask is not None:
            raise ValueError(f"Custom attention mask is not supported for Flash Attention 3")
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        if sliding_window:
            y = self.attn_impl_module.flash_attn_func(q, k, v, causal=True, window_size=(sliding_window, sliding_window))
        else:
            y = self.attn_impl_module.flash_attn_func(q, k, v, causal=True)
        return y

    def xformers(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor], seq_lens: Optional[torch.Tensor], dropout: float = 0.0) -> torch.Tensor:
        dropout_p = dropout if self.training else 0
        B, _, T, _ = q.size() # (B, nh, T, hs)
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        if attn_mask is None:
            if seq_lens is None:
                attn_mask = self.attn_impl_module.LowerTriangularMask()
            else:
                # FIXME: it is not working with torch.compile!
                seq_lens = list(itertools.chain(*seq_lens)) if seq_lens else None
                q = q.view(1, B * T, self.num_heads, self.head_size)
                k = k.view(1, B * T, self.num_heads, self.head_size)
                v = v.view(1, B * T, self.num_heads, self.head_size)
                attn_mask = self.attn_impl_module.fmha.attn_bias.BlockDiagonalCausalMask.from_seqlens(seq_lens, device=q.device)
        else:
            # FIXME: it is not working with torch.compile!
            if attn_mask.dtype == torch.bool:
                attn_mask = torch.where(attn_mask, 0.0, -torch.inf)
            attn_mask = attn_mask.to(device=q.device, dtype=q.dtype)
            attn_mask = torch.broadcast_to(attn_mask, (B, self.num_heads, T, T))
            attn_mask.requires_grad_(False)
        
        # xformers: (B, T, nh, hs) -> (B, T, nh, hs)
        y = self.attn_impl_module.memory_efficient_attention(
            q,
            k,
            v,
            attn_bias=attn_mask,
            p=dropout_p
        )
        if attn_mask is not None:
            y = y.view(B, T, self.num_heads, self.head_size)
        
        return y

    def flex_attention(self, q, k, v, attn_mask=None, sliding_window=None):
        B, _, T, _ = q.size()
        if attn_mask is None:
            mask_mod = _get_causal_mask_mod(sliding_window)
            block_mask = _create_block_mask_cached(mask_mod, b=None, h=None, q_len=T, kv_len=T, device=str(q.device))
        else:
            def document_mask(b, h, q_idx, kv_idx):
                return attn_mask[b, q_idx] == attn_mask[b, kv_idx]
            mask_mod = _flex_attn_impl_module.and_masks(_causal_mask_fn, document_mask)
            if sliding_window:
                mask_mod = _flex_attn_impl_module.and_masks(mask_mod, _make_sliding_window_fn(sliding_window))
            block_mask = _create_block_mask_cached(mask_mod, b=B, h=None, q_len=T, kv_len=T, device=str(q.device))

        # Flex attention: (B, nh, T, hs) -> (B, nh, T, hs)
        y = _flex_attn_impl_module.compiled_flex_attention_fn(q, k, v, block_mask=block_mask)
        return y.transpose(1, 2)

    def flex_attention_diffusion(self, q, k, v, T, q_len,
                                  anchor_pos: torch.Tensor,
                                  input_pos: Optional[torch.Tensor] = None,
                                  attn_mask: Optional[torch.Tensor] = None,
                                  sliding_window=None):
        B, _, total_q, _ = q.size()
        A = anchor_pos.size(1)
        kv_len = k.size(2)

        assert total_q == A * q_len
        assert kv_len  == T + total_q

        if input_pos is not None:
            mask_mod = _make_diffusion_mask_with_docs_fn(T, q_len, anchor_pos, input_pos, attn_mask)
        else:
            mask_mod = _make_diffusion_mask_fn(T, q_len, anchor_pos)

        if sliding_window is not None:
            mask_mod = _flex_attn_impl_module.and_masks(mask_mod, _make_sliding_window_fn(sliding_window))

        # block_mask cannot be cached: mask_mod closes over anchor_pos (and optionally
        # input_pos/attn_mask) whose contents change every batch. Although shapes are
        # fixed during training, lru_cache keys on the closure object itself (a new
        # object each call), so the cache would never hit. compiled_create_block_mask
        # handles this correctly by compiling the computation once and re-executing
        # the kernel with fresh tensor values each call without recompilation.
        block_mask = _flex_attn_impl_module.compiled_create_block_mask(
            mask_mod, B=B, H=None, Q_LEN=total_q, KV_LEN=kv_len, device=str(q.device)
        )

        # Flex attention: (B, nh, T, hs) -> (B, nh, T, hs)
        y = _flex_attn_impl_module.compiled_flex_attention_fn(q, k, v, block_mask=block_mask)
        return y.transpose(1, 2)

attention_version = AttentionVersion()

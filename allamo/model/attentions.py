import inspect
import itertools
import math
import torch
from functools import lru_cache
from torch.nn import functional as F
from typing import Optional

from allamo.configuration import AllamoConfiguration
from allamo.logging import logger

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
        self.enable_sdpa()
        
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
        self.version = 'flex'
        try:
            import torch.nn.attention.flex_attention as flexatt
            self.attn_impl_module = flexatt

            compiled_flex_attention = torch.compile(flexatt.flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")
            @torch.compiler.disable(recursive=False)
            def compiled_flex_attention_fn(
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                block_mask: flexatt.BlockMask,
            ) -> torch.Tensor:
                return compiled_flex_attention(q, k, v, block_mask=block_mask)
            self.attn_impl_module.compiled_flex_attention_fn = compiled_flex_attention_fn

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
    
    def apply(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor], dropout: float = 0.0, sliding_window: int = None) -> torch.Tensor:
        if self.version == 'eager':
            return self.eager(q, k, v, attn_mask, dropout)
        elif self.version == 'sdpa':
            return self.sdpa(q, k, v, attn_mask, dropout)
        elif self.version == 'fa2':
            return self.fa2(q, k, v, attn_mask, dropout, sliding_window)
        elif self.version == 'fa3':
            return self.fa3(q, k, v, attn_mask, sliding_window)
        elif self.version == 'xformers':
            return self.xformers(q, k, v, attn_mask, dropout)
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

    def xformers(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor], dropout: float = 0.0) -> torch.Tensor:
        dropout_p = dropout if self.training else 0
        B, T, C = q.size()
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

    def flex_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor], sliding_window: int = None) -> torch.Tensor:
        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        def sliding_window_mask(window_size: int):
            def mask_fn(b, h, q_idx, kv_idx):
                return (q_idx - kv_idx <= window_size) & (kv_idx - q_idx <= window_size)
            return mask_fn
        
        def document_mask(b, h, q_idx, kv_idx):
            return attn_mask[b, q_idx] == attn_mask[b, kv_idx]
                    
        @lru_cache
        def create_block_mask_cached(mask, b, h, q_len, kv_len, device="cuda"):
            return attention_version.attn_impl_module.create_block_mask(mask, b, h, q_len, kv_len, device=device, _compile=True)
        
        B, T, C = q.size()
        block_mask = None
        if attn_mask is None:
            mask_mod = attention_version.attn_impl_module.and_masks(causal_mask, sliding_window_mask(sliding_window)) if sliding_window else causal_mask
            block_mask = create_block_mask_cached(mask_mod, b=None, h=None, q_len=T, kv_len=T, device=q.device)                    
        else:
            mask_mod = attention_version.attn_impl_module.and_masks(causal_mask, document_mask)
            if sliding_window:
                mask_mod = attention_version.attn_impl_module.and_masks(mask_mod, sliding_window_mask(sliding_window))
            block_mask = create_block_mask_cached(mask_mod, b=B, h=None, q_len=T, kv_len=T, device=q.device)

        # Flex attention: (B, nh, T, hs) -> (B, nh, T, hs)
        y = attention_version.attn_impl_module.compiled_flex_attention_fn(q, k, v, block_mask=block_mask)
        return y.transpose(1, 2)
    
attention_version = AttentionVersion()

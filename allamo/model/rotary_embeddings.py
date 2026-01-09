import math
import torch
from typing import  Optional, Dict, Any, Tuple

def _compute_original_rope_parameters(rotary_dim: int, max_position: int, rope_freq_base: float, rope_scaling: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, float]:
    inv_freq = 1.0 / (rope_freq_base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
    return inv_freq, None

def _compute_linear_scaling_rope_parameters(rotary_dim: int, max_position: int, rope_freq_base: float, rope_scaling: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, float]:
    factor = rope_scaling["factor"]
    inv_freq, _ = _compute_original_rope_parameters(rotary_dim, max_position, rope_freq_base)
    inv_freq /= factor
    return inv_freq, None

# Inverse dim formula to find dim based on number of rotations
def _yarn_find_correction_dim(num_rotations: int, dim: int, base: float, max_position_embeddings: int) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

# Find dim range bounds based on rotations
def _yarn_find_correction_range(low_rot: int, high_rot: int, dim: int, base: float, max_position_embeddings: int) -> Tuple[int, int]:
    low = math.floor(_yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case

def _yarn_linear_ramp_mask(low: float, high: float, dim: int, dtype: torch.dtype) -> torch.Tensor:
    if low == high:
        high += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=dtype) - low) / (high - low)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func

def _yarn_get_mscale(scale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0

def _compute_yarn_rope_parameters(rotary_dim: int, max_position: int, rope_freq_base: float, rope_scaling: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, float]:
    scaling_factor = rope_scaling["factor"]
    original_max_position = rope_scaling.get("original_max_position_embeddings")

    # beta_fast/beta_slow: as suggested in the paper, default to 32/1 (correspondingly)
    beta_fast = rope_scaling.get("beta_fast") or 32
    beta_slow = rope_scaling.get("beta_slow") or 1
    
    # Get n-d magnitude scaling corrected for interpolation
    mscale = _yarn_get_mscale(scaling_factor)

    if original_max_position is None:
        assert max_position % scaling_factor == 0, f"max_position ({max_position}) must be a multiple of scaling_factor ({scaling_factor})"
        original_max_position = int(max_position / scaling_factor)
    else:
        assert max_position % original_max_position == 0, f"max_position ({max_position}) must be a multiple of original_max_position ({original_max_position})"
        assert max_position // original_max_position == scaling_factor, f"scaling_factor ({scaling_factor}) must be equal to max_position / original_max_position ({max_position // original_max_position})"

    pos_freqs = rope_freq_base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

    low, high = _yarn_find_correction_range(beta_fast, beta_slow, rotary_dim, rope_freq_base, original_max_position)

    # Get n-dimensional rotational scaling corrected for extrapolation
    inv_freq_extrapolation_factor = 1 - _yarn_linear_ramp_mask(low, high, rotary_dim // 2, dtype=torch.float)
    inv_freq = (
        inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
        + inv_freq_extrapolation * inv_freq_extrapolation_factor
    )
    return inv_freq, mscale

def _compute_llama3_rope_parameters(rotary_dim: int, max_position: int, rope_freq_base: float, rope_scaling: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, float]:
    factor = rope_scaling["factor"]
    low_freq_factor = rope_scaling["low_freq_factor"]
    high_freq_factor = rope_scaling["high_freq_factor"]
    old_context_len = rope_scaling["original_max_position_embeddings"]
    inv_freq, _ = _compute_original_rope_parameters(rotary_dim, max_position, rope_freq_base)

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    return inv_freq, None

class RotaryEmbedding(torch.nn.Module):
    
    def __init__(self, rotary_dim: int, max_position: int, rope_freq_base: float, rope_scaling: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.rotary_dim = rotary_dim
        self.max_position = max_position
        self.rope_freq_base = rope_freq_base
        self.rope_scaling = rope_scaling

        if self.rope_scaling is not None and "rope_type" in self.rope_scaling:
            self.rope_type = self.rope_scaling["rope_type"]
        else:
            self.rope_type = "original"
        
        if self.rope_type == "original":
            self.rope_init_fn = _compute_original_rope_parameters
        elif self.rope_type == "linear":
            self.rope_init_fn = _compute_linear_scaling_rope_parameters
        elif self.rope_type == "yarn":
            self.rope_init_fn = _compute_yarn_rope_parameters
        elif self.rope_type == "llama3":
            self.rope_init_fn = _compute_llama3_rope_parameters
        else:
            raise ValueError(f"Invalid rope type: {self.rope_type}")

        self._rope_init()
        
    # Define reset_parameters for FSDP initialization
    def reset_parameters(self):
        self._rope_init()

    def _rope_init(self):
        inv_freq, self.attention_scaling = self.rope_init_fn(rotary_dim=self.rotary_dim, max_position=self.max_position, rope_freq_base=self.rope_freq_base, rope_scaling=self.rope_scaling)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.build_rope_cache(self.max_position)
        
    def build_rope_cache(self, max_position: int) -> None:
        t = torch.arange(max_position, device=self.inv_freq.device, dtype=torch.int64).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq).float()
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling if self.attention_scaling is not None else emb.cos()
        sin = emb.sin() * self.attention_scaling if self.attention_scaling is not None else emb.sin()
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)
    
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

    def extra_repr(self):
        return f'rope_type={self.rope_type}'

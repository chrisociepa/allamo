from typing import Optional, Callable
from typing_extensions import Unpack, Tuple
import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    Qwen3Config,
    Qwen3PreTrainedModel,
    Qwen3MLP,
    GradientCheckpointingLayer,
    FlashAttentionKwargs,
    rotate_half,
    eager_attention_forward,
    ALL_ATTENTION_FUNCTIONS,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache

dflash_config_class = Qwen3Config

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Qwen3DFlashAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False  
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        attn_fn: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attn_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class DFlash2DynamicConv(nn.Module):
    """
    HF-side mirror of allamo.model.dflash.model.DFlash2DynamicConv. Parameter
    names and shapes (base_kernel, kernel_projection) match the checkpoints
    published for DFlash 2.

    Grouped dynamic depthwise convolution, wrapped around one decoder sublayer
    (attention or MLP): lets a block position see its immediate predecessor in
    the same block without another pass through the backbone. base_kernel is
    at full channel resolution; the dynamic correction from kernel_projection
    is shared within each conv_group_size-channel group. Taps never cross a
    block boundary. At identity init (base_kernel[0]=1, everything else 0)
    this module is a no-op, which is what makes DFlash1 -> DFlash2 warm-starting
    exact; a trained checkpoint carries real (non-identity) values here.

    Only conv_kernel_size == 2 (self + immediate predecessor) is implemented.
    """

    def __init__(self, config):
        super().__init__()
        self.n_embd = config.hidden_size
        self.draft_block_size = config.block_size
        self.taps = config.dflash_config.get("conv_kernel_size", 2)
        assert self.taps == 2, "only conv_kernel_size == 2 is implemented"
        self.group_size = config.dflash_config.get("conv_group_size", 1)
        assert self.n_embd % self.group_size == 0, "hidden_size must be divisible by conv_group_size"
        self.num_groups = self.n_embd // self.group_size
        self.base_kernel = nn.Parameter(torch.zeros(self.taps, self.n_embd))
        self.kernel_projection = nn.Linear(self.n_embd, self.taps * self.num_groups, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, A * draft_block_size, C)
        B, QT, C = x.shape
        block = self.draft_block_size
        A = QT // block

        x_blk = x.view(B, A, block, self.num_groups, self.group_size)
        x_prev = torch.zeros_like(x_blk)
        x_prev[:, :, 1:] = x_blk[:, :, :-1]  # shift within block; zero at block start

        delta = self.kernel_projection(x).view(B, A, block, self.taps, self.num_groups, 1)
        base = self.base_kernel.view(1, 1, 1, self.taps, self.num_groups, self.group_size)
        kernel = base + delta

        out = kernel[:, :, :, 0] * x_blk + kernel[:, :, :, 1] * x_prev
        return out.reshape(B, QT, C)


class DFlash2CandidateSelector(nn.Module):
    """
    HF-side mirror of allamo.model.dflash.model.DFlash2CandidateSelector.
    Parameter names (hidden_projection, predecessor_codebook,
    successor_codebook) match published DFlash 2 checkpoints.

    Exposed as DFlashDraftModel.candidate_selector for the serving engine to
    call directly during its path-walk decode step: score(candidate) = the
    candidate's own DFlash logit + a low-rank bilinear compatibility term
    against a chosen predecessor. The forward here only scores one predecessor
    against a set of candidates at a time (the same primitive used for
    teacher-forced training); walking the best path across a whole block using
    several live predecessor candidates is the serving engine's responsibility,
    not this module's.
    """

    def __init__(self, config):
        super().__init__()
        self.top_k = config.dflash_config.get("selector_top_k", 8)
        self.rank = config.dflash_config.get("selector_rank", 128)
        self.hidden_projection = nn.Linear(config.hidden_size, self.rank, bias=False)
        self.predecessor_codebook = nn.Embedding(config.vocab_size, self.rank)
        self.successor_codebook = nn.Embedding(config.vocab_size, self.rank)

    def forward(self,
        hidden_t: torch.Tensor,     # (..., C)  hidden state at position t
        pred_ids: torch.Tensor,     # (...,)    chosen predecessor token id(s)
        cand_ids: torch.Tensor,     # (..., k)  candidate ids at position t
        cand_logits: torch.Tensor,  # (..., k)  DFlash's own logits for those candidates
    ) -> torch.Tensor:
        ctx = self.hidden_projection(hidden_t).unsqueeze(-2)          # (..., 1, rank)
        pred_vec = self.predecessor_codebook(pred_ids).unsqueeze(-2)  # (..., 1, rank)
        succ_vec = self.successor_codebook(cand_ids)                  # (..., k, rank)
        bilinear = ((pred_vec * ctx) * succ_vec).sum(-1)              # (..., k)
        return cand_logits + bilinear


class Qwen3DFlashDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3DFlashAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.dflash2 = bool(config.dflash_config.get("dflash2", False))
        if self.dflash2:
            # one conv module per sublayer, called both before (on the raw residual stream)
            # and after (on the sublayer's own output)
            self.attention_conv = DFlash2DynamicConv(config)
            self.mlp_conv = DFlash2DynamicConv(config)
        else:
            self.attention_conv = None
            self.mlp_conv = None

    def forward(
        self,
        target_hidden: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        attn_in = self.attention_conv(hidden_states) if self.attention_conv is not None else hidden_states
        attn_in = self.input_layernorm(attn_in)
        hidden_states = self.self_attn(
            hidden_states=attn_in,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        if self.attention_conv is not None:
            hidden_states = self.attention_conv(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        ffn_in = self.mlp_conv(hidden_states) if self.mlp_conv is not None else hidden_states
        ffn_in = self.post_attention_layernorm(ffn_in)
        hidden_states = self.mlp(ffn_in)
        if self.mlp_conv is not None:
            hidden_states = self.mlp_conv(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class DFlashDraftModel(Qwen3PreTrainedModel):
    config_class = Qwen3Config
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.layers = nn.ModuleList(
            [Qwen3DFlashDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.target_layer_ids = self.config.dflash_config.get("target_layer_ids", None)
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.fc = nn.Linear(len(self.target_layer_ids) * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.block_size = config.block_size
        self.mask_token_id = self.config.dflash_config.get("mask_token_id", None)
        self.dflash2 = bool(config.dflash_config.get("dflash2", False))
        self.candidate_selector = (
            DFlash2CandidateSelector(config)
            if self.dflash2 and config.dflash_config.get("selector_enabled", True)
            else None
        )
        self.post_init()
        self._init_dflash2_identity()

    def _init_dflash2_identity(self):
        """
        post_init() applies transformers' generic weight init to every
        nn.Linear/nn.Embedding, which overwrites the identity/no-op values the
        DFlash 2 conv and selector modules need at construction time (see
        allamo.model.dflash.model's DFlash2DynamicConv.init_weights /
        DFlash2CandidateSelector.init_weights for the training-side
        equivalent this must match). Re-apply it here, after post_init().
        """
        for module in self.modules():
            if isinstance(module, DFlash2DynamicConv):
                with torch.no_grad():
                    module.base_kernel.zero_()
                    module.base_kernel[0, :] = 1.0  # tap 0 (self) passes through unchanged
                torch.nn.init.zeros_(module.kernel_projection.weight)
            elif isinstance(module, DFlash2CandidateSelector):
                torch.nn.init.zeros_(module.hidden_projection.weight)
                torch.nn.init.zeros_(module.successor_codebook.weight)  # bilinear term is zero at init

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)


class DFlash2DraftModel(DFlashDraftModel):
    """
    Identical to DFlashDraftModel - the conv/selector submodules are already
    built conditionally from config.dflash_config. This subclass exists only
    so save_pretrained() writes architectures: ["DFlash2DraftModel"], which is
    how vLLM's VllmConfig._is_dflash2_draft (vllm/config/vllm.py, merged in
    vllm-project/vllm#52816) tells a DFlash2 checkpoint apart from a DFlash1
    one and forces the V2 model runner that its candidate selector needs -
    without it, a DFlash2 checkpoint silently drafts as DFlash1 on V1.
    """
    pass
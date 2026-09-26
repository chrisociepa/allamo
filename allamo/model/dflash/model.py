import math
import torch
from dataclasses import dataclass
from torch.nn import functional as F
from typing import Optional, Tuple, List

from allamo.logging import logger
from allamo.model.modeling_utils import FeedForward, BaseModelConfig
from allamo.model.attentions import attention_version
from allamo.model.rotary_embeddings import RotaryEmbedding


class DFlashAttention(torch.nn.Module):

    def __init__(self, config: BaseModelConfig):
        super().__init__()
        self.head_size = config.head_size
        self.num_heads = config.n_head
        self.num_kv_heads = config.num_kv_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.dropout = config.dropout
        self.attn_output_gate = config.dflash_config.get("attn_output_gate", config.attn_output_gate)
        self.qk_norm = config.dflash_config.get("qk_norm", config.qk_norm)
        self.draft_block_size = config.dflash_config["block_size"]
        
        assert self.num_key_value_groups * self.num_kv_heads == self.num_heads
        
        # key, query, value projections for all heads
        self.q_proj = torch.nn.Linear(config.n_embd, self.num_heads * self.head_size * (1 + config.attn_output_gate), bias=config.bias)
        self.k_proj = torch.nn.Linear(config.n_embd, self.num_kv_heads * self.head_size, bias=config.bias)
        self.v_proj = torch.nn.Linear(config.n_embd, self.num_kv_heads * self.head_size, bias=config.bias)
        # output projection
        self.c_proj = torch.nn.Linear(self.num_heads * self.head_size, config.n_embd, bias=config.bias)

        self.q_norm = torch.nn.RMSNorm(config.head_size, eps=config.norm_eps) if self.qk_norm else None
        self.k_norm = torch.nn.RMSNorm(config.head_size, eps=config.norm_eps) if self.qk_norm else None
                    
    def init_weights(self, init_std: float):
        for module in (self.q_proj, self.k_proj, self.v_proj):
            torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
        torch.nn.init.trunc_normal_(self.c_proj.weight, mean=0.0, std=init_std)

        if self.q_norm:
            self.q_norm.reset_parameters()
        if self.k_norm:
            self.k_norm.reset_parameters()

    def forward(self,
                q_x: torch.Tensor,   # (B, A * draft_block_size, C)
                kv_x: torch.Tensor,  # (B, T, C) - target hidden states
                rotary_emb: RotaryEmbedding,
                anchor_pos: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                input_pos: Optional[torch.Tensor] = None,
                seq_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = kv_x.size()
        QT  = q_x.shape[1]
        assert seq_lens is None, "seq_lens not supported in DFlashAttention"

        q = self.q_proj(q_x)
        q = q.view(B, QT, self.num_heads, self.head_size * (1 + self.attn_output_gate))
        if self.attn_output_gate:
            q, gate = torch.chunk(q, 2, dim=-1)
            gate = gate.reshape(B, QT, -1)
        q = q.transpose(1, 2) # (B, nh, A * draft_block_size, hs)

        k_ctx_proj   = self.k_proj(kv_x).view(B, T, -1, self.head_size).transpose(1, 2)
        v_ctx        = self.v_proj(kv_x).view(B, T, -1, self.head_size).transpose(1, 2)
        k_noise_proj = self.k_proj(q_x).view(B, QT, -1, self.head_size).transpose(1, 2)
        v_noise      = self.v_proj(q_x).view(B, QT, -1, self.head_size).transpose(1, 2)

        if self.q_norm:
            q = self.q_norm(q)
        if self.k_norm:
            # Norm before RoPE, separately per segment
            k_ctx_proj   = self.k_norm(k_ctx_proj)
            k_noise_proj = self.k_norm(k_noise_proj)
        
        # Apply RoPE with correct positions per segment
        q, k = self._apply_rope_diffusion(q, k_ctx_proj, k_noise_proj, rotary_emb, anchor_pos, input_pos)

        v = torch.cat([v_ctx, v_noise], dim=2)  # (B, nh, T + A * draft_block_size, hs)
        
        if self.num_key_value_groups > 1:
            k = self.repeat_kv(k, self.num_key_value_groups)
            v = self.repeat_kv(v, self.num_key_value_groups)

        y = attention_version.flex_attention_diffusion(
            q, k, v, 
            T=T, 
            q_len=self.draft_block_size,
            anchor_pos=anchor_pos,
            input_pos=input_pos,
            attn_mask=attn_mask,
            sliding_window=None
        )

        # output projection (B, A * draft_block_size, nh * hs) -> (B, A * draft_block_size, C)
        y = y.contiguous().view(B, QT, self.num_heads * self.head_size)

        if self.attn_output_gate:
            y = y * torch.sigmoid(gate)

        y = self.c_proj(y)
        if self.dropout > 0:
            F.dropout(y, self.dropout, training=self.training, inplace=True)
        return y

    def _apply_rope_diffusion(
        self,
        q: torch.Tensor,        # (B, nh, A * draft_block_size, hs)
        k_ctx: torch.Tensor,    # (B, nh, T, hs)
        k_noise: torch.Tensor,  # (B, nh, A * draft_block_size, hs)
        rotary_emb: RotaryEmbedding,
        anchor_pos: torch.Tensor,                  # (B, A)
        input_pos: Optional[torch.Tensor] = None,  # (B, T)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = q.device
        T = k_ctx.size(2)
        B = anchor_pos.size(0)
        q_len = self.draft_block_size

        if input_pos is not None:
            ctx_pos = input_pos # (B, T)
            anchor_abs = input_pos.gather(1, anchor_pos + 1) # (B, A)
        else:
            ctx_pos = torch.arange(T, device=device).unsqueeze(0).expand(B, -1) # (B, T)
            anchor_abs = anchor_pos + 1 # (B, A)

        k_idx = torch.arange(q_len, device=device) # (q_len,)
        noise_pos = anchor_abs.unsqueeze(-1) + k_idx # (B, A, q_len)
        noise_pos = noise_pos.reshape(B, -1) # (B, A * q_len)

        q_pos = noise_pos # (B, A * q_len)
        kv_pos = torch.cat([ctx_pos, noise_pos], dim=1) # (B, T + A * q_len)

        k_full = torch.cat([k_ctx, k_noise], dim=2) # (B, nh, T + A * q_len, hs)

        q_rot, k_full_rot = rotary_emb(q, k_full, input_pos=q_pos, kv_input_pos=kv_pos)

        return q_rot, k_full_rot

    def repeat_kv(self, x: torch.Tensor, num_key_value_groups: int) -> torch.Tensor:
        # (B, num_kv_heads, T, hs) -> (B, nh, T, hs)
        if num_key_value_groups == 1:
            return x
        B, num_kv_heads, T, hs = x.shape
        x = x[:, :, None, :, :].expand(B, num_kv_heads, num_key_value_groups, T, hs)
        return x.reshape(B, num_kv_heads * num_key_value_groups, T, hs)


class DFlash2DynamicConv(torch.nn.Module):
    """
    Grouped dynamic depthwise convolution, wrapped around one DFlash sublayer
    (attention or feed-forward). Lets a block position see its immediate
    predecessor(s) within the same block without another pass through the
    backbone:

        out[i, c] = sum_t (base[t, c] + delta[i, t, g(c)]) * x[i - t, c]

    g(c) maps a channel to its group (conv_group_size channels share one
    dynamic correction). base is a static kernel at full channel resolution;
    delta is a per-position dynamic correction, shared within each group,
    predicted from the sublayer's own input. Taps never cross a block
    boundary: position i only sums over t <= i within its own block.

    Parameter names (base_kernel, kernel_projection) and base_kernel's full
    per-channel shape match vLLM's DFlash 2 runner. One projection of the
    normalized sublayer input produces coefficients for both the pre- and
    post-sublayer convolutions; the post-sublayer convolution reuses its
    second half rather than projecting the output again.

    The initialization contract (identity at construction) is the part that
    matters for warm-starting from a DFlash 1 checkpoint.
    """

    def __init__(self, config: BaseModelConfig):
        super().__init__()
        self.n_embd = config.n_embd
        self.draft_block_size = config.dflash_config["block_size"]
        self.taps = config.dflash_config.get("conv_kernel_size", 2)
        assert 0 < self.taps <= self.draft_block_size, "conv_kernel_size must be in [1, block_size]"
        self.group_size = config.dflash_config.get("conv_group_size", 1)
        assert self.n_embd % self.group_size == 0, "n_embd must be divisible by conv_group_size"
        self.num_groups = self.n_embd // self.group_size

        # Side 0 is applied before a sublayer and side 1 after it.
        self.base_kernel = torch.nn.Parameter(torch.zeros(2, self.taps, self.n_embd))
        self.kernel_projection = torch.nn.Linear(
            self.n_embd, 2 * self.taps * self.num_groups, bias=False
        )

    def init_weights(self):
        # identity at init: tap 0 passes x through unchanged, tap 1 and every
        # dynamic correction start at zero, so this module is a no-op until trained
        with torch.no_grad():
            self.base_kernel.zero_()
            self.base_kernel[:, 0, :] = 1.0
        torch.nn.init.zeros_(self.kernel_projection.weight)

    def _convolve(self, x: torch.Tensor, delta: torch.Tensor, side: int) -> torch.Tensor:
        # x: (B, A * draft_block_size, C)
        B, QT, C = x.shape
        block = self.draft_block_size
        A = QT // block
        assert QT % block == 0, "draft sequence length must be divisible by block_size"

        x_blk = x.view(B, A, block, self.num_groups, self.group_size)
        base = self.base_kernel[side].view(1, 1, 1, self.taps, self.num_groups, self.group_size)
        kernel = base + delta.unsqueeze(-1)
        out = kernel[:, :, :, 0] * x_blk
        for tap in range(1, self.taps):
            shifted = torch.zeros_like(x_blk)
            shifted[:, :, tap:] = x_blk[:, :, :-tap]
            out = out + kernel[:, :, :, tap] * shifted
        return out.reshape(B, QT, C)

    def prepare(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, QT, _ = x.shape
        assert QT % self.draft_block_size == 0, "draft sequence length must be divisible by block_size"
        A = QT // self.draft_block_size
        coefficients = self.kernel_projection(x).view(
            B, A, self.draft_block_size, 2, self.taps, self.num_groups
        )
        return self._convolve(x, coefficients[:, :, :, 0], 0), coefficients[:, :, :, 1]

    def finish(self, x: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
        return self._convolve(x, coefficients, 1)


class DFlash2CandidateSelector(torch.nn.Module):
    """
    Pairwise candidate-path selector for DFlash 2, trained with teacher forcing.

    At each block position t, DFlash's own head already proposes selector_top_k
    candidates (draft_logits.topk). This module rescores each candidate against
    the *ground-truth* predecessor token at position t-1 with a low-rank
    bilinear term added on top of DFlash's own logit:

        score(b) = logit_t(b) + <hidden_projection(h_t) * predecessor_codebook[pred_id], successor_codebook[b]>

    predecessor_codebook / successor_codebook are rank-`selector_rank` token
    codebooks; hidden_projection is a small context projection of the hidden
    state at t. Training supervises the index of the true token within the
    top-k list; positions whose true token misses the list are excluded.

    successor_codebook starts at zero, so the bilinear term is zero at
    initialization: a freshly built selector defers entirely to DFlash's own
    logits until it is trained.

    Parameter names match the checkpoints published for DFlash 2 (verified
    against multiple independent z-lab/Qwen3.8-27B-DFlash2 derivatives'
    documented tensor listings), so a state-dict copy needs no renaming.

    Note: this module only covers the trainable head. The full inference-time
    path walk (choosing predecessors from a live top-k set rather than a single
    teacher-forced token, across the whole block) belongs to the serving engine
    (vLLM/SGLang/etc.), not to this training-side module.
    """

    def __init__(self, config: BaseModelConfig):
        super().__init__()
        self.top_k = config.dflash_config.get("selector_top_k", 8)
        self.rank = config.dflash_config.get("selector_rank", 128)
        self.hidden_projection = torch.nn.Linear(config.n_embd, self.rank, bias=False)
        self.predecessor_codebook = torch.nn.Parameter(torch.empty(config.vocab_size, self.rank))
        self.successor_codebook = torch.nn.Parameter(torch.empty(config.vocab_size, self.rank))

    def init_weights(self):
        torch.nn.init.zeros_(self.hidden_projection.weight)
        torch.nn.init.trunc_normal_(self.predecessor_codebook, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.successor_codebook)  # bilinear term is zero at init

    def forward(self,
        hidden_t: torch.Tensor,     # (..., C)  hidden state at position t
        pred_ids: torch.Tensor,     # (...,)    ground-truth predecessor token id (teacher forced)
        cand_ids: torch.Tensor,     # (..., k)  top-k candidate ids at position t
        cand_logits: torch.Tensor,  # (..., k)  DFlash's own logits for those candidates
    ) -> torch.Tensor:
        ctx = self.hidden_projection(hidden_t).unsqueeze(-2)                # (..., 1, rank)
        pred_vec = self.predecessor_codebook[pred_ids].unsqueeze(-2)       # (..., 1, rank)
        succ_vec = self.successor_codebook[cand_ids]                        # (..., k, rank)
        bilinear = ((pred_vec * ctx) * succ_vec).sum(-1)       # (..., k)
        return cand_logits + bilinear


class DFlashLayer(torch.nn.Module):
    
    def __init__(self, layer_id: int, config: BaseModelConfig):
        super().__init__()
        self.layer_id = layer_id
        self.attention = DFlashAttention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = torch.nn.RMSNorm(config.n_embd, eps=config.norm_eps)
        self.ffn_norm = torch.nn.RMSNorm(config.n_embd, eps=config.norm_eps)

        self.dflash2 = config.dflash_config.get("dflash2", False)
        if self.dflash2:
            # one conv module per sublayer with two calls sharing the same projection
            self.attention_conv = DFlash2DynamicConv(config)
            self.mlp_conv = DFlash2DynamicConv(config)
        else:
            self.attention_conv = None
            self.mlp_conv = None
    
    def init_weights(self, init_std: float):
        self.attention.init_weights(init_std)
        self.feed_forward.init_weights(init_std)
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        if self.attention_conv is not None:
            self.attention_conv.init_weights()
        if self.mlp_conv is not None:
            self.mlp_conv.init_weights()

    def forward(self,
        x: torch.Tensor,
        target_hidden: torch.Tensor,
        rotary_emb: RotaryEmbedding,
        anchor_pos: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        attn_in = self.attention_norm(x)
        if self.attention_conv is not None:
            attn_in, attn_coefficients = self.attention_conv.prepare(attn_in)
        attn_out = self.attention(attn_in, target_hidden, rotary_emb, anchor_pos=anchor_pos, attn_mask=attn_mask, input_pos=input_pos, seq_lens=seq_lens)
        if self.attention_conv is not None:
            attn_out = self.attention_conv.finish(attn_out, attn_coefficients)
        x = x + attn_out

        ffn_in = self.ffn_norm(x)
        if self.mlp_conv is not None:
            ffn_in, ffn_coefficients = self.mlp_conv.prepare(ffn_in)
        ffn_out = self.feed_forward(ffn_in)
        if self.mlp_conv is not None:
            ffn_out = self.mlp_conv.finish(ffn_out, ffn_coefficients)
        x = x + ffn_out
        return x


class DFlashDraftModel(torch.nn.Module):

    def __init__(
        self,
        config: BaseModelConfig,
        tok_embeddings: torch.nn.Embedding,
        lm_head: torch.nn.Linear,
        rotary_emb: RotaryEmbedding,
    ):
        super().__init__()
        self.config = config
        self.target_layer_ids = set(config.dflash_config["target_layer_ids"])
        self.mask_token_id = config.dflash_config.get("mask_token_id", None)
        self.draft_block_size = config.dflash_config["block_size"]
        self.unfreeze_mask_token = config.dflash_config.get("unfreeze_mask_token", False)
        self.detach_hidden_states = config.dflash_config.get("detach_hidden_states", False)

        self.dflash2 = config.dflash_config.get("dflash2", False)
        self.selector_enabled = self.dflash2 and config.dflash_config.get("selector_enabled", True)

        self.embeddings = tok_embeddings
        self.lm_head = lm_head
        self.rotary_emb = rotary_emb

        if self.unfreeze_mask_token:
            self.mask_token_embd = torch.nn.Parameter(torch.empty(self.config.n_embd))
            logger.info("DFlash mask token embedding initialized. Remember to merge it into the target model.")
        else:
            self.mask_token_embd = None
            logger.info(f"DFlash will use mask token id {self.mask_token_id}")

        self.fc = torch.nn.Linear(len(self.target_layer_ids) * self.config.n_embd, self.config.n_embd, bias=False)
        self.hidden_norm = torch.nn.RMSNorm(self.config.n_embd, eps=self.config.norm_eps)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.config.dflash_config["num_hidden_layers"]):
            self.layers.append(DFlashLayer(layer_id, self.config))
        self.norm = torch.nn.RMSNorm(self.config.n_embd, eps=self.config.norm_eps)

        self.candidate_selector = DFlash2CandidateSelector(self.config) if self.selector_enabled else None

        self.init_weights()

    def init_weights(self):
        weight_init_std = 0.02 / math.sqrt(len(self.target_layer_ids))
        torch.nn.init.trunc_normal_(self.fc.weight, mean=0.0, std=weight_init_std)

        self.hidden_norm.reset_parameters()
        self.norm.reset_parameters()

        weight_init_std = 0.02 / math.sqrt(2 * len(self.layers))
        for layer in self.layers:
            layer.init_weights(weight_init_std)

        if self.candidate_selector is not None:
            self.candidate_selector.init_weights()

    def forward(self,
        target_ids: torch.Tensor,
        anchor_pos: torch.Tensor,
        target_hidden_states: List[torch.Tensor],
        attn_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, A = anchor_pos.shape

        if self.detach_hidden_states:
            target_hidden_states = [hs.detach() for hs in target_hidden_states]
        target_hidden = torch.cat(target_hidden_states, dim=-1)
        target_hidden = self.hidden_norm(self.fc(target_hidden))

        anchor_ids = target_ids.gather(1, anchor_pos) # (B, A)
        anchor_emb = self.embeddings(anchor_ids) # (B, A, C)

        C = anchor_emb.shape[-1]
        if self.mask_token_embd is not None:
            mask_emb = self.mask_token_embd
        else:
            mask_emb = self.embeddings(torch.tensor([self.mask_token_id], device=target_hidden.device))  # (1, C)
        mask_emb_full = mask_emb.expand(B, A, C)

        draft_hidden_states = mask_emb_full.unsqueeze(2).expand(B, A, self.draft_block_size - 1, C).clone()
        draft_hidden_states = torch.cat([anchor_emb.unsqueeze(2), draft_hidden_states], dim=2)
        draft_hidden_states = draft_hidden_states.reshape(B, A * self.draft_block_size, C)
        if self.detach_hidden_states:
            draft_hidden_states = draft_hidden_states.detach().requires_grad_(True)

        for layer in self.layers:
            draft_hidden_states = layer(
                draft_hidden_states,
                target_hidden=target_hidden,
                rotary_emb=self.rotary_emb,
                anchor_pos=anchor_pos,
                attn_mask=attn_mask,
                input_pos=input_pos,
                seq_lens=seq_lens,
            )
        draft_hidden_states = self.norm(draft_hidden_states)
        draft_logits = self.lm_head(draft_hidden_states) # (B, A * draft_block_size, vocab_size)

        # Selector loss needs the pre-lm_head hidden states (context gate input);
        # only materialize/return them when a selector is actually attached, so
        # plain DFlash 1 training keeps its original single-tensor cost.
        if self.candidate_selector is not None:
            return draft_logits, draft_hidden_states
        return draft_logits, None

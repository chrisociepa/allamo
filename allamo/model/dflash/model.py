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
        self.attn_output_gate = config.attn_output_gate
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


class DFlashLayer(torch.nn.Module):
    
    def __init__(self, layer_id: int, config: BaseModelConfig):
        super().__init__()
        self.layer_id = layer_id
        self.attention = DFlashAttention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = torch.nn.RMSNorm(config.n_embd, eps=config.norm_eps)
        self.ffn_norm = torch.nn.RMSNorm(config.n_embd, eps=config.norm_eps)
    
    def init_weights(self, init_std: float):
        self.attention.init_weights(init_std)
        self.feed_forward.init_weights(init_std)
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()

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
        x = x + self.attention(self.attention_norm(x), target_hidden, rotary_emb, anchor_pos=anchor_pos, attn_mask=attn_mask, input_pos=input_pos, seq_lens=seq_lens)
        x = x + self.feed_forward(self.ffn_norm(x))
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

        self.embeddings = tok_embeddings
        self.lm_head = lm_head
        self.rotary_emb = rotary_emb

        if self.unfreeze_mask_token:
            with torch.no_grad():
                orig_emb = self.embeddings.weight[self.mask_token_id]
            self.mask_token_embd = torch.nn.Parameter(orig_emb.clone())
            logger.warning(f"Unfrozen mask token embedding for token ID {self.mask_token_id}. "
                           f"Remember to merge it with the main model before unfreezing it.")

        self.fc = torch.nn.Linear(len(self.target_layer_ids) * self.config.n_embd, self.config.n_embd, bias=False)
        self.hidden_norm = torch.nn.RMSNorm(self.config.n_embd, eps=self.config.norm_eps)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.config.dflash_config["num_hidden_layers"]):
            self.layers.append(DFlashLayer(layer_id, self.config))
        self.norm = torch.nn.RMSNorm(self.config.n_embd, eps=self.config.norm_eps)

        self.init_weights()
        
    def init_weights(self):
        torch.nn.init.trunc_normal_(self.fc.weight, mean=0.0, std=0.02)
        self.hidden_norm.reset_parameters()
        self.norm.reset_parameters()

        weight_init_std = 0.02 / math.sqrt(2 * len(self.layers))
        for layer in self.layers:
            layer.init_weights(weight_init_std)

    def forward(self,
        input_ids: torch.Tensor,
        anchor_pos: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        last_hidden_states: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B = input_ids.size(0)
        target_hidden = self.hidden_norm(self.fc(target_hidden))

        # anchor_pos indexes target_ids space; input_ids is shifted left by 1,
        # so anchor_pos + 1 gives the corresponding input_ids positions
        A = anchor_pos.size(1)
        anchor_input_pos = anchor_pos + 1
        anchor_ids = input_ids.gather(1, anchor_input_pos) # (B, A)
        anchor_emb = self.embeddings(anchor_ids) # (B, A, C)

        C = anchor_emb.shape[-1]
        if self.unfreeze_mask_token:
            mask_emb = self.mask_token_embd
        else:
            mask_emb = self.embeddings(torch.tensor([self.mask_token_id], device=target_hidden.device))  # (1, C)
        mask_emb_full = mask_emb.expand(B, A, anchor_emb.size(-1))  # (B, A, C)

        draft_hidden_states = mask_emb_full.unsqueeze(2).expand(B, A, self.draft_block_size, C).clone()
        draft_hidden_states[:, :, 0, :] = anchor_emb
        draft_hidden_states = draft_hidden_states.reshape(B, A * self.draft_block_size, C)

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
    
        return draft_logits

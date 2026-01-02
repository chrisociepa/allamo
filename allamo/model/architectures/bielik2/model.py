import torch
from dataclasses import dataclass
from torch.nn import functional as F
from typing import Optional, Tuple, List

from allamo.logging import logger
from allamo.model.modeling_utils import AttentionBlock, BaseModel, BaseModelConfig, ModelSpec
from allamo.model.attentions import attention_version
from allamo.model.rotary_embeddings import RotaryEmbedding

def get_model_spec():
    return ModelSpec(
        model_type = "bielik2",
        model_config_cls = Bielik2Config,
        model_cls = Bielik2Model
    )

@dataclass
class Bielik2Config(BaseModelConfig):
    model_type: str = "bielik2"

class Bielik2Model(BaseModel):

    def configure(self):
        super().configure()

        self.tok_embeddings = torch.nn.Embedding(self.config.vocab_size, self.config.n_embd)
        self.tok_drop = torch.nn.Dropout(self.config.dropout) if self.config.dropout != 0 else None
        
        self.rotary_emb = RotaryEmbedding(self.config.head_size, self.config.block_size, self.config.rope_freq_base, self.config.rope_scaling)
        
        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.config.n_layer):
            self.layers.append(AttentionBlock(layer_id, self.config))
        
        self.norm = torch.nn.RMSNorm(self.config.n_embd, eps=self.config.norm_eps)
        self.lm_head = torch.nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)

    def init_model_weights(self, buffer_device: Optional[torch.device] = None):
        super().init_model_weights(buffer_device)
        
        buffer_device = buffer_device or self.rotary_emb.inv_freq.device
        with torch.device(buffer_device):
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

    def forward(self,
        input_ids: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        target_weights: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        ignore_index: Optional[int] = -100,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.FloatTensor]]]:
        if inputs_embeds is not None:
            _, T, _ = inputs_embeds.size()
        else:
            _, T = input_ids.size()
            inputs_embeds = self.get_embeddings()(input_ids) # token embeddings of shape (b, t, n_embd)
            if self.tok_drop is not None:
                inputs_embeds = self.tok_drop(inputs_embeds)
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        
        if attn_mask is not None and attention_version.version != 'flex': # FlexAttention use document ids instead of mask
            if attn_mask.ndim == 3:
                attn_mask = attn_mask.unsqueeze(1) # (B, T, T) -> (B, 1, T, T)
            elif attn_mask.ndim != 4:
                raise ValueError(f"Unsupport attn_mask shape {attn_mask.shape}")
        
        hidden_states = inputs_embeds
        for layer in self.get_layers():
            hidden_states = layer(hidden_states, self.rotary_emb, attn_mask=attn_mask, input_pos=input_pos, seq_lens=seq_lens)
        
        final_embeddings = self.get_lm_head_norm()(hidden_states)
        if target_ids is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.get_lm_head()(final_embeddings)
            if target_weights is None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), ignore_index=ignore_index)
            else:
                loss = (target_weights.view(-1) * F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), reduction="none")).sum()
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.get_lm_head()(final_embeddings[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
        return logits, loss, hidden_states
    
    def add_layer(self, new_layers=1):
        for _ in range(new_layers):
            new_layer_id = len(self.layers)
            layer = AttentionBlock(new_layer_id, self.config)
            self.layers.append(layer)
            self.config.n_layer += 1
            layer.init_weights(self.calculate_weight_init_std(self.config.n_layer))
    
    def get_embeddings(self):
        return self.tok_embeddings
    
    def get_lm_head_norm(self):
        return self.norm
    
    def get_lm_head(self):
        return self.lm_head
    
    def get_layers(self):
        return self.layers

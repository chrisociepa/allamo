import torch
from dataclasses import dataclass
from torch.nn import functional as F
from typing import Optional, Tuple, List

from allamo.logging import logger
from allamo.model.modeling_utils import AttentionBlock, BaseModel, BaseModelConfig, ModelSpec
from allamo.model.attentions import attention_version
from allamo.model.dflash.model import DFlashDraftModel
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
        
        max_seq_len = self.config.block_size + self.config.head_size # add head_size to support draft model
        self.rotary_emb = RotaryEmbedding(self.config.head_size, max_seq_len, self.config.rope_freq_base, self.config.rope_scaling)
        
        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.config.n_layer):
            self.layers.append(AttentionBlock(layer_id, self.config))
        
        self.norm = torch.nn.RMSNorm(self.config.n_embd, eps=self.config.norm_eps)
        self.lm_head = torch.nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)

        self.target_layer_ids: Optional[set[int]] = None
        self.dflash: Optional[DFlashDraftModel] = None
        if self.config.dflash_config:
            self.target_layer_ids = set(self.config.dflash_config["target_layer_ids"])
            self.dflash = DFlashDraftModel(self.config, self.tok_embeddings, self.lm_head, self.rotary_emb)

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
            
            if self.dflash is not None:
                self.dflash.init_weights()

    def forward(self,
        input_ids: torch.Tensor,
        anchor_pos: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if inputs_embeds is not None:
            T = inputs_embeds.size(1)
        else:
            T = input_ids.size(1)
            inputs_embeds = self.get_embeddings()(input_ids) # (B, T, C)
            if self.tok_drop is not None:
                inputs_embeds = self.tok_drop(inputs_embeds)
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        
        if attn_mask is not None and attention_version.version != 'flex': # FlexAttention use document ids instead of mask
            if attn_mask.ndim == 3:
                attn_mask = attn_mask.unsqueeze(1) # (B, T, T) -> (B, 1, T, T)
            elif attn_mask.ndim != 4:
                raise ValueError(f"Unsupport attn_mask shape {attn_mask.shape}")
        
        if self.target_layer_ids:
            hidden_states_list = []
            
        hidden_states = inputs_embeds
        for idx, layer in enumerate(self.get_layers()):
            hidden_states = layer(hidden_states, self.rotary_emb, attn_mask=attn_mask, input_pos=input_pos, seq_lens=seq_lens)
            if self.target_layer_ids and idx in self.target_layer_ids:
                hidden_states_list.append(hidden_states)
        
        last_hidden_states = hidden_states
        hidden_states = self.get_lm_head_norm()(hidden_states)
        logits = self.get_lm_head()(hidden_states)

        draft_logits = None
        if self.dflash is not None:
            target_hidden = torch.cat(hidden_states_list, dim=-1)
            draft_logits = self.dflash(
                input_ids=input_ids,
                anchor_pos=anchor_pos,
                attn_mask=attn_mask,
                input_pos=input_pos,
                seq_lens=seq_lens,
                target_hidden=target_hidden,
                last_hidden_states=last_hidden_states,
            )

        return logits, draft_logits
    
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
    
    def get_dflash(self):
        return self.dflash

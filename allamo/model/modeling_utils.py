import math
import torch
from dataclasses import dataclass, field
from importlib import import_module
from torch.nn import functional as F
from typing import Optional, List, Union, Dict

from allamo.logging import logger
from allamo.model.activations import get_activation
from allamo.model.attentions import attention_version
from allamo.model.architectures import _supported_models
from allamo.model.rotary_embeddings import RotaryEmbedding

@dataclass
class BaseModelConfig:

    model_type: str = ""
    block_size: int = 1024
    vocab_size: int = 32000
    rope_freq_base: int = 10000
    rope_scaling: Dict = field(default_factory=dict)
    n_layer: int = 12
    num_kv_heads: Union[None, int] = None
    head_size: Union[None, int] = None
    n_head: int = 12
    n_embd: int = 768
    intermediate_size: int = 2304
    dropout: float = 0.0
    bias: bool = True
    norm_eps: float = 1e-5
    sliding_window: int = None
    act_fn: str = "silu"
    act_fn_params: Dict = field(default_factory=dict)

class FeedForward(torch.nn.Module):

    def __init__(self, config: BaseModelConfig):
        super().__init__()       
        self.gate_proj = torch.nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.down_proj = torch.nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)
        self.up_proj = torch.nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.act_fn  = get_activation(act_fn_name=config.act_fn, **config.act_fn_params)
        self.dropout = torch.nn.Dropout(config.dropout) if config.dropout != 0 else None
        
    def init_weights(self, init_std: float):
        torch.nn.init.trunc_normal_(self.gate_proj.weight, mean=0.0, std=0.02)
        for module in (self.down_proj, self.up_proj):
            torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=init_std)
        if hasattr(self.act_fn, 'reset_params'):
            self.act_fn.reset_params()

    def forward(self, x):
        x = self.mlp(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x
        
    def mlp(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Attention(torch.nn.Module):

    def __init__(self, config: BaseModelConfig):
        super().__init__()
        self.head_size = config.head_size
        self.num_heads = config.n_head
        self.num_kv_heads = config.num_kv_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.dropout = config.dropout
        self.sliding_window = config.sliding_window if attention_version.flash_attn_supports_window_size else None
        
        assert self.num_key_value_groups * self.num_kv_heads == self.num_heads
        
        # key, query, value projections for all heads
        self.q_proj = torch.nn.Linear(config.n_embd, self.num_heads * self.head_size, bias=config.bias)
        self.k_proj = torch.nn.Linear(config.n_embd, self.num_kv_heads * self.head_size, bias=config.bias)
        self.v_proj = torch.nn.Linear(config.n_embd, self.num_kv_heads * self.head_size, bias=config.bias)
        # output projection
        self.c_proj = torch.nn.Linear(self.num_heads * self.head_size, config.n_embd, bias=config.bias)
                    
    def init_weights(self, init_std: float):
        for module in (self.q_proj, self.k_proj, self.v_proj):
            torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
        torch.nn.init.trunc_normal_(self.c_proj.weight, mean=0.0, std=init_std)
        
    def forward(self,
                q_x: torch.Tensor,
                kv_x: torch.Tensor,
                rotary_emb: RotaryEmbedding,
                attn_mask: Optional[torch.Tensor] = None,
                input_pos: Optional[torch.Tensor] = None,
                seq_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # notation:
        # B  | batch
        # T  | time-step (sequence length)
        # C  | embeddings size
        # hs | head size
        # nh | number of heads
        B, T, C = q_x.size()
        
        q, k, v = self.project_qkv(q_x, kv_x)
        
        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.num_kv_heads, self.head_size).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.num_kv_heads, self.head_size).transpose(1, 2) # (B, nh, T, hs)
        
        q, k = rotary_emb(q, k, input_pos=input_pos)
        
        if self.num_key_value_groups > 1:
            k = self.repeat_kv(k, self.num_key_value_groups)
            v = self.repeat_kv(v, self.num_key_value_groups)
        
        y = attention_version.apply(q, k, v, attn_mask, self.dropout, self.sliding_window)

        # output projection (B, T, nh * hs) -> (B, T, C)
        y = y.contiguous().view(B, T, self.num_heads * self.head_size)
        y = self.c_proj(y)
        if self.dropout > 0:
            F.dropout(y, self.dropout, training=self.training, inplace=True)
        return y
        
    def project_qkv(self, q_x: torch.Tensor, kv_x: Optional[torch.Tensor] = None):
        if kv_x is None:
            kv_x = q_x # self attention
        q = self.q_proj(q_x)
        k = self.k_proj(kv_x)
        v = self.v_proj(kv_x)
        return q, k, v
        
    def repeat_kv(self, x: torch.Tensor, num_key_value_groups: int) -> torch.Tensor:
        # (B, num_kv_heads, T, hs) -> (B, nh, T, hs)
        if num_key_value_groups == 1:
            return x
        B, num_kv_heads, T, hs = x.shape
        x = x[:, :, None, :, :].expand(B, num_kv_heads, num_key_value_groups, T, hs)
        return x.reshape(B, num_kv_heads * num_key_value_groups, T, hs)


class AttentionBlock(torch.nn.Module):

    def __init__(self, layer_id: int, config: BaseModelConfig):
        super().__init__()
        self.layer_id = layer_id
        self.attention = Attention(config)
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
        rotary_emb: RotaryEmbedding,
        attn_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        x = x + self.attention(self.attention_norm(x), None, rotary_emb, attn_mask=attn_mask, input_pos=input_pos, seq_lens=seq_lens)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x

class BaseModel(torch.nn.Module):

    def __init__(self, config: BaseModelConfig):
        super().__init__()
        self.config = config
        self.configure()
        self.__post_init__()
    
    def __post_init__(self):
        attention_version.log_version(self.config.sliding_window)
        self.log_estimated_size()
        self.init_model_weights(buffer_device=None)

    def configure(self):
        assert self.config.vocab_size is not None
        assert self.config.block_size is not None
        if self.config.head_size is None:
            assert self.config.n_embd % self.config.n_head == 0
            self.config.head_size = self.config.n_embd // self.config.n_head
            logger.info(f"defaulting to head_size={self.config.head_size} (n_embd / n_head)")
        if self.config.num_kv_heads is None:
            self.config.num_kv_heads = self.config.n_head
        if self.config.intermediate_size is None:
            self.config.intermediate_size = self.config.n_embd * 3
            logger.info(f"defaulting to intermediate_size={self.config.intermediate_size} (3 * n_embd)")
            
    def init_model_weights(self, buffer_device: Optional[torch.device] = None):
        pass

    def get_embeddings(self):
        return None

    def get_lm_head_norm(self):
        return None
    
    def get_lm_head(self):
        return None
    
    def get_layers(self):
        return None

    def calculate_weight_init_std(self, num_layers):
        return 0.02 / math.sqrt(2 * num_layers)

    def estimate_size(self, module):
        """
        Return the number of parameters and their size in the model.
        """
        params = 0
        bytes = 0
        for p in module.parameters():
            params += p.numel()
            bytes += p.numel() * p.element_size()
        for b in module.buffers():
            # don't count buffers as params
            bytes += b.numel() * b.element_size()
        return params, bytes
        
    def log_estimated_size(self):
        self.model_num_params, self.model_num_bytes = self.estimate_size(self)
        model_params = self.model_num_params / 1e6
        model_bytes = self.model_num_bytes / 1024**2
        logger.info(f"Model parameters: {model_params:.2f}M, Est. Size: {model_bytes:.3f}MB")
        if self.get_embeddings() is not None:
            embds_params, embds_bytes = self.estimate_size(self.get_embeddings())
            embds_params = embds_params / 1e6
            embds_bytes = embds_bytes / 1024**2
            logger.info(f"Embeddings parameters: {embds_params:.2f}M, Est. Size: {embds_bytes:.3f}MB")

    def freeze_module_params(self, module):
        for param in module.parameters():
            if param.requires_grad:
                param.requires_grad = False

    def freeze_model_params(self, freeze_embeddings: bool, freeze_lm_head: bool, freeze_layers: bool, keep_layers_trainable: List[int]):
        if freeze_embeddings and self.get_embeddings() is not None:
            self.freeze_module_params(self.get_embeddings())
            logger.info("Embeddings frozen")
        if freeze_lm_head and self.get_lm_head() is not None:
            if self.get_lm_head_norm() is not None:
                self.freeze_module_params(self.get_lm_head_norm())
            self.freeze_module_params(self.get_lm_head())
            logger.info("LM head frozen")
        if freeze_layers and self.get_layers() is not None:
            for layer_id in range(self.model_config.n_layer):
                if layer_id not in keep_layers_trainable:
                    self.freeze_module_params(self.get_layers()[layer_id])
                    logger.info(f"Layer {layer_id} frozen")
                else:
                    logger.info(f"Layer {layer_id} kept trainable")

@dataclass
class ModelSpec:
    model_type: str
    model_config_cls: type[BaseModelConfig]
    model_cls: type[BaseModel]
    
def get_model_spec(model_type: str):
    if model_type in _supported_models:
        module = import_module(f"allamo.model.architectures.{model_type}.model")
        return module.get_model_spec()
    
    raise ValueError(f"ModelSpec for {model_type} is not registered.")

def get_hf_model_adapter(model_type: str):
    if model_type in _supported_models:
        module = import_module(f"allamo.model.architectures.{model_type}.hf_adapter")
        return module.adapter
    
    raise ValueError(f"HF model adapter for {model_type} is not registered.")

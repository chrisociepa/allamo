import torch
import transformers
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaMLP

MODEL_TYPE = "llama_lra"
    
class LRA(torch.nn.Module):
    
    def __init__(self, dim, group_size):
        super().__init__()
        assert dim % group_size == 0
        self.group_size = group_size
        self.num_groups = dim // group_size
        
        self.p_coeff_0 = torch.nn.Parameter(torch.empty(1, self.group_size))
        self.p_coeff_1 = torch.nn.Parameter(torch.empty(1, self.group_size))
        self.p_coeff_2 = torch.nn.Parameter(torch.empty(1, self.group_size))
        self.p_coeff_3 = torch.nn.Parameter(torch.empty(1, self.group_size))
        self.p_coeff_4 = torch.nn.Parameter(torch.empty(1, self.group_size))
        self.p_coeff_5 = torch.nn.Parameter(torch.empty(1, self.group_size))
        
        self.q_coeff_1 = torch.nn.Parameter(torch.empty(1, self.group_size))
        self.q_coeff_2 = torch.nn.Parameter(torch.empty(1, self.group_size))
        self.q_coeff_3 = torch.nn.Parameter(torch.empty(1, self.group_size))
        self.q_coeff_4 = torch.nn.Parameter(torch.empty(1, self.group_size))
        
        self.reset_parameters()
            
    def reset_parameters(self):
        with torch.no_grad():
            for module in (self.p_coeff_0, self.p_coeff_1, self.p_coeff_2, self.p_coeff_3, self.p_coeff_4, self.p_coeff_5, self.q_coeff_1, self.q_coeff_2, self.q_coeff_3, self.q_coeff_4):
                torch.nn.init.trunc_normal_(module, mean=0.0, std=0.02)
        
    def forward(self, x):
        B, T, C = x.size()
        flat_x = x.view(-1)
        
        p = self.p_coeff_0.expand(B * T * self.num_groups, self.group_size).reshape(-1, self.group_size)
        
        p_xps = flat_x # x^1
        p = p + self.p_coeff_1 * p_xps.view(-1, self.group_size)
        q = (self.q_coeff_1 * p_xps.view(-1, self.group_size))
        
        p_xps = p_xps * flat_x # x^2
        p = p + self.p_coeff_2 * p_xps.view(-1, self.group_size)
        q = q + (self.q_coeff_2 * p_xps.view(-1, self.group_size))
        
        p_xps = p_xps * flat_x # x^3
        p = p + self.p_coeff_3 * p_xps.view(-1, self.group_size)
        q = q + (self.q_coeff_3 * p_xps.view(-1, self.group_size))
        
        p_xps = p_xps * flat_x # x^4
        p = p + self.p_coeff_4 * p_xps.view(-1, self.group_size)
        q = q + (self.q_coeff_4 * p_xps.view(-1, self.group_size))
        
        p_xps = p_xps * flat_x # x^5
        p = p + self.p_coeff_5 * p_xps.view(-1, self.group_size)
        
        return (p / (q.abs() + 1)).contiguous().view(B, T, C)

    def extra_repr(self):
        return f'num_groups={self.num_groups}, group_size={self.group_size}'
        
class LlamaLRAMLP(LlamaMLP):

    def __init__(self, config):
        super().__init__(config)
        self.act_fn = LRA(dim=config.intermediate_size, group_size=config.lra_group_size)

class LlamaLRAConfig(LlamaConfig):

    model_type = MODEL_TYPE

    def __init__(self, lra_group_size=128, **kwargs):
        super().__init__(**kwargs)
        self.lra_group_size = lra_group_size

class LlamaLRAForCausalLM(LlamaForCausalLM):

    model_type = MODEL_TYPE

    def __init__(self, config):
        transformers.models.llama.modeling_llama.LlamaMLP = LlamaLRAMLP
        super().__init__(config)

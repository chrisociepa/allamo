import torch.nn as nn
from allamo.model.lra import LRA

def get_activation(act_fn_name: str, **kwargs):
    if act_fn_name == "silu":
        return nn.SiLU()
    elif act_fn_name == "gelu_tanh":
        return nn.GELU(approximate='tanh')
    elif act_fn_name == "gelu":
        return nn.GELU()
    elif act_fn_name == "relu":
        return nn.ReLU()
    elif act_fn_name == "sigmoid":
        return nn.Sigmoid()
    elif act_fn_name == "tanh":
        return nn.Tanh()
    elif act_fn_name == "lra":
        return LRA(base_fn=kwargs.get("base_fn"), dim=kwargs.get("dim"), group_size=kwargs.get("group_size"))
    
    raise Exception(f'Unsupported activation function: {act_fn_name}')
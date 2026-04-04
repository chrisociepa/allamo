import torch
import torch.nn as nn
from allamo.model.lra import LRA

class XIELU(nn.Module):
    """
    xIELU activation function introduced in https://arxiv.org/abs/2411.13010
    """

    def __init__(self, alpha_p_init=0.8, alpha_n_init=0.8, beta=0.5, eps=-1e-6):
        super().__init__()
        self.alpha_p_init = alpha_p_init
        self.alpha_n_init = alpha_n_init
        self.beta_init = beta
        self.eps_init = eps
        self.alpha_p = nn.Parameter(torch.empty(1,))
        self.alpha_n = nn.Parameter(torch.empty(1,))
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            self.alpha_p.fill_(torch.log(torch.expm1(torch.tensor(self.alpha_p_init))))
            self.alpha_n.fill_(torch.log(torch.expm1(torch.tensor(self.alpha_n_init - self.beta_init))))
        self.register_buffer("beta", torch.tensor(self.beta_init), persistent=False)
        self.register_buffer("eps", torch.tensor(self.eps_init), persistent=False)

    def reset_parameters(self):
        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        alpha_p = nn.functional.softplus(self.alpha_p)
        alpha_n = self.beta + nn.functional.softplus(self.alpha_n)
        return torch.where(
            x > 0,
            alpha_p * x * x + self.beta * x,
            (torch.expm1(torch.min(x, self.eps)) - x) * alpha_n + self.beta * x,
        ).to(dtype=dtype)
    
    def extra_repr(self):
        return f'beta={self.beta_init}, eps={self.eps_init}'


class LeakyReLUSquare(nn.Module):
    """LeakyReLU then square: (LeakyReLU(x))²."""

    def __init__(self, negative_slope: float = 0.5):
        super().__init__()
        self.leaky = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.leaky(x)
        return y.square()

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
    elif act_fn_name == "xielu":
        return XIELU(**kwargs)
    elif act_fn_name == "lra":
        return LRA(**kwargs)
    elif act_fn_name == "leaky_relu_0.5_square":
        return LeakyReLUSquare(negative_slope=0.5)
    
    raise Exception(f'Unsupported activation function: {act_fn_name}')
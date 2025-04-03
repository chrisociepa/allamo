"""
Learnable Rational Activations - https://arxiv.org/abs/1907.06732
"""

import random
import torch

INIT_DATA = {
    "identity": {
        "init_numerator": [0,1,0,0,0,0],
        "init_denominator": [0,0,0,0]
    },
    "tanh": {
        "init_numerator": [-1.0804622559204184e-8,1.0003008043819048,-2.5878199375289335e-8,0.09632129918392647,3.4775841628196104e-9,0.0004255709234726337],
        "init_denominator": [-0.0013027181209176277,0.428349017422072,1.4524304083061898e-9,0.010796648111337176]
    },
    "sigmoid": {
        "init_numerator": [0.4999992534599381,0.25002157564685185,0.14061924500301096,0.049420492431596394,0.00876714851885483,0.0006442412789159799],
        "init_denominator": [2.1694506382753683e-9,0.28122766100417684,0.000010123620714203357,0.017531988049946]
    },
    "gelu": {
        "init_numerator": [-0.00153969,0.51692871,0.44075827,0.0972553,-0.00884418,-0.00378675],
        "init_denominator": [-0.14171056,-0.06481231,-0.0444695,0.01283776]
    },
    "swish": {
        "init_numerator": [3.054879741161051e-7,0.5000007853744493,0.24999783422824703,0.05326628273219478,0.005803034571292244,0.0002751961022402342],
        "init_denominator": [-0.000004111554955950634,0.10652899335007572,-0.0000012690007399796238,0.0005502331264140556]
    },
    "geglu": {
        "init_numerator": [0.00079642,-0.00602563,0.49750578,0.47691076,0.15421425,0.0168486],
        "init_denominator": [-0.12619161,-0.20067464,-0.03005465,0.0027335]
    },
    "swishglu": {
        "init_numerator": [-0.00111528,-0.00334562,0.53107663,0.27851476,0.06044361,0.00483468],
        "init_denominator": [0.14408492,0.02008671,0.02944107,-0.00329516]
    }
}

def get_supported_base_functions():
    return list(INIT_DATA.keys())

def get_fn_init_data(fn_name=None):
    if fn_name is None:
        return INIT_DATA[random.choice(["tanh", "sigmoid", "gelu", "swish", "geglu", "swishglu"])]
    elif fn_name in INIT_DATA:
        return INIT_DATA[fn_name]
    else:
        return None
    
class LRA(torch.nn.Module):
    
    def __init__(self, base_fn, dim, group_size):
        super().__init__()
        assert base_fn in INIT_DATA, f"Unsupported base function: {base_fn}"
        assert dim % group_size == 0
        self.base_fn = base_fn
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
        fn = get_fn_init_data(self.base_fn)
        with torch.no_grad():
            if fn is not None:
                self.p_coeff_0.fill_(fn["init_numerator"][0])
                self.p_coeff_1.fill_(fn["init_numerator"][1])
                self.p_coeff_2.fill_(fn["init_numerator"][2])
                self.p_coeff_3.fill_(fn["init_numerator"][3])
                self.p_coeff_4.fill_(fn["init_numerator"][4])
                self.p_coeff_5.fill_(fn["init_numerator"][5])
                
                self.q_coeff_1.fill_(fn["init_denominator"][0])
                self.q_coeff_2.fill_(fn["init_denominator"][1])
                self.q_coeff_3.fill_(fn["init_denominator"][2])
                self.q_coeff_4.fill_(fn["init_denominator"][3])
            else:
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
        return f'base_fn={self.base_fn}, num_groups={self.num_groups}, group_size={self.group_size}'
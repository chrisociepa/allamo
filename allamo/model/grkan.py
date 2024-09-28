import random
import torch
import grkan_cuda_lib

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

def get_fn_init_data(fn_name=None):
    if fn_name and fn_name in INIT_DATA:
        return INIT_DATA[fn_name]
    else:
        return INIT_DATA[random.choice(["tanh", "sigmoid", "gelu", "swish", "geglu", "swishglu"])]
    
@torch.library.register_fake("grkan::grkan_forward_cuda")
def fake_grkan_fwd(x, n, d, g):
    torch._check(x.dim() == 3)
    return torch.empty_like(x)

@torch.library.register_fake("grkan::grkan_backward_cuda")
def fake_grkan_bwd(o, x, n, d, g):
    torch._check(x.dim() == 3)
    return (torch.empty_like(x), torch.empty_like(n, dtype=torch.float), torch.empty_like(d, dtype=torch.float))

class grkan_cuda_op(torch.autograd.Function):
    
    @staticmethod
    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    def forward(ctx, x, n, d, group):
        """
        Forward pass of the custom autograd function.
        
        Args:
            ctx: Context object used to stash information for backward computation.
            x (Tensor): Input tensor.
            n (Tensor): Weights of the numerator polynomial.
            d (Tensor): Weights of the denominator polynomial.
            group (int): The group number.

        Returns:
            Tensor: The result of the rational function applied to the input tensor.
        """
        ctx.save_for_backward(x, n, d)
        ctx.group = group
        x = torch.ops.grkan.grkan_forward_cuda.default(x, n, d, group)
        return x

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        """
        Backward pass of the custom autograd function.
        
        Args:
            ctx: Context object from the forward pass.
            grad_output (Tensor): Gradient of the output tensor.

        Returns:
            tuple: Gradients of the input, numerator, denominator.
        """
        x, n, d = ctx.saved_tensors
        group = ctx.group
        d_x, d_n, d_d = torch.ops.grkan.grkan_backward_cuda.default(grad_output, x, n, d, group)
        return d_x, d_n, d_d, None

class GRKAN(torch.nn.Module):
    
    def __init__(self, num_groups=8, default_fn="swish"):
        super().__init__()
        
        self.num_groups = num_groups
        self.default_fn = default_fn
        
        self.numerator = torch.nn.Parameter(torch.empty(num_groups * 6))
        self.denominator = torch.nn.Parameter(torch.empty(num_groups * 4))
        
        self.initialize_weights()
    
    def initialize_weights(self):
        # Workaround for FSDP that ensures correct handling of param initialization
        if self.numerator.size()[0] == self.num_groups * 6:
            init_numerator = torch.tensor(sum((get_fn_init_data(self.default_fn)["init_numerator"] for _ in range(self.num_groups)), [])).float()
            with torch.no_grad():
                self.numerator.data.copy_(init_numerator)
        else:
            torch.nn.init.trunc_normal_(self.numerator, mean=0.0, std=0.02)
        
        if self.denominator.size()[0] == self.num_groups * 4:
            init_denominator = torch.tensor(sum((get_fn_init_data(self.default_fn)["init_denominator"] for _ in range(self.num_groups)), [])).float()
            with torch.no_grad():
                self.denominator.data.copy_(init_denominator)
        else:
            torch.nn.init.trunc_normal_(self.denominator, mean=0.0, std=0.02)
            
    def reset_parameters(self):
        self.initialize_weights()
        
    def forward(self, x):
        assert x.dim() == 3, "Input tensor must be 3D (batch, length, channels)"
        assert x.device.type == "cuda", "Only 'cuda' device is supported"
        return grkan_cuda_op.apply(x, self.numerator.view(self.num_groups, 6), self.denominator.view(self.num_groups, 4), self.num_groups)

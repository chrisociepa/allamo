import inspect
import math
import torch

from allamo.configuration import AllamoConfiguration
from allamo.model.model import AllamoTransformer
from allamo.training_context import TrainingContext
from allamo.logging import logger

def is_weight_decay_forbidden(param_name):
    return param_name.endswith('.bias') or param_name.endswith('_norm.weight') or param_name == 'norm.weight'

def configure_optimizer(model: AllamoTransformer, config: AllamoConfiguration, device_type: str):
    # start with all of the candidate parameters
    param_dict = {param_name: p for param_name, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {param_name: p for param_name, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if not is_weight_decay_forbidden(n)]
    nodecay_params = [p for n, p in param_dict.items() if is_weight_decay_forbidden(n)]
    optim_groups = [
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(f"Decayed parameter tensors: {len(decay_params):,}, with {num_decay_params:,} parameters")
    logger.info(f"Non-decayed parameter tensors: {len(nodecay_params):,}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=config.learning_rate, betas=(config.beta1, config.beta2), **extra_args)
    logger.info(f"Using fused AdamW: {use_fused}")
    return optimizer

def calculate_learning_rate(train_ctx: TrainingContext, config: AllamoConfiguration):
    """ learning rate decay scheduler (cosine with warmup) """
    if train_ctx.iter_num < config.warmup_iters:
        return config.learning_rate * train_ctx.iter_num / config.warmup_iters
        
    if config.decay_lr:   
        if train_ctx.iter_num >= config.lr_decay_iters:
            return config.min_lr
        if config.lr_decay_reset_iters is not None and config.lr_decay_reset_iters > 0:
            decay_ratio = (train_ctx.iter_num % config.lr_decay_reset_iters) / config.lr_decay_reset_iters
        else:
            decay_ratio = (train_ctx.iter_num - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return config.min_lr + coeff * (config.learning_rate - config.min_lr)
    else:
        return config.learning_rate
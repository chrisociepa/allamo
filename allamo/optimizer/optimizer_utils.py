import inspect
import math
import torch

from allamo.configuration import AllamoConfiguration
from allamo.model.modeling_utils import BaseModel
from allamo.training_context import TrainingContext
from allamo.logging import logger

def is_weight_decay_forbidden(param_name):
    return param_name.endswith('.bias') or param_name.endswith('_norm.weight') or param_name == 'norm.weight'

def _is_muon_eligible(param_name, param):
    """A 2D hidden weight (not embedding, not lm_head) is optimized by Muon+."""
    if param.ndim != 2:
        return False
    if 'tok_embeddings' in param_name or 'embed' in param_name:
        return False
    if 'lm_head' in param_name:
        return False
    return True

def _configure_adamw(param_dict, config, device_type):
    decay_params = [p for n, p in param_dict.items() if not is_weight_decay_forbidden(n)]
    nodecay_params = [p for n, p in param_dict.items() if is_weight_decay_forbidden(n)]
    optim_groups = [
        {'params': decay_params, 'weight_decay': config.weight_decay, 'lr_scale': 1.0},
        {'params': nodecay_params, 'weight_decay': 0.0, 'lr_scale': 1.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(f"Decayed parameter tensors: {len(decay_params):,}, with {num_decay_params:,} parameters")
    logger.info(f"Non-decayed parameter tensors: {len(nodecay_params):,}, with {num_nodecay_params:,} parameters")
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=config.learning_rate, betas=(config.beta1, config.beta2), **extra_args)
    logger.info(f"Using fused AdamW: {use_fused}")
    return optimizer

def _configure_muon_plus(param_dict, config, device_type):
    from allamo.optimizer.muon_plus import MuonPlusWithAuxAdam

    muon_params = []
    adam_decay_params = []
    adam_nodecay_params = []

    for name, p in param_dict.items():
        if _is_muon_eligible(name, p):
            muon_params.append(p)
        elif is_weight_decay_forbidden(name):
            adam_nodecay_params.append(p)
        else:
            adam_decay_params.append(p)

    num_muon = sum(p.numel() for p in muon_params)
    num_adam_decay = sum(p.numel() for p in adam_decay_params)
    num_adam_nodecay = sum(p.numel() for p in adam_nodecay_params)
    logger.info(f"Muon+ parameter tensors: {len(muon_params):,}, with {num_muon:,} parameters")
    logger.info(f"AdamW decayed parameter tensors: {len(adam_decay_params):,}, with {num_adam_decay:,} parameters")
    logger.info(f"AdamW non-decayed parameter tensors: {len(adam_nodecay_params):,}, with {num_adam_nodecay:,} parameters")

    muon_group = dict(
        params=muon_params,
        lr=config.muon_lr,
        weight_decay=config.weight_decay,
        momentum=config.muon_momentum,
        ns_steps=config.muon_ns_steps,
        norm_mode=config.muon_norm_mode,
        use_muon=True,
        lr_scale=config.muon_lr / config.learning_rate if config.learning_rate > 0 else 1.0,
    )

    adam_groups = []
    if adam_decay_params:
        adam_groups.append(dict(
            params=adam_decay_params,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=1e-10,
            weight_decay=config.weight_decay,
            use_muon=False,
            lr_scale=1.0,
        ))
    if adam_nodecay_params:
        adam_groups.append(dict(
            params=adam_nodecay_params,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=1e-10,
            weight_decay=0.0,
            use_muon=False,
            lr_scale=1.0,
        ))

    optimizer = MuonPlusWithAuxAdam([muon_group] + adam_groups)
    logger.info("Using Muon+ with auxiliary AdamW")
    return optimizer

class CombinedOptimizer:
    """Wraps multiple optimizers (e.g. torch.optim.Muon + AdamW) with a unified interface.

    GradScaler requires a real torch.optim.Optimizer for unscale_/step, so
    scaler_unscale and scaler_step delegate to each sub-optimizer individually.
    The training loop must check for CombinedOptimizer at those call sites.
    """

    def __init__(self, optimizers):
        self.optimizers = optimizers

    @property
    def param_groups(self):
        groups = []
        for opt in self.optimizers:
            groups.extend(opt.param_groups)
        return groups

    def step(self, closure=None):
        for opt in self.optimizers:
            opt.step(closure)

    def zero_grad(self, set_to_none=True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none)

    def state_dict(self):
        return [opt.state_dict() for opt in self.optimizers]

    def load_state_dict(self, state_dicts):
        for opt, sd in zip(self.optimizers, state_dicts):
            opt.load_state_dict(sd)


def _configure_muon(param_dict, config, device_type):
    muon_params = []
    adam_decay_params = []
    adam_nodecay_params = []

    for name, p in param_dict.items():
        if _is_muon_eligible(name, p):
            muon_params.append(p)
        elif is_weight_decay_forbidden(name):
            adam_nodecay_params.append(p)
        else:
            adam_decay_params.append(p)

    num_muon = sum(p.numel() for p in muon_params)
    num_adam_decay = sum(p.numel() for p in adam_decay_params)
    num_adam_nodecay = sum(p.numel() for p in adam_nodecay_params)
    logger.info(f"Muon parameter tensors: {len(muon_params):,}, with {num_muon:,} parameters")
    logger.info(f"AdamW decayed parameter tensors: {len(adam_decay_params):,}, with {num_adam_decay:,} parameters")
    logger.info(f"AdamW non-decayed parameter tensors: {len(adam_nodecay_params):,}, with {num_adam_nodecay:,} parameters")

    lr_scale = config.muon_lr / config.learning_rate if config.learning_rate > 0 else 1.0
    muon_opt = torch.optim.Muon(
        [{'params': muon_params, 'lr_scale': lr_scale}],
        lr=config.muon_lr,
        weight_decay=config.weight_decay,
        momentum=config.muon_momentum,
        ns_steps=config.muon_ns_steps,
    )
    logger.info(f"Using torch.optim.Muon (ns_steps={config.muon_ns_steps})")

    adam_groups = []
    if adam_decay_params:
        adam_groups.append({'params': adam_decay_params, 'weight_decay': config.weight_decay, 'lr_scale': 1.0})
    if adam_nodecay_params:
        adam_groups.append({'params': adam_nodecay_params, 'weight_decay': 0.0, 'lr_scale': 1.0})
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    adam_opt = torch.optim.AdamW(adam_groups, lr=config.learning_rate, betas=(config.beta1, config.beta2), **extra_args)
    logger.info(f"Using fused AdamW for non-Muon params: {use_fused}")

    return CombinedOptimizer([muon_opt, adam_opt])


def configure_optimizer(model: BaseModel, config: AllamoConfiguration, device_type: str):
    param_dict = {param_name: p for param_name, p in model.named_parameters()}
    param_dict = {param_name: p for param_name, p in param_dict.items() if p.requires_grad}

    if config.optimizer == 'muon':
        return _configure_muon(param_dict, config, device_type)
    if config.optimizer == 'muon_plus':
        return _configure_muon_plus(param_dict, config, device_type)
    return _configure_adamw(param_dict, config, device_type)

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
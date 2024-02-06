import datetime
import math

def remove_unwanted_prefix_from_model_state_dict(state_dict):
    unwanted_prefix = '_orig_mod.'
    unwanted_prefix_len = len(unwanted_prefix)
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[unwanted_prefix_len:]] = state_dict.pop(k)
            
def remove_unwanted_prefix_from_optimizer_state_dict(optimizer_state_dict):
    if "param_groups" in optimizer_state_dict:
        unwanted_prefix = '_orig_mod.'
        unwanted_prefix_len = len(unwanted_prefix)
        for param_group in optimizer_state_dict["param_groups"]:
            param_group['params'] = [p[unwanted_prefix_len:] if p.startswith(unwanted_prefix) else p for p in param_group['params']]
            
def get_lr(iter_num, config):
    """ learning rate decay scheduler (cosine with warmup) """
    if iter_num < config.warmup_iters:
        return config.learning_rate * iter_num / config.warmup_iters
        
    if config.decay_lr:   
        if iter_num >= config.lr_decay_iters:
            return config.min_lr
        if config.lr_decay_reset_iters is not None:
            decay_ratio = (iter_num % config.lr_decay_reset_iters) / config.lr_decay_reset_iters
        else:
            decay_ratio = (iter_num - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return config.min_lr + coeff * (config.learning_rate - config.min_lr)
    else:
        return config.learning_rate 
    
def get_grad_accum(current_grad_accum_steps, iter_num, config):
    """ grad_accum scheduler (when enabled) """
    if config.grad_accum_schedule and current_grad_accum_steps < config.grad_accum_max and iter_num % (config.grad_accum_max_iter/100) == 0:
        return min(current_grad_accum_steps + 1, config.grad_accum_max)
    else:
        return current_grad_accum_steps
        
def format_seconds_as_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}:{int(minutes):02}:{int(seconds):02}"
        
def calculate_eta(iter_num, start_iter, start_timestamp, config):
    current_time = datetime.datetime.now()
    elapsed_time = current_time - start_timestamp
    elapsed_iters = iter_num - start_iter
    if elapsed_iters < 1:
        return 'N/A'
    avg_time_per_iter = elapsed_time.total_seconds() / elapsed_iters
    eta_seconds = math.ceil(avg_time_per_iter * (config.max_iters - iter_num))
    return format_seconds_as_time(eta_seconds)
    
def has_next_iter_to_perform(iter_num, config, simple_data_loader):
    if config.num_train_epochs is not None and simple_data_loader.epoch >= config.num_train_epochs:
        return False
    return iter_num <= config.max_iters
    
def estimate_mfu(model_num_params, config, fwdbwd_per_iter, dt):
    # estimate model flops utilization (MFU) in units of GPU bfloat16 peak FLOPS
    # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
    N = model_num_params
    L, H, Q, T = config.n_layer, config.n_head, config.head_size, config.block_size
    flops_per_token = 6 * N + 12 * L * H * Q * T
    flops_per_fwdbwd = flops_per_token * T
    flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
    # express our flops throughput as ratio of GPU bfloat16 peak flops
    flops_achieved = flops_per_iter * (1.0/dt) # per second
    return flops_achieved / config.mfu_flops_peak


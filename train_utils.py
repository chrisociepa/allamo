import datetime
import hashlib
import math
import os
import subprocess
import torch
from typing import Optional

def rename_file_to_prev_version(file_path):
    if os.path.exists(file_path):
        os.rename(file_path, file_path + '.prev')
        
def calculate_md5(file_path, chunk_size=1024*1024):
    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()

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
    
def get_model_checkpoint_path(ckpt_file_name, ckpt_dir):
    return os.path.join(ckpt_dir, f'model_{ckpt_file_name}.pt')
    
def get_config_checkpoint_path(ckpt_file_name, ckpt_dir):
    return os.path.join(ckpt_dir, f'config_{ckpt_file_name}.json')
    
def get_optimizer_checkpoint_path(ckpt_file_name, ckpt_dir):
    return os.path.join(ckpt_dir, f'optimizer_{ckpt_file_name}.pt')
    
def model_checkpoint_files_exist(ckpt_file_name, ckpt_dir):
    return os.path.exists(get_config_checkpoint_path(ckpt_file_name, ckpt_dir)) \
            and os.path.exists(get_model_checkpoint_path(ckpt_file_name, ckpt_dir))

def run_checkpoint_hook_program(hook_program, run_uuid, training_uuid, epoch, iter_num, ckpt_file_name, config):
    env_variables = {
        "ALLAMO_EPOCH_HOOK_RUN_UUID": run_uuid,
        "ALLAMO_EPOCH_HOOK_TRAINING_UUID": training_uuid,
        "ALLAMO_EPOCH_HOOK_EPOCH": str(epoch),
        "ALLAMO_EPOCH_HOOK_ITERATION": str(iter_num),
        "ALLAMO_EPOCH_HOOK_MODEL_CKPT_PATH": str(os.path.abspath(get_model_checkpoint_path(ckpt_file_name, config.out_dir))),
        "ALLAMO_EPOCH_HOOK_CONFIG_CKPT_PATH": str(os.path.abspath(get_config_checkpoint_path(ckpt_file_name, config.out_dir)))
    }
    try:
        process = subprocess.Popen(hook_program, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env_variables)
        return process.pid
    except Exception as err:
        return f"n/a - Error: {err}"

def override_numa_affinity(local_rank: int, verbose: Optional[bool] = None) -> None:
    if torch.cuda.is_available():
        try:
            import pynvml as nvml
        except ImportError:
            print("To set CPU affinity on CUDA GPUs the `pynvml` package must be available. (`pip install pynvml`)")
            return

        # The below code is based on https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow2/LanguageModeling/BERT/gpu_affinity.py
        nvml.nvmlInit()
        num_elements = math.ceil(os.cpu_count() / 64)
        handle = nvml.nvmlDeviceGetHandleByIndex(local_rank)
        affinity_string = ""
        for j in nvml.nvmlDeviceGetCpuAffinity(handle, num_elements):
            # assume nvml returns list of 64 bit ints
            affinity_string = f"{j:064b}{affinity_string}"
        affinity_list = [int(x) for x in affinity_string]
        affinity_list.reverse()  # so core 0 is the 0th element
        affinity_to_set = set([i for i, e in enumerate(affinity_list) if e != 0])
        current_affinity = set(os.sched_getaffinity(0))
        affinity_to_set = affinity_to_set.intersection(current_affinity)
        if affinity_to_set:
            os.sched_setaffinity(0, affinity_to_set)
            if verbose:
                cpu_cores = os.sched_getaffinity(0)
                print(f"Assigning {len(cpu_cores)} cpu cores to process {local_rank}: {cpu_cores}")
        else:
            print("No affinity available to set!")

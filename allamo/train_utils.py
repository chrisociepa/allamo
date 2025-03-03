import dataclasses
import hashlib
import os
import shutil
from allamo.model.model import AllamoTransformerConfig

def rename_file_to_prev_version(file_path):
    if os.path.exists(file_path):
        os.rename(file_path, file_path + '.prev')

def copy_file_to_prev_version(file_path, postfix='.prev'):
    if os.path.exists(file_path):
        shutil.copy(file_path, file_path + postfix)

def copy_dir_to_prev_version(dir_path, postfix='-prev'):
    if os.path.exists(dir_path):
        new_path = dir_path + postfix
        if os.path.exists(new_path):
            shutil.rmtree(new_path)
        shutil.copytree(dir_path, new_path)
        
def calculate_md5(file_path, chunk_size=1024*1024):
    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()

def remove_unwanted_prefix_from_model_state_dict(state_dict, unwanted_prefix = '_orig_mod.'):
    unwanted_prefix_len = len(unwanted_prefix)
    for k, _ in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[unwanted_prefix_len:]] = state_dict.pop(k)
            
def remove_unwanted_prefix_from_optimizer_state_dict(optimizer_state_dict, unwanted_prefix = '_orig_mod.'):
    if "param_groups" in optimizer_state_dict:
        unwanted_prefix_len = len(unwanted_prefix)
        for param_group in optimizer_state_dict["param_groups"]:
            param_group['params'] = [p[unwanted_prefix_len:] if p.startswith(unwanted_prefix) else p for p in param_group['params']]
            
def format_seconds_as_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}:{int(minutes):02}:{int(seconds):02}"
        
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

def get_model_config_field_names():
    return [f.name for f in dataclasses.fields(AllamoTransformerConfig)]

def create_model_config(config):
    model_args = {k: getattr(config, k) for k in get_model_config_field_names() if hasattr(config, k)}
    return AllamoTransformerConfig(**model_args)

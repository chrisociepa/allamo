import dataclasses
import hashlib
import os
import shutil

import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
)
from torch.utils.checkpoint import checkpoint

from allamo.configuration import AllamoConfiguration
from allamo.logging import logger
from allamo.model.modeling_utils import ModelSpec

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

def get_model_config_field_names(model_spec: ModelSpec):
    return [f.name for f in dataclasses.fields(model_spec.model_config_cls)]

def create_model_config(config: AllamoConfiguration, model_spec: ModelSpec):
    model_args = {k: getattr(config, k) for k in get_model_config_field_names(model_spec) if hasattr(config, k)}
    return model_spec.model_config_cls(**model_args)

def apply_activation_checkpointing(model: nn.Module, config: AllamoConfiguration):
    excluded_layers = set()
    if config.gradient_checkpointing_excluded_layers > 0:
        total_layers = config.n_layer
        excluded_count = min(config.gradient_checkpointing_excluded_layers, total_layers)
        
        if excluded_count >= total_layers:
            logger.warning(f"All {total_layers} layers are excluded, so activation checkpointing won't be applied.")
            return
        else:
            step = total_layers / excluded_count
            excluded_layers = set([int(i * step) for i in range(excluded_count)])
            actual_excluded = len(excluded_layers)
            if actual_excluded < excluded_count:
                additional_needed = excluded_count - actual_excluded
                layer_idx = 0
                while additional_needed > 0 and layer_idx < total_layers:
                    if layer_idx not in excluded_layers:
                        excluded_layers.add(layer_idx)
                        additional_needed -= 1
                    layer_idx += 1

    for layer_id in range(len(model.layers)):
        if layer_id not in excluded_layers:
            model.layers[layer_id] = checkpoint_wrapper(
                model.layers[layer_id],
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                checkpoint_fn=checkpoint,
                use_reentrant=False,
                preserve_rng_state=False,
            )
    logger.info(f"Activation checkpointing applied to the model (excluded {max(config.gradient_checkpointing_excluded_layers, 0)} layers)")
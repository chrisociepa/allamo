import argparse
import os
from enum import Enum
from typing import Union, Dict, Any

import torch
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.format_utils import _EmptyStateDictLoadPlanner
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
from torch.distributed.checkpoint.state_dict_saver import _save_state_dict

def remove_unwanted_prefix_from_model_state_dict(state_dict, unwanted_prefix = '_orig_mod.'):
    unwanted_prefix_len = len(unwanted_prefix)
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[unwanted_prefix_len:]] = state_dict.pop(k)
            
def add_prefix_to_model_state_dict(state_dict, prefix):
    for k,v in list(state_dict.items()):
        state_dict[prefix + k] = state_dict.pop(k)

def require_conversion(key):
    for key_part in ["tok_embeddings", "lm_head", "attention.q_proj", "attention.k_proj", "attention.v_proj", "feed_forward.gate_proj", "feed_forward.up_proj"]:
        if key_part in key:
            return True
    return False
    
def transform_tensor(tensor, tp_size, world_size, to_dcp):
    chunk_size = len(tensor) // world_size
    dcp_tensor = tensor.clone().detach()
    for chunk_id in range(world_size):
        if chunk_id != 0 and chunk_id < world_size-1:
            chunk_offset = chunk_size * chunk_id
            dcp_chunk_offset = (chunk_offset * tp_size) % (len(tensor) - chunk_size)
            for i in range(chunk_size):
                if to_dcp:
                    dcp_tensor[dcp_chunk_offset+i] = tensor[chunk_offset+i]
                else:
                    tensor[chunk_offset+i] = dcp_tensor[dcp_chunk_offset+i]
    return dcp_tensor if to_dcp else tensor

def convert_state_dict(state_dict, tp_size, world_size, to_dcp):
    for k,v in list(state_dict.items()):
        if require_conversion(k):
            state_dict[k] = transform_tensor(state_dict.pop(k), tp_size, world_size, to_dcp)

def dcp_to_torch_save(dcp_checkpoint_dir: Union[str, os.PathLike], torch_save_path: Union[str, os.PathLike], state_key: str, tp_size: int, world_size: int):
    state_dict: STATE_DICT_TYPE = {}

    _load_state_dict(
        state_dict,
        storage_reader=FileSystemReader(dcp_checkpoint_dir),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    
    if state_key:
        if state_key in state_dict:
            state_dict = state_dict[state_key]
        else:
            print(f"Key '{state_key}' not found. Using full state dict with the following keys: {', '.join(state_dict.keys())}")
            
    convert_state_dict(state_dict, tp_size, world_size, to_dcp=False)
    
    torch.save(state_dict, torch_save_path)
    print(f"Converting completed. New model saved in {torch_save_path}")

def torch_save_to_dcp(torch_save_path: Union[str, os.PathLike], dcp_checkpoint_dir: Union[str, os.PathLike], state_key: str, tp_size: int, world_size: int):
    state_dict = torch.load(torch_save_path)
    remove_unwanted_prefix_from_model_state_dict(state_dict)
    
    convert_state_dict(state_dict, tp_size, world_size, to_dcp=True)
    
    if state_key:
        add_prefix_to_model_state_dict(state_dict, state_key + ".")
        print(f"Prefixed model state dict with '{state_key}.'")
    
    # we don't need stateful behavior here because the expectation is anything loaded by
    # torch.load would not contain stateful objects.
    _save_state_dict(
        state_dict, storage_writer=FileSystemWriter(dcp_checkpoint_dir), no_dist=True
    )
    print(f"Converting completed. New model saved in {dcp_checkpoint_dir}")

if __name__ == "__main__":
    
    class FormatMode(Enum):
        TORCH_TO_DCP = "torch_to_dcp"
        DCP_TO_TORCH = "dcp_to_torch"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True, choices=[m.value for m in FormatMode], help="Conversion mode")
    parser.add_argument('-s', '--src', type=str, required=True, help="Path to the source model")
    parser.add_argument('-d', '--dst', type=str, required=True, help="Path to the destination model")
    parser.add_argument('-k', '--state_key', type=str, help="Dictionary key with desired state")
    parser.add_argument('--tp_size', type=int, required=True, help="Tensor paralellism size")
    parser.add_argument('--world_size', type=int, required=True, help="World size")
    args = parser.parse_args()
    
    print(
        f"Converting checkpoint from {args.src} to {args.dst} using method: '{args.mode}'"
    )
    
    checkpoint_missing_warning = (
        f"No checkpoint found at {args.src}. Skipping conversion."
    )
    if args.mode == FormatMode.TORCH_TO_DCP.value:
        if os.path.isfile(args.src):
            os.makedirs(args.dst, exist_ok=True)
            torch_save_to_dcp(args.src, args.dst, args.state_key, args.tp_size, args.world_size)
        else:
            print(checkpoint_missing_warning)
    elif args.mode == FormatMode.DCP_TO_TORCH.value:
        if os.path.isdir(args.src):
            dcp_to_torch_save(args.src, args.dst, args.state_key, args.tp_size, args.world_size)
        else:
            print(checkpoint_missing_warning)


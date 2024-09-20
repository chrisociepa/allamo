import argparse
import os
from enum import Enum
from typing import Union

import torch
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.format_utils import _EmptyStateDictLoadPlanner
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
from torch.distributed.checkpoint.state_dict_saver import _save_state_dict

from allamo.logging import configure_logger, logger
from allamo.train_utils import remove_unwanted_prefix_from_model_state_dict
            
def add_prefix_to_model_state_dict(state_dict, prefix):
    for k, _ in list(state_dict.items()):
        state_dict[prefix + k] = state_dict.pop(k)
    
def dcp_to_torch_save(dcp_checkpoint_dir: Union[str, os.PathLike], torch_save_path: Union[str, os.PathLike], state_key: str):
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
            logger.warning(f"Key '{state_key}' not found. Using full state dict with the following keys: {', '.join(state_dict.keys())}")
            
    torch.save(state_dict, torch_save_path)
    logger.info(f"Conversion completed. New model saved in {torch_save_path}")

def torch_save_to_dcp(torch_save_path: Union[str, os.PathLike], dcp_checkpoint_dir: Union[str, os.PathLike], state_key: str):
    state_dict = torch.load(torch_save_path)
    remove_unwanted_prefix_from_model_state_dict(state_dict)
    
    if state_key:
        add_prefix_to_model_state_dict(state_dict, state_key + ".")
        logger.info(f"Prefixed model state dict with '{state_key}.'")
    
    # we don't need stateful behavior here because the expectation is anything loaded by
    # torch.load would not contain stateful objects.
    _save_state_dict(
        state_dict, storage_writer=FileSystemWriter(dcp_checkpoint_dir), no_dist=True
    )
    logger.info(f"Conversion completed. New model saved in {dcp_checkpoint_dir}")

if __name__ == "__main__":
    
    class FormatMode(Enum):
        TORCH_TO_DCP = "torch_to_dcp"
        DCP_TO_TORCH = "dcp_to_torch"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True, choices=[m.value for m in FormatMode], help="Conversion mode")
    parser.add_argument('-s', '--src', type=str, required=True, help="Path to the source model")
    parser.add_argument('-d', '--dst', type=str, required=True, help="Path to the target model")
    parser.add_argument('-k', '--state_key', type=str, help="Dictionary key with desired state")
    args = parser.parse_args()
    
    configure_logger()
    logger.info(f"Converting checkpoint from {args.src} to {args.dst} using method: '{args.mode}'")
    
    checkpoint_missing_warning = (
        f"No checkpoint found at {args.src}. Skipping conversion."
    )
    if args.mode == FormatMode.TORCH_TO_DCP.value:
        if os.path.isfile(args.src):
            os.makedirs(args.dst, exist_ok=True)
            torch_save_to_dcp(args.src, args.dst, args.state_key)
        else:
            logger.warning(checkpoint_missing_warning)
    elif args.mode == FormatMode.DCP_TO_TORCH.value:
        if os.path.isdir(args.src):
            dcp_to_torch_save(args.src, args.dst, args.state_key)
        else:
            logger.warning(checkpoint_missing_warning)

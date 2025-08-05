"""
Use this script to merge multiple PyTorch model checkpoints into a single checkpoint.

Merge config example:
{
    "checkpoints": [
        {
            "path": "path/to/checkpoint_1",
            "name_base": "last_eval_ckpt",
            "weight": 1.0
        },
        {
            "path": "path/to/checkpoint_2",
            "name_base": "ckpt",
            "weight": 1.0
        }
    ]
}
"""

import argparse
import json
import os
import torch
from allamo.logging import configure_logger, logger
from allamo.train_utils import (
    get_model_checkpoint_path,
    get_config_checkpoint_path,
    remove_unwanted_prefix_from_model_state_dict,
)

def merge_model(config_path, output_dir_path, output_checkpoint_name_base):
    os.makedirs(output_dir_path, exist_ok=True)

    logger.info(f"loading merge config from {config_path}...")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    checkpoints = config['checkpoints']
    assert len(checkpoints) > 1, "at least two checkpoints must be provided"
    
    final_config_checkpoint = None
    state_dict = None
    total_weight = 0
    for checkpoint in checkpoints:
        logger.info(f"loading checkpoint from {checkpoint['path']}...")
        with open(get_config_checkpoint_path(checkpoint['name_base'], checkpoint['path']), "r", encoding="utf-8") as f:
            config_checkpoint = json.load(f)
        model_checkpoint = torch.load(get_model_checkpoint_path(checkpoint['name_base'], checkpoint['path']), map_location='cpu', weights_only=True)
        remove_unwanted_prefix_from_model_state_dict(model_checkpoint)

        weight = checkpoint['weight'] if 'weight' in checkpoint else 1.0
        total_weight += weight
        
        logger.info(f"merging checkpoint with weight {weight}")
        if state_dict is None:
            state_dict = model_checkpoint
            final_config_checkpoint = config_checkpoint
        else:
            for k, v in model_checkpoint.items():
                if k in state_dict:
                    state_dict[k] += v * weight
                else:
                    logger.warning(f"key {k} not found in state_dict, adding it")
                    state_dict[k] = v * weight
    
    assert total_weight > 0
    logger.info(f"normalizing state_dict by total weight {total_weight}")
    for k, v in state_dict.items():
        state_dict[k] = v / total_weight
    
    param_count = 0
    param_bytes = 0
    for k, v in state_dict.items():
        param_count += v.numel()
        param_bytes += v.numel() * v.element_size()
    
    param_count /= 1e6
    param_bytes /= 1024**2
    logger.info(f"Model parameters: {param_count:.2f}M. Est. Size: {param_bytes:.3f}MB")
            
    ckpt_file_path = get_config_checkpoint_path(output_checkpoint_name_base, output_dir_path)
    logger.info(f"saving config checkpoint to {ckpt_file_path}")
    with open(ckpt_file_path, "w", encoding="utf-8") as f:
        json.dump(final_config_checkpoint, f, indent=4, ensure_ascii=False)
    ckpt_file_path = get_model_checkpoint_path(output_checkpoint_name_base, output_dir_path)
    logger.info(f"saving model checkpoint to {ckpt_file_path}")
    torch.save(state_dict, ckpt_file_path)
    logger.info(f"checkpoint files saved in {output_dir_path}")

if __name__ == "__main__":
    configure_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="Path to merge config file",
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write merged model",
    )
    parser.add_argument(
        "--output_checkpoint_name_base",
        default='up-scaled_ckpt',
        help="Output checkpoint file name base",
    )
    args = parser.parse_args()
    merge_model(
        config_path=args.config,
        output_dir_path=args.output_dir,
        output_checkpoint_name_base=args.output_checkpoint_name_base
    )

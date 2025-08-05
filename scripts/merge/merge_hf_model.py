"""
Use this script to merge multiple Huggingface model checkpoints into a single checkpoint.

Merge config example:
{
    "checkpoints": [
        {
            "path": "path/to/checkpoint_1",
            "weight": 1.0
        },
        {
            "path": "path/to/checkpoint_2",
            "weight": 1.0
        }
    ]
}
"""

import argparse
import json
import os
import torch
from transformers import AutoModelForCausalLM
from allamo.logging import configure_logger, logger

def merge_model(config_path, output_dir_path, output_dtype):
    os.makedirs(output_dir_path, exist_ok=True)

    logger.info(f"loading merge config from {config_path}...")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    checkpoints = config['checkpoints']
    assert len(checkpoints) > 1, "at least two checkpoints must be provided"

    if output_dtype:
        torch_dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[output_dtype]
    else:
        output_dtype = 'float32'
        torch_dtype = torch.float32
    
    final_hf_model = None
    state_dict = None
    total_weight = 0
    for checkpoint in checkpoints:
        logger.info(f"loading checkpoint from {checkpoint['path']}...")
        hf_model = AutoModelForCausalLM.from_pretrained(checkpoint['path'], torch_dtype=torch.float32, low_cpu_mem_usage=True)
        model_checkpoint = hf_model.state_dict()

        weight = checkpoint['weight'] if 'weight' in checkpoint else 1.0
        total_weight += weight
        
        logger.info(f"merging checkpoint with weight {weight}")
        if state_dict is None:
            state_dict = model_checkpoint
            final_hf_model = hf_model
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

    if output_dtype != 'float32':
        logger.info(f"converting model to {output_dtype}")
        final_hf_model = final_hf_model.to(torch_dtype)
        final_hf_model.config.torch_dtype = output_dtype

    logger.info(f"saving model to {output_dir_path}")
    final_hf_model.save_pretrained(output_dir_path)
    logger.info("model files saved")

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
        "--output_dtype",
        choices=['float32', 'bfloat16', 'float16'],
        help="Save the model under a specific dtype",
    )
    args = parser.parse_args()
    merge_model(
        config_path=args.config,
        output_dir_path=args.output_dir,
        output_dtype=args.output_dtype
    )

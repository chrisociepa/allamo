"""
Use this file for scaling your model with a method called depth up-scaling (DUS).

More details: https://arxiv.org/abs/2312.15166
"""

import argparse
import datetime
import gc
import json
import logging
import os
import shutil
import sys
import torch

sys.path.append(os.path.abspath('..'))
from train_utils import (
    get_model_checkpoint_path,
    get_config_checkpoint_path,
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger('AllamoModelDepthUpScaler')

def prepare_layer_keys_mapping(n_layers, last_layers_to_drop, first_layers_to_drop):
    mapping_pairs = []
    bottom_layers = n_layers - last_layers_to_drop
    top_layers = n_layers - first_layers_to_drop
    
    # bottom layers
    for layer_i in range(bottom_layers):
        mapping_pairs.append((layer_i, layer_i))
     
    # top layers
    for layer_i in range(top_layers):
        src_layer_idx = first_layers_to_drop + layer_i
        dest_layer_idx = bottom_layers + layer_i
        mapping_pairs.append((dest_layer_idx, src_layer_idx))
    
    return mapping_pairs

def depth_up_scale_model(input_dir_path, checkpoint_name_base, output_dir_path, last_layers_to_drop, first_layers_to_drop, bfloat16):
    os.makedirs(output_dir_path, exist_ok=True)
    
    logger.info(f"loading checkpoint from {input_dir_path}...")
    with open(get_config_checkpoint_path(checkpoint_name_base, input_dir_path), "r", encoding="utf-8") as f:
        config_checkpoint = json.load(f)
    model_checkpoint = torch.load(get_model_checkpoint_path(checkpoint_name_base, input_dir_path), map_location='cpu')
    
    unwanted_prefix = '_orig_mod.'
    for k,v in list(model_checkpoint.items()):
        if k.startswith(unwanted_prefix):
            model_checkpoint[k[len(unwanted_prefix):]] = model_checkpoint.pop(k)
    
    state_dict = {
        "tok_embeddings.weight": model_checkpoint["tok_embeddings.weight"],
        "norm.weight": model_checkpoint["norm.weight"],
        "lm_head.weight": model_checkpoint["lm_head.weight"],
    }
    
    layer_mapping_pairs = prepare_layer_keys_mapping(config_checkpoint['model_args']['n_layer'], last_layers_to_drop, first_layers_to_drop)
    
    # you can customize the mapping here, e.g., replace some layers with others
    
    for dest_layer_idx, src_layer_idx in layer_mapping_pairs:
        logger.info(f"copying weights from layer {src_layer_idx} to layer {dest_layer_idx}")
        state_dict[f"layers.{dest_layer_idx}.attention.q_proj.weight"] = model_checkpoint[f"layers.{src_layer_idx}.attention.q_proj.weight"].clone()
        state_dict[f"layers.{dest_layer_idx}.attention.k_proj.weight"] = model_checkpoint[f"layers.{src_layer_idx}.attention.k_proj.weight"].clone()
        state_dict[f"layers.{dest_layer_idx}.attention.v_proj.weight"] = model_checkpoint[f"layers.{src_layer_idx}.attention.v_proj.weight"].clone()
        state_dict[f"layers.{dest_layer_idx}.attention.c_proj.weight"] = model_checkpoint[f"layers.{src_layer_idx}.attention.c_proj.weight"].clone()
        state_dict[f"layers.{dest_layer_idx}.feed_forward.gate_proj.weight"] = model_checkpoint[f"layers.{src_layer_idx}.feed_forward.gate_proj.weight"].clone()
        state_dict[f"layers.{dest_layer_idx}.feed_forward.down_proj.weight"] = model_checkpoint[f"layers.{src_layer_idx}.feed_forward.down_proj.weight"].clone()
        state_dict[f"layers.{dest_layer_idx}.feed_forward.up_proj.weight"] = model_checkpoint[f"layers.{src_layer_idx}.feed_forward.up_proj.weight"].clone()
        state_dict[f"layers.{dest_layer_idx}.attention_norm.weight"] = model_checkpoint[f"layers.{src_layer_idx}.attention_norm.weight"].clone()
        state_dict[f"layers.{dest_layer_idx}.ffn_norm.weight"] = model_checkpoint[f"layers.{src_layer_idx}.ffn_norm.weight"].clone()
    
    if bfloat16:
        logger.info("converting weights to bfloat16")
    param_count = 0
    param_bytes = 0
    for k, v in state_dict.items():
        if bfloat16:
            v = v.to(torch.bfloat16)
            state_dict[k] = v
        param_count += v.numel()
        param_bytes += v.numel() * v.element_size()
    
    config_checkpoint['model_args']['n_layer'] = len(layer_mapping_pairs)
    param_count /= 1e6
    param_bytes /= 1024**2
    logger.info(f"New model layers: {config_checkpoint['model_args']['n_layer']}. Model parameters: {param_count:.2f}M. Est. Size: {param_bytes:.3f}MB")
            
    ckpt_file_name = 'up-scaled_ckpt'
    ckpt_file_path = get_config_checkpoint_path(ckpt_file_name, output_dir_path)
    logger.info(f"saving config checkpoint to {ckpt_file_path}")
    with open(ckpt_file_path, "w", encoding="utf-8") as f:
        json.dump(config_checkpoint, f, indent=4, ensure_ascii=False)
    ckpt_file_path = get_model_checkpoint_path(ckpt_file_name, output_dir_path)
    logger.info(f"saving model checkpoint to {ckpt_file_path}")
    torch.save(state_dict, ckpt_file_path)
    logger.info(f"checkpoint files saved in {output_dir_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Location of ALLaMo weights, which contains a checkpoint file",
    )
    parser.add_argument(
        "--checkpoint_name_base",
        default='ckpt',
        help="Source checkpoint file name base",
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write up-scaled model",
    )
    parser.add_argument(
        "--last_layers_to_drop", type=int,
        help="Number of last layers to drop from the bottom copy of the model",
    )
    parser.add_argument(
        "--first_layers_to_drop", type=int,
        help="Number of first layers to drop from the top copy of the model",
    )
    parser.add_argument(
        "--bfloat16", type=bool,
        help="Convert weights to bfloaf16",
    )
    args = parser.parse_args()
    depth_up_scale_model(
        input_dir_path=args.input_dir,
        checkpoint_name_base=args.checkpoint_name_base,
        output_dir_path=args.output_dir,
        last_layers_to_drop=args.last_layers_to_drop,
        first_layers_to_drop=args.first_layers_to_drop,
        bfloat16=args.bfloat16
    )


if __name__ == "__main__":
    main()
    

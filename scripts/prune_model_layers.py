"""
Use this file for prunning model layers
"""

import argparse
import json
import os
import torch
from allamo.logging import configure_logger, logger
from allamo.train_utils import (
    get_model_checkpoint_path,
    get_config_checkpoint_path,
)

def prepare_layer_keys_mapping(n_layers, num_layers_to_remove):
    num_layers = n_layers - 1 # last layer is excluded since we will always keep it
    step = num_layers // (num_layers - num_layers_to_remove)
    indices_to_keep = [int(i * step) for i in range(num_layers - num_layers_to_remove)]
    mapping_pairs = []
    for layer_i in range(num_layers):
        if layer_i in indices_to_keep:
            mapping_pairs.append((len(mapping_pairs), layer_i))
    assert len(mapping_pairs) == len(indices_to_keep)
    
    mapping_pairs.append((len(mapping_pairs), n_layers - 1))
    
    return mapping_pairs

def prune_model(input_dir_path, input_checkpoint_name_base, output_dir_path, output_checkpoint_name_base, num_layers_to_remove, bfloat16):
    os.makedirs(output_dir_path, exist_ok=True)
    
    logger.info(f"loading checkpoint from {input_dir_path}...")
    with open(get_config_checkpoint_path(input_checkpoint_name_base, input_dir_path), "r", encoding="utf-8") as f:
        config_checkpoint = json.load(f)
    model_checkpoint = torch.load(get_model_checkpoint_path(input_checkpoint_name_base, input_dir_path), map_location='cpu', weights_only=True)
    
    unwanted_prefix = '_orig_mod.'
    for k,v in list(model_checkpoint.items()):
        if k.startswith(unwanted_prefix):
            model_checkpoint[k[len(unwanted_prefix):]] = model_checkpoint.pop(k)
    
    state_dict = {
        "tok_embeddings.weight": model_checkpoint["tok_embeddings.weight"],
        "norm.weight": model_checkpoint["norm.weight"],
        "lm_head.weight": model_checkpoint["lm_head.weight"],
    }
    
    layer_mapping_pairs = prepare_layer_keys_mapping(config_checkpoint['model_args']['n_layer'], num_layers_to_remove)
    
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
            
    ckpt_file_path = get_config_checkpoint_path(output_checkpoint_name_base, output_dir_path)
    logger.info(f"saving config checkpoint to {ckpt_file_path}")
    with open(ckpt_file_path, "w", encoding="utf-8") as f:
        json.dump(config_checkpoint, f, indent=4, ensure_ascii=False)
    ckpt_file_path = get_model_checkpoint_path(output_checkpoint_name_base, output_dir_path)
    logger.info(f"saving model checkpoint to {ckpt_file_path}")
    torch.save(state_dict, ckpt_file_path)
    logger.info(f"checkpoint files saved in {output_dir_path}")

if __name__ == "__main__":
    configure_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Location of ALLaMo weights, which contains a checkpoint file",
    )
    parser.add_argument(
        "--input_checkpoint_name_base",
        default='ckpt',
        help="Source checkpoint file name base",
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write up-scaled model",
    )
    parser.add_argument(
        "--output_checkpoint_name_base",
        default='pruned_ckpt',
        help="Output checkpoint file name base",
    )
    parser.add_argument(
        "--num_layers_to_remove", type=int,
        help="Number of layers to remove from the model",
    )
    parser.add_argument(
        "--bfloat16", type=bool,
        help="Convert weights to bfloaf16",
    )
    args = parser.parse_args()
    prune_model(
        input_dir_path=args.input_dir,
        input_checkpoint_name_base=args.input_checkpoint_name_base,
        output_dir_path=args.output_dir,
        output_checkpoint_name_base=args.output_checkpoint_name_base,
        num_layers_to_remove=args.num_layers_to_remove,
        bfloat16=args.bfloat16
    )

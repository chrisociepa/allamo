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

def copy_layer(model_checkpoint, state_dict, src_layer_idx, dest_layer_idx):
    logger.info(f"copying weights from layer {src_layer_idx} to layer {dest_layer_idx}")
    for k in model_checkpoint.keys():
        if k.startswith(f"layers.{src_layer_idx}."):
            dst_k = k.replace(f"layers.{src_layer_idx}.", f"layers.{dest_layer_idx}.")
            state_dict[dst_k] = model_checkpoint[k].clone()

def reset_layer(layer_idx, state_dict):
    for k in state_dict.keys():
        if k.startswith(f"layers.{layer_idx}."):
            if k.endswith("_norm.weight"):
                torch.nn.init.ones_(state_dict[k])
            else:
                torch.nn.init.trunc_normal_(state_dict[k], mean=0.0, std=0.02)
    

def adjust_model(input_dir_path, input_checkpoint_name_base, output_dir_path, output_checkpoint_name_base, duplicate_first_layer, duplicate_last_layer, reset_first_layer, reset_last_layer):
    os.makedirs(output_dir_path, exist_ok=True)
    
    logger.info(f"loading checkpoint from {input_dir_path}...")
    with open(get_config_checkpoint_path(input_checkpoint_name_base, input_dir_path), "r", encoding="utf-8") as f:
        config_checkpoint = json.load(f)
    model_checkpoint = torch.load(get_model_checkpoint_path(input_checkpoint_name_base, input_dir_path), map_location='cpu')
    
    remove_unwanted_prefix_from_model_state_dict(model_checkpoint)
    
    state_dict = {
        "tok_embeddings.weight": model_checkpoint["tok_embeddings.weight"],
        "norm.weight": model_checkpoint["norm.weight"],
        "lm_head.weight": model_checkpoint["lm_head.weight"],
    }
    
    expected_num_layers = config_checkpoint['model_args']['n_layer']
    if duplicate_first_layer:
        expected_num_layers += 1
    if duplicate_last_layer:
        expected_num_layers += 1
    
    for src_layer_idx in range(config_checkpoint['model_args']['n_layer']):
        if duplicate_first_layer:
            if src_layer_idx == 0:
                copy_layer(model_checkpoint, state_dict, src_layer_idx, src_layer_idx)
            copy_layer(model_checkpoint, state_dict, src_layer_idx, src_layer_idx+1)
        else:
            copy_layer(model_checkpoint, state_dict, src_layer_idx, src_layer_idx)
            
    if duplicate_last_layer:
        copy_layer(model_checkpoint, state_dict, config_checkpoint['model_args']['n_layer']-1, expected_num_layers-1)
        
    if reset_first_layer:
        reset_layer(0, state_dict)
        
    if reset_last_layer:
        reset_layer(expected_num_layers-1, state_dict)
    
    param_count = 0
    param_bytes = 0
    for _, v in state_dict.items():
        param_count += v.numel()
        param_bytes += v.numel() * v.element_size()
    
    config_checkpoint['model_args']['n_layer'] = expected_num_layers
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
        default='ckpt',
        help="Output checkpoint file name base",
    )
    parser.add_argument(
        "--duplicate_first_layer", action='store_true',
        help="Duplicate first layer",
    )
    parser.add_argument(
        "--duplicate_last_layer", action='store_true',
        help="Duplicate last layer",
    )
    parser.add_argument(
        "--reset_first_layer", action='store_true',
        help="Reset first layer (apply after duplication)",
    )
    parser.add_argument(
        "--reset_last_layer", action='store_true',
        help="Reset last layer (apply after duplication)",
    )
    args = parser.parse_args()
    adjust_model(
        input_dir_path=args.input_dir,
        input_checkpoint_name_base=args.input_checkpoint_name_base,
        output_dir_path=args.output_dir,
        output_checkpoint_name_base=args.output_checkpoint_name_base,
        duplicate_first_layer=args.duplicate_first_layer,
        duplicate_last_layer=args.duplicate_last_layer,
        reset_first_layer=args.reset_first_layer,
        reset_last_layer=args.reset_last_layer,
    )

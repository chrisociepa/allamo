"""
This script is designed to replace the tokenizer of a given model along with its embeddings.
It allows for swapping out the existing tokenizer and embedding layer with new ones,
which can be useful for fine-tuning models on specific domains or languages,
or for experimenting with different tokenization strategies.
"""

import argparse
import json
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from tqdm import tqdm
from allamo.logging import configure_logger, logger
from allamo.train_utils import (
    get_model_checkpoint_path,
    get_config_checkpoint_path,
)

def copy_layer(model_checkpoint, state_dict, src_layer_idx, dest_layer_idx):
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

def adjust_model(input_dir_path, input_checkpoint_name_base, output_dir_path, output_checkpoint_name_base, add_first_layer, add_last_layer, source_tokenizer_path, target_tokenizer_path):
    os.makedirs(output_dir_path, exist_ok=True)
    
    logger.info(f"loading checkpoint from {input_dir_path}...")
    with open(get_config_checkpoint_path(input_checkpoint_name_base, input_dir_path), "r", encoding="utf-8") as f:
        config_checkpoint = json.load(f)
    model_checkpoint = torch.load(get_model_checkpoint_path(input_checkpoint_name_base, input_dir_path), map_location='cpu')
    
    unwanted_prefix = '_orig_mod.'
    for k,v in list(model_checkpoint.items()):
        if k.startswith(unwanted_prefix):
            model_checkpoint[k[len(unwanted_prefix):]] = model_checkpoint.pop(k)
            
    if source_tokenizer_path and target_tokenizer_path:
        logger.info(f"swapping embeddings for the new tokenizer")
        source_tokenizer = AutoTokenizer.from_pretrained(source_tokenizer_path)
        target_tokenizer = AutoTokenizer.from_pretrained(target_tokenizer_path)
        
        new_embs = nn.Embedding(len(target_tokenizer), config_checkpoint['model_args']['n_embd']).weight.data
        new_lm_head = nn.Linear(config_checkpoint['model_args']['n_embd'], len(target_tokenizer), bias=False).weight.data
        
        existing_token_count = 0
        new_token_count = 0
        stats = {}
        for token, token_id in tqdm(target_tokenizer.get_vocab().items()):
            if token in source_tokenizer.get_vocab():
                existing_token_count += 1
                src_token_id = source_tokenizer.get_vocab()[token]
                new_embs[token_id] = model_checkpoint["tok_embeddings.weight"][src_token_id]
                new_lm_head[token_id] = model_checkpoint["lm_head.weight"][src_token_id]
                stats[1] = stats.get(1, 0) + 1
            else:
                new_token_count += 1
                src_token_ids = source_tokenizer(token, add_special_tokens=False).input_ids
                new_embs[token_id] = model_checkpoint["tok_embeddings.weight"][src_token_ids[-1]]
                new_lm_head[token_id] = model_checkpoint["lm_head.weight"][src_token_ids[-1]]
                stats[len(src_token_ids)] = stats.get(len(src_token_ids), 0) + 1
        logger.info(f"Embeddings adjusted. Prev vocab size: {len(source_tokenizer)}. New vocab size: {len(target_tokenizer)}. Reused tokens: {existing_token_count}. New tokens: {new_token_count}")
        logger.info(f"Prev tokens to new token counts: {stats}")
        model_checkpoint["tok_embeddings.weight"] = new_embs
        model_checkpoint["lm_head.weight"] = new_lm_head
    
    state_dict = {
        "tok_embeddings.weight": model_checkpoint["tok_embeddings.weight"],
        "norm.weight": model_checkpoint["norm.weight"],
        "lm_head.weight": model_checkpoint["lm_head.weight"],
    }
    
    expected_num_layers = config_checkpoint['model_args']['n_layer']
    if add_first_layer:
        expected_num_layers += 1
    if add_last_layer:
        expected_num_layers += 1
    
    for src_layer_idx in range(config_checkpoint['model_args']['n_layer']):
        if add_first_layer:
            if src_layer_idx == 0:
                copy_layer(model_checkpoint, state_dict, src_layer_idx, src_layer_idx)
            copy_layer(model_checkpoint, state_dict, src_layer_idx, src_layer_idx+1)
        else:
            copy_layer(model_checkpoint, state_dict, src_layer_idx, src_layer_idx)
            
    if add_last_layer:
        copy_layer(model_checkpoint, state_dict, config_checkpoint['model_args']['n_layer']-1, expected_num_layers-1)
    
    param_count = 0
    param_bytes = 0
    for k, v in state_dict.items():
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
        "--add_first_layer", action='store_true',
        help="Duplicate first layer",
    )
    parser.add_argument(
        "--add_last_layer", action='store_true',
        help="Duplicate last layer",
    )
    parser.add_argument(
        "--source_tokenizer",
        help="Source/original tokenizer",
    )
    parser.add_argument(
        "--target_tokenizer",
        help="Target/new tokenizer",
    )
    args = parser.parse_args()
    adjust_model(
        input_dir_path=args.input_dir,
        input_checkpoint_name_base=args.input_checkpoint_name_base,
        output_dir_path=args.output_dir,
        output_checkpoint_name_base=args.output_checkpoint_name_base,
        add_first_layer=args.add_first_layer,
        add_last_layer=args.add_last_layer,
        source_tokenizer_path=args.source_tokenizer,
        target_tokenizer_path=args.target_tokenizer
    )

"""
This script is designed to swap embeddings from a source model with embeddings from a target model.
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
    remove_unwanted_prefix_from_model_state_dict,
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

def load_model_checkpoint(model_dir_path, checkpoint_name_base):
    logger.info(f"loading model checkpoint from {model_dir_path}...")
    with open(get_config_checkpoint_path(checkpoint_name_base, model_dir_path), "r", encoding="utf-8") as f:
        config_checkpoint = json.load(f)
    model_checkpoint = torch.load(get_model_checkpoint_path(checkpoint_name_base, model_dir_path), map_location='cpu', weights_only=True)
    remove_unwanted_prefix_from_model_state_dict(model_checkpoint)
    return config_checkpoint, model_checkpoint

def adjust_model(
    source_model_dir_path,
    source_model_checkpoint_name_base,
    target_model_dir_path,
    target_model_checkpoint_name_base,
    output_dir_path,
    output_checkpoint_name_base,
    source_tokenizer_path,
    target_tokenizer_path,
    swap_all_embeddings):
    
    os.makedirs(output_dir_path, exist_ok=True)

    logger.info("Loading model checkpoints...")
    source_config_checkpoint, source_model_checkpoint = load_model_checkpoint(source_model_dir_path, source_model_checkpoint_name_base)
    target_config_checkpoint, target_model_checkpoint = load_model_checkpoint(target_model_dir_path, target_model_checkpoint_name_base)
    assert source_config_checkpoint['model_args']['n_embd'] == target_config_checkpoint['model_args']['n_embd'], "Source and target models have different embedding sizes!"

    logger.info("Loading tokenizers...")
    source_tokenizer = AutoTokenizer.from_pretrained(source_tokenizer_path)
    target_tokenizer = AutoTokenizer.from_pretrained(target_tokenizer_path)
    assert len(source_tokenizer) <= len(target_tokenizer), "Source tokenizer has more tokens than target tokenizer. This combination is not supported."

    logger.info("Initializing new embeddings...")
    new_embs = nn.Embedding(len(target_tokenizer), source_config_checkpoint['model_args']['n_embd']).weight.data
    new_lm_head = nn.Linear(source_config_checkpoint['model_args']['n_embd'], len(target_tokenizer), bias=False).weight.data

    logger.info("Copying embeddings...")
    existing_token_count = 0
    new_token_count = 0
    for token, token_id in tqdm(target_tokenizer.get_vocab().items()):
        if token_id >= len(source_tokenizer):
            new_embs[token_id] = target_model_checkpoint["tok_embeddings.weight"][token_id]
            new_lm_head[token_id] = target_model_checkpoint["lm_head.weight"][token_id]
            logger.info(f"New embedding for token {token} (id: {token_id})")
            new_token_count += 1
        else:
            model_checkpoint = source_model_checkpoint
            if swap_all_embeddings:
                model_checkpoint = target_model_checkpoint
                logger.info(f"Swapped embedding for token {token} (id: {token_id})")
            new_embs[token_id] = model_checkpoint["tok_embeddings.weight"][token_id]
            new_lm_head[token_id] = model_checkpoint["lm_head.weight"][token_id]
            existing_token_count += 1
    
    logger.info(f"Embeddings copied. New vocab size: {len(new_embs)} ({existing_token_count} existing tokens, {new_token_count} new tokens)")
    del target_model_checkpoint
    
    state_dict = source_model_checkpoint
    state_dict["tok_embeddings.weight"] = new_embs
    state_dict["lm_head.weight"] = new_lm_head
   
    ckpt_file_path = get_config_checkpoint_path(output_checkpoint_name_base, output_dir_path)
    logger.info(f"Saving config checkpoint to {ckpt_file_path}")
    with open(ckpt_file_path, "w", encoding="utf-8") as f:
        json.dump(source_config_checkpoint, f, indent=4, ensure_ascii=False)
    ckpt_file_path = get_model_checkpoint_path(output_checkpoint_name_base, output_dir_path)
    logger.info(f"Saving model checkpoint to {ckpt_file_path}")
    torch.save(state_dict, ckpt_file_path)
    logger.info(f"Checkpoint files saved in {output_dir_path}")

if __name__ == "__main__":
    configure_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_model_dir", help="Location of ALLaMo weights, which contains a source model checkpoint file")
    parser.add_argument("--source_model_checkpoint_name_base", default='ckpt', help="Source model checkpoint file name base")
    parser.add_argument("--target_model_dir", help="Location of ALLaMo weights, which contains a target model checkpoint file")
    parser.add_argument("--target_model_checkpoint_name_base", default='ckpt', help="Target model checkpoint file name base")
    parser.add_argument("--output_dir", help="Location to write up-scaled model")
    parser.add_argument("--output_checkpoint_name_base", default='ckpt', help="Output checkpoint file name base")
    parser.add_argument("--source_tokenizer", help="Source/original tokenizer")
    parser.add_argument("--target_tokenizer", help="Target/new tokenizer")
    parser.add_argument('--all_embeddings', action='store_true', help='Replace all embeddings')
    args = parser.parse_args()

    adjust_model(
        source_model_dir_path=args.source_model_dir,
        source_model_checkpoint_name_base=args.source_model_checkpoint_name_base,
        target_model_dir_path=args.target_model_dir,
        target_model_checkpoint_name_base=args.target_model_checkpoint_name_base,
        output_dir_path=args.output_dir,
        output_checkpoint_name_base=args.output_checkpoint_name_base,
        source_tokenizer_path=args.source_tokenizer,
        target_tokenizer_path=args.target_tokenizer,
        swap_all_embeddings=args.all_embeddings
    )

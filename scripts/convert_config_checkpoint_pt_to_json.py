"""
Use this file to convert config checkpoint from PyTorch to JSON format
"""
import argparse
import dataclasses
import json
import torch
from allamo.logging import configure_logger, logger

def convert_ckpt(config_ckpt):
    config_checkpoint = torch.load(config_ckpt, map_location='cpu', weights_only=True)
    json_checkpoint = {}
    if 'model_args' in config_checkpoint:
        json_checkpoint['model_args'] = dataclasses.asdict(config_checkpoint['model_args'])
    if 'iter_num' in config_checkpoint:
        json_checkpoint['iter_num'] = config_checkpoint['iter_num']
    if 'best_train_loss' in config_checkpoint:
        json_checkpoint['best_train_loss'] = config_checkpoint['best_train_loss']
        if isinstance(json_checkpoint['best_train_loss'], torch.Tensor):
            json_checkpoint['best_train_loss'] = json_checkpoint['best_train_loss'].item()
    if 'best_val_loss' in config_checkpoint:
        json_checkpoint['best_val_loss'] = config_checkpoint['best_val_loss']
        if isinstance(json_checkpoint['best_val_loss'], torch.Tensor):
            json_checkpoint['best_val_loss'] = json_checkpoint['best_val_loss'].item()
    if 'processed_tokens' in config_checkpoint:
        json_checkpoint['processed_tokens'] = config_checkpoint['processed_tokens']
    if 'config' in config_checkpoint:
        json_checkpoint['config'] = config_checkpoint['config']
        
    data_loader_ckpt = {}
    if 'allamo_dataloader_train_processed_files' in config_checkpoint:
        data_loader_ckpt['train_processed_files'] = config_checkpoint['allamo_dataloader_train_processed_files']
    if 'allamo_dataloader_dataset_offset' in config_checkpoint:
        data_loader_ckpt['dataset_offset'] = config_checkpoint['allamo_dataloader_dataset_offset']
    if 'allamo_dataloader_epoch' in config_checkpoint:
        data_loader_ckpt['epoch'] = config_checkpoint['allamo_dataloader_epoch']
    if data_loader_ckpt:
        json_checkpoint['allamo_dataloader'] = data_loader_ckpt
    
    output_file = config_ckpt[:-3] + '.json'
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(json_checkpoint, f, indent=4, ensure_ascii=False)
    logger.info(f"Conversion completed! New config saved in {output_file}")
        
if __name__ == '__main__':
    configure_logger()
    parser = argparse.ArgumentParser(description='Convert config checkpoint to JSON format')
    parser.add_argument(
        "--config_ckpt",
        help="Path to config checkpoint in PyTorch format",
    )
    args = parser.parse_args()
    
    convert_ckpt(args.config_ckpt)
    
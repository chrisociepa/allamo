"""
Use this file to add reference log probabilities to your DPO (Direct Preference Optimization) dataset
"""

import argparse
import joblib
import json
import logging
import os
import sys
import time
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath('..'))
from model import AllamoTransformerConfig, AllamoTransformer
from configuration import AllamoConfiguration
from dpo_fsdp_train import get_log_prob
from train_utils import (
    model_checkpoint_files_exist,
    remove_unwanted_prefix_from_model_state_dict,
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger()
    
def format_seconds_as_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}:{int(minutes):02}:{int(seconds):02}"

def load_model_checkpoint(model, ckpt_path):
    state_dict = torch.load(ckpt_path, map_location='cpu')
    remove_unwanted_prefix_from_model_state_dict(state_dict)
    model.load_state_dict(state_dict)
    
def get_batch(sample, config, pin_memory):
    chosen_input_ids = torch.stack([torch.from_numpy(sample['chosen_input_ids'])]).to(torch.int64)
    chosen_target_ids = torch.stack([torch.from_numpy(sample['chosen_target_ids'])]).to(torch.int64)
    rejected_input_ids = torch.stack([torch.from_numpy(sample['rejected_input_ids'])]).to(torch.int64)
    rejected_target_ids = torch.stack([torch.from_numpy(sample['rejected_target_ids'])]).to(torch.int64)
    
    if 'cuda' in config.device and pin_memory:
        chosen_input_ids = chosen_input_ids.pin_memory().to(config.device, non_blocking=True)
        chosen_target_ids = chosen_target_ids.pin_memory().to(config.device, non_blocking=True)
        rejected_input_ids = rejected_input_ids.pin_memory().to(config.device, non_blocking=True)
        rejected_target_ids = rejected_target_ids.pin_memory().to(config.device, non_blocking=True)
    else:
        chosen_input_ids = chosen_input_ids.to(config.device)
        chosen_target_ids = chosen_target_ids.to(config.device)
        rejected_input_ids = rejected_input_ids.to(config.device)
        rejected_target_ids = rejected_target_ids.to(config.device)
    return {
        "chosen_input_ids": chosen_input_ids,
        "chosen_target_ids": chosen_target_ids,
        "rejected_input_ids": rejected_input_ids,
        "rejected_target_ids": rejected_target_ids
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tokenize dialogues for DPO training')
    parser.add_argument("-f", "--input_file", help="Input file in the ALM format")
    parser.add_argument("-i", "--input_dir", help="Directory with input files in the ALM format")
    parser.add_argument("-o", "--output_dir", required=True, help="Output dir")
    parser.add_argument("--checkpoint_dir", required=True, help="Model checkpoint dir")
    parser.add_argument("--checkpoint_name", required=True, help="Model checkpoint name")
    parser.add_argument("--pin_memory", type=bool, default=True, help="Specifies if the tensor is copied to pinned memory")
    parser.add_argument('--verbose', action='store_true', help='Be verbose')
    args = parser.parse_args()
    
    input_files = []
    if args.input_file:
        input_files.append(args.input_file)
    if args.input_dir:
        for root, dirs, files in os.walk(args.input_dir):
            for f in files:
                if f.endswith('.alm'):
                    input_files.append(os.path.join(root, f))
    logger.info(f"Initialized with {len(input_files)} input file(s)")
    
    assert model_checkpoint_files_exist(args.checkpoint_name, args.checkpoint_dir)
    
    ckpt_path = os.path.join(args.checkpoint_dir, f'config_{args.checkpoint_name}.json')
    with open(ckpt_path, "r", encoding="utf-8") as f:
        config_checkpoint = json.load(f)
        
    config_checkpoint["config"]["load_configuration"] = False
    config = AllamoConfiguration(**(config_checkpoint["config"]))
    
    logger.info("Start preparing model")
    model_ckpt_path = os.path.join(args.checkpoint_dir, f'model_{args.checkpoint_name}.pt')
    model_config = AllamoTransformerConfig(**(config_checkpoint["model_args"]))
    model = AllamoTransformer(model_config)
    logger.info(f"Model initialized. Start loading checkpoint {model_ckpt_path}")
    load_model_checkpoint(model, model_ckpt_path)
    model.to(config.device)
    model.eval()
    logger.info(f"Model loaded")
    
    os.makedirs(args.output_dir, exist_ok=True)
    timer = time.time()
    for input_file in input_files:
        logger.info(f'Loading data from {input_file}')
        samples = joblib.load(input_file)
        logger.info(f'Loaded {len(samples)} samples. Start generating log probabilities')
        
        with torch.no_grad():
            for sample in tqdm(samples, disable=(not args.verbose)):
                batch = get_batch(sample, config, args.pin_memory)
                reference_chosen_logits, _, _ = model(input_ids=batch["chosen_input_ids"], target_ids=batch["chosen_target_ids"])
                reference_rejected_logits, _, _ = model(input_ids=batch["rejected_input_ids"], target_ids=batch["rejected_target_ids"])
                sample["reference_chosen_logps"] = get_log_prob(reference_chosen_logits, batch["chosen_target_ids"], config.ignore_index).item()
                sample["reference_rejected_logps"] = get_log_prob(reference_rejected_logits, batch["rejected_target_ids"], config.ignore_index).item()
            
        output_file = os.path.join(args.output_dir, os.path.basename(input_file))
        with open(output_file, 'wb') as f:
            joblib.dump(samples, f)
        logger.info(f"Saved ({len(samples)}) samples in {output_file}")
        
        sum_reference_chosen_logps = sum(sample["reference_chosen_logps"] for sample in samples)
        sum_reference_rejected_logps = sum(sample["reference_rejected_logps"] for sample in samples)
        stats = {
            'min_reference_chosen_logps': min(sample["reference_chosen_logps"] for sample in samples),
            'max_reference_chosen_logps': max(sample["reference_chosen_logps"] for sample in samples),
            'sum_reference_chosen_logps': sum_reference_chosen_logps,
            'avg_reference_chosen_logps': sum_reference_chosen_logps / len(samples),
            'min_reference_rejected_logps': min(sample["reference_rejected_logps"] for sample in samples),
            'max_reference_rejected_logps': max(sample["reference_rejected_logps"] for sample in samples),
            'sum_reference_rejected_logps': sum_reference_rejected_logps,
            'avg_reference_rejected_logps': sum_reference_rejected_logps / len(samples)
        }
        stats_str = json.dumps(stats, indent=4, ensure_ascii=False)
        logger.info(f"Stats: {stats_str}")

    logger.info(f"Generated log probabilities in {format_seconds_as_time(time.time()-timer)}")

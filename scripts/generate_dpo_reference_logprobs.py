"""
Use this file to add reference log probabilities to your DPO (Direct Preference Optimization) dataset
"""

import argparse
import concurrent.futures
import joblib
import json
import logging
import os
import sys
import time
import torch
from itertools import chain
from tqdm import tqdm
from transformers import AutoModelForCausalLM

sys.path.append(os.path.abspath('..'))
from dpo_fsdp_train import get_log_prob

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger()
    
def format_seconds_as_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}:{int(minutes):02}:{int(seconds):02}"

def get_dtype(dtype_str):
    return {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype_str]
    
def get_batch(sample, device, pin_memory):
    chosen_input_ids = torch.stack([torch.from_numpy(sample['chosen_input_ids'])]).to(torch.int64)
    chosen_target_ids = torch.stack([torch.from_numpy(sample['chosen_target_ids'])]).to(torch.int64)
    rejected_input_ids = torch.stack([torch.from_numpy(sample['rejected_input_ids'])]).to(torch.int64)
    rejected_target_ids = torch.stack([torch.from_numpy(sample['rejected_target_ids'])]).to(torch.int64)
    
    if 'cuda' in device and pin_memory:
        chosen_input_ids = chosen_input_ids.pin_memory().to(device, non_blocking=True)
        chosen_target_ids = chosen_target_ids.pin_memory().to(device, non_blocking=True)
        rejected_input_ids = rejected_input_ids.pin_memory().to(device, non_blocking=True)
        rejected_target_ids = rejected_target_ids.pin_memory().to(device, non_blocking=True)
    else:
        chosen_input_ids = chosen_input_ids.to(device)
        chosen_target_ids = chosen_target_ids.to(device)
        rejected_input_ids = rejected_input_ids.to(device)
        rejected_target_ids = rejected_target_ids.to(device)
    return {
        "chosen_input_ids": chosen_input_ids,
        "chosen_target_ids": chosen_target_ids,
        "rejected_input_ids": rejected_input_ids,
        "rejected_target_ids": rejected_target_ids
    }
    
def calculate_sample_stats(samples):
    sum_reference_chosen_logps = sum(sample["reference_chosen_logps"] for sample in samples)
    sum_reference_rejected_logps = sum(sample["reference_rejected_logps"] for sample in samples)
    return {
        'min_reference_chosen_logps': min(sample["reference_chosen_logps"] for sample in samples),
        'max_reference_chosen_logps': max(sample["reference_chosen_logps"] for sample in samples),
        'sum_reference_chosen_logps': sum_reference_chosen_logps,
        'avg_reference_chosen_logps': sum_reference_chosen_logps / len(samples),
        'min_reference_rejected_logps': min(sample["reference_rejected_logps"] for sample in samples),
        'max_reference_rejected_logps': max(sample["reference_rejected_logps"] for sample in samples),
        'sum_reference_rejected_logps': sum_reference_rejected_logps,
        'avg_reference_rejected_logps': sum_reference_rejected_logps / len(samples)
    }
        
def process_file(input_file, model, device, pin_memory, ignore_index, disable_logging=True):
    samples = joblib.load(input_file)
    
    with torch.no_grad():
        for sample in tqdm(samples, disable=disable_logging):
            batch = get_batch(sample, device, pin_memory)
            reference_chosen_logits = model(input_ids=batch["chosen_input_ids"]).logits
            reference_rejected_logits = model(input_ids=batch["rejected_input_ids"]).logits
            sample["reference_chosen_logps"] = get_log_prob(reference_chosen_logits, batch["chosen_target_ids"], ignore_index).item()
            sample["reference_rejected_logps"] = get_log_prob(reference_rejected_logits, batch["rejected_target_ids"], ignore_index).item()
    
    with open(input_file, 'wb') as f:
        joblib.dump(samples, f)
    return samples
        
def process_chunk(args):
    input_file, hf_model_path, hf_model_dtype, device, pin_memory, ignore_index = args
    model = AutoModelForCausalLM.from_pretrained(hf_model_path, torch_dtype=get_dtype(hf_model_dtype), device_map=device)
    process_file(input_file, model, device, pin_memory, ignore_index)
    
def save_samples(samples, input_file, args):
    if args.save_samples > 0:
        logger.info(f"Saving samples")
        samples_file = os.path.join(args.output_dir, os.path.basename(input_file) + "-samples.jsonl")
        with open(samples_file, 'w') as f:
            for sample in samples[:args.save_samples]:
                chosen_input_ids = sample["chosen_input_ids"].tolist()
                rejected_input_ids = sample["rejected_input_ids"].tolist()
                new_sample = {
                    "chosen_len": len(chosen_input_ids),
                    "rejected_len": len(rejected_input_ids),
                    "batch_len": len(chosen_input_ids)+len(rejected_input_ids),
                    "chosen_input_ids": chosen_input_ids,
                    "chosen_target_ids": sample["chosen_target_ids"].tolist(),
                    "rejected_input_ids": rejected_input_ids,
                    "rejected_target_ids": sample["rejected_target_ids"].tolist(),
                    "reference_chosen_logps": sample["reference_chosen_logps"],
                    "reference_rejected_logps": sample["reference_rejected_logps"]
                }
                
                f.write(json.dumps(new_sample, ensure_ascii=False))
                f.write('\n')
        logger.info(f"Samples saved in {samples_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tokenize dialogues for DPO training')
    parser.add_argument("-f", "--input_file", help="Input file in the ALM format")
    parser.add_argument("-i", "--input_dir", help="Directory with input files in the ALM format")
    parser.add_argument("-o", "--output_dir", required=True, help="Output dir")
    parser.add_argument("--hf_model_path", required=True, help="Model path in HF format")
    parser.add_argument("--hf_model_dtype", required=True, help="HF model dtype")
    parser.add_argument("--hf_model_device", required=True, help="Device to load the HF model on")
    parser.add_argument("--hf_model_copies", type=int, default=1, help="Number of model copies to run on separate devices")
    parser.add_argument("--pin_memory", type=bool, default=True, help="Specifies if the tensor is copied to pinned memory")
    parser.add_argument("--ignore_index", type=int, default=-100, help="Specifies a target value that is ignored in loss computation. Default is -100")
    parser.add_argument('--save_samples', type=int, default=-1, help='Save this number of samples if positive')
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
    
    os.makedirs(args.output_dir, exist_ok=True)
    timer = time.time()
    if args.hf_model_copies > 1:
        assert args.hf_model_device.startswith("cuda"), "Only CUDA devices are supported in parallel mode"
        
        for input_file in input_files:
            logger.info(f'Loading data from {input_file}')
            samples = joblib.load(input_file)
            logger.info(f'Loaded {len(samples)} samples. Start generating log probabilities')
        
            logger.info(f"Chunking {len(samples):,} samples into {args.hf_model_copies} files")
            chunk_files = []
            for rank in tqdm(range(args.hf_model_copies), total=args.hf_model_copies, desc="Chunking", disable=(not args.verbose)):
                chunk_file = os.path.join(args.output_dir, f"chunk_{rank:05}.tmp")
                with open(chunk_file, 'wb') as f:
                    joblib.dump(samples[rank::args.hf_model_copies], f)
                chunk_files.append(chunk_file)
            del samples
            logger.info(f"Saved {len(chunk_files)} chunks in {args.output_dir}")
            
            logger.info(f"Start generating log probabilities for {len(chunk_files)} chunks")
            max_workers = min(len(chunk_files), args.hf_model_copies)
            chunk_batches = list((chunk_file, args.hf_model_path, args.hf_model_dtype, f"cuda:{rank}", args.pin_memory, args.ignore_index) for rank, chunk_file in enumerate(chunk_files))
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                for _ in executor.map(process_chunk, chunk_batches):
                    pass
            del executor
            logger.info(f"Log probabilities generated in {len(chunk_files)} chunks")
            
            logger.info(f"Merging {len(chunk_files)} chunks")
            chunks = joblib.Parallel(n_jobs=len(chunk_files))(joblib.delayed(joblib.load)(f) for f in chunk_files)
            samples = list(chain.from_iterable(chunks))
            logger.info(f"{len(samples):,} samples merged")
            
            output_file = os.path.join(args.output_dir, os.path.basename(input_file))
            with open(output_file, 'wb') as f:
                joblib.dump(samples, f)
            logger.info(f"Saved ({len(samples)}) samples in {output_file}")
            
            save_samples(samples, input_file, args)
            
            stats = calculate_sample_stats(samples)
            stats_str = json.dumps(stats, indent=4, ensure_ascii=False)
            logger.info(f"Stats for {input_file}:\n{stats_str}")
            
            # cleanup
            for chunk_file in chunk_files:
                os.remove(chunk_file)
    else:
        device = args.hf_model_device
        
        model = AutoModelForCausalLM.from_pretrained(args.hf_model_path, torch_dtype=get_dtype(args.hf_model_dtype), device_map=device)
        logger.info(f"Model loaded")
        
        for input_file in input_files:
            logger.info(f'Processing {input_file}')
            samples =  process_file(input_file, model, device, args.pin_memory, args.ignore_index, disable_logging=(not args.verbose))
            
            save_samples(samples, input_file, args)

            stats = calculate_sample_stats(samples)
            stats_str = json.dumps(stats, indent=4, ensure_ascii=False)
            logger.info(f"Stats for {input_file}:\n{stats_str}")
    
    logger.info(f"Generated log probabilities for {len(input_files)} file(s) in {format_seconds_as_time(time.time()-timer)}")

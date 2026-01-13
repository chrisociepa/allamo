import argparse
import joblib
import json
import os
import time
import torch
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from allamo.logging import configure_logger, logger

def format_seconds_as_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}:{int(minutes):02}:{int(seconds):02}"
   
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

def calculate_logprobs_and_loss(log_probs, target_ids, ignore_index):
    loss_mask = (target_ids != ignore_index)
    logps = (log_probs * loss_mask).sum(-1)
    loss = -logps / loss_mask.sum()
    return logps, loss

def generate_logprobs_and_loss(vllm_client, model, input_ids, target_ids, ignore_index):
    if len(input_ids) == len(target_ids):
        input_ids = input_ids + target_ids[-1:]
    
    response = vllm_client.completions.create(
        model=model,
        prompt=input_ids,
        max_tokens=0,
        temperature=0,
        logprobs=1,
        echo=True
    )

    token_logprobs = [t for t in response.choices[0].logprobs.token_logprobs if t is not None]
    return calculate_logprobs_and_loss(torch.tensor(token_logprobs), torch.tensor(target_ids), ignore_index)

def process_single_sample(sample, vllm_client, model, ignore_index):
    chosen_logps, chosen_loss = generate_logprobs_and_loss(vllm_client, model, sample["chosen_input_ids"].tolist(), sample["chosen_target_ids"].tolist(), ignore_index)
    rejected_logps, rejected_loss = generate_logprobs_and_loss(vllm_client, model, sample["rejected_input_ids"].tolist(), sample["rejected_target_ids"].tolist(), ignore_index)
    
    sample["reference_chosen_loss"] = chosen_loss.item()
    sample["reference_chosen_logps"] = chosen_logps.item()
    sample["reference_rejected_loss"] = rejected_loss.item()
    sample["reference_rejected_logps"] = rejected_logps.item()
    return sample

def process_file(input_file, output_file, vllm_client, model, ignore_index, concurrency):
    samples = joblib.load(input_file)
    total_rows = len(samples)
    log_interval = max(1, total_rows // 100)

    results = [None] * total_rows
    failed_count = 0
    start_time = time.time()

    logger.info(f"Starting processing {total_rows} rows with {concurrency} workers...")

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_idx = {
            executor.submit(process_single_sample, sample, vllm_client, model, ignore_index): i
            for i, sample in enumerate(samples)
        }
        
        for i, future in enumerate(as_completed(future_to_idx), 1):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.error(f"Row {idx} failed: {e}")
                results[idx] = None
                failed_count += 1

            if i % log_interval == 0 or i == total_rows:
                elapsed = time.time() - start_time
                speed = i / elapsed if elapsed > 0 else 0
                remaining_items = total_rows - i
                eta_seconds = remaining_items / speed if speed > 0 else 0
                
                eta_str = format_seconds_as_time(eta_seconds)
                percent = (i / total_rows) * 100

                logger.info(
                    f"Progress: {percent:3.0f}% | "
                    f"Done: {i}/{total_rows} | "
                    f"Failed: {failed_count} | "
                    f"Speed: {speed:.2f} it/s | "
                    f"ETA: {eta_str}"
                )

    total_time = format_seconds_as_time(time.time() - start_time)
    success_count = total_rows - failed_count
    logger.info("-" * 30)
    logger.info(f"PROCESSING SUMMARY:")
    logger.info(f"Total rows:      {total_rows}")
    logger.info(f"Successfully:    {success_count}")
    logger.info(f"Failed:          {failed_count}")
    logger.info(f"Total time:      {total_time}")
    logger.info("-" * 30)

    with open(output_file, 'wb') as f:
        joblib.dump(results, f)
    
    return results

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
                    "reference_chosen_loss": sample["reference_chosen_loss"],
                    "reference_rejected_logps": sample["reference_rejected_logps"],
                    "reference_rejected_loss": sample["reference_rejected_loss"]
                }
                f.write(json.dumps(new_sample, ensure_ascii=False) + '\n')
        logger.info(f"Samples saved in {samples_file}")

if __name__ == "__main__":
    configure_logger()
    parser = argparse.ArgumentParser(description='Tokenize dialogues for DPO training')
    parser.add_argument("-f", "--input_file", help="Input file in the ALM format")
    parser.add_argument("-i", "--input_dir", help="Directory with input files in the ALM format")
    parser.add_argument("-o", "--output_dir", required=True, help="Output dir")
    parser.add_argument("--vllm_base_url", required=True, help="Base URL for vLLM")
    parser.add_argument("--vllm_model", required=True, help="Model name")
    parser.add_argument("--vllm_api_key", required=True, help="API key for vLLM")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent requests")
    parser.add_argument("--ignore_index", type=int, default=-100, help="Ignore index for loss computation")
    parser.add_argument('--save_samples', type=int, default=-1, help='Number of samples to save')
    args = parser.parse_args()
    
    input_files = []
    if args.input_file:
        input_files.append(args.input_file)
    if args.input_dir:
        for root, _, files in os.walk(args.input_dir):
            for f in files:
                if f.endswith('.alm'):
                    input_files.append(os.path.join(root, f))
    
    logger.info(f"Initialized with {len(input_files)} input file(s)")
    os.makedirs(args.output_dir, exist_ok=True)
    timer = time.time()
    
    vllm_client = OpenAI(base_url=args.vllm_base_url, api_key=args.vllm_api_key)

    for input_file in input_files:
        logger.info(f'Processing {input_file}')
        output_file = os.path.join(args.output_dir, os.path.basename(input_file))
        samples = process_file(input_file, output_file, vllm_client, args.vllm_model, args.ignore_index, args.concurrency)
        logger.info(f"Saved ({len(samples)}) samples in {output_file}")

        save_samples(samples, input_file, args)

        stats = calculate_sample_stats(samples)
        logger.info(f"Stats for {input_file}:\n{json.dumps(stats, indent=4, ensure_ascii=False)}")
    
    logger.info(f"Finished in {format_seconds_as_time(time.time()-timer)}")
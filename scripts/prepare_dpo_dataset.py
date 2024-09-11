"""
Use this file to create a dataset for DPO (Direct Preference Optimization) training

The script performs the following steps:
1. Reads the input JSONL file with dialogues (single or multi-turn)
2. Applies the OpenChatML or Llama2 chat template to each dialogue
3. Tokenizes the formatted dialogues
6. Saves the processed data in a binary format
7. Generates and saves summary statistics for the dataset

Example record with signe-turn dialogue:
```json
{"messages": [{"role": "user", "content": "1+2=?"}], "chosen": {"role": "assistant", "content": "3"}, "rejected": {"role": "assistant", "content": "4"}}
```

Example record with multi-turn dialogue:
```json
{"messages": [{"role": "user", "content": "1+2=?"}, {"role": "assistant", "content": "3"}, {"role": "user", "content": "2+2=?"}], "chosen": {"role": "assistant", "content": "4"}, "rejected": {"role": "assistant", "content": "5"}
```
"""

import argparse
import concurrent.futures
import joblib
import json
import numpy as np
import os
import pyarrow as pa
import pyarrow.parquet as pq
import random
import time
from collections import Counter
from itertools import chain
from tqdm import tqdm
from transformers import AutoTokenizer
from allamo.logging import configure_logger, logger

def tokenize_openchatml_conversation(messages, tokenizer, ignore_index):
    result = {'input_ids': [], 'target_ids': []}
    last_idx = len(messages) - 1
    for idx, entry in enumerate(messages):
        if entry["role"] == 'assistant':
            pre_content = '<|im_start|>assistant\n'
            pre_input_ids = tokenizer.encode(pre_content, add_special_tokens=False)
            pre_input_ids_len = len(pre_input_ids)
            
            content = entry['content'] + '<|im_end|>\n'
            if idx == last_idx:
                content += "</s>"
            full_input_ids = tokenizer.encode(pre_content + content, add_special_tokens=False)
            
            if full_input_ids[:pre_input_ids_len] == pre_input_ids:
                result['input_ids'].extend(full_input_ids)
                result['target_ids'].extend(list(
                    ignore_index if i < pre_input_ids_len else full_input_ids[i] for i in range(len(full_input_ids))
                ))
            else:
                logger.warning("Tokenization inconsistency detected. Performing separate tokenization")
                content_input_ids = tokenizer.encode(content, add_special_tokens=False)
                result['input_ids'].extend(pre_input_ids)
                result['input_ids'].extend(content_input_ids)
                result['target_ids'].extend(list(ignore_index for _ in range(pre_input_ids_len)))
                result['target_ids'].extend(content_input_ids)
        else:
            content = "<s><|im_start|>" if idx == 0 else "<|im_start|>"
            content += entry["role"] + '\n' + entry["content"] + '<|im_end|>\n'
            input_ids = tokenizer.encode(content, add_special_tokens=False)
            result['input_ids'].extend(input_ids)
            result['target_ids'].extend(list(ignore_index for _ in range(len(input_ids))))
    assert len(result['input_ids']) == len(result['target_ids'])
    return result

def tokenize_llama2_conversation(messages, tokenizer, ignore_index):
    result = {'input_ids': [], 'target_ids': []}
    
    if messages[0]['role'] == 'system':
        sys_message = f"<<SYS>>\n{messages[0]['content']}\n<</SYS>>\n\n"
        messages = messages[1:]
    else:
        sys_message = ''
        
    for idx, entry in enumerate(messages):
        if entry['role'] == 'user':
            content = '<s>[INST] '+sys_message if idx <= 1 else '[INST] '
            content += entry['content'] + ' [/INST]'
            input_ids = tokenizer.encode(content, add_special_tokens=False)
            result['input_ids'].extend(input_ids)
            result['target_ids'].extend(list(ignore_index for _ in range(len(input_ids))))
        elif entry['role'] == 'assistant':
            content = ' ' + entry['content'] + '</s>'
            input_ids = tokenizer.encode(content, add_special_tokens=False)
            result['input_ids'].extend(input_ids)
            result['target_ids'].extend(input_ids)
    assert len(result['input_ids']) == len(result['target_ids'])
    return result

def tokenize_conversation(data, tokenizer, ignore_index, chat_format):
    if chat_format == 'OpenChatML':
        return tokenize_openchatml_conversation(data, tokenizer, ignore_index)
    elif chat_format == 'llama2':
        return tokenize_llama2_conversation(data, tokenizer, ignore_index)
    else:
        raise Exception(f"Unsupported chat format: {chat_format}")
    
def convert_to_numpy_array(pylist, target_length, pad_token, data_type):
    padded = np.full(target_length, pad_token, dtype=data_type)
    padded[:len(pylist)] = pylist
    return padded

def pad_and_align(sample, input_ids_key, target_ids_key, block_size, max_sample_size, pad_token_id, ignore_index, data_dtype):
    padding = max_sample_size - len(sample[input_ids_key])
    if pad_token_id >= 0:
        assert padding >= 0
        if padding > 0:
            if padding > 1:
                sample[input_ids_key] = convert_to_numpy_array(sample[input_ids_key], block_size, pad_token_id, data_dtype)
            else:
                sample[input_ids_key] = np.array(sample[input_ids_key], dtype=data_dtype)
            sample[target_ids_key] = convert_to_numpy_array(sample[target_ids_key][1:], block_size, ignore_index, data_dtype)
        else:
            assert len(sample[input_ids_key]) == max_sample_size
            assert len(sample[target_ids_key]) == max_sample_size
            sample[input_ids_key] = np.array(sample[input_ids_key][:-1], dtype=data_dtype)
            sample[target_ids_key] = np.array(sample[target_ids_key][1:], dtype=data_dtype)
    else:
        expected_len = len(sample[input_ids_key]) - 1 if padding > 0 else block_size
        sample[input_ids_key] = np.array(sample[input_ids_key][:expected_len], dtype=data_dtype)
        sample[target_ids_key] = np.array(sample[target_ids_key][1:expected_len+1], dtype=data_dtype)

def process_chunk(args):
    chunk_file, tokenizer_path, chat_format, block_size, ignore_index, pad_token_id, min_unmasked_tokens = args
    max_sample_size = block_size + 1
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    data_dtype = np.int16 if len(tokenizer) < 32767 else np.int32
    truncated = 0
    rejected = 0
    data = []
    pa_table = pq.read_table(chunk_file)
    for i in range(len(pa_table['rows'])):
        cols = pa_table['rows'][i].as_py().split(';', 1)
        row = json.loads(cols[1])
        if 'messages' not in row or 'chosen' not in row or 'rejected' not in row:
            rejected += 1
        else:
            chosen_sample = tokenize_conversation(row['messages']+[row['chosen']], tokenizer, ignore_index, chat_format)
            chosen_input_ids_len = len(chosen_sample['input_ids'])
            if chosen_input_ids_len > max_sample_size:
                chosen_sample['input_ids'] = chosen_sample['input_ids'][:max_sample_size]
                chosen_sample['target_ids'] = chosen_sample['target_ids'][:max_sample_size]
                truncated += 1
            
            rejected_sample = tokenize_conversation(row['messages']+[row['rejected']], tokenizer, ignore_index, chat_format)
            rejected_input_ids_len = len(rejected_sample['input_ids'])
            if rejected_input_ids_len > max_sample_size:
                rejected_sample['input_ids'] = rejected_sample['input_ids'][:max_sample_size]
                rejected_sample['target_ids'] = rejected_sample['target_ids'][:max_sample_size]
                truncated += 1
                
            data.append({
                'chosen_input_ids': chosen_sample['input_ids'],
                'chosen_target_ids': chosen_sample['target_ids'],
                'rejected_input_ids': rejected_sample['input_ids'],
                'rejected_target_ids': rejected_sample['target_ids'],
                'source_file': cols[0]
            })
    del pa_table
    
    created = len(data)
    result = []
    for sample in data:
        pad_and_align(sample, "chosen_input_ids", "chosen_target_ids", block_size, max_sample_size, pad_token_id, ignore_index, data_dtype)
        pad_and_align(sample, "rejected_input_ids", "rejected_target_ids", block_size, max_sample_size, pad_token_id, ignore_index, data_dtype)

        assert isinstance(sample["chosen_input_ids"], np.ndarray)
        assert isinstance(sample["chosen_target_ids"], np.ndarray)
        assert isinstance(sample["rejected_input_ids"], np.ndarray)
        assert isinstance(sample["rejected_target_ids"], np.ndarray)
        if np.sum(sample['chosen_target_ids'] != ignore_index) >= min_unmasked_tokens and np.sum(sample['rejected_target_ids'] != ignore_index) >= min_unmasked_tokens:
            result.append(sample)
        else:
            rejected += 1
    
    with open(chunk_file, 'wb') as f:
        joblib.dump(result, f)
    
    return {'created': created, 'truncated': truncated, 'rejected': rejected}

def save_chunk_for_rank(rows, rank, output_dir, chunk_files):
    chunk_file = os.path.join(output_dir, f"chunk_{rank:05}.tmp")
    pa_array = pa.array(rows)
    pa_table = pa.table([pa_array], names=['rows'])
    pq.write_table(pa_table, chunk_file)
    chunk_files.append(chunk_file)
    
def format_seconds_as_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}:{int(minutes):02}:{int(seconds):02}"

if __name__ == "__main__":
    configure_logger()
    parser = argparse.ArgumentParser(description='Tokenize dialogues for DPO training')
    parser.add_argument("-c", "--config_path", help="Config file with a list of input files")
    parser.add_argument("-f", "--input_file", help="Input file")
    parser.add_argument("-i", "--input_dir", help="Directory with input jsonl files")
    parser.add_argument("-o", "--output_dir", help="Output dir")
    parser.add_argument("-n", "--num_output_files", type=int, default=1, help="Number of final output files")
    parser.add_argument("-t", "--tokenizer_path", required=True, help="Tokenizer path")
    parser.add_argument("-p", "--max_workers", type=int, default=20, help="The max number of processes")
    parser.add_argument("-b", "--block_size", type=int, default=4096, help="Block/context size")
    parser.add_argument('--chat_format', type=str, choices=['OpenChatML', 'llama2'], default='OpenChatML', help='Chat format')
    parser.add_argument("--min_unmasked_tokens", type=int, default=1, help="Minimum number of unmasked target tokens required for a sample to be included in training")
    parser.add_argument("--ignore_index", type=int, default=-100, help="Specifies a target value that is ignored in loss computation. Default is -100")
    parser.add_argument("--pad_token_id", type=int, default=0, help="Specifies the padding token id. Default is 0")
    parser.add_argument("--chunk_size", type=int, default=100000, help="Chunk size")
    parser.add_argument('--save_samples', type=int, default=-1, help='Save this number of samples if positive')
    parser.add_argument('--verbose', action='store_true', help='Be verbose')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    logger.info(f"Loaded tokenizer with vocab size {len(tokenizer)}")
    logger.info(f"Active chat template type: {args.chat_format}")

    timer = time.time()
    max_sample_size = args.block_size + 1
    
    configs = []
    if args.config_path:
        with open(args.config_path, "r", encoding="utf-8") as f:
            configs = json.load(f)

    if args.input_file:
        configs.append({'path': args.input_file})

    if args.input_dir:
        for root, dirs, files in os.walk(args.input_dir):
            for f in files:
                if f.endswith('.jsonl'):
                    configs.append({'path': os.path.join(root, f)})
    logger.info(f"Initialized with {len(configs)} input files")
    
    logger.info("Loading data")
    def load_data_file(config):
        filename_prefix = os.path.basename(config['path']) + ";"
        with open(config['path'], 'r') as f:
            return list(filename_prefix + line for line in f if line)
        
    chunks = joblib.Parallel(n_jobs=args.max_workers)(joblib.delayed(load_data_file)(config) for config in configs)
    all_rows = list(chain.from_iterable(chunks))
    del chunks
    del configs
    instruction_count = len(all_rows)
    logger.info(f"Loaded {instruction_count:,} rows")
    
    logger.info("Shuffling data")
    random.shuffle(all_rows)
    logger.info("Shuffling completed")
    
    # adjust num of workers if needed
    if len(all_rows) < 10*args.max_workers:
        args.max_workers = max(1, len(all_rows) // 10)
    
    logger.info(f"Chunking {len(all_rows):,} rows into {args.max_workers} files")
    chunk_files = []
    for rank in tqdm(range(args.max_workers), total=args.max_workers, desc="Chunking", disable=(not args.verbose)):
        save_chunk_for_rank(all_rows[rank::args.max_workers], rank, args.output_dir, chunk_files)
    del all_rows
    logger.info(f"Saved {len(chunk_files)} chunks in {args.output_dir}")
    
    logger.info(f"Tokenizing {len(chunk_files)} files")
    processed_chunk_stats = []
    max_workers = min(len(chunk_files), args.max_workers)
    chunk_batches = list((chunk_file, args.tokenizer_path, args.chat_format, args.block_size, args.ignore_index, args.pad_token_id, args.min_unmasked_tokens) for chunk_file in chunk_files)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for result in tqdm(executor.map(process_chunk, chunk_batches), total=len(chunk_batches), desc="Tokenizing", disable=(not args.verbose)):
            processed_chunk_stats.append(result)
    del executor
    
    stats = {'created': 0, 'truncated': 0, 'rejected': 0}
    for s in processed_chunk_stats:
        for k, v in s.items():
            stats[k] += v
    del processed_chunk_stats
    logger.info(f"Tokenization finished in {len(chunk_files)} chunks. Stats: {stats}")
    
    logger.info(f"Merging {len(chunk_files)} chunks")
    chunks = joblib.Parallel(n_jobs=args.max_workers)(joblib.delayed(joblib.load)(f) for f in chunk_files)
    all_samples = list(chain.from_iterable(chunks))
    sample_count = len(all_samples)
    logger.info(f"{sample_count:,} samples loaded")
    
    assert isinstance(all_samples[0]["chosen_input_ids"], np.ndarray)
    assert isinstance(all_samples[0]["chosen_target_ids"], np.ndarray)
    assert isinstance(all_samples[0]["rejected_input_ids"], np.ndarray)
    assert isinstance(all_samples[0]["rejected_target_ids"], np.ndarray)

    assert sample_count > 0
    if args.save_samples > 0:
        logger.info(f"Saving samples")
        samples_file = os.path.join(args.output_dir, "samples.jsonl")
        with open(samples_file, 'w') as f:
            for sample in all_samples[:args.save_samples]:
                chosen_input_ids = sample["chosen_input_ids"].tolist()
                rejected_input_ids = sample["rejected_input_ids"].tolist()
                new_sample = {
                    "chosen_input": tokenizer.decode(chosen_input_ids),
                    "chosen_input_ids": chosen_input_ids,
                    "chosen_target_ids": sample["chosen_target_ids"].tolist(),
                    "rejected_input": tokenizer.decode(rejected_input_ids),
                    "rejected_input_ids": rejected_input_ids,
                    "rejected_target_ids": sample["rejected_target_ids"].tolist(),
                }
                f.write(json.dumps(new_sample, ensure_ascii=False))
                f.write('\n')
        logger.info(f"Samples saved in {samples_file}")

    if args.num_output_files > 1:
        for i in tqdm(range(args.num_output_files), desc="Saving", disable=(not args.verbose)):
            bucket = all_samples[i::args.num_output_files]
            output_file = os.path.join(args.output_dir, f"samples_part_{i:05}.alm")
            with open(output_file, 'wb') as f:
                joblib.dump(bucket, f)
            logger.info(f"Saved {len(bucket)} samples into {output_file}")
    else:
        output_file = os.path.join(args.output_dir, "all_samples.alm")
        with open(output_file, 'wb') as f:
            joblib.dump(all_samples, f)
        logger.info(f"All ({sample_count}) samples saved in {output_file}")
    
    # cleanup
    for chunk_file in chunk_files:
        os.remove(chunk_file)
    
    logger.info(f"Calculating stats")
    chosen_sample_lenghts = [np.sum(sample['chosen_input_ids'] != args.pad_token_id).item() for sample in all_samples]
    rejected_sample_lenghts = [np.sum(sample['rejected_input_ids'] != args.pad_token_id).item() for sample in all_samples]
    chosen_shortest_sample_tokens = min(chosen_sample_lenghts)
    chosen_longest_sample_tokens = max(chosen_sample_lenghts)
    chosen_total_tokens_count = sum(chosen_sample_lenghts)
    rejected_shortest_sample_tokens = min(rejected_sample_lenghts)
    rejected_longest_sample_tokens = max(rejected_sample_lenghts)
    rejected_total_tokens_count = sum(rejected_sample_lenghts)
    stats = {
        'samples_count': sample_count,
        'chosen_shortest_sample_tokens': chosen_shortest_sample_tokens,
        'chosen_longest_sample_tokens': chosen_longest_sample_tokens,
        'chosen_total_tokens_count': chosen_total_tokens_count,
        'rejected_shortest_sample_tokens': rejected_shortest_sample_tokens,
        'rejected_longest_sample_tokens': rejected_longest_sample_tokens,
        'rejected_total_tokens_count': rejected_total_tokens_count,
        'chosen_avg_sample_size': (chosen_total_tokens_count // sample_count),
        'rejected_avg_sample_size': (rejected_total_tokens_count // sample_count)
    }
    stats_str = json.dumps(stats, indent=4, ensure_ascii=False)
    logger.info(f"Stats:\n{stats_str}")
    stats_file = os.path.join(args.output_dir, "dataset_stats.json")
    with open(stats_file, 'w') as fin:
        json.dump(stats, fin)
    logger.info(f"Stats saved in {stats_file}")
    
    chosen_sample_lenght_histogram = dict(Counter(chosen_sample_lenghts))
    histogram_file = os.path.join(args.output_dir, "dataset_chosen_histogram.csv")
    with open(histogram_file, 'w') as fin:
        fin.write("token_count; sample_count\n")
        for length in range(0, max_sample_size+1):
            fin.write(f"{length}; {chosen_sample_lenght_histogram.get(length, 0)}\n")
    logger.info(f"Chosen samples histogram saved in {histogram_file}")
    
    rejected_sample_lenght_histogram = dict(Counter(rejected_sample_lenghts))
    histogram_file = os.path.join(args.output_dir, "dataset_rejected_histogram.csv")
    with open(histogram_file, 'w') as fin:
        fin.write("token_count; sample_count\n")
        for length in range(0, max_sample_size+1):
            fin.write(f"{length}; {rejected_sample_lenght_histogram.get(length, 0)}\n")
    logger.info(f"Rejected samples istogram saved in {histogram_file}")

    logger.info(f"Dataset with {sample_count:,} samples ({(chosen_total_tokens_count+rejected_total_tokens_count):,} tokens) has been created in {format_seconds_as_time(time.time()-timer)}")

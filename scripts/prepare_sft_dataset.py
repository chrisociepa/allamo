"""
Use this file to create a dataset for Supervised Fine-Tuning (SFT) training

The script performs the following steps:
1. Reads the input JSONL file with dialogues (single or multi-turn)
2. Applies the OpenChatML or Llama2 chat template to each dialogue
3. Tokenizes the formatted dialogues
4. Generates token weights
5. Optionally packs dialogues to maximize GPU utilization
6. Saves the processed data in a binary format
7. Generates and saves summary statistics for the dataset

Example record with signe-turn dialogue:
```json
{"messages": [{"role": "user", "content": "1+2=?"}, {"role": "assistant", "content": "3"}]}
```

Example record with multi-turn dialogue:
```json
{"messages": [{"role": "user", "content": "1+2=?"}, {"role": "assistant", "content": "3"}, {"role": "user", "content": "2+2=?"}, {"role": "assistant", "content": "4"}]}
```
"""

import argparse
import concurrent.futures
import joblib
import json
import logging
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

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger()

MIN_WEIGHT = 0.001

def tokenize_openchatml_conversation(data, tokenizer, ignore_index):
    conversation = data["messages"]
    weight = data["weight"]
    result = {'input_ids': [], 'target_ids': []}
    if weight > MIN_WEIGHT:
        result['target_weights'] = []

    last_idx = len(conversation) - 1
    for idx, entry in enumerate(conversation):
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
                if weight > 0:
                    result['target_weights'].extend(list(
                        0.0 if i < pre_input_ids_len else weight for i in range(len(full_input_ids))
                    ))
            else:
                logger.warning("Tokenization inconsistency detected. Performing separate tokenization")
                content_input_ids = tokenizer.encode(content, add_special_tokens=False)
                result['input_ids'].extend(pre_input_ids)
                result['input_ids'].extend(content_input_ids)
                result['target_ids'].extend(list(ignore_index for _ in range(pre_input_ids_len)))
                result['target_ids'].extend(content_input_ids)
                if weight > 0:
                    result['target_weights'].extend(list(0.0 for _ in range(pre_input_ids_len)))
                    result['target_weights'].extend(list(weight for _ in range(len(content_input_ids))))
        else:
            content = "<s><|im_start|>" if idx == 0 else "<|im_start|>"
            content += entry["role"] + '\n' + entry["content"] + '<|im_end|>\n'
            input_ids = tokenizer.encode(content, add_special_tokens=False)
            result['input_ids'].extend(input_ids)
            result['target_ids'].extend(list(ignore_index for _ in range(len(input_ids))))
            if weight > 0:
                result['target_weights'].extend(list(0.0 for _ in range(len(input_ids))))
    assert len(result['input_ids']) == len(result['target_ids'])
    if weight > 0:
        assert len(result['input_ids']) == len(result['target_weights'])
    return result

def tokenize_llama2_conversation(data, tokenizer, ignore_index):
    conversation = data["messages"]
    weight = data["weight"]
    result = {'input_ids': [], 'target_ids': []}
    if weight > MIN_WEIGHT:
        result['target_weights'] = []
        
    if conversation[0]['role'] == 'system':
        sys_message = f"<<SYS>>\n{conversation[0]['content']}\n<</SYS>>\n\n"
        conversation = conversation[1:]
    else:
        sys_message = ''
        
    for idx, entry in enumerate(conversation):
        if entry['role'] == 'user':
            content = '<s>[INST] '+sys_message if idx <= 1 else '[INST] '
            content += entry['content'] + ' [/INST]'
            input_ids = tokenizer.encode(content, add_special_tokens=False)
            result['input_ids'].extend(input_ids)
            result['target_ids'].extend(list(ignore_index for _ in range(len(input_ids))))
            if weight > 0:
                result['target_weights'].extend(list(0.0 for _ in range(len(input_ids))))
        elif entry['role'] == 'assistant':
            content = ' ' + entry['content'] + '</s>'
            input_ids = tokenizer.encode(content, add_special_tokens=False)
            result['input_ids'].extend(input_ids)
            result['target_ids'].extend(input_ids)
            if weight > 0:
                result['target_weights'].extend(
                    list(weight for _ in range(len(input_ids)))
                )
    assert len(result['input_ids']) == len(result['target_ids'])
    if weight > 0:
        assert len(result['input_ids']) == len(result['target_weights'])
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

def process_chunk(args):
    chunk_file, pack, tokenizer_path, chat_format, block_size, ignore_index, pad_token_id, min_unmasked_tokens = args
    max_sample_size = block_size + 1
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    data_dtype = np.int16 if len(tokenizer) < 32767 else np.int32
    truncated = 0
    rejected = 0
    data = []
    pa_table = pq.read_table(chunk_file)
    for i in range(len(pa_table['rows'])):
        cols = pa_table['rows'][i].as_py().split(';', 1)
        messages = json.loads(cols[1])
        if 'messages' not in messages:
            rejected += 1
        else:
            sample = tokenize_conversation({
                "messages": messages['messages'], 
                "weight": float(cols[0])
            }, tokenizer, ignore_index, chat_format)
            
            input_ids_len = len(sample['input_ids'])
            if input_ids_len > max_sample_size:
                sample['input_ids'] = sample['input_ids'][:max_sample_size]
                sample['target_ids'] = sample['target_ids'][:max_sample_size]
                if 'target_weights' in sample:
                    sample['target_weights'] = sample['target_weights'][:max_sample_size]
                truncated += 1
            data.append(sample)
    del pa_table
    
    created = len(data)
    packed = 0
    
    if pack:
        packed_data = []
        while data:
            instructions_buffer = data.pop()
            instructions_buffer["seq_lens"] = [len(instructions_buffer["input_ids"])]
            while len(data) > 0 and len(instructions_buffer["input_ids"]) + len(data[-1]["input_ids"]) <= max_sample_size:
                instruction = data.pop()
                instructions_buffer["input_ids"].extend(instruction["input_ids"])
                instructions_buffer["target_ids"].extend(instruction["target_ids"])
                if "target_weights" in instructions_buffer:
                    instructions_buffer["target_weights"].extend(instruction["target_weights"])
                instructions_buffer["seq_lens"].append(len(instruction["input_ids"]))
            packed_data.append(instructions_buffer)
        packed = len(packed_data)
        data = packed_data
        del packed_data
        
    result = []
    for sample in data:
        if pad_token_id >= 0:
            padding = max_sample_size - len(sample['input_ids'])
            assert padding >= 0
            if padding > 0:
                if padding > 1:
                    sample["input_ids"] = convert_to_numpy_array(sample["input_ids"], block_size, pad_token_id, data_dtype)
                else:
                    sample["input_ids"] = np.array(sample["input_ids"], dtype=data_dtype)
                sample["target_ids"] = convert_to_numpy_array(sample["target_ids"][1:], block_size, ignore_index, data_dtype)
                if "target_weights" in sample:
                    sample["target_weights"] = convert_to_numpy_array(sample["target_weights"][1:], block_size, 0, np.float16)
            else:
                assert len(sample["input_ids"]) == max_sample_size
                assert len(sample["target_ids"]) == max_sample_size
                assert len(sample["target_weights"]) == max_sample_size
                assert sum(sample["seq_lens"]) == max_sample_size
                sample["input_ids"] = np.array(sample["input_ids"][:-1], dtype=data_dtype)
                sample["target_ids"] = np.array(sample["target_ids"][1:], dtype=data_dtype)
                if "target_weights" in sample:
                    sample["target_weights"] = np.array(sample["target_weights"][1:], dtype=np.float16)
                if "seq_lens" in sample:
                    sample["seq_lens"][-1] -= 1
        else:
            sample["input_ids"] = np.array(sample["input_ids"][:block_size], dtype=data_dtype)
            sample["target_ids"] = np.array(sample["target_ids"][1:max_sample_size], dtype=data_dtype)
            if "target_weights" in sample:
                sample["target_weights"] = np.array(sample["target_weights"][1:max_sample_size], dtype=np.float16)
        
        assert isinstance(sample["input_ids"], np.ndarray)
        assert isinstance(sample["target_ids"], np.ndarray)
        if np.sum(sample['target_ids'] != ignore_index) >= min_unmasked_tokens:
            result.append(sample)
        else:
            rejected += 1
    
    with open(chunk_file, 'wb') as f:
        joblib.dump(result, f)
    
    return {'created': created, 'truncated': truncated, 'rejected': rejected, 'packed': packed}

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

def create_sample_for(input_ids, target_weights, seq_lens, data_dtype):
    sample = {'input_ids': np.array(input_ids, dtype=data_dtype)}
    if target_weights and isinstance(target_weights, list):
        sample['target_weights'] = np.array(target_weights, dtype=np.float16)
    if seq_lens and isinstance(seq_lens, list):
        sample['seq_lens'] = seq_lens
    return sample

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tokenize dialogues with weights')
    parser.add_argument("-c", "--config_path", help="Config file with a list of input files")
    parser.add_argument("-f", "--input_file", help="Input file")
    parser.add_argument("-i", "--input_dir", help="Directory with input jsonl files")
    parser.add_argument("-o", "--output_dir", help="Output dir")
    parser.add_argument("-n", "--num_output_files", type=int, default=1, help="Number of final output files")
    parser.add_argument("-t", "--tokenizer_path", required=True, help="Tokenizer path")
    parser.add_argument("-w", "--default_weight", type=float, default=-1, help="Default weight for input files")
    parser.add_argument("-p", "--max_workers", type=int, default=20, help="The max number of processes")
    parser.add_argument("-b", "--block_size", type=int, default=4096, help="Block/context size")
    parser.add_argument('--chat_format', type=str, choices=['OpenChatML', 'llama2'], default='OpenChatML', help='Chat format')
    parser.add_argument("--min_unmasked_tokens", type=int, default=1, help="Minimum number of unmasked target tokens required for a sample to be included in training")
    parser.add_argument("--ignore_index", type=int, default=-100, help="Specifies a target value that is ignored in loss computation. Default is -100")
    parser.add_argument("--pad_token_id", type=int, default=0, help="Specifies the padding token id. Default is 0")
    parser.add_argument("--chunk_size", type=int, default=100000, help="Chunk size")
    parser.add_argument('--save_samples', action='store_true', help='Save some samples')
    parser.add_argument('--pack', action='store_true', help='Pack')
    parser.add_argument('--verbose', action='store_true', help='Be verbose')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    logger.info(f"Loaded tokenizer with vocab size {len(tokenizer)}")
    logger.info(f"Active chat template type: {args.chat_format}")
    if not args.pack:
        logger.warning("Padding not applied as packing is disabled")

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
    all_rows = []
    for config in tqdm(configs, desc="Loading data", disable=(not args.verbose)):
        weight = config['weight'] if 'weight' in config else args.default_weight
        weight_doc_prefix = f"{weight};"
        with open(config['path'], 'r') as f:
            for line in f:
                if line:
                    all_rows.append(weight_doc_prefix + line)
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
    chunk_batches = list((chunk_file, args.pack, args.tokenizer_path, args.chat_format, args.block_size, args.ignore_index, args.pad_token_id, args.min_unmasked_tokens) for chunk_file in chunk_files)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for result in tqdm(executor.map(process_chunk, chunk_batches), total=len(chunk_batches), desc="Tokenizing", disable=(not args.verbose)):
            processed_chunk_stats.append(result)
    del executor
    
    stats = {'created': 0, 'truncated': 0, 'rejected': 0, 'packed': 0}
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
    
    assert isinstance(all_samples[0]["input_ids"], np.ndarray)
    assert isinstance(all_samples[0]["target_ids"], np.ndarray)

    assert sample_count > 0
    if args.save_samples:
        logger.info(f"Saving samples")
        samples_file = os.path.join(args.output_dir, "samples.jsonl")
        with open(samples_file, 'w') as f:
            for sample in all_samples[:100]:
                input_ids = sample["input_ids"].tolist()
                new_sample = {
                    "input": tokenizer.decode(input_ids),
                    "input_ids": input_ids,
                    "target_ids": sample["target_ids"].tolist(),
                }
                if 'target_weights' in sample:
                    new_sample["target_weights"] = sample["target_weights"].tolist()
                if 'seq_lens' in sample:
                    new_sample["seq_lens"] = sample["seq_lens"]
                
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
    if args.pack:
        sample_lenghts = [sum(sample['seq_lens']) for sample in all_samples]
    else:
        sample_lenghts = [np.sum(sample['input_ids'] != args.pad_token_id) for sample in all_samples]
    shortest_sample_tokens = min(sample_lenghts)
    longest_sample_tokens = max(sample_lenghts)
    total_tokens_count = sum(sample_lenghts)
    stats = {
        'instruction_count': instruction_count,
        'samples_count': sample_count,
        'shortest_sample_tokens': shortest_sample_tokens,
        'longest_sample_tokens': longest_sample_tokens,
        'total_tokens_count': total_tokens_count,
        'avg_instruction_size': (total_tokens_count // instruction_count),
        'avg_sample_size': (total_tokens_count // sample_count),
        'packing_ratio': (instruction_count / sample_count),
        'packing_level': (total_tokens_count / (sample_count * args.block_size) * 100),
    }
    stats_str = json.dumps(stats, indent=4, ensure_ascii=False)
    logger.info(f"Stats:\n{stats_str}")
    stats_file = os.path.join(args.output_dir, "dataset_stats.json")
    with open(stats_file, 'w') as fin:
        json.dump(stats, fin)
    logger.info(f"Stats saved in {stats_file}")
    
    sample_lenght_histogram = dict(Counter(sample_lenghts))
    histogram_file = os.path.join(args.output_dir, "dataset_histogram.csv")
    with open(histogram_file, 'w') as fin:
        fin.write("token_count; sample_count\n")
        for length in range(0, max_sample_size+1):
            fin.write(f"{length}; {sample_lenght_histogram.get(length, 0)}\n")
    logger.info(f"Histogram saved in {histogram_file}")

    logger.info(f"Dataset with {sample_count:,} samples ({total_tokens_count:,} tokens) has been created in {format_seconds_as_time(time.time()-timer)}")

"""
This file contains the full configuration and helps with its management.
"""

import time
import json
import argparse
from dataclasses import dataclass

@dataclass
class AllamoConfiguration:

    load_configuration: bool = True
    init_from: str = 'scratch'
    checkpoint_path: str = None
    seed: int = 1337
    data_dir: str = 'data'
    out_dir: str = 'out'
    in_memory_data: bool = False
    reload_datasets_interval: int = -1
    log_checkpoint_md5_on_load: bool = False
    log_checkpoint_md5_on_epoch: bool = False
    ignore_last_checkpoint_backup: bool = False
    log_interval: int = 1
    eval_interval: int = 1000
    eval_iters: int = 200
    eval_only: bool = False
    always_save_checkpoint: bool = True
    vocab_size: int = 31980
    tiktoken_tokenizer_name: str = None
    hf_tokenizer_path: str = None
    wandb_log: bool = False
    wandb_project: str = 'allamo'
    wandb_run_name: str = 'allamo-run-' + str(time.time())
    gradient_checkpointing: bool = False
    gradient_accumulation_steps: int = 8
    batch_size: int = 64
    block_size: int = 1024
    sliding_window: int = None
    dataloader_type: str = 'allamo'
    dataset: str = None
    dataset_train_files: str = None
    dataset_validation_files: str = None
    dataset_train_file_prefix: str = 'train.'
    dataset_validation_file_prefix: str = 'val.'
    dataset_train_processed_files_count: int = 0
    dataset_seq_train: bool = False
    dataset_seq_train_start: int = None
    dataset_seq_step_size: int = None 
    batch_size_initial: int = 2
    batch_size_max_iter: int = 2000
    batch_size_schedule: bool = False
    batch_size_max: int = 64
    grad_accum_initial: int = 2
    grad_accum_max_iter: int = 2000
    grad_accum_schedule: bool = False
    grad_accum_max: int = 8
    rope_freq_base: int = 10000
    rope_freq_scale: float = 1.0
    n_layer: int = 12
    n_head: int = 12
    head_size: int = 64
    num_kv_heads: int = None
    n_embd: int = 768
    intermediate_size: int = None
    dropout: float = 0.0 
    bias: bool = False 
    multiple_of: int = 256
    norm_eps: float = 1e-5
    learning_rate: float = 6e-4
    num_train_epochs: int = None
    max_iters: int = 600000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0 
    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = 600000
    lr_decay_reset_iters: int = 60000
    min_lr: float = 6e-5 
    backend: str = 'nccl' 
    device: str = 'cuda' 
    dtype: str = 'float16'
    compile: bool = False
    compile_mode: str = 'default'
    mfu_flops_peak: float = -1.0
    ignore_index: int = -100
    pad_token_id: int = -1
    weighted_loss: bool = False
    adaptive_learning_rate: bool = False
    fsdp_sharding_strategy: str = 'FULL_SHARD'
    epoch_completion_hook_program: str = None
    
    # inference params
    prompt: str = "\n" 
    num_samples: int = 1 
    max_new_tokens: int = 50 
    temperature: float = 0.8 
    top_k: int = 100
    
    def __post_init__(self):
        if self.load_configuration:
            self.load_values()
    
    def load_values(self):
        parser = argparse.ArgumentParser(description='Allamo allows you to train and evaluate LLaMA-based models.')
        parser.add_argument('--config', help='Path to a json configuration file')
        parser.add_argument('--init_from', type=str, choices=['scratch', 'resume', 'resume_last'], help='Start from scratch or resume from best or last checkpoint')
        parser.add_argument('--checkpoint_path', type=str, help='Custom input checkpoint path')
        parser.add_argument('--seed', type=int, help='The desired seed for generating random numbers')
        parser.add_argument('--data_dir', type=str, help='Directory where datasets exist')
        parser.add_argument('--out_dir', type=str, help='Output directory for checkpoints')
        parser.add_argument('--in_memory_data', type=bool, help='Store data in memory')
        parser.add_argument('--reload_datasets_interval', type=int, help='Number of iterations when datasets are reloaded')
        parser.add_argument('--log_checkpoint_md5_on_load', type=bool, help='When loading a checkpoint, log its MD5 checksum')
        parser.add_argument('--log_checkpoint_md5_on_epoch', type=bool, help='When saving a checkpoint at the end of an epoch, log its MD5 checksum')
        parser.add_argument('--ignore_last_checkpoint_backup', type=bool, help='Ignores preserving a copy of the last checkpoint version by overwriting it')
        parser.add_argument('--log_interval', type=int, help='Number of iterations when training loss is logged')
        parser.add_argument('--eval_interval', type=int, help='Number of iterations when evaluating model')
        parser.add_argument('--eval_iters', type=int, help='Number of iterations when evaluating')
        parser.add_argument('--eval_only', type=bool, help='Exit right after the first evaluation. Indicates no training.')
        parser.add_argument('--always_save_checkpoint', type=bool, help='Enable saving the last checkpoint')
        parser.add_argument('--vocab_size', type=int, help='Vacabulary size. Might be overwritten by checkpoint')
        parser.add_argument('--tiktoken_tokenizer_name', type=str, help='Tiktoken tokenizer name. Might be overwritten by checkpoint')
        parser.add_argument('--hf_tokenizer_path', type=str, help='HuggingFace tokenizer path. Might be overwritten by checkpoint')
        parser.add_argument('--wandb_log', type=bool, help='Enable logging to wandb')
        parser.add_argument('--wandb_project', type=str, help='Wandb project name')
        parser.add_argument('--wandb_run_name', type=str, help='Wandb run name')
        parser.add_argument('--gradient_checkpointing', type=bool, help='Enable gradient checkpointing')
        parser.add_argument('--gradient_accumulation_steps', type=int, help='Help simulating larger batch sizes')
        parser.add_argument('--batch_size', type=int, help='Batch size')
        parser.add_argument('--sliding_window', type=int, help='Sliding window attention window size')
        parser.add_argument('--block_size', type=int, help='The maximum sequence length that this model might ever be used with')
        parser.add_argument('--dataloader_type', type=str, choices=['allamo', 'simple', 'simple-instructions'], help='The type of Data Loader')
        parser.add_argument('--dataset', type=str, help='The name of the dataset directory within the data_dir')
        parser.add_argument('--dataset_train_files', type=str, help='Comma-separated list of training dataset files to use')
        parser.add_argument('--dataset_validation_files', type=str, help='Comma-separated list of validation dataset files to use')
        parser.add_argument('--dataset_train_file_prefix', type=str, help='Custom prefix for training dataset files')
        parser.add_argument('--dataset_validation_file_prefix', type=str, help='Custom prefix for validation dataset files')
        parser.add_argument('--dataset_train_processed_files_count', type=int, help='The number of files already processed in the training dataset')
        parser.add_argument('--dataset_seq_train', type=bool, help='Iterate dataset sequentially')
        parser.add_argument('--dataset_seq_train_start', type=int, help='Position in tokens to start with')
        parser.add_argument('--dataset_seq_step_size', type=int, help='Step size when iterate dataset sequentially. E.g. block_size/2')
        parser.add_argument('--batch_size_initial', type=int, help='Initial batch_size value')
        parser.add_argument('--batch_size_max_iter', help='Number of iterations to reach maximum batch_size value')
        parser.add_argument('--batch_size_schedule', type=bool, help='Enable linear batch_size scheduler')
        parser.add_argument('--grad_accum_initial', type=int, help='Initial gradient_accumulation_steps value')
        parser.add_argument('--grad_accum_max_iter', type=int, help='Number of iterations to reach maximum gradient_accumulation_steps value')
        parser.add_argument('--grad_accum_schedule', type=bool, help='Enable linear gradient_accumulation_steps scheduler')
        parser.add_argument('--rope_freq_base', type=int, help='RoPE base frequency')
        parser.add_argument('--rope_freq_scale', type=int, help='RoPE frequency scaling factor')
        parser.add_argument('--n_layer', type=int, help='Number of layers')
        parser.add_argument('--n_head', type=int, help='Number of heads')
        parser.add_argument('--head_size', type=int, help='Often calculated as n_embd/n_head')
        parser.add_argument('--num_kv_heads', type=int, help='Number of key-value heads')
        parser.add_argument('--n_embd', type=int, help='Number of model dimensions')
        parser.add_argument('--intermediate_size', type=int, help='Dimension of the MLP representations')
        parser.add_argument('--dropout', type=float, help='Enable dropouts globally. Disabled when 0')
        parser.add_argument('--bias', type=bool, help='Enable bias globally. Helpful in finetuning process')
        parser.add_argument('--multiple_of', type=int, help='Make SwiGLU hidden layer size multiple of large power of 2. Used only when intermediate_size is not specified')
        parser.add_argument('--norm_eps', type=float, help='RMSNorm normalizing function param')
        parser.add_argument('--learning_rate', type=float, help='Learning rate to start with')
        parser.add_argument('--num_train_epochs', type=int, help='Total number of training epochs to perform')
        parser.add_argument('--max_iters', type=int, help='Total number of training iterations')
        parser.add_argument('--weight_decay', type=float, help='Max learning rate')
        parser.add_argument('--beta1', type=float, help='Adamw optimizer Beta1 param')
        parser.add_argument('--beta2', type=float, help='Adamw optimizer Beta2 param')
        parser.add_argument('--grad_clip', type=float, help='Clip gradients at this value. Disabled when 0.')
        parser.add_argument('--decay_lr', type=bool, help='Whether to decay the learning rate')
        parser.add_argument('--warmup_iters', type=int, help='Learning rate is calculated linearly for warmup_iters steps')
        parser.add_argument('--lr_decay_iters', type=int, help='Learning rate decay iterations. When exceeded, the min_lr is used')
        parser.add_argument('--lr_decay_reset_iters', type=int, help='Number of iterations for the learning rate decay restart')
        parser.add_argument('--min_lr', type=float, help='Minimum learning rate')
        parser.add_argument('--backend', type=str, help='"nccl", "gloo", etc.')
        parser.add_argument('--device', type=str, help='"cpu", "cuda", "cuda:0", "cuda:1" etc., or try "mps" on macbooks')
        parser.add_argument('--dtype', type=str, choices=['float32', 'bfloat16', 'bfloat16-true', 'float16'], help='Type of tensor to be used in the model')
        parser.add_argument('--compile', type=bool, help='Whether to use PyTorch 2.0 to compile the model to be faster')
        parser.add_argument('--compile_mode', type=str, choices=['default', 'reduce-overhead', 'max-autotune'], help='Specifies what the PyTorch compiler should be optimizing while compiling')
        parser.add_argument('--mfu_flops_peak', type=float, help="Specifies the MFU's peak performance in FLOPs. A default value of -1 disables MFU estimation")
        parser.add_argument('--ignore_index', type=int, help="Specifies a target value that is ignored and does not contribute to the input gradient")
        parser.add_argument('--pad_token_id', type=float, help="Enables padding and specifies the token id used for padding in sequences")
        parser.add_argument('--weighted_loss', type=bool, help='Whether to use weighted loss if available')
        parser.add_argument('--adaptive_learning_rate', type=bool, help='Whether to use adaptive learning rate')
        parser.add_argument('--fsdp_sharding_strategy', type=str, choices=['FULL_SHARD', 'HYBRID_SHARD', '_HYBRID_SHARD_ZERO2', 'SHARD_GRAD_OP', 'NO_SHARD'], help='FSDP sharding strategy')
        parser.add_argument('--epoch_completion_hook_program', type=str, help='Path to the program/script to be executed after the epoch ends and the checkpoint is saved')
        parser.add_argument('--prompt', type=str, help='Prompt for generating text. Can also specify a file, use as: "FILE:prompt.txt"')
        parser.add_argument('--num_samples', type=int, help='Number of samples to generate')
        parser.add_argument('--max_new_tokens', type=int, help='Number of tokens to generate in each sample')
        parser.add_argument('--temperature', type=float, help='Temperature value for text generation')
        parser.add_argument('--top_k', type=int, help='Top k most likely tokens to be retained during text generation')

        args = parser.parse_args()
        
        if args.config:
            with open(args.config) as f:
                config = json.load(f)
            for k, v in config.items():
                if hasattr(self, k):
                    setattr(self, k, v)

        for arg_name, arg_value in vars(args).items():
            if arg_value is not None and hasattr(self, arg_name):
                setattr(self, arg_name, arg_value)


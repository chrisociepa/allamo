import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import List, Dict

logger = logging.getLogger("AllamoConfiguration")

@dataclass
class AllamoConfiguration:

    load_configuration: bool = True
    config_override_check_interval: int = None
    config_override_path: str = None

    init_from: str = 'scratch'
    training_type: str = None

    out_dir: str = 'out'
    checkpoint_path: str = None
    log_checkpoint_md5_on_load: bool = False
    log_checkpoint_md5_on_epoch: bool = False
    ignore_last_checkpoint_backup: bool = False
    checkpoint_interval: int = 1000
    save_optimizer_checkpoint: bool = True
    optimizer_checkpoint_interval: int = None
    save_best_checkpoint: bool = False
    save_checkpoint_on_dataset_reload: bool = False
    distributed_checkpoint: bool = False

    eval_interval: int = 1000
    eval_iters: int = 200
    eval_only: bool = False
    
    gradient_accumulation_steps: int = 8
    batch_size: int = 64
    batch_size_initial: int = 2
    batch_size_max_iter: int = 2000
    batch_size_schedule: bool = False
    batch_size_max: int = 64
    grad_accum_initial: int = 2
    grad_accum_max_iter: int = 2000
    grad_accum_schedule: bool = False
    grad_accum_max: int = 8    
    num_train_epochs: int = None
    max_iters: int = 600000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    adaptive_learning_rate: bool = False
    learning_rate: float = 6e-4
    warmup_iters: int = 2000
    decay_lr: bool = False
    min_lr: float = 6e-5
    lr_decay_iters: int = 600000
    lr_decay_reset_iters: int = 60000
    
    ignore_index: int = -100
    pad_token_id: int = -1
    weighted_loss: bool = False
    weighted_loss_method: str = 'allamo'

    seed: int = 1337
    backend: str = 'nccl' 
    device: str = 'cuda' 
    dtype: str = 'float16'
    compile: bool = False
    compile_mode: str = 'default'
    mfu_flops_peak: float = -1.0
    attention_implementation: str = 'sdpa'
    fsdp_sharding_strategy: str = 'FULL_SHARD'
    tensor_parallel_degree: int = 1
    enable_cpu_offload: bool = False
    epoch_completion_hook_program: str = None
    regular_checkpoint_hook_program: str = None

    # dataloader
    data_dir: str = 'data'
    dataset: str = None
    dataset_train_files: str = None
    dataset_validation_files: str = None
    dataset_train_file_prefix: str = 'train.'
    dataset_validation_file_prefix: str = 'val.'
    dataset_train_processed_files_count: int = 0
    dataset_seq_train: bool = True
    dataset_seq_train_start: int = None
    dataset_buffer: bool = False

    # model specification
    model_type: str = 'bielik2'
    block_size: int = 1024
    vocab_size: int = 32000
    rope_freq_base: int = 10000
    rope_scaling: Dict = field(default_factory=dict)
    n_layer: int = 12
    num_kv_heads: int = None
    head_size: int = 64
    n_head: int = 12
    n_embd: int = 768
    intermediate_size: int = None
    dropout: float = 0.0
    bias: bool = False
    norm_eps: float = 1e-5
    sliding_window: int = None
    act_fn: str = "silu"
    act_fn_params: Dict = field(default_factory=dict)

    # metrics logging
    log_interval: int = 1
    log_metrics: bool = False
    metrics_logger: str = None
    metrics_logger_project: str = 'allamo'
    metrics_logger_run_name: str = 'allamo-run-' + str(time.time())
    metrics_logger_run_id: str = None
    metrics_logger_hardware_monitoring: bool = False
    metrics_logger_tags: List[str] = None
    
    # gradient checkpointing
    gradient_checkpointing: bool = False
    gradient_checkpointing_excluded_layers: int = 0

    # freezing params
    freeze_embeddings: bool = False
    freeze_lm_head: bool = False
    freeze_layers: bool = False
    keep_layers_trainable: List[int] = field(default_factory=list)

    # DPO params
    dpo_chosen_beta: float = 0.5
    dpo_rejected_beta: float = 0.1
    dpo_penalty_lambda: float = 50.0
    reference_checkpoint_name: str = 'ref_ckpt'
    
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
        parser.add_argument('--config_override_check_interval', type=int, help='Number of iterations for checking override configuration. Feature disabled if not specified.')
        parser.add_argument('--config_override_path', type=str, help='Specifies the location of the configuration override file')

        parser.add_argument('--init_from', type=str, choices=['scratch', 'resume', 'resume_last'], help='Start from scratch or resume from best or last checkpoint')
        parser.add_argument('--training_type', type=str, choices=['pre', 'sft', 'dpo'], required=True, help='Specifies the type of training: pre (pre-training), sft (supervised fine-tuning), or dpo (direct preference optimization)')

        parser.add_argument('--out_dir', type=str, help='Output directory for checkpoints')
        parser.add_argument('--checkpoint_path', type=str, help='Custom input checkpoint path')
        parser.add_argument('--log_checkpoint_md5_on_load', action='store_true', default=None, help='When loading a checkpoint, log its MD5 checksum')
        parser.add_argument('--log_checkpoint_md5_on_epoch', action='store_true', default=None, help='When saving a checkpoint at the end of an epoch, log its MD5 checksum')
        parser.add_argument('--ignore_last_checkpoint_backup', action='store_true', default=None, help='Ignores preserving a copy of the last checkpoint version by overwriting it')
        parser.add_argument('--checkpoint_interval', type=int, help='Number of iterations between checkpoints where the state of the model is saved')
        parser.add_argument('--save_optimizer_checkpoint', action='store_true', default=None, help='Enable saving optimizer checkpoint')
        parser.add_argument('--optimizer_checkpoint_interval', type=int, help='Number of iterations between checkpoints where the state of the optimizer is saved. The same as checkpoint_interval, if not specified')
        parser.add_argument('--save_best_checkpoint', action='store_true', default=None, help='Enable saving the best checkpoint when evaluating model')
        parser.add_argument('--save_checkpoint_on_dataset_reload', action='store_true', default=None, help='Enable model checkpoint saving on dataset reload')
        parser.add_argument('--distributed_checkpoint', action='store_true', default=None, help='Use PyTorch Distributed Checkpoint (DCP)')

        parser.add_argument('--eval_interval', type=int, help='Number of iterations when evaluating model')
        parser.add_argument('--eval_iters', type=int, help='Number of iterations when evaluating')
        parser.add_argument('--eval_only', action='store_true', default=None, help='Exit right after the first evaluation. Indicates no training.')

        parser.add_argument('--gradient_accumulation_steps', type=int, help='Help simulating larger batch sizes')
        parser.add_argument('--batch_size', type=int, help='Batch size')
        parser.add_argument('--batch_size_initial', type=int, help='Initial batch_size value')
        parser.add_argument('--batch_size_max_iter', help='Number of iterations to reach maximum batch_size value')
        parser.add_argument('--batch_size_schedule', action='store_true', default=None, help='Enable linear batch_size scheduler')
        parser.add_argument('--grad_accum_initial', type=int, help='Initial gradient_accumulation_steps value')
        parser.add_argument('--grad_accum_max_iter', type=int, help='Number of iterations to reach maximum gradient_accumulation_steps value')
        parser.add_argument('--grad_accum_schedule', action='store_true', default=None, help='Enable linear gradient_accumulation_steps scheduler')
        parser.add_argument('--num_train_epochs', type=int, help='Total number of training epochs to perform')
        parser.add_argument('--max_iters', type=int, help='Total number of training iterations')
        parser.add_argument('--weight_decay', type=float, help='Max learning rate')
        parser.add_argument('--beta1', type=float, help='Adamw optimizer Beta1 param')
        parser.add_argument('--beta2', type=float, help='Adamw optimizer Beta2 param')
        parser.add_argument('--grad_clip', type=float, help='Clip gradients at this value. Disabled when 0.')

        parser.add_argument('--adaptive_learning_rate', action='store_true', default=None, help='Whether to use adaptive learning rate')
        parser.add_argument('--learning_rate', type=float, help='Learning rate to start with')
        parser.add_argument('--warmup_iters', type=int, help='Learning rate is calculated linearly for warmup_iters steps')
        parser.add_argument('--decay_lr', action='store_true', default=None, help='Whether to decay the learning rate')
        parser.add_argument('--min_lr', type=float, help='Minimum learning rate')
        parser.add_argument('--lr_decay_iters', type=int, help='Learning rate decay iterations. When exceeded, the min_lr is used')
        parser.add_argument('--lr_decay_reset_iters', type=int, help='Number of iterations for the learning rate decay restart')

        parser.add_argument('--ignore_index', type=int, help="Specifies a target value that is ignored and does not contribute to the input gradient")
        parser.add_argument('--pad_token_id', type=float, help="Enables padding and specifies the token id used for padding in sequences")
        parser.add_argument('--weighted_loss', action='store_true', default=None, help='Whether to use weighted loss if available')
        parser.add_argument('--weighted_loss_method', type=str, choices=['allamo', 'openchat'], help='How weighted loss is calculated')

        parser.add_argument('--seed', type=int, help='The desired seed for generating random numbers')
        parser.add_argument('--backend', type=str, help='Specifies one of three built-in backends: nccl, gloo, mpi')
        parser.add_argument('--device', type=str, help='"cpu", "cuda", "cuda:0", "cuda:1" etc., or try "mps" on macbooks')
        parser.add_argument('--dtype', type=str, choices=['float32', 'bfloat16', 'bfloat16-true', 'float16'], help='Type of tensor to be used in the model')
        parser.add_argument('--compile', action='store_true', default=None, help='Whether to use PyTorch 2.0 to compile the model to be faster')
        parser.add_argument('--compile_mode', type=str, choices=['default', 'reduce-overhead', 'max-autotune'], help='Specifies what the PyTorch compiler should be optimizing while compiling')
        parser.add_argument('--mfu_flops_peak', type=float, help="Specifies the MFU's peak performance in FLOPs. A default value of -1 disables MFU estimation")
        parser.add_argument('--attention_implementation', type=str, choices=['eager', 'sdpa', 'fa2', 'fa3', 'xformers', 'flex'], help='Specifies attention implementation')
        parser.add_argument('--fsdp_sharding_strategy', type=str, choices=['FULL_SHARD', 'HYBRID_SHARD', '_HYBRID_SHARD_ZERO2', 'SHARD_GRAD_OP', 'NO_SHARD'], help='FSDP sharding strategy')
        parser.add_argument('--tensor_parallel_degree', type=int, help='Specifies the degree of tensor parallelism. Activates TP when it is greater than 1')
        parser.add_argument('--enable_cpu_offload', action='store_true', default=None, help='Whether to enable CPU offloading of parameters, gradients, and optimizer states in FSDP')
        parser.add_argument('--epoch_completion_hook_program', type=str, help='Path to the program/script to be executed after the epoch ends and the checkpoint is saved')
        parser.add_argument('--regular_checkpoint_hook_program', type=str, help='Path to the program/script to be executed after the regualar checkpoint is saved')

        parser.add_argument('--data_dir', type=str, help='Directory where datasets exist')
        parser.add_argument('--dataset', type=str, help='The name of the dataset directory within the data_dir')
        parser.add_argument('--dataset_train_files', type=str, help='Comma-separated list of training dataset files to use')
        parser.add_argument('--dataset_validation_files', type=str, help='Comma-separated list of validation dataset files to use')
        parser.add_argument('--dataset_train_file_prefix', type=str, help='Custom prefix for training dataset files')
        parser.add_argument('--dataset_validation_file_prefix', type=str, help='Custom prefix for validation dataset files')
        parser.add_argument('--dataset_train_processed_files_count', type=int, help='The number of files already processed in the training dataset')
        parser.add_argument('--dataset_seq_train', action='store_true', default=None, help='Iterate dataset sequentially')
        parser.add_argument('--dataset_seq_train_start', type=int, help='Position in tokens to start with')
        parser.add_argument('--dataset_buffer', action='store_true', default=None, help='Enable buffer for dataset samples')

        parser.add_argument('--model_type', type=str, help='Model type to use')
        parser.add_argument('--block_size', type=int, help='The maximum sequence length that this model might ever be used with')
        parser.add_argument('--vocab_size', type=int, help='Vacabulary size. Might be overwritten by checkpoint')
        parser.add_argument('--rope_freq_base', type=int, help='RoPE base frequency')
        parser.add_argument('--n_layer', type=int, help='Number of layers')
        parser.add_argument('--num_kv_heads', type=int, help='Number of key-value heads')
        parser.add_argument('--head_size', type=int, help='Often calculated as n_embd/n_head')
        parser.add_argument('--n_head', type=int, help='Number of heads')
        parser.add_argument('--n_embd', type=int, help='Number of model dimensions')
        parser.add_argument('--intermediate_size', type=int, help='Dimension of the MLP representations')
        parser.add_argument('--dropout', type=float, help='Enable dropouts globally. Disabled when 0')
        parser.add_argument('--bias', action='store_true', default=None, help='Enable bias globally. Helpful in finetuning process')
        parser.add_argument('--norm_eps', type=float, help='RMSNorm normalizing function param')
        parser.add_argument('--sliding_window', type=int, help='Enable sliding window attention with specified window size')

        parser.add_argument('--log_interval', type=int, help='Number of iterations when training loss is logged')
        parser.add_argument('--log_metrics', action='store_true', default=None, help='Enable logging metrics')
        parser.add_argument('--metrics_logger', type=str, choices=['wandb', 'neptune'], help='Metrics logger type')
        parser.add_argument('--metrics_logger_project', type=str, help='Metrics logger project name')
        parser.add_argument('--metrics_logger_run_name', type=str, help='Metrics logger run name')
        parser.add_argument('--metrics_logger_run_id', type=str, help='Metrics logger run id')
        parser.add_argument('--metrics_logger_hardware_monitoring', action='store_true', default=None, help='Enable hardware monitoring for Neptune')
        parser.add_argument('--metrics_logger_tags', type=str, nargs='*', help='Metrics logger tags')

        parser.add_argument('--gradient_checkpointing', action='store_true', default=None, help='Enable gradient checkpointing')
        parser.add_argument('--gradient_checkpointing_excluded_layers', type=int, help='Specifies how many layers will not use gradient checkpointing')

        parser.add_argument('--freeze_embeddings', action='store_true', default=None, help='Freeze embeddings')
        parser.add_argument('--freeze_lm_head', action='store_true', default=None, help='Freeze lm_head')
        parser.add_argument('--freeze_layers', action='store_true', default=None, help='Freeze all layers')
        parser.add_argument('--keep_layers_trainable', type=int, nargs='*', default=[], help='List of layer indices to keep trainable (e.g., --keep_layers_trainable 0 31)')

        parser.add_argument('--dpo_chosen_beta', type=float, help='Temperature parameter for the chosen part of the DPO loss, typically something in the range of 0.1 to 0.5')
        parser.add_argument('--dpo_rejected_beta', type=float, help='Temperature parameter for the rejected part of the DPO loss, typically something in the range of 0.1 to 0.5')
        parser.add_argument('--dpo_penalty_lambda', type=float, help='Temperature parameter for penalty-positive in the DPO loss, typically in the range of 1 to 100')

        args = parser.parse_args()
        
        if args.config:
            with open(args.config) as f:
                config = json.load(f)
            self.override_values(config)

        for arg_name, arg_value in vars(args).items():
            if arg_value is not None and hasattr(self, arg_name):
                setattr(self, arg_name, arg_value)
        
        self.validate_configuration()

    def override_values(self, config_dict):
        modified = {}
        for k, v in config_dict.items():
            if hasattr(self, k) and getattr(self, k) != v:
                modified[k] = {"prev": getattr(self, k), "curr": v}
                setattr(self, k, v)
        return modified
    
    def validate_configuration(self):
        if self.training_type is None:
            raise ValueError("Training type must be specified")
    
    def should_override_config(self, iter_num):
        return self.config_override_check_interval is not None and \
            self.config_override_path is not None and \
            self.config_override_check_interval > 0 and \
            iter_num % self.config_override_check_interval == 0
            
    def override_config_properties(self):
        if os.path.exists(self.config_override_path):
            try:
                with open(self.config_override_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                modified = self.override_values(config)
                if modified:
                    logger.info(f"The following config properties were overridden: {modified}")
            except Exception as err:
                logger.warning(f"Unable to load override config. Error: {err}")
        

"""
This file is intended to perform some magic for training/finetuning using FSDP.
"""

import gc
import json
import os
import time
import math
import pickle
import random
import logging
import datetime
import dataclasses
import uuid
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig, # general model non-sharded, non-flattened params
    FullOptimStateDictConfig,
)

from model import AllamoTransformerConfig, AllamoTransformer, SelfAttentionBlock
from configuration import AllamoConfiguration
from data_loader import AllamoDataLoader
from fsdp_utils import (
    build_world_mesh,
    dist_all_reduce,
    parallelize_model,
    model_distributed_checkpoint_files_exist,
    load_distributed_checkpoint,
    save_distributed_checkpoint,
)
from train_utils import (
    rename_file_to_prev_version,
    calculate_md5,
    remove_unwanted_prefix_from_model_state_dict,
    get_lr,
    get_grad_accum,
    format_seconds_as_time,
    calculate_eta,
    has_next_iter_to_perform,
    estimate_mfu,
    get_model_checkpoint_path,
    get_config_checkpoint_path,
    get_optimizer_checkpoint_path,
    model_checkpoint_files_exist,
    run_checkpoint_hook_program,
)

DISTRIBUTED_CHECKPOINT_AVAILABLE = True

class AllamoFSDPTrainer:

    def __init__(self, config: AllamoConfiguration):
        self.run_uuid = str(uuid.uuid4())
        self.training_uuid = self.run_uuid
        self.config = config
        self.__init_torch(config)
        self.__init_logger(config)
        self.logger.info(f"Torch initialized for run {self.run_uuid}")
        self.logger.info(
            f"RANK: {self.rank}, LOCAL_RANK: {self.local_rank}, "
            f"WORLD_SIZE: {self.world_size}, LOCAL_WORLD_SIZE: {os.environ['LOCAL_WORLD_SIZE']}, "
            f"DP_SIZE: {self.dp_size}, DP_RANK: {self.dp_rank}"
        )
        
        self.iter_num = 0
        self.best_train_loss = 1e2
        self.best_val_loss = 1e2
        self.processed_tokens = 0
        self.data_loader = AllamoDataLoader(config, self.dp_rank, self.dp_size)
        self.__init_training(config)
        
    def __init_logger(self, config: AllamoConfiguration):
        if self.master_process:
            run_timestamp_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            log_file_path = os.path.join(config.out_dir, f'train-{run_timestamp_str}.log')
            logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(log_file_path)])
        self.logger = logging.getLogger('AllamoTrainer')
            
    def __init_torch(self, config: AllamoConfiguration):
        dist.init_process_group(backend=config.backend)
        self.rank = int(os.environ['RANK'])
        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.world_size = int(os.environ['WORLD_SIZE'])
        config.device = f'cuda:{self.local_rank}'
        torch.cuda.set_device(config.device)
        self.master_process = self.rank == 0 # this process will do logging, checkpointing etc.
        self.world_mesh = build_world_mesh(self.world_size, config)
        self.device_mesh = self.world_mesh["dp"] if self.world_mesh is not None else None
        self.dp_size = self.device_mesh.size() if self.world_mesh is not None else self.world_size
        self.dp_rank = self.device_mesh.get_local_rank() if self.world_mesh is not None else self.rank
        if self.master_process:
            os.makedirs(config.out_dir, exist_ok=True)
        torch.manual_seed(config.seed + self.dp_rank)
        torch.cuda.manual_seed(config.seed + self.dp_rank)
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        torch.set_float32_matmul_precision("highest") # set to "high" for faster matrix multiplications with bfloat16
        if config.dtype == 'bfloat16-true':
            raise Exception('Full bfloat16 training is not supported with FSDP')
        
        self.fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        if config.gradient_checkpointing:
            self.fsdp_activation_checkpointing = True
            config.gradient_checkpointing = False # control gradient checkpointing with FSDP 
        else:
            self.fsdp_activation_checkpointing = False
            
    def __init_training(self, config: AllamoConfiguration):
        model_config_fields = [f.name for f in dataclasses.fields(AllamoTransformerConfig)]
        ckpt_dir = config.checkpoint_path if config.checkpoint_path else config.out_dir
        checkpoint_name = None
        if config.init_from == 'resume':
            checkpoint_name = 'ckpt'
        elif config.init_from == 'resume_last':
            checkpoint_name = 'last_eval_ckpt'
        else:
            if self.check_model_checkpoint_files('ckpt', ckpt_dir) or self.check_model_checkpoint_files('last_eval_ckpt', ckpt_dir):
                self.logger.info("Delete existing checkpoint files to start from scratch or use --init_from=resume to resume training")
                exit()
        
        if checkpoint_name is not None:
            if self.check_model_checkpoint_files(checkpoint_name, ckpt_dir):
                self.logger.info(f"Resuming training from {ckpt_dir} and start loading '{checkpoint_name}' checkpoint files")
                self.load_config_checkpoint(os.path.join(ckpt_dir, f'config_{checkpoint_name}.json'), config, model_config_fields)
            elif config.init_from == 'resume_last':
                if self.master_process:
                    self.logger.warning(f"'{checkpoint_name}' checkpoint files not found but allowing to start from scratch")
                checkpoint_name = None
            else:
                raise Exception(f"'{checkpoint_name}' checkpoint files not found!")
                
        model_args = {k: getattr(config, k) for k in model_config_fields if hasattr(config, k)}
        modelConf = AllamoTransformerConfig(**model_args)
        if self.fsdp_activation_checkpointing:
            modelConf.gradient_checkpointing = False
            
        if self.world_mesh is None or not DISTRIBUTED_CHECKPOINT_AVAILABLE:
            model = AllamoTransformer(modelConf)
            self.model_num_params = model.model_num_params
            if checkpoint_name is None:
                self.logger.info("Initialized a new model from scratch")
            else:
                self.load_model_checkpoint(model, os.path.join(ckpt_dir, f'model_{checkpoint_name}.pt'), config)
            
            self.logger.info("Configuring model with FSDP")
            model = parallelize_model(model, self.world_mesh, config, self.fsdp_activation_checkpointing)
            
            # optimizer
            self.optimizer = model.configure_optimizers(config, "cuda", use_fused=(self.world_mesh is None))
            if checkpoint_name is None:
                self.logger.info("Initializing optimizer from scratch")
            else:
                if self.world_mesh is None:
                    self.load_optimizer_checkpoint(model, self.optimizer, os.path.join(ckpt_dir, f'optimizer_{checkpoint_name}.pt'), config)
                else:
                    self.logger.info("Loading optimizer state for FSDP2 is not supported yet!")
        else:
            with torch.device("meta"):
                model = AllamoTransformer(modelConf, with_init_weights=False)
            self.model_num_params = model.model_num_params
            
            self.logger.info("Configuring model with FSDP2")
            model = parallelize_model(model, self.world_mesh, config, self.fsdp_activation_checkpointing)
            
            # allocate sharded model on GPU and initialize weights via DTensor
            model.to_empty(device="cuda")
            model.init_model_weights()
            
            self.optimizer = model.configure_optimizers(config, "cuda", use_fused=(self.world_mesh is None))
            
            if checkpoint_name is None:
                self.logger.info("Initializing from scratch")
            else:
                load_distributed_checkpoint(model, self.optimizer, ckpt_dir, checkpoint_name, config)

        if config.compile:
            # convert to desire dtype, to make it work with FSDP and torch.compile
            ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config.dtype]
            model.rotary_emb.cos_cached = model.rotary_emb.cos_cached.to(ptdtype)
            model.rotary_emb.sin_cached = model.rotary_emb.sin_cached.to(ptdtype)

        self.model = model
        
        # gradient_accumulation scheduler
        if config.grad_accum_schedule: 
            config.grad_accum_max = config.gradient_accumulation_steps
            config.gradient_accumulation_steps = config.grad_accum_initial
            self.logger.info(
                f"Gradient accumulation scheduler enabled. "
                f"Current gradient accumulation steps: {config.gradient_accumulation_steps}"
            )
        self.gradient_accumulation_steps = config.gradient_accumulation_steps
        
        if config.decay_lr:
            self.logger.info(f"Cosing decay learning rate enabled. Currect learning rate: {get_lr(self.iter_num, self.config)}")
        else:
            self.logger.info(f"Using constant learning rate: {config.learning_rate}")
            
    def check_model_checkpoint_files(self, ckpt_name, ckpt_dir):
        if self.world_mesh is not None:
            return model_distributed_checkpoint_files_exist(ckpt_name, ckpt_dir)
        else:
            return model_checkpoint_files_exist(ckpt_name, ckpt_dir)
    
    def load_config_checkpoint(self, ckpt_path, config, model_config_fields):
        with open(ckpt_path, "r", encoding="utf-8") as f:
            config_checkpoint = json.load(f)
        if 'training_uuid' in config_checkpoint:
            self.training_uuid = config_checkpoint['training_uuid']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in model_config_fields:
            if hasattr(config, k) and k in config_checkpoint['model_args']:
                setattr(config, k, config_checkpoint['model_args'][k])
        if 'iter_num' in config_checkpoint:
            self.iter_num = config_checkpoint['iter_num']
        if 'best_train_loss' in config_checkpoint:
            self.best_train_loss = config_checkpoint['best_train_loss']
        if 'best_val_loss' in config_checkpoint:
            self.best_val_loss = config_checkpoint['best_val_loss']
        if 'processed_tokens' in config_checkpoint:
            self.processed_tokens = config_checkpoint['processed_tokens']
        
        if 'allamo_dataloader' in config_checkpoint:
            if  'train_processed_files' in config_checkpoint['allamo_dataloader']:
                self.data_loader.train_dataset.processed_files = config_checkpoint['allamo_dataloader']['train_processed_files']
                if len(self.data_loader.train_dataset.processed_files) > 0:
                    # Removing the last element from the list because it represents the file where processing was interrupted.
                    # We will load this file and resume processing from there, indicated by the dataset_offset.
                    self.data_loader.train_dataset.processed_files.pop()
                    self.data_loader.train_dataset.load_next_dataset()
            if 'dataset_offset' in config_checkpoint['allamo_dataloader']:
                self.data_loader.dataset_offset = config_checkpoint['allamo_dataloader']['dataset_offset'] // self.dp_size
            if 'epoch' in config_checkpoint['allamo_dataloader']:
                self.data_loader.epoch = config_checkpoint['allamo_dataloader']['epoch']
    
    def load_model_checkpoint(self, model, ckpt_path, config):
        state_dict = torch.load(ckpt_path, map_location='cpu')
        remove_unwanted_prefix_from_model_state_dict(state_dict)
        model.load_state_dict(state_dict)
        if config.log_checkpoint_md5_on_load and self.master_process:
            md5sum = calculate_md5(ckpt_path)
            self.logger.info(f"Loaded model from checkpoint {ckpt_path} - MD5: {md5sum}")
        else:
            self.logger.info(f"Loaded model from checkpoint {ckpt_path}")
        
    def load_optimizer_checkpoint(self, model, optimizer, ckpt_path, config):
        if os.path.exists(ckpt_path):
            # requires each rank to have the full dict in CPU memory to reduce communication
            full_osd = torch.load(ckpt_path, map_location='cpu')
            remove_unwanted_prefix_from_model_state_dict(full_osd)
            sharded_osd = FSDP.shard_full_optim_state_dict(full_osd, model)
            optimizer.load_state_dict(sharded_osd)
            self.logger.info("Sharded optimizer state loaded.")
        else:
            if self.master_process:
                self.logger.warning("Optimizer checkpoint file not found. Initializing optimizer from scratch")
        
    def save_config_checkpoint(self, ckpt_file_name, ckpt_md5):
        if self.master_process:
            checkpoint = {
                'model_args': dataclasses.asdict(self.model.config),
                'run_uuid': self.run_uuid,
                'training_uuid': self.training_uuid,
                'iter_num': self.iter_num,
                'best_train_loss': self.best_train_loss,
                'best_val_loss': self.best_val_loss,
                'processed_tokens': self.processed_tokens,
                'config': dataclasses.asdict(self.config),
                'allamo_dataloader': {
                    'train_processed_files': self.data_loader.train_dataset.processed_files,
                    'dataset_offset': self.data_loader.dataset_offset * self.dp_size,
                    'epoch': self.data_loader.epoch
                }
            }
            if ckpt_md5 is not None:
                checkpoint['checkpoint_md5sum'] = ckpt_md5
                self.logger.info(f"model checkpoint saved - MD5: {ckpt_md5}")
                
            ckpt_file_path = get_config_checkpoint_path(ckpt_file_name, self.config.out_dir)
            self.logger.info(f"saving config checkpoint to {ckpt_file_path}")
            if not self.config.ignore_last_checkpoint_backup:
                rename_file_to_prev_version(ckpt_file_path)
            with open(ckpt_file_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, indent=4, ensure_ascii=False)
            
    def save_checkpoint(self, ckpt_file_name, model_only=False, epoch_ckpt=False):
        if self.world_mesh is not None and DISTRIBUTED_CHECKPOINT_AVAILABLE:
            self.save_config_checkpoint(ckpt_file_name, None)
                
            if self.master_process and not self.config.ignore_last_checkpoint_backup:
                self.logger.warning("Backing up a previous checkpoint is not supported in distributed checkpoints")
            save_distributed_checkpoint(self.model, self.optimizer, self.config.out_dir, ckpt_file_name, config, model_only)
        else:
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, self.fullstate_save_policy):
                full_msd = self.model.state_dict()
            if self.master_process:
                ckpt_file_path = get_model_checkpoint_path(ckpt_file_name, self.config.out_dir)
                self.logger.info(f"saving model checkpoint to {ckpt_file_path}")
                if not self.config.ignore_last_checkpoint_backup:
                    rename_file_to_prev_version(ckpt_file_path)
                torch.save(full_msd, ckpt_file_path)
                del full_msd
                
                ckpt_md5 = calculate_md5(ckpt_file_path) if epoch_ckpt and self.config.log_checkpoint_md5_on_epoch else None
                
                self.save_config_checkpoint(ckpt_file_name, ckpt_md5)
            
            if self.config.save_optimizer_checkpoint and model_only == False:
                # pull all sharded optimizer states to rank0 cpu
                full_osd = FSDP.full_optim_state_dict(self.model, self.optimizer)
                if self.master_process:
                    ckpt_file_path = get_optimizer_checkpoint_path(ckpt_file_name, self.config.out_dir)
                    self.logger.info(f"saving optimizer checkpoint to {ckpt_file_path}")
                    if not self.config.ignore_last_checkpoint_backup:
                        rename_file_to_prev_version(ckpt_file_path)
                    torch.save(full_osd, ckpt_file_path)
                    self.logger.info(f"checkpoint files saved in {config.out_dir}")
                    del full_osd
            
    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss(self):
        losses_out = {}
        accuraces = {}
        self.model.eval()
        for split in self.data_loader.splits:
            fsdp_loss_preds = torch.zeros(3).to(self.config.device)
            for k in range(self.config.eval_iters):
                X, Y, W = self.data_loader.get_batch(split, True)
                logits, loss, _ = self.model(X, Y)
                unmasked_labels = torch.sum(Y.view(-1) != self.config.ignore_index).item()
                fsdp_loss_preds[0] += loss.item()
                fsdp_loss_preds[1] += (logits.max(2).indices == Y).sum().item() / torch.sum(Y.view(-1) != self.config.ignore_index).item()
                fsdp_loss_preds[2] += 1
            fsdp_loss_preds = dist_all_reduce(fsdp_loss_preds, dist.ReduceOp.SUM, self.device_mesh)
            steps = self.config.eval_iters * self.dp_size
            losses_out[split] = fsdp_loss_preds[0] / (self.config.eval_iters * self.dp_size)
            accuraces[split] = fsdp_loss_preds[1] / fsdp_loss_preds[2]
        self.model.train()
        if 'val' not in losses_out:
            losses_out['val'] = losses_out['train']
            accuraces['val'] = accuraces['train']
        return losses_out, accuraces
        
    def train(self):
        self.logger.info(f"Starting FSDP training (run id: {self.run_uuid}, world size: {self.world_size}, DP size: {self.dp_size}) with configuration: {self.config}")
        X, Y, W = self.data_loader.get_batch('train') # fetch the very first batch
        self.start_iter = self.iter_num
        self.start_timestamp = datetime.datetime.now()
        current_epoch = self.data_loader.epoch
        fsdp_loss_acc = torch.zeros(4).to(self.config.device)
        while has_next_iter_to_perform(self.iter_num, self.config, self.data_loader):
            if current_epoch < self.data_loader.epoch:
                ckpt_file_name = f'epoch_{current_epoch}'
                self.save_checkpoint(ckpt_file_name, model_only=True, epoch_ckpt=True)
                if self.config.epoch_completion_hook_program and self.master_process:
                    pid = run_checkpoint_hook_program(self.config.epoch_completion_hook_program, self.run_uuid, self.training_uuid, current_epoch, self.iter_num, ckpt_file_name, self.config)
                    self.logger.info(f"Epoch completion hook program started with pid {pid}")
                current_epoch = self.data_loader.epoch
            
            timer = time.time()
            lr = get_lr(self.iter_num, self.config)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
                
            # determine and set batch_size and gradient_accumulation_steps for this iteration 
            micro_batch_size = self.data_loader.update_batch_size(self.iter_num)
            total_batch_size = self.config.block_size * micro_batch_size * self.gradient_accumulation_steps * self.dp_size
            self.gradient_accumulation_steps = get_grad_accum(self.gradient_accumulation_steps, self.iter_num, self.config)

            # evaluate the loss on train/val sets and write best checkpoint
            if self.config.eval_interval > 0 and self.iter_num % self.config.eval_interval == 0:
                eval_time = time.time()
                losses, accuraces = self.estimate_loss()
                eval_time = time.time() - eval_time
                train_loss = losses['train'].item()
                val_loss = losses['val'].item()
                if self.iter_num > self.start_iter:
                    if train_loss < self.best_train_loss:
                        self.best_train_loss = train_loss
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        if self.config.save_best_checkpoint:
                            self.save_checkpoint('ckpt')
                        
                if self.master_process:                
                    train_ppl = torch.exp(losses['train']).item()
                    val_ppl = torch.exp(losses['val']).item()
                    self.logger.info(
                        f"iter {self.iter_num:,}: train loss={train_loss:.4f} ppl={train_ppl:.4f} "
                        f"acc={accuraces['train']:.4f} (best loss={self.best_train_loss:.4f}), "
                        f"val loss={val_loss:.4f} ppl={val_ppl:.4f} acc={accuraces['val']:.4f} "
                        f"(best loss={self.best_val_loss:.4f}), tokens {self.processed_tokens:,}"
                    )
                    if self.config.wandb_log:
                        wandb.log({
                            "iter": self.iter_num,
                            "eval/time": eval_time*1000,
                            "eval/samples_per_second": (self.config.eval_iters * len(self.data_loader.splits)) / eval_time,
                            "eval/train_loss": train_loss,
                            "eval/val_loss": val_loss,
                            "eval/train_ppl": train_ppl,
                            "eval/val_ppl": val_ppl,
                            "eval/train_acc": accuraces['train'].item(),
                            "eval/val_acc": accuraces['val'].item(),
                            "eval/diff_loss": (val_loss-train_loss),
                            "eval/diff_acc": (accuraces['train']-accuraces['val']).item(),
                            "eval/diff_ppl": (val_ppl-train_ppl),
                            "eval/best_train_loss": self.best_train_loss,
                            "eval/best_val_loss": self.best_val_loss
                        })
                gc.collect()
                torch.cuda.empty_cache()
                
            if self.config.checkpoint_interval > 0 and self.iter_num > self.start_iter and self.iter_num % self.config.checkpoint_interval == 0:
                ckpt_file_name = 'last_eval_ckpt'
                self.save_checkpoint(ckpt_file_name)
                if self.config.regular_checkpoint_hook_program and self.master_process:
                    pid = run_checkpoint_hook_program(self.config.regular_checkpoint_hook_program, self.run_uuid, self.training_uuid, current_epoch, self.iter_num, ckpt_file_name, self.config)
                    self.logger.info(f"Regular checkpoint hook program started with pid {pid}")
            
            accuracy = 0
            fsdp_loss_acc.zero_()
            batch_mfu_excluded_time = 0
            fwdbwd_time = time.time()
            # forward backward update, with optional gradient accumulation to simulate larger batch size
            for micro_step in range(self.gradient_accumulation_steps):
                logits, loss, _ = self.model(X, Y, W)
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps # scale the loss to account for micro steps
                    
                if W is not None:
                    fsdp_loss_weight_acc = W.view(-1).sum()
                    # sum loss weights over all processes
                    fsdp_loss_weight_acc = dist_all_reduce(fsdp_loss_weight_acc, dist.ReduceOp.SUM, self.device_mesh)
                    loss = (self.dp_size / fsdp_loss_weight_acc) * loss
                
                mfu_excluded_time = time.time()
                unmasked_labels = torch.sum(Y.view(-1) != self.config.ignore_index).item()
                fsdp_loss_acc[0] += loss.item()
                fsdp_loss_acc[1] += unmasked_labels
                fsdp_loss_acc[2] += (logits.max(2).indices == Y).sum().item() / unmasked_labels
                fsdp_loss_acc[3] += 1
                
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                X, Y, W = self.data_loader.get_batch('train')
                batch_mfu_excluded_time += time.time() - mfu_excluded_time
                
                # backward pass
                loss.backward()
                
            # clip the gradient
            if self.config.grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip, foreach=True)
            
            mfu_excluded_time = time.time()
            # sync loss and acc over all processes
            fsdp_loss_acc = dist_all_reduce(fsdp_loss_acc, dist.ReduceOp.SUM, self.device_mesh)
            
            # adjust learning rate
            if self.config.adaptive_learning_rate:
                lr = lr * math.sqrt(fsdp_loss_acc[1].item() / total_batch_size)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            
            if self.master_process:
                self.processed_tokens += int(fsdp_loss_acc[1])
            batch_mfu_excluded_time += time.time() - mfu_excluded_time
            
            # weight update
            self.optimizer.step()
            
            # flush the gradients as soon as we can, no need for this memory anymore
            self.optimizer.zero_grad(set_to_none=True)
            fwdbwd_time = time.time() - fwdbwd_time - batch_mfu_excluded_time

            # timing and logging
            if self.config.log_interval > 0 and self.iter_num % self.config.log_interval == 0 and self.master_process:
                iter_time = time.time() - timer
                # get loss as float. note: this is a CPU-GPU sync point
                lossf = fsdp_loss_acc[0].item() / self.dp_size
                ppl = torch.exp(torch.tensor(lossf)).item()
                accuracy = fsdp_loss_acc[2].item() / fsdp_loss_acc[3].item()
                if self.config.mfu_flops_peak > 0 and self.iter_num > self.start_iter:
                    mfu = estimate_mfu(self.model_num_params, self.config, micro_batch_size * self.gradient_accumulation_steps, fwdbwd_time)
                    mfu_str = f'{mfu*100:.2f}%'
                else:
                    mfu = -1.0
                    mfu_str = 'n/a'
                mtu = fwdbwd_time/iter_time # model time utilization
                iter_time_ms = iter_time * 1000
                self.logger.info(
                    f"iter {self.iter_num:,}: loss {lossf:.4f}, ppl {ppl:.4f}, acc {accuracy:.4f}, "
                    f"iter time {iter_time_ms:.2f}ms, tokens {self.processed_tokens:,}, lr {lr:.6f}, "
                    f"mfu {mfu_str}, mtu {mtu*100:.2f}%, epoch {self.data_loader.epoch}, "
                    f"ETA: {calculate_eta(self.iter_num, self.start_iter, self.start_timestamp, self.config)}"
                )
                if self.config.wandb_log:
                    metrics = {
                        "iter": self.iter_num,
                        "train/iter_time": iter_time_ms,
                        "train/loss": lossf,
                        "train/ppl": ppl,
                        "train/acc": accuracy,
                        "train/lr": lr,
                        "train/tokens": self.processed_tokens,
                        "train/tokens_per_sec": (total_batch_size/iter_time),
                        "train/tokens_per_gpu_per_sec": (total_batch_size/self.world_size/iter_time),
                        "train/total_batch_size": total_batch_size,
                        "train/mtu": mtu,
                        "train/epoch": self.data_loader.epoch
                    }
                    if mfu > 0:
                        metrics['train/mfu'] = mfu
                    if self.config.dataset_seq_train:
                        metrics['train/ds_offset'] = self.data_loader.dataset_offset
                    if self.dp_size < self.world_size:
                        metrics['train/tokens_per_group_per_sec'] = total_batch_size/self.dp_size/iter_time
                    wandb.log(metrics)
            self.iter_num += 1
            
        training_time = format_seconds_as_time((datetime.datetime.now() - self.start_timestamp).total_seconds())
        self.logger.info(f"Training finished in {training_time}")
        
        ckpt_file_name = 'final_ckpt'
        self.save_checkpoint(ckpt_file_name, model_only=True, epoch_ckpt=True)
        if self.config.epoch_completion_hook_program and self.master_process:
            pid = run_checkpoint_hook_program(self.config.epoch_completion_hook_program, self.run_uuid, self.training_uuid, current_epoch, self.iter_num, ckpt_file_name, self.config)
            self.logger.info(f"Epoch completion hook program started with pid {pid}")

if __name__ == '__main__':
    config = AllamoConfiguration()
    trainer = AllamoFSDPTrainer(config)
    
    # logging
    if config.wandb_log and trainer.master_process:
        import wandb
        wandb_run_name = config.wandb_run_name + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        wandb.init(project=config.wandb_project, name=wandb_run_name, config=config)
    
    # clean up after initialization
    gc.collect()
    torch.cuda.empty_cache()
    
    trainer.train()  
    
    dist.barrier()
    dist.destroy_process_group()

"""
This single file is intended to perform some magic for training/finetuning.
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
from torch.nn.parallel import DistributedDataParallel as DDP

from model import AllamoTransformerConfig, AllamoTransformer
from configuration import AllamoConfiguration
from data_loader import AllamoDataLoader
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

class AllamoTrainer:

    def __init__(self, config: AllamoConfiguration, ddp=False):
        self.run_uuid = str(uuid.uuid4())
        self.config = config
        self.ddp = ddp
        self.__init_torch(config)
        self.__init_logger(config)
        self.logger.info(f"Torch initialized for run {self.run_uuid}")
        
        self.iter_num = 0
        self.best_train_loss = 1e2
        self.best_val_loss = 1e2
        self.processed_tokens = 0
        self.data_loader = AllamoDataLoader(config, self.rank, self.world_size)
        self.__init_training(config)
        
    def __init_logger(self, config: AllamoConfiguration):
        run_timestamp_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        if self.ddp:
            log_file_name_base = f'train-{run_timestamp_str}-rank_{self.rank}'
        else:
            log_file_name_base = f'train-{run_timestamp_str}'
        log_file_path = os.path.join(config.out_dir, f'{log_file_name_base}.log')
        logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(), logging.FileHandler(log_file_path)])
        self.logger = logging.getLogger('AllamoTrainer')
            
    def __init_torch(self, config: AllamoConfiguration):
        if self.ddp:
            dist.init_process_group(backend=config.backend)
            self.rank = int(os.environ['RANK'])
            self.local_rank = int(os.environ['LOCAL_RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])
            print(
                f"RANK: {self.rank}, LOCAL_RANK: {self.local_rank}, "
                f"WORLD_SIZE: {self.world_size}, LOCAL_WORLD_SIZE: {os.environ['LOCAL_WORLD_SIZE']}"
            )
            config.device = f'cuda:{self.local_rank}'
            torch.cuda.set_device(config.device)
            self.master_process = self.rank == 0 # this process will do logging, checkpointing etc.
            self.seed_offset = self.rank # each process gets a different seed
        else:
            # if not ddp, we are running on a single gpu, and one process
            self.rank = 0
            self.local_rank = None
            self.world_size = 1
            self.master_process = True
            self.seed_offset = 0
    
        if self.master_process:
            os.makedirs(config.out_dir, exist_ok=True)
        torch.manual_seed(config.seed + self.seed_offset)
        torch.cuda.manual_seed(config.seed + self.seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        torch.set_float32_matmul_precision("highest") # set to "high" for faster matrix multiplications with bfloat16
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'bfloat16-true': torch.bfloat16, 'float16': torch.float16}[config.dtype]
        self.device_type = 'cuda' if 'cuda' in config.device else 'cpu' # for later use in torch.autocast
        self.ctx = nullcontext() if self.device_type == 'cpu' else torch.amp.autocast(device_type=self.device_type, dtype=ptdtype)
        if config.dtype == 'bfloat16-true':
            # torch.set_float32_matmul_precision("high")
            torch.set_default_dtype(torch.bfloat16)
        
    def __init_training(self, config: AllamoConfiguration):
        model_config_fields = [f.name for f in dataclasses.fields(AllamoTransformerConfig)]
        ckpt_dir = config.checkpoint_path if config.checkpoint_path else config.out_dir
        checkpoint_name = None
        if config.init_from == 'resume':
            checkpoint_name = 'ckpt'
        elif config.init_from == 'resume_last':
            checkpoint_name = 'last_eval_ckpt'
        else:
            if model_checkpoint_files_exist('ckpt', ckpt_dir):
                self.logger.info("Delete existing checkpoint files to start from scratch or use --init_from=resume to resume training")
                exit()
        
        if checkpoint_name is not None:
            if model_checkpoint_files_exist(checkpoint_name, ckpt_dir):
                self.logger.info(f"Resuming training from {ckpt_dir} and start loading '{checkpoint_name}' checkpoint files")
                self.load_config_checkpoint(os.path.join(ckpt_dir, f'config_{checkpoint_name}.json'), config, model_config_fields)
            elif config.init_from == 'resume_last':
                checkpoint_name = None
                self.logger.warning(f"'{checkpoint_name}' checkpoint files not found but allowing to start from scratch")
            else:
                raise Exception(f"'{checkpoint_name}' checkpoint files not found!")
    
        model_args = {k: getattr(config, k) for k in model_config_fields if hasattr(config, k)}
        modelConf = AllamoTransformerConfig(**model_args)
        model = AllamoTransformer(modelConf)
        self.model_num_params = model.model_num_params
        if checkpoint_name is None:
            self.logger.info("Initialized a new model from scratch")
        else:
            self.load_model_checkpoint(model, os.path.join(ckpt_dir, f'model_{checkpoint_name}.pt'), config)
        model.to(config.device)

        # compile the model - requires PyTorch 2.0
        if config.compile:
            self.logger.info("compiling the model... (takes a ~minute)")
            try:
                model = torch.compile(model, mode=config.compile_mode)
                self.logger.info("Model compiled and ready to use")
            except Exception as err:
                self.logger.warn(f"Model compile not supported: {err}")

        self.raw_model = model # neeeded in DDP training
        self.model = model
        # wrap model into DDP container
        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.local_rank])
            
        # initialize a GradScaler. If enabled=False scaler is a no-op
        self.scaler = torch.cuda.amp.GradScaler(enabled=(config.dtype == 'float16' or config.dtype == 'bfloat16'))
        
        # optimizer
        self.optimizer = self.model.configure_optimizers(config, self.device_type)
        if checkpoint_name is not None:
            self.load_optimizer_checkpoint(self.optimizer, os.path.join(ckpt_dir, f'optimizer_{checkpoint_name}.pt'))
                
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

    def load_config_checkpoint(self, ckpt_path, config, model_config_fields):
        with open(ckpt_path, "r", encoding="utf-8") as f:
            config_checkpoint = json.load(f)
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
                self.data_loader.dataset_offset = config_checkpoint['allamo_dataloader']['dataset_offset'] // self.world_size
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
        
        
    def load_optimizer_checkpoint(self, optimizer, ckpt_path):
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location=config.device)
            optimizer.load_state_dict(state_dict)
            self.logger.info("Optimizer state loaded.")
        else:
            self.logger.warning("Optimizer checkpoint file not found. Initializing optimizer from scratch")

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss(self):
        losses_out = {}
        accuraces = {}
        self.model.eval()
        for split in self.data_loader.splits:
            losses = torch.zeros(self.config.eval_iters)
            correct_preds = 0
            total_preds = 0
            for k in range(self.config.eval_iters):
                X, Y, W = self.data_loader.get_batch(split, True)
                with self.ctx:
                    logits, loss, _ = self.model(X, Y)
                losses[k] = loss.item()
                total_preds += torch.sum(Y.view(-1) != self.config.ignore_index).item()
                correct_preds += (logits.max(2).indices == Y).sum().item()
            losses_out[split] = losses.mean()
            accuraces[split] = correct_preds / total_preds
        self.model.train()
        if 'val' not in losses_out:
            losses_out['val'] = losses_out['train']
            accuraces['val'] = accuraces['train']
        return losses_out, accuraces

    # helps saving checkpoint to a file
    def save_checkpoint(self, ckpt_file_name, model_only=False, epoch_ckpt=False):
        ckpt_file_path = get_model_checkpoint_path(ckpt_file_name, self.config.out_dir)
        self.logger.info(f"saving model checkpoint to {ckpt_file_path}")
        if not self.config.ignore_last_checkpoint_backup:
            rename_file_to_prev_version(ckpt_file_path)
        torch.save(self.raw_model.state_dict(), ckpt_file_path)
        
        md5sum = calculate_md5(ckpt_file_path) if epoch_ckpt and config.log_checkpoint_md5_on_epoch else None
        
        checkpoint = {
            'model_args': dataclasses.asdict(self.raw_model.config),
            'run_uuid': self.run_uuid,
            'iter_num': self.iter_num,
            'best_train_loss': self.best_train_loss,
            'best_val_loss': self.best_val_loss,
            'processed_tokens': self.processed_tokens,
            'config': dataclasses.asdict(self.config),
            'allamo_dataloader': {
                'train_processed_files': self.data_loader.train_dataset.processed_files,
                'dataset_offset': self.data_loader.dataset_offset * self.world_size,
                'epoch': self.data_loader.epoch
            }
        }
        if md5sum is not None:
            checkpoint['checkpoint_md5sum'] = md5sum
            self.logger.info(f"model checkpoint saved - MD5: {md5sum}")
        
        ckpt_file_path = get_config_checkpoint_path(ckpt_file_name, self.config.out_dir)
        self.logger.info(f"saving config checkpoint to {ckpt_file_path}")
        if not self.config.ignore_last_checkpoint_backup:
            rename_file_to_prev_version(ckpt_file_path)
        with open(ckpt_file_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=4, ensure_ascii=False)
        
        if self.config.save_optimizer_checkpoint and model_only == False:
            ckpt_file_path = get_optimizer_checkpoint_path(ckpt_file_name, self.config.out_dir)
            self.logger.info(f"saving optimizer checkpoint to {ckpt_file_path}")
            if not self.config.ignore_last_checkpoint_backup:
                rename_file_to_prev_version(ckpt_file_path)
            torch.save(self.optimizer.state_dict(), ckpt_file_path)
            
        self.logger.info(f"checkpoint files saved in {config.out_dir}")
        
    def train(self):
        self.logger.info(f"Starting training (run id: {self.run_uuid}) with configuration: {self.config}")
        X, Y, W = self.data_loader.get_batch('train') # fetch the very first batch
        self.start_iter = self.iter_num
        self.start_timestamp = datetime.datetime.now()
        current_epoch = self.data_loader.epoch
        while has_next_iter_to_perform(self.iter_num, self.config, self.data_loader):
            if current_epoch < self.data_loader.epoch:
                ckpt_file_name = f'epoch_{current_epoch}'
                self.save_checkpoint(ckpt_file_name, model_only=True, epoch_ckpt=True)
                if self.config.epoch_completion_hook_program:
                    pid = run_checkpoint_hook_program(self.config.epoch_completion_hook_program, self.run_uuid, current_epoch, self.iter_num, ckpt_file_name, self.config)
                    self.logger.info(f"Epoch completion hook program started with pid {pid}")
                current_epoch = self.data_loader.epoch
            
            timer = time.time()
            log_iter = (self.config.log_interval > 0 and self.iter_num % self.config.log_interval == 0 and self.master_process)
            eval_iter = (self.config.eval_interval > 0 and self.iter_num % self.config.eval_interval == 0 and self.master_process)
            lr = get_lr(self.iter_num, self.config)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
                
            # determine and set batch_size and gradient_accumulation_steps for this iteration 
            micro_batch_size = self.data_loader.update_batch_size(self.iter_num)
            total_batch_size = self.config.block_size * micro_batch_size * self.gradient_accumulation_steps * self.world_size
            self.gradient_accumulation_steps = get_grad_accum(self.gradient_accumulation_steps, self.iter_num, self.config)

            # evaluate the loss on train/val sets and write best checkpoint
            if eval_iter:
                eval_time = time.time()
                losses, accuraces = self.estimate_loss()
                eval_time = time.time() - eval_time
                train_loss = losses['train'].item()
                val_loss = losses['val'].item()
                train_ppl = torch.exp(losses['train']).item()
                val_ppl = torch.exp(losses['val']).item()
                self.logger.info(
                    f"iter {self.iter_num:,}: train loss={train_loss:.4f} ppl={train_ppl:.4f} "
                    f"acc={accuraces['train']:.4f} (best loss={self.best_train_loss:.4f}), "
                    f"val loss={val_loss:.4f} ppl={val_ppl:.4f} acc={accuraces['val']:.4f} "
                    f"(best loss={self.best_val_loss:.4f}), tokens {self.processed_tokens:,}"
                )
                if self.iter_num > self.start_iter:
                    if train_loss < self.best_train_loss:
                        self.best_train_loss = train_loss
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        if self.config.save_best_checkpoint:
                            self.save_checkpoint('ckpt')
                if self.config.wandb_log:
                    wandb.log({
                        "iter": self.iter_num,
                        "eval/time": eval_time*1000,
                        "eval/samples_per_second": (self.config.eval_iters * len(self.data_loader.splits)) / eval_time,
                        "eval/train_loss": train_loss,
                        "eval/val_loss": val_loss,
                        "eval/train_ppl": train_ppl,
                        "eval/val_ppl": val_ppl,
                        "eval/train_acc": accuraces['train'],
                        "eval/val_acc": accuraces['val'],
                        "eval/diff_loss": (val_loss-train_loss),
                        "eval/diff_acc": (accuraces['train']-accuraces['val']),
                        "eval/diff_ppl": (val_ppl-train_ppl),
                        "eval/best_train_loss": self.best_train_loss,
                        "eval/best_val_loss": self.best_val_loss
                    })
                gc.collect()
                torch.cuda.empty_cache()
            
            if self.config.eval_only:
                break
                
            if self.config.checkpoint_interval > 0 and self.iter_num > self.start_iter and self.iter_num % self.config.checkpoint_interval == 0:
                ckpt_file_name = 'last_eval_ckpt'
                self.save_checkpoint(ckpt_file_name)
                if self.config.regular_checkpoint_hook_program:
                    pid = run_checkpoint_hook_program(self.config.regular_checkpoint_hook_program, self.run_uuid, current_epoch, self.iter_num, ckpt_file_name, self.config)
                    self.logger.info(f"Regular checkpoint hook program started with pid {pid}")
            
            accuracy = 0
            unmasked_labels = 0
            batch_mfu_excluded_time = 0
            fwdbwd_time = time.time()
            # forward backward update, with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16
            micro_steps = self.gradient_accumulation_steps
            for micro_step in range(self.gradient_accumulation_steps):
                if self.ddp:
                    # in DDP training we only need to sync gradients at the last micro step.
                    # the official way to do this is with model.no_sync() context manager, but
                    # I really dislike that this bloats the code and forces us to repeat code
                    # looking at the source of that context manager, it just toggles this variable
                    self.model.require_backward_grad_sync = (micro_step == self.gradient_accumulation_steps - 1)
                with self.ctx:
                    logits, loss, _ = self.model(X, Y, W)
                    if micro_steps > 1:
                        loss = loss / micro_steps # scale the loss to account for micro steps
                    if W is not None:
                        loss = loss / W.view(-1).sum()
                
                mfu_excluded_time = time.time()
                unmasked_labels += torch.sum(Y.view(-1) != self.config.ignore_index).item()
                
                if log_iter and (micro_step == self.gradient_accumulation_steps - 1):
                    # calculate accuracy. note: this is a CPU-GPU sync point!
                    accuracy = (logits.max(2).indices == Y).sum().item() / unmasked_labels
                    
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                X, Y, W = self.data_loader.get_batch('train')
                batch_mfu_excluded_time += time.time() - mfu_excluded_time
                
                # backward pass, with gradient scaling if training in fp16
                self.scaler.scale(loss).backward()
                
            # clip the gradient
            if self.config.grad_clip != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            
            mfu_excluded_time = time.time()
            # we can't count it precisely in DDP, so let's estimate
            unmasked_labels *= self.world_size
            # adjust learning rate
            if self.config.adaptive_learning_rate:
                lr = lr * math.sqrt(unmasked_labels / total_batch_size)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                    
            # count processed tokens
            self.processed_tokens += unmasked_labels
            batch_mfu_excluded_time += time.time() - mfu_excluded_time
            
            # step the optimizer and scaler if training in fp16
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            self.optimizer.zero_grad(set_to_none=True)
            
            # timing and logging
            if log_iter:
                fwdbwd_time = time.time() - fwdbwd_time - batch_mfu_excluded_time
                iter_time = time.time() - timer
                # get loss as float. note: this is a CPU-GPU sync point
                # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                lossf = loss.item() * micro_steps
                ppl = torch.exp(torch.tensor(lossf))
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
                    wandb.log(metrics)
            self.iter_num += 1
            
        training_time = format_seconds_as_time((datetime.datetime.now() - self.start_timestamp).total_seconds())
        self.logger.info(f"Training finished in {training_time}")
        
        if self.master_process and not self.config.eval_only:
            ckpt_file_name = 'final_ckpt'
            self.save_checkpoint(ckpt_file_name, model_only=True, epoch_ckpt=True)
            if self.config.epoch_completion_hook_program:
                pid = run_checkpoint_hook_program(self.config.epoch_completion_hook_program, self.run_uuid, current_epoch, self.iter_num, ckpt_file_name, self.config)
                self.logger.info(f"Epoch completion hook program started with pid {pid}")

if __name__ == '__main__':
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    config = AllamoConfiguration()
    trainer = AllamoTrainer(config, ddp)
    
    # logging
    if config.wandb_log and trainer.master_process:
        import wandb
        wandb_run_name = config.wandb_run_name + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        wandb.init(project=config.wandb_project, name=wandb_run_name, config=config)
    
    # clean up after initialization
    gc.collect()
    torch.cuda.empty_cache()
    
    trainer.train()  
      
    if ddp:
        dist.destroy_process_group()

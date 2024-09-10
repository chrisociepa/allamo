import gc
import json
import os
import time
import math
import datetime
import dataclasses
import wandb

import torch
import torch.distributed as dist

from allamo.logging import configure_logger, logger
from allamo.training_context import TrainingContext
from allamo.model import AllamoTransformerConfig
from allamo.configuration import AllamoConfiguration
from allamo.data_loader import AllamoDataLoader
from allamo.torch_utils import init_torch
from allamo.train_utils import (
    get_lr,
    get_grad_accum,
    format_seconds_as_time,
    calculate_eta,
    has_next_iter_to_perform,
    estimate_mfu,
    model_checkpoint_files_exist,
    run_checkpoint_hook_program,
)

class BaseTrainer:
    
    def __init__(self, config: AllamoConfiguration):
        self.train_ctx = TrainingContext()
        if self.train_ctx.master_process:
            configure_logger(config, True)
        
        self.config = config
        self.init_torch(config)
        logger.info(f"Torch initialized for run {self.train_ctx.run_uuid}")
        
        self.data_loader = AllamoDataLoader(config, self.train_ctx.rank, self.train_ctx.world_size)

        self.iter_num = 0
        self.best_train_loss = 1e2
        self.best_val_loss = 1e2
        self.processed_tokens = 0
        self.init_training()
        
    def distributed(self):
        raise NotImplementedError("Not implemented")

    def init_training(self):
        raise NotImplementedError("Not implemented")
    
    @torch.no_grad()
    def estimate_loss(self):
        raise NotImplementedError("Not implemented")
    
    def init_torch(self, config: AllamoConfiguration):
        self.device_type = 'cuda' if 'cuda' in config.device else 'cpu'
        init_torch(self.train_ctx, config, distributed=self.distributed())
        
        
    def init_checkpoint(self):
        self.checkpoint_dir = self.config.checkpoint_path if self.config.checkpoint_path else self.config.out_dir
        self.checkpoint_name = None
        if self.config.init_from == 'resume':
            self.checkpoint_name = 'ckpt'
        elif self.config.init_from == 'resume_last':
            self.checkpoint_name = 'last_eval_ckpt'
        else:
            if model_checkpoint_files_exist('ckpt', self.checkpoint_dir):
                logger.info("Delete existing checkpoint files to start from scratch or use --init_from=resume to resume training")
                exit()
        
        if self.checkpoint_name is not None:
            if model_checkpoint_files_exist(self.checkpoint_name, self.checkpoint_dir):
                logger.info(f"Resuming training from {self.checkpoint_dir} and start loading '{self.checkpoint_name}' checkpoint files")
                self.load_config_checkpoint(os.path.join(self.checkpoint_dir, f'config_{self.checkpoint_name}.json'))
            elif self.config.init_from == 'resume_last':
                self.checkpoint_name = None
                if self.train_ctx.master_process:
                    logger.warning(f"'{self.checkpoint_name}' checkpoint files not found but allowing to start from scratch")
            else:
                raise Exception(f"'{self.checkpoint_name}' checkpoint files not found!")
        
    def load_config_checkpoint(self, ckpt_path):
        with open(ckpt_path, "r", encoding="utf-8") as f:
            config_checkpoint = json.load(f)
        if 'training_uuid' in config_checkpoint:
            self.train_ctx.training_uuid = config_checkpoint['training_uuid']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in self.get_model_config_field_names():
            if hasattr(self.config, k) and k in config_checkpoint['model_args']:
                setattr(self.config, k, config_checkpoint['model_args'][k])
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
                self.data_loader.dataset_offset = config_checkpoint['allamo_dataloader']['dataset_offset'] // self.train_ctx.world_size
            if 'epoch' in config_checkpoint['allamo_dataloader']:
                self.data_loader.epoch = config_checkpoint['allamo_dataloader']['epoch']
    
    def create_model_config(self):
        model_args = {k: getattr(self.config, k) for k in self.get_model_config_field_names() if hasattr(self.config, k)}
        return AllamoTransformerConfig(**model_args)
    
    def init_gradient_accumulation_scheduler(self):
        if self.config.grad_accum_schedule: 
            self.config.grad_accum_max = self.config.gradient_accumulation_steps
            self.config.gradient_accumulation_steps = self.config.grad_accum_initial
            logger.info(
                f"Gradient accumulation scheduler enabled. "
                f"Current gradient accumulation steps: {self.config.gradient_accumulation_steps}"
            )
        self.gradient_accumulation_steps = self.config.gradient_accumulation_steps
        
    def log_init_learning_rate(self):
        if self.config.decay_lr:
            logger.info(f"Cosing decay learning rate enabled. Currect learning rate: {get_lr(self.iter_num, self.config)}")
        else:
            logger.info(f"Using constant learning rate: {self.config.learning_rate}")
    
    def init_wandb(self):
        if self.config.wandb_log and self.train_ctx.master_process:
            wandb_run_name = self.config.wandb_run_name + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            wandb.init(project=self.config.wandb_project, name=wandb_run_name, config=self.config)
            
    def get_model_config_field_names(self):
        return [f.name for f in dataclasses.fields(AllamoTransformerConfig)]
    
    def trigger_gc(self):
        gc.collect()
        torch.cuda.empty_cache()
    
    def should_evaluate(self):
        return self.config.eval_interval > 0 and self.iter_num % self.config.eval_interval == 0

    def should_save_last_checkpoint(self):
        return self.config.checkpoint_interval > 0 and self.iter_num > self.start_iter and self.iter_num % self.config.checkpoint_interval == 0
    
    def should_log_metrics(self):
        return self.config.log_interval > 0 and self.iter_num % self.config.log_interval == 0 and self.train_ctx.master_process
    
    def clip_grad_norm(self):
        return torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip).item()
    
    def evaluate(self):
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
                
        if self.train_ctx.master_process:                
            train_ppl = torch.exp(losses['train']).item()
            val_ppl = torch.exp(losses['val']).item()
            logger.info(
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
        self.trigger_gc()
    
    def train(self):
        logger.info(f"Starting training (run id: {self.train_ctx.run_uuid}, world size: {self.train_ctx.world_size}) with configuration:\n{self.config}")
        batch = self.data_loader.get_batch('train') # fetch the very first batch
        self.start_iter = self.iter_num
        self.start_timestamp = datetime.datetime.now()
        current_epoch = self.data_loader.epoch
        current_num_loaded_files = self.data_loader.get_num_loaded_files()
        iter_metrics = torch.zeros(5).to(self.config.device)
        self.trigger_gc()
        while has_next_iter_to_perform(self.iter_num, self.config, self.data_loader):
            if current_epoch < self.data_loader.epoch:
                ckpt_file_name = f'epoch_{current_epoch}'
                self.save_checkpoint(ckpt_file_name, model_only=True, epoch_ckpt=True)
                if self.config.epoch_completion_hook_program and self.train_ctx.master_process:
                    pid = run_checkpoint_hook_program(self.config.epoch_completion_hook_program, self.train_ctx.run_uuid, self.train_ctx.training_uuid, current_epoch, self.iter_num, ckpt_file_name, self.config)
                    logger.info(f"Epoch completion hook program started with pid {pid}")
                current_epoch = self.data_loader.epoch
                current_num_loaded_files = self.data_loader.get_num_loaded_files()
            elif self.config.save_checkpoint_on_dataset_reload and current_num_loaded_files != self.data_loader.get_num_loaded_files():
                ckpt_file_name = f'ds_reload_{current_epoch}-{current_num_loaded_files}'
                self.save_checkpoint(ckpt_file_name, model_only=True, epoch_ckpt=False)
                current_num_loaded_files = self.data_loader.get_num_loaded_files()
            elif self.config.should_override_config(self.iter_num):
                self.config.override_config_properties()
            
            timer = time.time()
            
            lr = get_lr(self.iter_num, self.config)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
                
            # determine and set batch_size and gradient_accumulation_steps for this iteration 
            micro_batch_size = self.data_loader.update_batch_size(self.iter_num)
            total_batch_size = self.config.block_size * micro_batch_size * self.gradient_accumulation_steps * self.train_ctx.world_size
            self.gradient_accumulation_steps = get_grad_accum(self.gradient_accumulation_steps, self.iter_num, self.config)

            # evaluate the loss on train/val sets and write best checkpoint
            if self.should_evaluate():
                self.evaluate()
                
            if self.should_save_last_checkpoint():
                ckpt_file_name = 'last_eval_ckpt'
                self.save_checkpoint(ckpt_file_name)
                if self.config.regular_checkpoint_hook_program and self.train_ctx.master_process:
                    pid = run_checkpoint_hook_program(self.config.regular_checkpoint_hook_program, self.train_ctx.run_uuid, self.train_ctx.training_uuid, current_epoch, self.iter_num, ckpt_file_name, self.config)
                    logger.info(f"Regular checkpoint hook program started with pid {pid}")
            
            accuracy = 0
            iter_metrics.zero_()
            batch_mfu_excluded_time = 0
            fwdbwd_time = time.time()
            # forward backward update, with optional gradient accumulation to simulate larger batch size
            for micro_step in range(self.gradient_accumulation_steps):
                loss, unmasked_labels, accuracy = self.forward(batch, (micro_step == self.gradient_accumulation_steps - 1))
                
                mfu_excluded_time = time.time()
                iter_metrics[0] += loss.item()
                iter_metrics[1] += unmasked_labels
                iter_metrics[2] += accuracy
                iter_metrics[3] += 1
                
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                batch = self.data_loader.get_batch('train')
                batch_mfu_excluded_time += time.time() - mfu_excluded_time
                
                # backward pass, with gradient scaling
                self.scaler.scale(loss).backward()
                
            # clip the gradient
            if self.config.grad_clip != 0.0:
                self.scaler.unscale_(self.optimizer)
                iter_metrics[4] += self.clip_grad_norm()
            
            mfu_excluded_time = time.time()
            if self.distributed():
                # sync loss and acc over all processes
                dist.all_reduce(iter_metrics, op=dist.ReduceOp.SUM)
            
            # adjust learning rate
            if self.config.adaptive_learning_rate:
                lr = lr * math.sqrt(iter_metrics[1].item() / total_batch_size)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            
            if self.train_ctx.master_process:
                self.processed_tokens += int(iter_metrics[1])
            batch_mfu_excluded_time += time.time() - mfu_excluded_time
            
            # step the optimizer and scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            self.optimizer.zero_grad(set_to_none=True)
            fwdbwd_time = time.time() - fwdbwd_time - batch_mfu_excluded_time

            if self.should_log_metrics():
                iter_time = time.time() - timer
                # get loss as float. note: this is a CPU-GPU sync point
                lossf = iter_metrics[0].item() / self.train_ctx.world_size
                ppl = torch.exp(torch.tensor(lossf)).item()
                accuracy = iter_metrics[2].item() / iter_metrics[3].item()
                grad_norm = iter_metrics[4].item() / self.train_ctx.world_size
                if self.config.mfu_flops_peak > 0 and self.iter_num > self.start_iter:
                    mfu = estimate_mfu(self.model_num_params, self.config, micro_batch_size * self.gradient_accumulation_steps, fwdbwd_time)
                    mfu_str = f'{mfu*100:.2f}%'
                else:
                    mfu = -1.0
                    mfu_str = 'n/a'
                mtu = fwdbwd_time/iter_time # model time utilization
                iter_time_ms = iter_time * 1000
                logger.info(
                    f"iter {self.iter_num:,}: loss {lossf:.4f}, ppl {ppl:.4f}, acc {accuracy:.4f}, "
                    f"iter time {iter_time_ms:.2f}ms, tokens {self.processed_tokens:,}, lr {lr:.8f}, "
                    f"mfu {mfu_str}, mtu {mtu*100:.2f}%, epoch {self.data_loader.epoch}, "
                    f"ETA: {calculate_eta(self.iter_num, self.start_iter, self.start_timestamp, self.config)}"
                )
                if self.config.wandb_log:
                    metrics = {
                        "iter": self.iter_num,
                        "train/loss": lossf,
                        "train/acc": accuracy,
                        "train/ppl": ppl,
                        "train/grad_norm": grad_norm,
                        "train/lr": lr,
                        "train/mtu": mtu,
                        "train/tokens_per_sec": (total_batch_size/iter_time),
                        "train/tokens_per_gpu_per_sec": (total_batch_size/self.train_ctx.world_size/iter_time),
                        "train/tokens": self.processed_tokens,
                        "train/epoch": self.data_loader.epoch,
                        "train/total_batch_size": total_batch_size,
                        "train/iter_time": iter_time_ms,
                    }
                    if mfu > 0:
                        metrics['train/mfu'] = mfu
                    if self.config.dataset_seq_train:
                        metrics['train/ds_offset'] = self.data_loader.dataset_offset
                    wandb.log(metrics)
            self.iter_num += 1
            
        training_time = format_seconds_as_time((datetime.datetime.now() - self.start_timestamp).total_seconds())
        logger.info(f"Training finished in {training_time}")
        
        ckpt_file_name = 'final_ckpt'
        self.save_checkpoint(ckpt_file_name, model_only=True, epoch_ckpt=True)
        if self.config.epoch_completion_hook_program and self.train_ctx.master_process:
            pid = run_checkpoint_hook_program(self.config.epoch_completion_hook_program, self.train_ctx.run_uuid, self.train_ctx.training_uuid, current_epoch, self.iter_num, ckpt_file_name, self.config)
            logger.info(f"Epoch completion hook program started with pid {pid}")

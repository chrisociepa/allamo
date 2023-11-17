"""
This single file is intended to perform some magic for training/finetuning.
"""

import gc
import os
import time
import math
import pickle
import random
import logging
import datetime
import dataclasses
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import AllamoTransformerConfig, AllamoTransformer
from configuration import AllamoConfiguration
from simple_data_loader import SimpleDataLoader

class AllamoTrainer:

    def __init__(self, config: AllamoConfiguration, ddp=False):
        self.config = config
        self.ddp = ddp
        self.__init_torch(config)
        self.__init_logger(config)
        
        self.iter_num = 0
        self.best_train_loss = 1e9
        self.best_val_loss = 1e9
        self.processed_tokens = 0
        self.__init_training(config)
        
        self.simple_data_loader = SimpleDataLoader(config)
        
    def __init_logger(self, config: AllamoConfiguration):
        run_timestamp_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        if self.ddp:
            log_file_name_base = f'train-{run_timestamp_str}-r{self.ddp_rank}-lr{self.ddp_local_rank}'
        else:
            log_file_name_base = f'train-{run_timestamp_str}'
        log_file_path = os.path.join(config.out_dir, f'{log_file_name_base}.log')
        logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(), logging.FileHandler(log_file_path)])
        self.logger = logging.getLogger('AllamoTrainer')
            
    def __init_torch(self, config: AllamoConfiguration):
        if self.ddp:
            init_process_group(backend=config.backend)
            self.ddp_rank = int(os.environ['RANK'])
            self.ddp_local_rank = int(os.environ['LOCAL_RANK'])
            self.ddp_world_size = int(os.environ['WORLD_SIZE'])
            self.logger.info(
                f"RANK: {self.ddp_rank}, LOCAL_RANK: {self.ddp_local_rank}, "
                f"WORLD_SIZE: {self.ddp_world_size}, LOCAL_WORLD_SIZE: {os.environ['LOCAL_WORLD_SIZE']}"
            )
            config.device = f'cuda:{self.ddp_local_rank}'
            torch.cuda.set_device(config.device)
            self.master_process = self.ddp_rank == 0 # this process will do logging, checkpointing etc.
            self.seed_offset = self.ddp_rank # each process gets a different seed
        else:
            # if not ddp, we are running on a single gpu, and one process
            self.master_process = True
            self.seed_offset = 0
    
        if self.master_process:
            os.makedirs(config.out_dir, exist_ok=True)
        torch.manual_seed(config.seed + self.seed_offset)
        torch.cuda.manual_seed(config.seed + self.seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config.dtype]
        self.device_type = 'cuda' if 'cuda' in config.device else 'cpu' # for later use in torch.autocast
        self.ctx = nullcontext() if self.device_type == 'cpu' else torch.amp.autocast(device_type=self.device_type, dtype=ptdtype)
        
    def __init_training(self, config: AllamoConfiguration):
        if self.ddp:
            # world_size number of processes will be training simultaneously, so we can scale
            # down the desired gradient accumulation iterations per process proportionally
            assert config.gradient_accumulation_steps % self.ddp_world_size == 0
            config.gradient_accumulation_steps //= self.ddp_world_size
        transformer_config_fields = [f.name for f in dataclasses.fields(AllamoTransformerConfig)]
        checkpoint_name = None
        if config.init_from == 'resume':
            checkpoint_name = 'ckpt.pt'
        elif config.init_from == 'resume_last':
            checkpoint_name = 'last_eval_ckpt.pt'
        else:
            if os.path.exists(os.path.join(config.out_dir, 'config_ckpt.pt')) \
                or os.path.exists(os.path.join(config.out_dir, 'model_ckpt.pt')) \
                or os.path.exists(os.path.join(config.out_dir, 'optimizer_ckpt.pt')):
                self.logger.info("Delete existing checkpoint files to start from scratch or use --init_from=resume to resume training")
                exit()
            
        if checkpoint_name is not None:
            self.logger.info(f"Resuming training from {config.out_dir}")
            # resume training from a checkpoint
            ckpt_dir = config.checkpoint_path if config.checkpoint_path else config.out_dir
            self.logger.info(f"Loading {checkpoint_name} checkpoint files from {ckpt_dir}...")
            config_checkpoint = torch.load(os.path.join(ckpt_dir, f'config_{checkpoint_name}'), map_location='cpu')
            checkpoint_model_args = config_checkpoint['model_args']
            # force these config attributes to be equal otherwise we can't even resume training
            # the rest of the attributes (e.g. dropout) can stay as desired from command line
            for k in transformer_config_fields:
                if hasattr(config, k) and hasattr(checkpoint_model_args, k):
                    setattr(config, k, getattr(checkpoint_model_args, k))
            if 'iter_num' in config_checkpoint:
                self.iter_num = config_checkpoint['iter_num']
            if 'best_train_loss' in config_checkpoint:
                self.best_train_loss = config_checkpoint['best_train_loss']
            if 'best_val_loss' in config_checkpoint:
                self.best_val_loss = config_checkpoint['best_val_loss']
            if 'processed_tokens' in config_checkpoint:
                self.processed_tokens = config_checkpoint['processed_tokens']
            del config_checkpoint
            del checkpoint_model_args
            
        model_args = {k: getattr(config, k) for k in transformer_config_fields if hasattr(config, k)}
        modelConf = AllamoTransformerConfig(**model_args)
        model = AllamoTransformer(modelConf)
        if checkpoint_name is None:
            self.logger.info("Initialized a new model from scratch")
        else:
            state_dict = torch.load(os.path.join(ckpt_dir, f'model_{checkpoint_name}'), map_location='cpu')
            unwanted_prefix = '_orig_mod.'
            for k,v in list(state_dict.items()):
                if k.endswith('.rotary_emb.inv_freq'):
                    # For backward compatibility, where we had it in the checkpoint
                    state_dict.pop(k)
                elif k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            model.load_state_dict(state_dict)
            del state_dict
            self.logger.info("Loaded model from the checkpoint")
        model.to(config.device)

        # initialize a GradScaler. If enabled=False scaler is a no-op
        self.scaler = torch.cuda.amp.GradScaler(enabled=(config.dtype == 'float16' or config.dtype == 'bfloat16'))
        
        # optimizer
        self.optimizer = model.configure_optimizers(config.weight_decay, config.learning_rate, (config.beta1, config.beta2), self.device_type)
        if checkpoint_name is not None:
            optimizer_checkpoint_path = os.path.join(ckpt_dir, f'optimizer_{checkpoint_name}')
            if os.path.exists(optimizer_checkpoint_path):
                optimizer_checkpoint = torch.load(optimizer_checkpoint_path, map_location=config.device)
                self.optimizer.load_state_dict(optimizer_checkpoint)
                del optimizer_checkpoint
            else:
                self.logger.warn("Optimizer checkpoint file not found. Initializing optimizer from scratch")
                
        # compile the model - requires PyTorch 2.0
        if config.compile:
            self.logger.info("compiling the model... (takes a ~minute)")
            try:
                model = torch.compile(model, mode=config.compile_mode)
                self.logger.info("Model compiled and ready to use")
            except Exception as err:
                self.logger.warn(f"Model compile not supported: {err}")

        # wrap model into DDP container
        if self.ddp:
            model = DDP(model, device_ids=[self.ddp_local_rank])
        self.model = model
        self.raw_model = model.module if self.ddp else model # unwrap DDP container if needed
                
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
            self.logger.info(f"Cosing decay learning rate enabled. Currect learning rate: {self.get_lr(self.iter_num)}")
        else:
            self.logger.info(f"Using constant learning rate: {config.learning_rate}")

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss(self):
        losses_out = {}
        accuraces = {}
        self.model.eval()
        for split in self.simple_data_loader.splits:
            losses = torch.zeros(self.config.eval_iters)
            correct_preds = 0
            total_preds = 0
            for k in range(self.config.eval_iters):
                X, Y = self.simple_data_loader.get_batch(split, True)
                with self.ctx:
                    logits, loss, _ = self.model(X, Y)
                losses[k] = loss.item()
                total_preds += Y.size(0)
                correct_preds += (logits[:,-1,:].max(1).indices == Y[:,-1]).sum().item()
            losses_out[split] = losses.mean()
            accuraces[split] = correct_preds / total_preds
        self.model.train()
        if 'val' not in losses_out:
            losses_out['val'] = losses_out['train']
            accuraces['val'] = accuraces['train']
        return losses_out, accuraces

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(self, it):
        # 1) linear warmup for warmup_iters steps
        if it < self.config.warmup_iters:
            return self.config.learning_rate * it / self.config.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it >= self.config.lr_decay_iters:
            return self.config.min_lr
        # 3) in between, use cosine decay down to min learning rate with restarts (optional)
        if self.config.lr_decay_reset_iters is not None:
            decay_ratio = (it % self.config.lr_decay_reset_iters) / self.config.lr_decay_reset_iters
        else:
            decay_ratio = (it - self.config.warmup_iters) / (self.config.lr_decay_iters - self.config.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return self.config.min_lr + coeff * (self.config.learning_rate - self.config.min_lr)

    # grad_accum scheduler (when enabled) 
    def get_grad_accum(self, it):
        if self.gradient_accumulation_steps < self.config.grad_accum_max and \
            self.config.grad_accum_schedule and it % (self.config.grad_accum_max_iter/100) == 0:
            return min(self.gradient_accumulation_steps + 1, self.config.grad_accum_max)
        else:
            return self.gradient_accumulation_steps
            
    # helps saving checkpoint to a file
    def save_checkpoint(self, ckpt_file_name):
        checkpoint = {
            'model_args': self.raw_model.config,
            'iter_num': self.iter_num,
            'best_train_loss': self.best_train_loss,
            'best_val_loss': self.best_val_loss,
            'processed_tokens': self.processed_tokens,
            'config': self.config.__dict__,
        }
        ckpt_file_path = os.path.join(self.config.out_dir, 'config_' + ckpt_file_name)
        self.logger.info(f"saving config checkpoint to {ckpt_file_path}")
        torch.save(checkpoint, ckpt_file_path)
        
        ckpt_file_path = os.path.join(self.config.out_dir, 'model_' + ckpt_file_name)
        self.logger.info(f"saving model checkpoint to {ckpt_file_path}")
        torch.save(self.raw_model.state_dict(), ckpt_file_path)
        
        ckpt_file_path = os.path.join(self.config.out_dir, 'optimizer_' + ckpt_file_name)
        self.logger.info(f"saving optimizer checkpoint to {ckpt_file_path}")
        torch.save(self.optimizer.state_dict(), ckpt_file_path)
        self.logger.info(f"checkpoint files saved in {config.out_dir}")
        
    def calculate_eta(self):
        current_time = datetime.datetime.now()
        elapsed_time = current_time - self.start_timestamp
        elapsed_iters = self.iter_num - self.start_iter
        if elapsed_iters < 1:
            return 'N/A'
        avg_time_per_iter = elapsed_time.total_seconds() / elapsed_iters
        eta_seconds = math.ceil(avg_time_per_iter * (self.config.max_iters - self.iter_num))
        return self.format_seconds_as_time(eta_seconds)
        
    def format_seconds_as_time(self, seconds):
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours)}:{int(minutes):02}:{int(seconds):02}"
        
    def train(self):
        # training loop
        X, Y = self.simple_data_loader.get_batch('train') # fetch the very first batch
        self.start_iter = self.iter_num
        self.start_timestamp = datetime.datetime.now()
        while self.iter_num <= self.config.max_iters:
            log_iter = (self.iter_num % self.config.log_interval == 0 and self.master_process)
            eval_iter = (self.iter_num % self.config.eval_interval == 0 and self.master_process)

            # determine and set learning rate for this iteration
            lr = self.get_lr(self.iter_num) if self.config.decay_lr else self.config.learning_rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
                
            # determine and set batch_size and gradient_accumulation_steps for this iteration 
            micro_batch_size = self.simple_data_loader.update_batch_size(self.iter_num)
            self.gradient_accumulation_steps = self.get_grad_accum(self.iter_num)
            total_batch_size = self.config.block_size * micro_batch_size * self.gradient_accumulation_steps
            if self.ddp:
                total_batch_size *= self.ddp_world_size

            # evaluate the loss on train/val sets and write checkpoints
            if eval_iter:
                eval_time = time.time()
                losses, accuraces = self.estimate_loss()
                eval_time = time.time() - eval_time
                train_loss = losses['train']
                val_loss = losses['val']
                train_ppl = torch.exp(train_loss)
                val_ppl = torch.exp(val_loss)
                self.logger.info(
                    f"iter {self.iter_num:,}: train loss={train_loss:.4f} ppl={train_ppl:.4f} "
                    f"acc={accuraces['train']:.4f} (best loss={self.best_train_loss:.4f}), "
                    f"val loss={val_loss:.4f} ppl={val_ppl:.4f} acc={accuraces['val']:.4f} "
                    f"(best loss={self.best_val_loss:.4f}), tokens {self.processed_tokens:,}"
                )
                if self.iter_num > self.start_iter:
                    if losses['train'] < self.best_train_loss:
                        self.best_train_loss = losses['train']
                    if losses['val'] < self.best_val_loss:
                        self.best_val_loss = losses['val']
                        self.save_checkpoint('ckpt.pt')
                    if self.config.always_save_checkpoint:
                        self.save_checkpoint('last_eval_ckpt.pt')
                if self.config.wandb_log:
                    wandb.log({
                        "iter": self.iter_num,
                        "eval/time": eval_time*1000,
                        "eval/samples_per_second": (self.config.eval_iters * len(self.simple_data_loader.splits)) / eval_time,
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
            
            # numpy.memmap does not release RAM after reading data. To keep memory consumption low, let's reconstruct the memmap objects
            if self.config.reload_datasets_interval > 0 and self.iter_num % self.config.reload_datasets_interval == 0:
                self.simple_data_loader.reload_datasets()
                gc.collect()
                torch.cuda.empty_cache()
            
            accuracy = 0
            timer = time.time()
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
                    logits, loss, _ = self.model(X, Y)
                    if micro_steps > 1:
                        loss = loss / micro_steps # scale the loss to account for micro steps
                
                # count processed tokens
                self.processed_tokens += X.numel() * self.ddp_world_size if self.ddp else X.numel()
                if log_iter and (micro_step == self.gradient_accumulation_steps - 1):
                    # calculate accuracy. note: this is a CPU-GPU sync point!
                    accuracy = (logits.max(2).indices == Y).sum().item() / Y.view(-1).size(0)
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                X, Y = self.simple_data_loader.get_batch('train')
                
                # backward pass, with gradient scaling if training in fp16
                self.scaler.scale(loss).backward()
                
            # clip the gradient
            if self.config.grad_clip != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            # step the optimizer and scaler if training in fp16
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            self.optimizer.zero_grad(set_to_none=True)

            # timing and logging
            if log_iter:
                iter_time = (time.time() - timer)*1000
                # get loss as float. note: this is a CPU-GPU sync point
                # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                lossf = loss.item() * micro_steps
                ppl = torch.exp(torch.tensor(lossf))
                self.logger.info(
                    f"iter {self.iter_num:,}: loss {lossf:.4f}, ppl {ppl:.4f}, acc {accuracy:.4f}, "
                    f"iter time {iter_time:.2f}ms, tokens {self.processed_tokens:,}, lr {lr:.6f}, "
                    f"ETA: {self.calculate_eta()}"
                )
                if self.config.wandb_log:
                    metrics = {
                        "iter": self.iter_num,
                        "train/iter_time": iter_time,
                        "train/loss": lossf,
                        "train/ppl": ppl,
                        "train/acc": accuracy,
                        "train/lr": lr,
                        "train/tokens": self.processed_tokens,
                        "train/total_batch_size": total_batch_size
                    }
                    if self.config.dataset_seq_train:
                        metrics['train/ds_offset'] = self.simple_data_loader.dataset_train_x_start
                    wandb.log(metrics)
            self.iter_num += 1
            
        training_time = self.format_seconds_as_time((datetime.datetime.now() - self.start_timestamp).total_seconds())
        self.logger.info(f"Training finished in {training_time}")

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
        destroy_process_group()

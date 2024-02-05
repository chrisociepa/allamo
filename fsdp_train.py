"""
This single file is intended to perform some magic for training/finetuning using FSDP.
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
import torch.distributed as dist
import functools
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig, # general model non-sharded, non-flattened params
    MixedPrecision,
    BackwardPrefetch,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

from model import AllamoTransformerConfig, AllamoTransformer, SelfAttentionBlock
from configuration import AllamoConfiguration
from simple_data_loader import SimpleDataLoader
from simple_instructions_data_loader import SimpleInstructionsDataLoader

class AllamoFSDPTrainer:

    def __init__(self, config: AllamoConfiguration):
        self.config = config
        self.__init_torch(config)
        self.__init_logger(config)
        
        self.iter_num = 0
        self.best_train_loss = 1e9
        self.best_val_loss = 1e9
        self.processed_tokens = 0
        self.__init_training(config)
        
        if config.dataloader_type == 'instructions':
            self.simple_data_loader = SimpleInstructionsDataLoader(config, self.rank, self.world_size)
        else:
            self.simple_data_loader = SimpleDataLoader(config, self.rank, self.world_size)
        
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
        self.seed_offset = self.rank # each process gets a different seed
        if self.master_process:
            print(
                f"RANK: {self.rank}, LOCAL_RANK: {self.local_rank}, "
                f"WORLD_SIZE: {self.world_size}, LOCAL_WORLD_SIZE: {os.environ['LOCAL_WORLD_SIZE']}"
            )
            os.makedirs(config.out_dir, exist_ok=True)
        torch.manual_seed(config.seed + self.seed_offset)
        torch.cuda.manual_seed(config.seed + self.seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        torch.set_float32_matmul_precision("highest") # set to "high" for faster matrix multiplications with bfloat16
        if config.dtype == 'bfloat16-true':
            raise Exception('Full bfloat16 training is not supported with FSDP')
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config.dtype]
        self.device_type = 'cuda' if 'cuda' in config.device else 'cpu'
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                SelfAttentionBlock,
            },
        )
        self.fsdp_config = dict(
            auto_wrap_policy=auto_wrap_policy,
            sharding_strategy=ShardingStrategy.HYBRID_SHARD, #Options: FULL_SHARD, HYBRID_SHARD, _HYBRID_SHARD_ZERO2, SHARD_GRAD_OP, NO_SHARD
            device_id=torch.cuda.current_device(),
            mixed_precision=MixedPrecision(
                param_dtype=ptdtype,
                reduce_dtype=ptdtype,
                buffer_dtype=ptdtype,
            ),
            limit_all_gathers=True,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # will use slightly more memory vs. no prefetch
            use_orig_params=True # required to use torch.compile()
        )
        self.fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        if config.gradient_checkpointing:
            self.fsdp_activation_checkpointing = True
            config.gradient_checkpointing = False # control gradient checkpointing with FSDP 
        else:
            self.fsdp_activation_checkpointing = False

    def __init_training(self, config: AllamoConfiguration):
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
        if self.fsdp_activation_checkpointing:
            modelConf.gradient_checkpointing = False
        model = AllamoTransformer(modelConf)
        if checkpoint_name is None:
            self.logger.info("Initialized a new model from scratch")
        else:
            self.load_model_checkpoint(model, os.path.join(ckpt_dir, f'model_{checkpoint_name}'))
        
        self.logger.info("Configuring model with FSDP")
        model = FSDP(model, **self.fsdp_config)
        self.logger.info("Model configured with FSDP")
        
        if self.fsdp_activation_checkpointing:
            non_reentrant_wrapper = functools.partial(
                checkpoint_wrapper,
                offload_to_cpu=False,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )
            check_fn = lambda submodule: isinstance(submodule, SelfAttentionBlock)
            apply_activation_checkpointing(model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)
        
        # compile the model - requires PyTorch 2.0
        if config.compile:
            self.logger.info("compiling the model... (takes a ~minute)")
            try:
                model = torch.compile(model, mode=config.compile_mode)
                self.logger.info("Model compiled and ready to use")
            except Exception as err:
                self.logger.warning(f"Model compile not supported: {err}")
        self.model = model
        
        # optimizer
        self.optimizer = model.configure_optimizers(config.weight_decay, config.learning_rate, (config.beta1, config.beta2), self.device_type)
        if checkpoint_name is None:
            self.logger.info("Initializing optimizer from scratch")
        else:
            self.load_optimizer_checkpoint(model, self.optimizer, os.path.join(ckpt_dir, f'optimizer_{checkpoint_name}'))
                
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
            
    def load_model_checkpoint(self, model, ckpt_path):
        #if not self.master_process:
        #    return
        state_dict = torch.load(ckpt_path, map_location='cpu')
        self.remove_unwanted_prefix_from_model_state_dict(state_dict)
        model.load_state_dict(state_dict)
        self.logger.info("Loaded model from the checkpoint")
        
    def load_optimizer_checkpoint(self, model, optimizer, ckpt_path):
        if os.path.exists(ckpt_path):
            # requires each rank to have the full dict in CPU memory to reduce communication
            full_osd = torch.load(ckpt_path, map_location='cpu')
            #self.remove_unwanted_prefix_from_optimizer_state_dict(full_osd)
            sharded_osd = FSDP.shard_full_optim_state_dict(full_osd, model)
            optimizer.load_state_dict(sharded_osd)
            self.logger.info("Shared optimizer state loaded.")
        else:
            if self.master_process:
                self.logger.warning("Optimizer checkpoint file not found. Initializing optimizer from scratch")
                
    def remove_unwanted_prefix_from_model_state_dict(self, state_dict):
        unwanted_prefix = '_orig_mod.'
        unwanted_prefix_len = len(unwanted_prefix)
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[unwanted_prefix_len:]] = state_dict.pop(k)
                
    def remove_unwanted_prefix_from_optimizer_state_dict(self, optimizer_state_dict):
        if "param_groups" in optimizer_state_dict:
            unwanted_prefix = '_orig_mod.'
            unwanted_prefix_len = len(unwanted_prefix)
            for param_group in optimizer_state_dict["param_groups"]:
                param_group['params'] = [p[unwanted_prefix_len:] if p.startswith(unwanted_prefix) else p for p in param_group['params']]
        
    # helps saving checkpoint to a file
    def save_checkpoint(self, ckpt_file_name):
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, self.fullstate_save_policy):
            full_msd = self.model.state_dict()
        if self.master_process:
            checkpoint = {
                'model_args': self.model.config,
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
            torch.save(full_msd, ckpt_file_path)
            del full_msd
            
        # pull all sharded optimizer states to rank0 cpu.
        full_osd = FSDP.full_optim_state_dict(self.model, self.optimizer)
        if self.master_process:
            ckpt_file_path = os.path.join(self.config.out_dir, 'optimizer_' + ckpt_file_name)
            self.logger.info(f"saving optimizer checkpoint to {ckpt_file_path}")
            torch.save(full_osd, ckpt_file_path)
            self.logger.info(f"checkpoint files saved in {config.out_dir}")
            del full_osd
            
    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss(self):
        losses_out = {}
        accuraces = {}
        self.model.eval()
        for split in self.simple_data_loader.splits:
            fsdp_loss_preds = torch.zeros(3).to(self.config.device)
            for k in range(self.config.eval_iters):
                X, Y = self.simple_data_loader.get_batch(split, True)
                logits, loss, _ = self.model(X, Y)
                fsdp_loss_preds[0] += loss.item()
                fsdp_loss_preds[1] += (logits[:,-1,:].max(1).indices == Y[:,-1]).sum().item()
            dist.all_reduce(fsdp_loss_preds, op=dist.ReduceOp.SUM)
            steps = self.config.eval_iters * self.world_size
            losses_out[split] = fsdp_loss_preds[0] / steps
            accuraces[split] = fsdp_loss_preds[1] / steps
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
            
    def format_seconds_as_time(self, seconds):
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours)}:{int(minutes):02}:{int(seconds):02}"
            
    def calculate_eta(self):
        current_time = datetime.datetime.now()
        elapsed_time = current_time - self.start_timestamp
        elapsed_iters = self.iter_num - self.start_iter
        if elapsed_iters < 1:
            return 'N/A'
        avg_time_per_iter = elapsed_time.total_seconds() / elapsed_iters
        eta_seconds = math.ceil(avg_time_per_iter * (self.config.max_iters - self.iter_num))
        return self.format_seconds_as_time(eta_seconds)
        
    def has_next_iter_to_perform(self):
        if self.config.num_train_epochs is not None and self.simple_data_loader.epoch >= self.config.num_train_epochs:
            return False
        return self.iter_num <= self.config.max_iters
        
    def train(self):
        # training loop
        X, Y = self.simple_data_loader.get_batch('train') # fetch the very first batch
        self.start_iter = self.iter_num
        self.start_timestamp = datetime.datetime.now()
        while self.has_next_iter_to_perform():
            # determine and set learning rate for this iteration
            lr = self.get_lr(self.iter_num) if self.config.decay_lr else self.config.learning_rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
                
            # determine and set batch_size and gradient_accumulation_steps for this iteration 
            micro_batch_size = self.simple_data_loader.update_batch_size(self.iter_num)
            self.gradient_accumulation_steps = self.get_grad_accum(self.iter_num)

            # evaluate the loss on train/val sets and write checkpoints
            if self.iter_num % self.config.eval_interval == 0:
                eval_time = time.time()
                losses, accuraces = self.estimate_loss()
                eval_time = time.time() - eval_time
                if self.iter_num > self.start_iter:
                    if losses['train'] < self.best_train_loss:
                        self.best_train_loss = losses['train']
                    if losses['val'] < self.best_val_loss:
                        self.best_val_loss = losses['val']
                        self.save_checkpoint('ckpt.pt')
                    if self.config.always_save_checkpoint:
                        self.save_checkpoint('last_eval_ckpt.pt')
                if self.master_process:                
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
            
            # numpy.memmap does not release RAM after reading data. To keep memory consumption low, let's reconstruct the memmap objects
            if self.config.reload_datasets_interval > 0 and self.iter_num % self.config.reload_datasets_interval == 0:
                self.simple_data_loader.reload_datasets()
                gc.collect()
                torch.cuda.empty_cache()
            
            accuracy = 0
            fsdp_loss_acc = torch.zeros(4).to(self.config.device)
            timer = time.time()
            # forward backward update, with optional gradient accumulation to simulate larger batch size
            for micro_step in range(self.gradient_accumulation_steps):
                logits, loss, _ = self.model(X, Y)
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps # scale the loss to account for micro steps
                
                fsdp_loss_acc[0] += loss.item()
                fsdp_loss_acc[1] += X.numel()
                fsdp_loss_acc[2] += (logits.max(2).indices == Y).sum().item() / Y.view(-1).size(0)
                fsdp_loss_acc[3] += 1
                
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                X, Y = self.simple_data_loader.get_batch('train')
                
                # backward pass
                loss.backward()
                
            # clip the gradient
            if self.config.grad_clip != 0.0:
                self.model.clip_grad_norm_(self.config.grad_clip)
                
            # weight update
            self.optimizer.step()
            
            # flush the gradients as soon as we can, no need for this memory anymore
            self.optimizer.zero_grad(set_to_none=True)
            
            # sync loss and acc over all processes
            dist.all_reduce(fsdp_loss_acc, op=dist.ReduceOp.SUM)
            
            if self.master_process:
                self.processed_tokens += int(fsdp_loss_acc[1])

            # timing and logging
            if self.iter_num % self.config.log_interval == 0 and self.master_process:
                iter_time = (time.time() - timer)*1000
                # get loss as float. note: this is a CPU-GPU sync point
                lossf = fsdp_loss_acc[0].item() / self.world_size
                ppl = torch.exp(torch.tensor(lossf)).item()
                accuracy = fsdp_loss_acc[2].item() / fsdp_loss_acc[3].item()
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
                        "train/total_batch_size": self.config.block_size * micro_batch_size * self.gradient_accumulation_steps * self.world_size
                    }
                    if self.config.dataset_seq_train:
                        metrics['train/ds_offset'] = self.simple_data_loader.dataset_train_x_start
                    wandb.log(metrics)
            self.iter_num += 1
            
        training_time = self.format_seconds_as_time((datetime.datetime.now() - self.start_timestamp).total_seconds())
        self.logger.info(f"Training finished in {training_time}")
        self.save_checkpoint('final_ckpt.pt')

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

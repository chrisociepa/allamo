import json
import os
import dataclasses
import shutil
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from allamo.trainer.base import BaseTrainer
from allamo.logging import logger
from allamo.model import AllamoTransformer
from allamo.configuration import AllamoConfiguration
from allamo.torch_utils import TORCH_DTYPE_MAP
from allamo.train_utils import (
    rename_file_to_prev_version,
    calculate_md5,
    remove_unwanted_prefix_from_model_state_dict,
    get_model_checkpoint_path,
    get_config_checkpoint_path,
    get_optimizer_checkpoint_path,
)

class SimpleTrainer(BaseTrainer):

    def __init__(self, config: AllamoConfiguration):
        super().__init__(config)
        
    def distributed(self):
        return self.train_ctx.world_size > 1
        
    def init_torch(self, config: AllamoConfiguration):
        super().init_torch(config)
        self.ctx = nullcontext() if self.device_type == 'cpu' else torch.amp.autocast(device_type=self.device_type, dtype=TORCH_DTYPE_MAP[config.dtype])
        if config.dtype == 'bfloat16-true':
            # torch.set_float32_matmul_precision("high")
            torch.set_default_dtype(torch.bfloat16)
        
    def init_training(self):
        self.init_checkpoint()
        self.load_datasets()
        
        model = AllamoTransformer(self.create_model_config())
        self.model_num_params = model.model_num_params
        if self.checkpoint_name is None:
            logger.info("Initialized a new model from scratch")
        else:
            self.load_model_checkpoint(model, os.path.join(self.checkpoint_dir, f'model_{self.checkpoint_name}.pt'), self.config)
        model.to(self.config.device)

        if self.config.compile:
            logger.info("compiling the model... (takes a ~minute)")
            try:
                model = torch.compile(model, mode=self.config.compile_mode)
                logger.info("Model compiled and ready to use")
            except Exception as err:
                logger.warn(f"Model compile not supported: {err}")

        self.raw_model = model # neeeded in DDP training
        self.model = model
        # wrap model into DDP container
        if self.distributed():
            self.model = DDP(self.model, device_ids=[self.train_ctx.local_rank])
            
        # initialize a GradScaler. If enabled=False scaler is a no-op
        self.scaler = torch.amp.GradScaler(self.device_type, enabled=(self.config.dtype == 'float16' or self.config.dtype == 'bfloat16'))
        
        # optimizer
        self.optimizer = self.model.configure_optimizers(self.config, self.device_type)
        if self.checkpoint_name is not None:
            self.load_optimizer_checkpoint(self.optimizer, os.path.join(self.checkpoint_dir, f'optimizer_{self.checkpoint_name}.pt'))
        
        self.init_gradient_accumulation_scheduler()
        self.log_init_learning_rate()

    def load_model_checkpoint(self, model, ckpt_path, config):
        state_dict = torch.load(ckpt_path, map_location='cpu')
        remove_unwanted_prefix_from_model_state_dict(state_dict)
        model.load_state_dict(state_dict)
        if config.log_checkpoint_md5_on_load and self.train_ctx.master_process:
            md5sum = calculate_md5(ckpt_path)
            logger.info(f"Loaded model from checkpoint {ckpt_path} - MD5: {md5sum}")
        else:
            logger.info(f"Loaded model from checkpoint {ckpt_path}")
        
    def load_optimizer_checkpoint(self, optimizer, ckpt_path):
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location=self.config.device)
            optimizer.load_state_dict(state_dict)
            logger.info("Optimizer state loaded.")
        else:
            logger.warning("Optimizer checkpoint file not found. Initializing optimizer from scratch")

    # helps saving checkpoint to a file
    def save_checkpoint(self, ckpt_file_name, model_only=False, epoch_ckpt=False):
        if not self.train_ctx.master_process:
            return
        
        model_ckpt_file_path = get_model_checkpoint_path(ckpt_file_name, self.config.out_dir)
        logger.info(f"saving model checkpoint to {model_ckpt_file_path}")
        if not self.config.ignore_last_checkpoint_backup:
            rename_file_to_prev_version(model_ckpt_file_path)
        torch.save(self.raw_model.state_dict(), model_ckpt_file_path)
        
        md5sum = calculate_md5(model_ckpt_file_path) if epoch_ckpt and self.config.log_checkpoint_md5_on_epoch else None
        
        checkpoint = {
            'model_args': dataclasses.asdict(self.raw_model.config),
            'run_uuid': self.train_ctx.run_uuid,
            'training_uuid': self.train_ctx.training_uuid,
            'iter_num': self.iter_num,
            'best_train_loss': self.best_train_loss,
            'best_val_loss': self.best_val_loss,
            'processed_tokens': self.processed_tokens,
            'config': dataclasses.asdict(self.config),
            'allamo_dataloader': {
                'train_processed_files': self.data_loader.train_dataset.processed_files,
                'dataset_offset': self.data_loader.dataset_offset * self.train_ctx.world_size,
                'epoch': self.data_loader.epoch
            }
        }
        if md5sum is not None:
            checkpoint['checkpoint_md5sum'] = md5sum
            logger.info(f"model checkpoint saved - MD5: {md5sum}")
        
        config_ckpt_file_path = get_config_checkpoint_path(ckpt_file_name, self.config.out_dir)
        logger.info(f"saving config checkpoint to {config_ckpt_file_path}")
        if not self.config.ignore_last_checkpoint_backup:
            rename_file_to_prev_version(config_ckpt_file_path)
        with open(config_ckpt_file_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=4, ensure_ascii=False)
        
        if self.config.save_optimizer_checkpoint and model_only == False and \
            (self.config.optimizer_checkpoint_interval is None or \
             self.iter_num % self.config.optimizer_checkpoint_interval == 0):
            optim_ckpt_file_path = get_optimizer_checkpoint_path(ckpt_file_name, self.config.out_dir)
            logger.info(f"saving optimizer checkpoint to {optim_ckpt_file_path}")
            if not self.config.ignore_last_checkpoint_backup:
                rename_file_to_prev_version(optim_ckpt_file_path)
            torch.save(self.optimizer.state_dict(), optim_ckpt_file_path)
            
            if self.config.optimizer_checkpoint_interval is not None:
                shutil.copy(model_ckpt_file_path, model_ckpt_file_path + '.optim')
                shutil.copy(config_ckpt_file_path, config_ckpt_file_path + '.optim')
        logger.info(f"checkpoint files saved in {self.config.out_dir}")

    def should_evaluate(self):
        return super().should_evaluate() and self.train_ctx.master_process
    
    def forward(self, batch, last_micro_step):
        if self.distributed():
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            self.model.require_backward_grad_sync = last_micro_step
        with self.ctx:
            logits, loss, _ = self.model(**batch)
        if self.gradient_accumulation_steps > 1:
            loss = loss / self.gradient_accumulation_steps # scale the loss to account for micro steps
        if batch["target_weights"] is not None:
            if self.config.weighted_loss_method == 'openchat':
                target_weights = batch["target_weights"].sum()
                if self.distributed():
                    # sum loss weights over all processes
                    dist.all_reduce(target_weights, op=dist.ReduceOp.SUM)
                loss = (self.train_ctx.world_size / target_weights) * loss
            else:
                loss = loss / torch.sum(batch["target_weights"] > 0).item()
        
        unmasked_labels = torch.sum(batch["target_ids"].view(-1) != self.config.ignore_index).item()
        accuracy = (logits.max(2).indices == batch["target_ids"]).sum().item() / unmasked_labels
        return loss, unmasked_labels, accuracy

    def close(self):
        if self.distributed():
            dist.barrier()
            dist.destroy_process_group()

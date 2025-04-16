import os
import shutil
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from allamo.trainer.base import BaseTrainer
from allamo.logging import logger
from allamo.model.model import AllamoTransformer
from allamo.configuration import AllamoConfiguration
from allamo.optimizer.optimizer_utils import configure_optimizer
from allamo.torch_utils import TORCH_DTYPE_MAP
from allamo.train_utils import (
    get_model_checkpoint_path,
    get_config_checkpoint_path,
    get_optimizer_checkpoint_path,
    apply_activation_checkpointing,
)

class SimpleTrainer(BaseTrainer):

    def __init__(self, config: AllamoConfiguration):
        super().__init__(config)
        if config.distributed_checkpoint:
            config.distributed_checkpoint = False
            logger.warning("PyTorch Distributed Checkpoint (DCP) is only available for FSDP training! Fallback to regular checkpoint")
        
    def distributed(self):
        return self.train_ctx.world_size > 1
        
    def init_torch(self):
        super().init_torch()
        self.ctx = nullcontext() if self.device_type == 'cpu' else torch.amp.autocast(device_type=self.device_type, dtype=TORCH_DTYPE_MAP[self.config.dtype])
        if self.config.dtype == 'bfloat16-true':
            # torch.set_float32_matmul_precision("high")
            torch.set_default_dtype(torch.bfloat16)
        
    def init_training(self):
        super().init_training()
        
        model = AllamoTransformer(self.model_config)
        self.model_num_params = model.model_num_params

        self.freeze_model_params(model) # Optionally freezes model parameters depending on the configuration

        if self.config.gradient_checkpointing:
            apply_activation_checkpointing(model, self.config)

        if self.checkpoint_manager.is_checkpoint_available():
            self.checkpoint_manager.load_regular_model_checkpoint(model)
        else:
            logger.info("New model initialized from scratch")
        model.to(self.config.device)

        if self.config.compile:
            logger.info("Compiling model")
            try:
                model = torch.compile(model, mode=self.config.compile_mode)
                logger.info("Model compiled and ready to use")
            except Exception as err:
                logger.warning(f"Unable to compile the model: {err}")

        self.raw_model = model # neeeded in DDP training
        self.model = model
        # wrap model into DDP container
        if self.distributed():
            self.model = DDP(self.model, device_ids=[self.train_ctx.local_rank])
            
        # initialize a GradScaler. If enabled=False scaler is a no-op
        self.scaler = torch.amp.GradScaler(self.device_type, enabled=(self.config.dtype == 'float16' or self.config.dtype == 'bfloat16'))
        
        # optimizer
        self.optimizer = configure_optimizer(self.raw_model, self.config, self.device_type)
        if self.checkpoint_manager.is_checkpoint_available():
            self.load_optimizer_checkpoint(self.optimizer)
        
        self.init_gradient_accumulation_scheduler()
        self.log_init_learning_rate()

    def load_optimizer_checkpoint(self, optimizer):
        ckpt_path = get_optimizer_checkpoint_path(self.checkpoint_manager.checkpoint_name, self.checkpoint_manager.checkpoint_dir)
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location=self.config.device, weights_only=True)
            optimizer.load_state_dict(state_dict)
            logger.info(f"Optimizer state loaded from checkpoint {ckpt_path}")
        else:
            logger.warning("Optimizer checkpoint file not found. Initializing optimizer from scratch")

    # helps saving checkpoint to a file
    def save_checkpoint(self, ckpt_file_name, model_only=False, epoch_ckpt=False):
        if not self.train_ctx.master_process:
            return
        
        model_ckpt_file_path = get_model_checkpoint_path(ckpt_file_name, self.config.out_dir)
        md5sum = self.checkpoint_manager.save_regular_model_checkpoint(self.raw_model.state_dict(), model_ckpt_file_path, epoch_ckpt)

        config_ckpt_file_path = get_config_checkpoint_path(ckpt_file_name, self.config.out_dir)
        self.checkpoint_manager.save_config_checkpoint(config_ckpt_file_path, md5sum, self.model_config)
        
        if model_only == False and self.checkpoint_manager.should_save_optimizer():
            optim_ckpt_file_path = get_optimizer_checkpoint_path(ckpt_file_name, self.config.out_dir)
            self.checkpoint_manager.save_regular_optimizer_checkpoint(self.optimizer.state_dict(), optim_ckpt_file_path)
            
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
                # sum loss weights over all processes
                target_weights = self.dist_all_reduce(target_weights, op=dist.ReduceOp.SUM)
                loss = (self.dp_world_size / target_weights) * loss
            else:
                loss = loss / torch.sum(batch["target_weights"] > 0).item()
        
        unmasked_labels = torch.sum(batch["target_ids"].view(-1) != self.config.ignore_index).item()
        accuracy = (logits.max(2).indices == batch["target_ids"]).sum().item() / unmasked_labels
        return loss, unmasked_labels, accuracy

    def close(self):
        if self.distributed():
            dist.barrier()
            dist.destroy_process_group()

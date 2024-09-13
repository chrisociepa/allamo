import os
import shutil
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig, # general model non-sharded, non-flattened params
)

from allamo.trainer.base import BaseTrainer
from allamo.logging import logger
from allamo.model.model import AllamoTransformer
from allamo.configuration import AllamoConfiguration
from allamo.fsdp_utils import parallelize_with_fsdp
from allamo.train_utils import (
    get_model_checkpoint_path,
    get_config_checkpoint_path,
    get_optimizer_checkpoint_path,
)

class FSDPTrainer(BaseTrainer):

    def __init__(self, config: AllamoConfiguration):
        super().__init__(config)
        
    def distributed(self):
        return True
                    
    def init_torch(self, config: AllamoConfiguration):
        super().init_torch(config)
        if config.dtype == 'bfloat16-true':
            raise Exception('Full bfloat16 training is not supported with FSDP')
        
        self.fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        if config.gradient_checkpointing:
            self.fsdp_activation_checkpointing = True
            config.gradient_checkpointing = False # control gradient checkpointing with FSDP 
            logger.info(
                "Deactivated gradient checkpointing at the model configuration level. "
                "Activated gradient checkpointing at the FSDP level."
            )
        else:
            self.fsdp_activation_checkpointing = False
            
    def init_training(self):
        super().init_training()
        
        if self.fsdp_activation_checkpointing:
            self.model_config.gradient_checkpointing = False
        
        model = AllamoTransformer(self.model_config)
        self.model_num_params = model.model_num_params
        if self.checkpoint_manager.checkpoint_name is None:
            logger.info("Initialized a new model from scratch")
        else:
            self.checkpoint_manager.load_regular_model_checkpoint(model)
        
        logger.info("Configuring model with FSDP")
        self.model = parallelize_with_fsdp(model, self.config, self.fsdp_activation_checkpointing)
        
        # initialize a GradScaler only for FSDP's built-in mixed precision with fp16
        self.scaler = torch.amp.GradScaler(self.device_type, enabled=(self.config.dtype == 'float16'))
        
        # optimizer
        self.optimizer = model.configure_optimizers(self.config, self.device_type)
        if self.checkpoint_manager.checkpoint_name is None:
            logger.info("Initializing optimizer from scratch")
        else:
            self.load_optimizer_checkpoint(model, self.optimizer)
                
        self.init_gradient_accumulation_scheduler()
        self.log_init_learning_rate()
    
    def load_optimizer_checkpoint(self, model, optimizer):
        ckpt_path = get_optimizer_checkpoint_path(self.checkpoint_manager.checkpoint_name, self.checkpoint_manager.checkpoint_dir)
        if os.path.exists(ckpt_path):
            # requires each rank to have the full dict in CPU memory to reduce communication
            full_osd = torch.load(ckpt_path, map_location='cpu')
            sharded_osd = FSDP.optim_state_dict_to_load(model, optimizer, full_osd)
            optimizer.load_state_dict(sharded_osd)
            logger.info(f"Shared optimizer state loaded from checkpoint {ckpt_path}")
        else:
            if self.train_ctx.master_process:
                logger.warning("Optimizer checkpoint file not found. Initializing optimizer from scratch")
        
    # helps saving checkpoint to a file
    def save_checkpoint(self, ckpt_file_name, model_only=False, epoch_ckpt=False):
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, self.fullstate_save_policy):
            full_msd = self.model.state_dict()
        if self.train_ctx.master_process:
            model_ckpt_file_path = get_model_checkpoint_path(ckpt_file_name, self.config.out_dir)
            md5sum = self.checkpoint_manager.save_regular_model_checkpoint(full_msd, model_ckpt_file_path, epoch_ckpt)
            del full_msd
            
            config_ckpt_file_path = get_config_checkpoint_path(ckpt_file_name, self.config.out_dir)
            self.checkpoint_manager.save_config_checkpoint(config_ckpt_file_path, md5sum, self.model_config)
        
        if model_only == False and self.checkpoint_manager.should_save_optimizer():
            # pull all sharded optimizer states to rank0 cpu.
            full_osd = FSDP.full_optim_state_dict(self.model, self.optimizer)
            if self.train_ctx.master_process:
                optim_ckpt_file_path = get_optimizer_checkpoint_path(ckpt_file_name, self.config.out_dir)
                self.checkpoint_manager.save_regular_optimizer_checkpoint(full_osd, optim_ckpt_file_path)
                del full_osd
                
                if self.config.optimizer_checkpoint_interval is not None:
                    shutil.copy(model_ckpt_file_path, model_ckpt_file_path + '.optim')
                    shutil.copy(config_ckpt_file_path, config_ckpt_file_path + '.optim')
                    
    def clip_grad_norm(self):
        return self.model.clip_grad_norm_(self.config.grad_clip).item()
            
    def forward(self, batch, last_micro_step):
        logits, loss, _ = self.model(**batch)
        if self.gradient_accumulation_steps > 1:
            loss = loss / self.gradient_accumulation_steps # scale the loss to account for micro steps
        
        if batch["target_weights"] is not None:
            if self.config.weighted_loss_method == 'openchat':
                target_weights = batch["target_weights"].sum()
                # sum loss weights over all processes
                dist.all_reduce(target_weights, op=dist.ReduceOp.SUM)
                loss = (self.train_ctx.world_size / target_weights) * loss
            else:
                loss = loss / torch.sum(batch["target_weights"] > 0).item()
        
        unmasked_labels = torch.sum(batch["target_ids"].view(-1) != self.config.ignore_index).item()
        accuracy = (logits.max(2).indices == batch["target_ids"]).sum().item() / unmasked_labels
        return loss, unmasked_labels, accuracy
            
    def close(self):
        dist.barrier()
        dist.destroy_process_group()

import dataclasses
import json
import os
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)

from allamo.configuration import AllamoConfiguration
from allamo.dataset.data_loader import AllamoDataLoader
from allamo.logging import logger
from allamo.train_utils import (
    calculate_md5,
    get_config_checkpoint_path,
    get_model_checkpoint_path,
    rename_file_to_prev_version,
    get_model_config_field_names,
    remove_unwanted_prefix_from_model_state_dict,
)
from allamo.training_context import TrainingContext

class ModelWrapper(Stateful):
    def __init__(self, model: nn.Module):
        self.model = model

    def state_dict(self):
        return get_model_state_dict(self.model)

    def load_state_dict(self, state_dict: Dict[str, Any]):
        set_model_state_dict(self.model, state_dict)


class OptimizerWrapper(Stateful):
    def __init__(self, model: nn.Module, optim: torch.optim.Optimizer):
        self.model = model
        self.optim = optim

    def state_dict(self):
        return get_optimizer_state_dict(self.model, self.optim)

    def load_state_dict(self, state_dict: Dict[str, Any]):
        set_optimizer_state_dict(self.model, self.optim, optim_state_dict=state_dict)

class CheckpointManager:
    
    def __init__(self, config: AllamoConfiguration, train_ctx: TrainingContext, data_loader: AllamoDataLoader):
        self.config = config
        self.train_ctx = train_ctx
        self.data_loader = data_loader
        self.checkpoint_dir = self.config.checkpoint_path if self.config.checkpoint_path else self.config.out_dir
        self.checkpoint_name = None
        
    def init_checkpoint(self):
        if self.config.init_from == 'resume':
            self.checkpoint_name = 'ckpt'
        elif self.config.init_from == 'resume_last':
            self.checkpoint_name = 'last_eval_ckpt'
        else:
            if os.path.exists(get_config_checkpoint_path('ckpt', self.checkpoint_dir)):
                logger.info("Delete existing checkpoint files to start from scratch or use --init_from=resume to resume training")
                exit()
        
        if self.checkpoint_name is not None:
            config_ckpt_path = get_config_checkpoint_path(self.checkpoint_name, self.checkpoint_dir)
            if os.path.exists(config_ckpt_path):
                logger.info(f"Resuming training from {self.checkpoint_dir} and start loading '{self.checkpoint_name}' checkpoint files")
                self.load_config_checkpoint(config_ckpt_path)
            elif self.config.init_from == 'resume_last':
                self.checkpoint_name = None
                if self.train_ctx.master_process:
                    logger.warning(f"'{self.checkpoint_name}' checkpoint files not found but allowing to start from scratch")
            else:
                raise Exception(f"'{self.checkpoint_name}' checkpoint files not found!")
    
    def load_config_checkpoint(self, ckpt_path):
        with open(ckpt_path, "r", encoding="utf-8") as f:
            config_checkpoint = json.load(f)
            
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in get_model_config_field_names():
            if hasattr(self.config, k) and k in config_checkpoint['model_args']:
                setattr(self.config, k, config_checkpoint['model_args'][k])
        
        if 'training_uuid' in config_checkpoint:
            self.train_ctx.training_uuid = config_checkpoint['training_uuid']
        if 'iter_num' in config_checkpoint:
            self.train_ctx.iter_num = config_checkpoint['iter_num']
        if 'best_train_loss' in config_checkpoint:
            self.train_ctx.best_train_loss = config_checkpoint['best_train_loss']
        if 'best_val_loss' in config_checkpoint:
            self.train_ctx.best_val_loss = config_checkpoint['best_val_loss']
        if 'processed_tokens' in config_checkpoint:
            self.train_ctx.processed_tokens = config_checkpoint['processed_tokens']
        
        if 'allamo_dataloader' in config_checkpoint and self.data_loader is not None:
            if  'train_processed_files' in config_checkpoint['allamo_dataloader']:
                self.data_loader.train_dataset.processed_files = config_checkpoint['allamo_dataloader']['train_processed_files']
                if len(self.data_loader.train_dataset.processed_files) > 0:
                    # Removing the last element from the list because it represents the file where processing was interrupted.
                    # We will load this file and resume processing from there, indicated by the dataset_offset.
                    self.data_loader.train_dataset.processed_files.pop()
            if 'dataset_offset' in config_checkpoint['allamo_dataloader']:
                self.data_loader.dataset_offset = config_checkpoint['allamo_dataloader']['dataset_offset'] // self.train_ctx.dp
            if 'epoch' in config_checkpoint['allamo_dataloader']:
                self.data_loader.epoch = config_checkpoint['allamo_dataloader']['epoch']
                
    def is_checkpoint_available(self):
        return self.checkpoint_name is not None

    def load_regular_model_checkpoint(self, model):
        model_ckpt_file_path = get_model_checkpoint_path(self.checkpoint_name, self.checkpoint_dir)
        state_dict = torch.load(model_ckpt_file_path, map_location='cpu')
        remove_unwanted_prefix_from_model_state_dict(state_dict)
        model.load_state_dict(state_dict)
        if self.config.log_checkpoint_md5_on_load and self.train_ctx.master_process:
            md5sum = calculate_md5(model_ckpt_file_path)
            logger.info(f"Model state loaded from checkpoint {model_ckpt_file_path} - MD5: {md5sum}")
        else:
            logger.info(f"Model state loaded from checkpoint {model_ckpt_file_path}")

    def save_config_checkpoint(self, config_ckpt_file_path, md5sum, model_config):
        if not self.train_ctx.master_process:
            return
        
        checkpoint = {
            'model_args': dataclasses.asdict(model_config),
            'run_uuid': self.train_ctx.run_uuid,
            'training_uuid': self.train_ctx.training_uuid,
            'iter_num': self.train_ctx.iter_num,
            'best_train_loss': self.train_ctx.best_train_loss,
            'best_val_loss': self.train_ctx.best_val_loss,
            'processed_tokens': self.train_ctx.processed_tokens,
            'config': dataclasses.asdict(self.config),
            'allamo_dataloader': {
                'train_processed_files': self.data_loader.train_dataset.processed_files,
                'dataset_offset': self.data_loader.dataset_offset * self.train_ctx.dp,
                'epoch': self.data_loader.epoch
            }
        }
        if md5sum is not None:
            checkpoint['checkpoint_md5sum'] = md5sum
            logger.info(f"model checkpoint saved - MD5: {md5sum}")
        
        logger.info(f"saving config checkpoint to {config_ckpt_file_path}")
        if not self.config.ignore_last_checkpoint_backup:
            rename_file_to_prev_version(config_ckpt_file_path)
        with open(config_ckpt_file_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=4, ensure_ascii=False)
    
    def save_regular_model_checkpoint(self, state_dict, model_ckpt_file_path, epoch_ckpt):
        md5sum = None
        logger.info(f"saving model checkpoint to {model_ckpt_file_path}")
        if not self.config.ignore_last_checkpoint_backup:
            rename_file_to_prev_version(model_ckpt_file_path)
        torch.save(state_dict, model_ckpt_file_path)
        logger.info(f"model checkpoint saved in {model_ckpt_file_path}")
        
        if epoch_ckpt and self.config.log_checkpoint_md5_on_epoch:
            md5sum = calculate_md5(model_ckpt_file_path)
        return md5sum
    
    def should_save_optimizer(self):
        return self.config.save_optimizer_checkpoint and \
            (self.config.optimizer_checkpoint_interval is None or \
             self.train_ctx.iter_num % self.config.optimizer_checkpoint_interval == 0)
    
    def save_regular_optimizer_checkpoint(self, state_dict, optim_ckpt_file_path):
        logger.info(f"saving optimizer checkpoint to {optim_ckpt_file_path}")
        if not self.config.ignore_last_checkpoint_backup:
            rename_file_to_prev_version(optim_ckpt_file_path)
        torch.save(state_dict, optim_ckpt_file_path)
        logger.info(f"optimizer checkpoint saved in {optim_ckpt_file_path}")
        
    def load_distributed_model_checkpoint_from(self, model, checkpoint_dir, checkpoint_name):
        model_ckpt_dir_path = os.path.join(checkpoint_dir, f'model_{checkpoint_name}')
        assert os.path.exists(model_ckpt_dir_path) and os.path.isdir(model_ckpt_dir_path), "Model distributed checkpoint not found"
        model_state = {"model": ModelWrapper(model)}
        dcp.load(model_state, checkpoint_id=model_ckpt_dir_path)
        logger.info(f"Model state loaded from distributed checkpoint {model_ckpt_dir_path}")

    def load_distributed_model_checkpoint(self, model):
        self.load_distributed_model_checkpoint_from(model, self.checkpoint_dir, self.checkpoint_name)
        
    def load_distributed_optimizer_checkpoint_from(self, model, optimizer, checkpoint_dir, checkpoint_name):
        optimizer_ckpt_dir_path = os.path.join(checkpoint_dir, f'optimizer_{checkpoint_name}')
        if os.path.exists(optimizer_ckpt_dir_path) and os.path.isdir(optimizer_ckpt_dir_path):
            optimizer_state = {"optimizer": OptimizerWrapper(model, optimizer)}
            dcp.load(optimizer_state, checkpoint_id=optimizer_ckpt_dir_path)
            logger.info(f"Optimizer state loaded from distributed checkpoint {optimizer_ckpt_dir_path}")
        else:
            logger.warning("Optimizer distributed checkpoint not found. Initializing optimizer from scratch")
    
    def load_distributed_optimizer_checkpoint(self, model, optimizer):
        self.load_distributed_optimizer_checkpoint_from(model, optimizer, self.checkpoint_dir, self.checkpoint_name)
    
    def save_distributed_model_checkpoint_to(self, model, checkpoint_dir, checkpoint_name):
        model_ckpt_dir_path = os.path.join(checkpoint_dir, f'model_{checkpoint_name}')
        logger.info(f"Saving model distributed checkpoint to {model_ckpt_dir_path}")
        model_state = {"model": ModelWrapper(model)}
        dcp.save(model_state, checkpoint_id=model_ckpt_dir_path)
        logger.info(f"Model distributed checkpoint saved in {model_ckpt_dir_path}")
        return model_ckpt_dir_path
    
    def save_distributed_model_checkpoint(self, model, checkpoint_name):
        return self.save_distributed_model_checkpoint_to(model, self.config.out_dir, checkpoint_name)
    
    def save_distributed_optimizer_checkpoint_to(self, model, optimizer, checkpoint_dir, checkpoint_name):
        optimizer_ckpt_dir_path = os.path.join(checkpoint_dir, f'optimizer_{checkpoint_name}')
        logger.info(f"Saving optimizer distributed checkpoint to {optimizer_ckpt_dir_path}")
        optimizer_state = {"optimizer": OptimizerWrapper(model, optimizer)}
        dcp.save(optimizer_state, checkpoint_id=optimizer_ckpt_dir_path)
        logger.info(f"Optimizer distributed checkpoint saved in {optimizer_ckpt_dir_path}")
    
    def save_distributed_optimizer_checkpoint(self, model, optimizer, checkpoint_name):
        self.save_distributed_optimizer_checkpoint(model, optimizer, self.config.out_dir, checkpoint_name)
    

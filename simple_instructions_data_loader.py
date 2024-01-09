"""
Poor man's instructions data loader

The expected format of the file `instructions.pt` is `[tensor1, tensor2, ..., tensorN]`, 
where `tensorX` contains the tokens of X'th instruction.
"""
import logging
import numpy as np
import os
import random
import time
import torch
from configuration import AllamoConfiguration

class SimpleInstructionsDataLoader:

    def __init__(self, config: AllamoConfiguration, ddp_rank=None, ddp_world_size=None):
        self.logger = logging.getLogger('AllamoSimpleInstructionsDataLoader')
        self.config = config
        self.epoch = 0
        self.rank = ddp_rank if ddp_rank is not None else 0
        self.world_size = ddp_world_size if ddp_world_size is not None else 1
        
        self.batch_size = config.batch_size
        if self.batch_size > 1:
            self.logger.warn("Batch Size is greater than 1. Be careful, padding is not supported!")
        self.dataset_train_x_start = config.dataset_seq_train_start if config.dataset_seq_train_start is not None else 0
        
        self.__load_datasets()
        self.logger.info(f"Training dataset loaded. Size: {self.train_data_size:,} instructions")
        if self.val_data is None:
            self.splits = ['train']
            self.logger.info(f"Val dataset is missing. Testing only on the train dataset")
        else:
            self.splits = ['train', 'val']
            self.logger.info(f"Val dataset loaded. Size: {self.val_data_size:,} instructions")
        
    def __load_datasets(self):
        data_dir = os.path.join(self.config.data_dir, self.config.dataset)
        train_data_path = os.path.join(data_dir, 'train-instructions.pt')
        self.train_data = torch.load(train_data_path)
        self.train_data_size = len(self.train_data)
        self.__truncate_long_instructions(self.train_data)
        if self.config.dataset_seq_train:
            # start from the longest instructions to optimize training
            self.train_data.sort(key=len, reverse=True)
        
        val_data_path = os.path.join(data_dir, 'val-instructions.pt')
        if os.path.exists(val_data_path):
            self.val_data = torch.load(val_data_path)
            self.val_data_size = len(self.val_data)
            self.__truncate_long_instructions(self.val_data)
        else:
            self.val_data = None
            
    def __truncate_long_instructions(self, data):
        max_length = self.config.block_size + 1
        for idx in range(len(data)):
            data[idx] = data[idx][:max_length].to(torch.int64)
            
    def reload_datasets(self):
        timer = time.time()
        del self.train_data
        if self.val_data is not None:
            del self.val_data
        
        self.__load_datasets()
        dt = time.time() - timer
        self.logger.info(f"Datasets reloaded in {dt*1000:.2f}ms")
            
    def get_splits(self):
        return self.splits
        
    def get_batch(self, split='train', random_samples=False):
        if split == 'train' or self.val_data is None:
            data = self.train_data
            data_size = self.train_data_size
        else:
            data = self.val_data
            data_size = self.val_data_size
        seq_samples = random_samples == False and split == 'train' and self.config.dataset_seq_train
        idx_batch = []
        min_sample_length = self.config.block_size + 1
        idx = self.dataset_train_x_start + self.rank
        for _ in range(self.batch_size):
            sample_idx = None
            if seq_samples:
                sample_idx = idx
                idx += self.world_size
                if idx >= data_size:
                    idx = 0
            else:
                sample_idx = random.randint(0, len(data) - 1)
            
            idx_batch.append(sample_idx)
            if len(data[sample_idx]) < min_sample_length:
                min_sample_length = len(data[sample_idx])
                
        if seq_samples:
            self.dataset_train_x_start += self.world_size * self.batch_size
            if self.dataset_train_x_start >= data_size:
                self.epoch += 1
                self.dataset_train_x_start = 0
                self.logger.info(f"Starting new epoch: {self.epoch}")
        
        x = torch.stack([data[idx][:min_sample_length-1] for idx in idx_batch])
        y = torch.stack([data[idx][1:min_sample_length] for idx in idx_batch])
        if 'cuda' in self.config.device:
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(self.config.device, non_blocking=True), y.pin_memory().to(self.config.device, non_blocking=True)
        else:
            x, y = x.to(self.config.device), y.to(self.config.device)
        return x, y
        
    def update_batch_size(self, iter_num):
        if self.config.batch_size_schedule and self.batch_size < self.config.batch_size_max:
            self.batch_size = min(self.batch_size + 1, self.config.batch_size_max) if iter_num % (self.config.batch_size_max_iter/100) == 0 else self.batch_size 
        return self.batch_size
        

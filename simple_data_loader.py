"""
Very simple, poor man's data loader
"""
import logging
import numpy as np
import os
import random
import time
import torch
from configuration import AllamoConfiguration

class SimpleDataLoader:

    def __init__(self, config: AllamoConfiguration, ddp_rank=None, ddp_world_size=None):
        self.logger = logging.getLogger('AllamoSimpleDataLoader')
        self.config = config
        self.epoch = 0
        self.rank = ddp_rank if ddp_rank is not None else 0
        self.world_size = ddp_world_size if ddp_world_size is not None else 1
        self.pin_memory = True
        
        if config.batch_size_schedule: 
            self.config.batch_size_max = config.batch_size
            self.batch_size = config.batch_size_initial
        else:
            self.batch_size = config.batch_size
        
        if self.config.dataset_seq_train:
            self.dataset_train_x_start = config.dataset_seq_train_start if config.dataset_seq_train_start is not None else 0
            if config.dataset_seq_step_size is None:
                config.dataset_seq_step_size = config.block_size
                self.logger.info(f"Sequential step set to {config.block_size:,} tokens")
            self.logger.info(f"Training dataset offset set to {self.dataset_train_x_start:,} tokens")
        else:
            self.dataset_train_x_start = 0
        
        self.__load_datasets()
        self.logger.info(f"Training dataset loaded. Size: {self.train_data_size:,} tokens")
        if self.val_data is None:
            self.splits = ['train']
            self.logger.info(f"Val dataset is missing. Testing only on the train dataset")
        else:
            self.splits = ['train', 'val']
            self.logger.info(f"Val dataset loaded. Size: {self.val_data_size:,} tokens")
        
    def __load_datasets(self):
        data_dir = os.path.join(self.config.data_dir, self.config.dataset)
        train_data_path = os.path.join(data_dir, 'train.bin')
        if self.config.in_memory_data:
            self.train_data = np.fromfile(train_data_path, dtype=np.uint16)
        else:
            self.train_data = np.memmap(train_data_path, dtype=np.uint16, mode='r')
        self.train_data_size = len(self.train_data)
        
        val_data_path = os.path.join(data_dir, 'val.bin')
        if os.path.exists(val_data_path):
            if self.config.in_memory_data:
                self.val_data = np.fromfile(val_data_path, dtype=np.uint16)
            else:
                self.val_data = np.memmap(val_data_path, dtype=np.uint16, mode='r')
            self.val_data_size = len(self.val_data)
        else:
            self.val_data = None
            
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
        if random_samples == False and split == 'train' and self.config.dataset_seq_train:
            idx_batch = torch.zeros(self.batch_size, dtype=torch.int64)
            idx = self.dataset_train_x_start + self.rank * self.config.dataset_seq_step_size
            for i in range(self.batch_size):
                if idx+self.config.block_size >= data_size:
                    idx = self.rank * self.config.dataset_seq_step_size + random.randint(0, self.config.block_size)
                idx_batch[i] = idx
                idx += self.world_size * self.config.dataset_seq_step_size
            self.dataset_train_x_start += self.world_size * self.batch_size * self.config.dataset_seq_step_size
            if self.dataset_train_x_start >= data_size:
                self.epoch += 1
                self.dataset_train_x_start = 0
                self.logger.info(f"Epoch {self.epoch} finished")
        else:
            idx_batch = torch.randint(data_size - self.config.block_size, (self.batch_size,))
        
        x = torch.stack([torch.from_numpy(data[i:i+self.config.block_size].astype(np.int64)) for i in idx_batch])
        y = torch.stack([torch.from_numpy(data[i+1:i+1+self.config.block_size].astype(np.int64)) for i in idx_batch])
        if 'cuda' in self.config.device and self.pin_memory:
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(self.config.device, non_blocking=True), y.pin_memory().to(self.config.device, non_blocking=True)
        else:
            x, y = x.to(self.config.device), y.to(self.config.device)
        return x, y
        
    def update_batch_size(self, iter_num):
        if self.config.batch_size_schedule and self.batch_size < self.config.batch_size_max:
            self.batch_size = min(self.batch_size + 1, self.config.batch_size_max) if iter_num % (self.config.batch_size_max_iter/100) == 0 else self.batch_size 
        return self.batch_size
        

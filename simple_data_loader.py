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

    def __init__(self, config: AllamoConfiguration):
        self.logger = logging.getLogger('AllamoSimpleDataLoader')
        self.config = config
        self.epoch = 0
        
        if config.batch_size_schedule: 
            self.config.batch_size_max = config.batch_size
            self.batch_size = config.batch_size_initial
        else:
            self.batch_size = config.batch_size
            
        if self.config.dataset_seq_train:
            self.dataset_train_x_start = config.dataset_seq_train_start if config.dataset_seq_train_start is not None else random.randint(0, self.config.block_size)
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
            ix = torch.zeros(self.batch_size, dtype=torch.int64)
            end_of_batch = self.dataset_train_x_start + (self.batch_size-1) * self.config.dataset_seq_step_size + self.config.block_size + 1 >= data_size
            if end_of_batch:
                # align to the right
                self.dataset_train_x_start = data_size - ((self.batch_size-1) * self.config.dataset_seq_step_size + self.config.block_size + 1)

            for i in range(self.batch_size):
                last_x_start = self.dataset_train_x_start + i * self.config.dataset_seq_step_size
                ix[i] = last_x_start
                
            if end_of_batch:
                self.epoch += 1
                self.logger.info(f"Staring new epoch: {self.epoch}")
                self.dataset_train_x_start = random.randint(0, self.batch_size-1)
            else:    
                self.dataset_train_x_start = last_x_start + self.config.dataset_seq_step_size 
        else:
            ix = torch.randint(data_size - self.config.block_size, (self.batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i+self.config.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i+1:i+1+self.config.block_size].astype(np.int64)) for i in ix])
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
        

"""
Very simple, poor man's data loader
"""
import numpy as np
import os
import random
import torch
from configuration import AllamoConfiguration

class SimpleDataLoader:

    def __init__(self, config: AllamoConfiguration):
        self.config = config
        self.epoch = 0
        
        if config.batch_size_schedule: 
            self.batch_size_max = config.batch_size
            self.batch_size = config.batch_size_initial
        else:
            self.batch_size = config.batch_size
            
        data_dir = os.path.join(config.data_dir, config.dataset)
        self.dataset_train_x_start = config.dataset_seq_train_start if config.dataset_seq_train_start is not None else random.randint(0, self.batch_size-1)
        self.train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        val_data_path = os.path.join(data_dir, 'val.bin')
        if os.path.exists(val_data_path):
            self.val_data = np.memmap(val_data_path, dtype=np.uint16, mode='r')
            self.splits = ['train', 'val']
        else:
            self.val_data = None
            self.splits = ['train']
            print(f"Val dataset is missing. Testing only on the train dataset")
            
    def get_splits(self):
        return self.splits
        
    def get_batch(self, split='train', random_samples=False):
        data = self.train_data if split == 'train' or val_data is None else self.val_data
        data_size = len(data)
        if random_samples == False and split == 'train' and self.config.dataset_seq_train:
            ix = torch.zeros(self.batch_size, dtype=torch.int)
            end_of_batch = self.dataset_train_x_start + (self.batch_size-1) * self.config.dataset_seq_step_size + self.config.block_size + 1 >= data_size
            if end_of_batch:
                # align to the right
                self.dataset_train_x_start = data_size - ((self.batch_size-1) * self.config.dataset_seq_step_size + self.config.block_size + 1)

            for i in range(self.batch_size):
                last_x_start = self.dataset_train_x_start + i * self.config.dataset_seq_step_size
                ix[i] = last_x_start
                
            if end_of_batch:
                self.epoch += 1
                print(f"Staring new epoch: {self.epoch}")
                self.dataset_train_x_start = random.randint(0, self.batch_size-1)
            else:    
                self.dataset_train_x_start = last_x_start + self.config.dataset_seq_step_size 
        else:
            ix = torch.randint(data_size - self.config.block_size, (self.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+self.config.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+self.config.block_size]).astype(np.int64)) for i in ix])
        if 'cuda' in self.config.device:
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(self.config.device, non_blocking=True), y.pin_memory().to(self.config.device, non_blocking=True)
        else:
            x, y = x.to(self.config.device), y.to(self.config.device)
        return x, y
        
    def update_batch_size(self, iter_num):
        if self.config.batch_size_schedule:
            self.batch_size = min(self.batch_size + 1, self.config.batch_size_max) if iter_num % (self.config.batch_size_max_iter/100) == 0 else self.batch_size 
        return self.batch_size
        
        

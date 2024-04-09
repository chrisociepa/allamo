import gc
import glob
import logging
import os
import random
import time
import torch
from configuration import AllamoConfiguration

class AllamoDataset:
    """ In-Memory map-style dataset """

    def __init__(self, config: AllamoConfiguration, train_split=True, rank=None, world_size=None):
        self.logger = logging.getLogger('AllamoDataset')
        self.rank = rank
        self.world_size = world_size
        self.data_dir = config.data_dir
        self.sample_size = config.block_size + 1 
        self.ignore_index = config.ignore_index
        self.pad_token_id = config.pad_token_id
        self.data = None
        self.dataset_files = self.get_dataset_files(config, train_split)
        self.processed_files = []
        if config.dataset_train_processed_files_count > 0:
            self.processed_files = self.dataset_files[:config.dataset_train_processed_files_count]
        self.load_next_dataset()
        
    def get_dataset_files(self, config, train_split):
        dataset_files = []
        if train_split and config.dataset_train_files:
            dataset_files = config.dataset_train_files.split(',')
        elif not train_split and config.dataset_validation_files:
            dataset_files = config.dataset_validation_files.split(',')
        elif config.dataset:
            dataset_dir = os.path.join(config.data_dir, config.dataset)
            prefix = config.dataset_train_file_prefix if train_split else config.dataset_validation_file_prefix
            for dataset_file in glob.glob(os.path.join(dataset_dir, "*.*")):
                if self.is_file_type_supported(dataset_file) and os.path.basename(dataset_file).startswith(prefix):
                    dataset_files.append(dataset_file)
            self.logger.info(f"Found {len(dataset_files)} files in {dataset_dir} with prefix '{prefix}'")
        if dataset_files:
            return sorted(dataset_files)
        elif train_split:
            raise Exception('Training dataset files not found!')
        else:
            return []
    
    def is_file_type_supported(self, dataset_file):
        return dataset_file.endswith('.bin') or dataset_file.endswith('.pt')
    
    def load_next_dataset(self):
        self.data = None
        gc.collect()
        for ds_file in self.dataset_files:
            if ds_file not in self.processed_files:
                if self.load_dataset_file(ds_file):
                    return True
        return False
                
    def load_dataset_file(self, load_dataset_file):
        self.processed_files.append(load_dataset_file)
        new_data = None
        if load_dataset_file.endswith('.bin'):
            import numpy as np
            step_size = self.world_size * self.sample_size
            new_data = torch.from_numpy(np.fromfile(load_dataset_file, dtype=np.uint16).astype(np.int16))
            if step_size > len(new_data):
                self.logger.warning(
                    f"Dataset file {load_dataset_file} does not have enough data and will be ignored. "
                    f"Expected at least {step_size} tokens but found only {len(new_data)}"
                )
                return False
            new_data = self.align_data_to_step_size(new_data, step_size)
            new_data = self.transform_continuous_data_to_samples(new_data)
            new_data = self.limit_samples_to_rank(new_data)
        elif load_dataset_file.endswith('.pt'):
            new_data = torch.load(load_dataset_file, map_location='cpu')
            if isinstance(new_data, torch.Tensor):
                step_size = self.world_size * self.sample_size
                if step_size > len(new_data):
                    self.logger.warning(
                        f"Dataset file {load_dataset_file} does not have enough data and will be ignored. "
                        f"Expected at least {step_size} tokens but found only {len(new_data)}"
                    )
                    return False
                new_data = self.align_data_to_step_size(new_data, step_size)
                new_data = self.transform_continuous_data_to_samples(new_data)
                new_data = self.limit_samples_to_rank(new_data)
            elif isinstance(new_data, list):
                if self.world_size > len(new_data):
                    self.logger.warning(
                        f"Dataset file {load_dataset_file} does not have enough data and will be ignored. "
                        f"Expected at least {self.world_size} samples but found only {len(new_data)}"
                    )
                    return False
                new_data = self.align_data_to_step_size(new_data, self.world_size)
                self.pad_or_truncate_to_block_size(new_data)
                new_data = self.limit_samples_to_rank(new_data)
            else:
                self.logger.info(f"Unsupported format of {load_dataset_file}!")
                new_data = None
                
        if new_data:
            self.data = new_data
            self.logger.info(f"New dataset file {load_dataset_file} loaded. Processed files: {len(self.processed_files)}")
            return True
        else:
            return False
    
    def align_data_to_step_size(self, data, step_size):
        target_length = ((len(data) + step_size - 1) // step_size) * step_size
        padding_length = target_length - len(data)
        return data + data[:padding_length] if isinstance(data, list) else torch.concat((data, data[:padding_length]))
        
    def transform_continuous_data_to_samples(self, data):
        return [data[i:i + self.sample_size] for i in range(0, len(data), self.sample_size)]
        
    def pad_or_truncate_to_block_size(self, data):
        """
        Adds padding to instructions to maintain a consistent input shape, avoiding recompilations.
        This method ensures all instructions have a uniform length matching the block size.
        By doing so, it prevents the need for frequent recompilations that occur due to
        dynamic input shapes, enhancing computational efficiency and stability.
        """
        for idx in range(len(data)):
            if isinstance(data[idx], dict):
                if 'input_ids' not in data[idx]:
                    raise Exception(f"'input_id' field not found in sample! Available keys: {', '.join(data[idx].keys())}")
                if 'target_ids' not in data[idx]:
                    data[idx]['target_ids'] = data[idx]['input_ids'][1:]
                if 'target_weights' not in data[idx]:
                    data[idx]['target_weights'] = torch.where(data[idx]['target_ids'] == self.ignore_index, 0, 1)
                    
                if len(data[idx]['input_ids']) >= self.sample_size: # block_size = sample_size - 1
                    data[idx]['input_ids'] = data[idx]['input_ids'][:self.sample_size-1]
                elif self.pad_token_id >= 0 and len(data[idx]['input_ids']) < self.sample_size-1:
                    padding = self.sample_size - 1 - len(data[idx]['input_ids'])
                    data[idx]['input_ids'] = torch.cat([data[idx]['input_ids'], torch.full((padding,), self.ignore_index)], dim=0)
                
                if len(data[idx]['target_ids']) >= self.sample_size:
                    data[idx]['target_ids'] = data[idx]['target_ids'][:self.sample_size-1]
                elif self.pad_token_id >= 0 and len(data[idx]['target_ids']) < self.sample_size-1:
                    padding = self.sample_size - 1 - len(data[idx]['target_ids'])
                    data[idx]['target_ids'] = torch.cat([data[idx]['target_ids'], torch.full((padding,), self.ignore_index)], dim=0)
                    
                if len(data[idx]['target_weights']) >= self.sample_size:
                    data[idx]['target_weights'] = data[idx]['target_weights'][:self.sample_size-1]
                elif self.pad_token_id >= 0 and len(data[idx]['target_weights']) < self.sample_size-1:
                    padding = self.sample_size - 1 - len(data[idx]['target_weights'])
                    data[idx]['target_weights'] = torch.cat([data[idx]['target_weights'], torch.full((padding,), 0)], dim=0)
                
                if 'target_mask' in data[idx]:
                    if len(data[idx]['target_mask']) >= self.sample_size:
                        data[idx]['target_mask'] = data[idx]['target_mask'][:self.sample_size-1]
                    elif self.pad_token_id >= 0 and len(data[idx]['target_mask']) < self.sample_size-1:
                        padding_value = False if isinstance(data[idx]['target_mask'][0].item(), bool) else 0
                        padding = self.sample_size - 1 - len(data[idx]['target_mask'])
                        data[idx]['target_mask'] = torch.cat([data[idx]['target_mask'], torch.full((padding,), padding_value)], dim=0)
                    data[idx]['target_ids'] = data[idx]['target_ids'].masked_fill(data[idx]['target_mask'] == 0, self.ignore_index)
                    data[idx]['target_weights'] = data[idx]['target_weights'].masked_fill(data[idx]['target_mask'] == 0, 0)
                    del data[idx]['target_mask']
                assert len(data[idx]['input_ids']) == len(data[idx]['target_ids'])
            else:
                if len(data[idx]) > self.sample_size:
                    data[idx] = data[idx][:self.sample_size]
                if self.pad_token_id >= 0:
                    if len(data[idx]) < self.sample_size:
                        padding = self.sample_size - len(data[idx])
                        data[idx] = torch.cat([data[idx], torch.full((padding,), self.ignore_index)], dim=0)
                    input_ids = data[idx][:-1]
                    target_ids = data[idx][1:]
                    target_weights = torch.where(target_ids == self.ignore_index, 0, 1)
                    input_ids = input_ids.masked_fill(input_ids == self.ignore_index, self.pad_token_id)
                    data[idx] = {'input_ids': input_ids, 'target_ids': target_ids, 'target_weights': target_weights}
        
    def limit_samples_to_rank(self, samples):
        return samples[self.rank::self.world_size] if self.world_size > 1 else samples
        
    def has_data(self):
        return self.data and len(self.data) > 0
    
    def __len__(self):
        """ Size of currently loaded dataset file """
        return len(self.data) if self.data else 0
        
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.data[idx]
        if idx < self.__len__():
            return self.data[idx]
        return None
        

class AllamoDataLoader:

    def __init__(self, config: AllamoConfiguration, rank=None, world_size=None):
        self.logger = logging.getLogger('AllamoDataLoader')
        self.config = config
        self.epoch = 0
        self.rank = rank if rank is not None else 0
        self.world_size = world_size if world_size is not None else 1
        self.pin_memory = True
        
        if config.batch_size_schedule: 
            self.config.batch_size_max = config.batch_size
            self.batch_size = config.batch_size_initial
        else:
            self.batch_size = config.batch_size
        
        if self.config.dataset_seq_train:
            self.dataset_offset = config.dataset_seq_train_start if config.dataset_seq_train_start is not None else 0
            if config.dataset_seq_step_size is not None:
                self.logger.warning(f"Sequential step is not supported in this data loader")
        else:
            self.dataset_offset = 0
        self.logger.info(f"Training dataset offset set to {self.dataset_offset:,}")
        
        self.load_datasets()
            
    def load_datasets(self):
        timer = time.time()
        self.train_dataset = AllamoDataset(self.config, True, self.rank, self.world_size)
        self.splits = ['train']
        self.logger.info(f"Training dataset created with files: {','.join(self.train_dataset.dataset_files)}")
        self.logger.info(f"Training samples loaded: {(len(self.train_dataset)*self.world_size):,}")
        
        self.val_dataset = AllamoDataset(self.config, False, self.rank, self.world_size)
        if self.val_dataset.has_data():
            self.splits.append('val')
            self.logger.info(f"Validation dataset created with files: {','.join(self.val_dataset.dataset_files)}")
            self.logger.info(f"Validation samples loaded: {(len(self.val_dataset)*self.world_size):,}")
        else:
            self.val_dataset = None
            self.logger.info(f"Validation dataset is missing. Testing only on the training dataset")
        dt = time.time() - timer
        self.logger.info(f"Datasets loaded in {dt:.2f} secs")
        
    def get_splits(self):
        return self.splits
        
    def get_batch(self, split='train', random_samples=False):
        if split == 'train' or self.val_dataset is None:
            dataset = self.train_dataset
        else:
            dataset = self.val_dataset
        
        if random_samples == False and split == 'train' and self.config.dataset_seq_train:
            if self.dataset_offset + self.batch_size <= len(dataset):
                samples = dataset[self.dataset_offset:self.dataset_offset+self.batch_size]
                self.dataset_offset += self.batch_size
            else:
                samples = []
                for _ in range(self.batch_size):
                    if self.dataset_offset >= len(dataset):
                        self.reload_dataset(dataset)
                    samples.append(dataset[self.dataset_offset])
                    self.dataset_offset += 1
        else:
            idx_batch = torch.randint(len(dataset), (self.batch_size,))
            samples = [dataset[i] for i in idx_batch]
            
        if isinstance(samples[0], dict):
            x = torch.stack([sample['input_ids'] for sample in samples]).to(torch.int64)
            y = torch.stack([sample['target_ids'] for sample in samples]).to(torch.int64)
            w = torch.stack([sample['target_weights'] for sample in samples]).to(torch.int64) if self.config.weighted_loss else None
        else:
            x = torch.stack([sample[:-1] for sample in samples]).to(torch.int64)
            y = torch.stack([sample[1:] for sample in samples]).to(torch.int64)
            w = None
        
        if 'cuda' in self.config.device and self.pin_memory:
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(self.config.device, non_blocking=True), y.pin_memory().to(self.config.device, non_blocking=True)
            if w is not None:
                w = w.pin_memory().to(self.config.device, non_blocking=True)
        else:
            x, y = x.to(self.config.device), y.to(self.config.device)
            if w is not None:
                w = w.to(self.config.device)
        return x, y, w
        
    def reload_dataset(self, dataset):
        if len(dataset.dataset_files) > 1:
            if dataset.load_next_dataset():
                # Epoch is not finished, we've just loaded next dataset file
                self.dataset_offset = 0
                return
            else:
                dataset.processed_files.clear()
                assert dataset.load_next_dataset(), 'Something very bad has happend and we are unable to reload dataset'
        self.dataset_offset = 0
        self.epoch += 1
        self.logger.info(f"Epoch {self.epoch} finished")
        
    def reload_datasets(self):
        # only for backward compatibility with SimpleDataLoader
        self.load_datasets()
        
    def update_batch_size(self, iter_num):
        if self.config.batch_size_schedule and self.batch_size < self.config.batch_size_max:
            self.batch_size = min(self.batch_size + 1, self.config.batch_size_max) if iter_num % (self.config.batch_size_max_iter/100) == 0 else self.batch_size 
        return self.batch_size
        

import itertools
import threading
import time
import torch
from allamo.configuration import AllamoConfiguration
from allamo.dataset.dataset import AllamoDataset
from allamo.logging import logger

class AllamoDataLoader:

    def __init__(self, config: AllamoConfiguration, rank=None, world_size=None):
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
        else:
            self.dataset_offset = 0
        logger.info(f"Training dataset offset set to {self.dataset_offset:,}")
        
        self.init_datasets()
        self.buffer = None
        self.buffer_lock = threading.Lock()
        self.buffer_thread = None
            
    def init_datasets(self):
        self.train_dataset = AllamoDataset(self.config, True, self.rank, self.world_size)
        self.splits = ['train']
        logger.info(f"Training dataset initialized with files: {','.join(self.train_dataset.dataset_files)}")
        
        self.val_dataset = AllamoDataset(self.config, False, self.rank, self.world_size)
        if self.val_dataset.dataset_files:
            self.splits.append('val')
            logger.info(f"Validation dataset initialized with files: {','.join(self.val_dataset.dataset_files)}")
        else:
            self.val_dataset = None
            logger.info(f"Validation dataset is missing. Testing only on the training dataset")
        
    def load_datasets(self):
        logger.info(f"Loading dataset samples")
        timer = time.time()
        self.train_dataset.load_next_dataset()
        logger.info(f"Training samples loaded: {(len(self.train_dataset)*self.world_size):,}")
        
        if self.val_dataset is not None:
            self.val_dataset.load_next_dataset()
            logger.info(f"Validation samples loaded: {(len(self.val_dataset)*self.world_size):,}")
        dt = time.time() - timer
        logger.info(f"Datasets loaded in {dt:.2f} secs")
        
    def get_splits(self):
        return self.splits
    
    def prepare_dpo_samples(self, samples):
        chosen_input_ids = torch.stack([sample['chosen_input_ids'] for sample in samples]).to(torch.int64)
        chosen_target_ids = torch.stack([sample['chosen_target_ids'] for sample in samples]).to(torch.int64)
        rejected_input_ids = torch.stack([sample['rejected_input_ids'] for sample in samples]).to(torch.int64)
        rejected_target_ids = torch.stack([sample['rejected_target_ids'] for sample in samples]).to(torch.int64)
        reference_chosen_logps = torch.stack([sample['reference_chosen_logps'] for sample in samples]).to(torch.float32) if 'reference_chosen_logps' in samples[0] else None
        reference_rejected_logps = torch.stack([sample['reference_rejected_logps'] for sample in samples]).to(torch.float32) if 'reference_rejected_logps' in samples[0] else None
        
        if 'cuda' in self.config.device and self.pin_memory:
            chosen_input_ids = chosen_input_ids.pin_memory().to(self.config.device, non_blocking=True)
            chosen_target_ids = chosen_target_ids.pin_memory().to(self.config.device, non_blocking=True)
            rejected_input_ids = rejected_input_ids.pin_memory().to(self.config.device, non_blocking=True)
            rejected_target_ids = rejected_target_ids.pin_memory().to(self.config.device, non_blocking=True)
            if reference_chosen_logps is not None:
                reference_chosen_logps = reference_chosen_logps.pin_memory().to(self.config.device, non_blocking=True)
            if reference_rejected_logps is not None:
                reference_rejected_logps = reference_rejected_logps.pin_memory().to(self.config.device, non_blocking=True)
        else:
            chosen_input_ids = chosen_input_ids.to(self.config.device)
            chosen_target_ids = chosen_target_ids.to(self.config.device)
            rejected_input_ids = rejected_input_ids.to(self.config.device)
            rejected_target_ids = rejected_target_ids.to(self.config.device)
            if reference_chosen_logps is not None:
                reference_chosen_logps = reference_chosen_logps.to(self.config.device)
            if reference_rejected_logps is not None:
                reference_rejected_logps = reference_rejected_logps.to(self.config.device)
        return {
            "chosen_input_ids": chosen_input_ids,
            "chosen_target_ids": chosen_target_ids,
            "rejected_input_ids": rejected_input_ids,
            "rejected_target_ids": rejected_target_ids,
            "reference_chosen_logps": reference_chosen_logps,
            "reference_rejected_logps": reference_rejected_logps
        }
    
    def prepare_samples(self, samples):
        if self.config.training_type == 'dpo':
            return self.prepare_dpo_samples(samples)
        
        if isinstance(samples[0], dict):
            input_ids = torch.stack([sample['input_ids'] for sample in samples]).to(torch.int64)
            target_ids = torch.stack([sample['target_ids'] for sample in samples]).to(torch.int64)
            target_weights = torch.stack([sample['target_weights'] for sample in samples]).to(torch.float32) if 'target_weights' in samples[0] else None
            attn_mask = torch.stack([sample['attn_mask'] for sample in samples]) if 'attn_mask' in samples[0] else None
            input_pos = torch.stack([sample['input_pos'] for sample in samples]) if 'input_pos' in samples[0] else None
            seq_lens = [sample["seq_lens"] for sample in samples] if 'seq_lens' in samples[0] else None
        else:
            input_ids = torch.stack([sample[:-1] for sample in samples]).to(torch.int64)
            target_ids = torch.stack([sample[1:] for sample in samples]).to(torch.int64)
            target_weights = None
            attn_mask = None
            input_pos = None
            seq_lens = None
        
        if 'cuda' in self.config.device and self.pin_memory:
            input_ids = input_ids.pin_memory().to(self.config.device, non_blocking=True)
            target_ids = target_ids.pin_memory().to(self.config.device, non_blocking=True)
            if target_weights is not None:
                target_weights = target_weights.pin_memory().to(self.config.device, non_blocking=True)
            if attn_mask is not None:
                attn_mask = attn_mask.pin_memory().to(self.config.device, non_blocking=True)
            if input_pos is not None:
                input_pos = input_pos.pin_memory().to(self.config.device, non_blocking=True)
        else:
            input_ids = input_ids.to(self.config.device)
            target_ids = target_ids.to(self.config.device)
            if target_weights is not None:
                target_weights = target_weights.to(self.config.device)
            if attn_mask is not None:
                attn_mask = attn_mask.to(self.config.device)
            if input_pos is not None:
                input_pos = input_pos.to(self.config.device)
        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "target_weights": target_weights,
            "attn_mask": attn_mask,
            "input_pos": input_pos,
            "seq_lens": seq_lens
        }
        
    def update_buffer(self, dataset):
        with self.buffer_lock:
            self.buffer = {
                "batch": self.prepare_samples(dataset[self.dataset_offset:self.dataset_offset+self.batch_size]),
                "offset": self.dataset_offset + self.batch_size
            }
    
    def reload_buffer(self, dataset):
        self.buffer = None
        if self.dataset_offset + self.batch_size <= len(dataset):
            self.buffer_thread = threading.Thread(target=self.update_buffer, args=(dataset,))
            self.buffer_thread.start()
        else:
            self.buffer_thread = None
            
    def get_batch_from_buffer(self, dataset):
        with self.buffer_lock:
            batch = self.buffer["batch"]
            self.dataset_offset = self.buffer["offset"]
        assert self.buffer_thread is None or not self.buffer_thread.is_alive()
        self.reload_buffer(dataset)
        return batch
        
    def get_batch(self, split='train', random_samples=False):
        if split == 'train' or self.val_dataset is None:
            dataset = self.train_dataset
        else:
            dataset = self.val_dataset
        
        if random_samples == False and split == 'train' and self.config.dataset_seq_train:
            if self.config.dataset_buffer and self.buffer is not None:
                return self.get_batch_from_buffer(dataset)
            elif self.dataset_offset + self.batch_size <= len(dataset):
                samples = dataset[self.dataset_offset:self.dataset_offset+self.batch_size]
                self.dataset_offset += self.batch_size
            else:
                samples = []
                for _ in range(self.batch_size):
                    if self.dataset_offset >= len(dataset):
                        self.reload_dataset(dataset)
                    samples.append(dataset[self.dataset_offset])
                    self.dataset_offset += 1
            self.reload_buffer(dataset)
        else:
            idx_batch = torch.randint(len(dataset), (self.batch_size,))
            samples = [dataset[i] for i in idx_batch]
            
        return self.prepare_samples(samples)
    
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
        logger.info(f"Epoch {self.epoch} finished")
        
    def update_batch_size(self, iter_num):
        if self.config.batch_size_schedule and self.batch_size < self.config.batch_size_max:
            self.batch_size = min(self.batch_size + 1, self.config.batch_size_max) if iter_num % (self.config.batch_size_max_iter/100) == 0 else self.batch_size 
        return self.batch_size
    
    def get_num_loaded_files(self):
        return len(self.train_dataset.processed_files)

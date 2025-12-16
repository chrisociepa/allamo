import gc
import glob
import joblib
import numpy as np
import os
import torch
import torch.nn.functional as F
from allamo.configuration import AllamoConfiguration
from allamo.logging import logger
from allamo.model.attentions import attention_version

class AllamoDataset:
    """ In-Memory map-style dataset """

    def __init__(self, config: AllamoConfiguration, train_split=True, rank=None, world_size=None):
        self.rank = rank
        self.world_size = world_size
        self.block_size = config.block_size
        self.sample_size = config.block_size + 1 
        self.ignore_index = config.ignore_index
        self.pad_token_id = config.pad_token_id
        self.weighted_loss = config.weighted_loss
        self.training_type = config.training_type
        if train_split:
            self.dataset_dir = config.dataset_train_dir
            self.dataset_files_lst = config.dataset_train_files
            self.dataset_file_prefix = config.dataset_train_file_prefix
        else:
            self.dataset_dir = config.dataset_validation_dir
            self.dataset_files_lst = config.dataset_validation_files
            self.dataset_file_prefix = config.dataset_validation_file_prefix

        self.data = None
        self.data_in_alm_format = False
        self.dataset_files = self.get_dataset_files()
        self.processed_files = []
        if train_split and config.dataset_train_processed_files_count > 0:
            self.processed_files = self.dataset_files[:config.dataset_train_processed_files_count]
        
    def get_dataset_files(self):
        dataset_files = []
        if self.dataset_files_lst:
            dataset_files = self.dataset_files_lst.split(',')
        elif self.dataset_dir:
            for dataset_file in glob.glob(os.path.join(self.dataset_dir, "*.*")):
                if self.is_file_type_supported(dataset_file) and (not self.dataset_file_prefix or os.path.basename(dataset_file).startswith(self.dataset_file_prefix)):
                    dataset_files.append(dataset_file)
            logger.info(f"Found {len(dataset_files)} dataset files in {self.dataset_dir}")
        return sorted(dataset_files)
    
    def is_file_type_supported(self, dataset_file):
        return dataset_file.endswith('.bin') or dataset_file.endswith('.pt') or dataset_file.endswith('.alm')
    
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
            assert self.training_type == 'pre', 'NumPy format is supported only for pre-training'
            step_size = self.world_size * self.sample_size
            new_data = torch.from_numpy(np.fromfile(load_dataset_file, dtype=np.uint16).astype(np.int16))
            if step_size > len(new_data):
                logger.warning(
                    f"Dataset file {load_dataset_file} does not have enough data and will be ignored. "
                    f"Expected at least {step_size} tokens but found only {len(new_data)}"
                )
                return False
            new_data = self.align_and_transform_continuous_data_to_samples(new_data, step_size)
            new_data = self.limit_samples_to_rank(new_data)
        elif load_dataset_file.endswith('.pt'):
            assert self.training_type != 'dpo', 'DPO training only supports the ALM format'
            new_data = torch.load(load_dataset_file, map_location='cpu', weights_only=True)
            if isinstance(new_data, torch.Tensor):
                step_size = self.world_size * self.sample_size
                if step_size > len(new_data):
                    logger.warning(
                        f"Dataset file {load_dataset_file} does not have enough data and will be ignored. "
                        f"Expected at least {step_size} tokens but found only {len(new_data)}"
                    )
                    return False
                new_data = self.align_and_transform_continuous_data_to_samples(new_data, step_size)
                new_data = self.limit_samples_to_rank(new_data)
            else:
                new_data = self.align_and_limit_to_rank(new_data, load_dataset_file)
                if new_data:
                    self.pad_or_truncate_to_block_size(new_data)
        elif load_dataset_file.endswith('.alm'):
            new_data = joblib.load(load_dataset_file)
            new_data = self.align_and_limit_to_rank(new_data, load_dataset_file)
        
        if new_data:
            self.data = new_data
            self.data_in_alm_format = load_dataset_file.endswith('.alm')
            logger.info(f"New dataset file {load_dataset_file} loaded. Processed files: {len(self.processed_files)}")
            gc.collect()
            return True
        else:
            return False
        
    def align_and_limit_to_rank(self, new_data, load_dataset_file):
        if isinstance(new_data, list):
            if self.world_size > len(new_data):
                logger.warning(
                    f"Dataset file {load_dataset_file} does not have enough data and will be ignored. "
                    f"Expected at least {self.world_size} samples but found only {len(new_data)}"
                )
                return None
            new_data = self.align_data_to_step_size(new_data, self.world_size)
            new_data = self.limit_samples_to_rank(new_data)
        else:
            logger.info(f"Unsupported format of {load_dataset_file}!")
            new_data = None
        return new_data
    
    def align_data_to_step_size(self, data, step_size):
        target_length = ((len(data) + step_size - 1) // step_size) * step_size
        padding_length = target_length - len(data)
        if padding_length > 0:
            pre_size = len(data)
            if isinstance(data, list):
                data.extend(data[:padding_length])
            else:
                # FIXME: this operation is highly inefficient - it duplicates data in memory
                data = torch.concat((data, data[:padding_length]))
            logger.info(f"Data aligned. Pre-alignment size: {pre_size}, "
                             f"post-alignment size: {len(data)}, "
                             f"padding added: {padding_length}")
        return data
        
    def align_and_transform_continuous_data_to_samples(self, data, step_size):
        target_length = ((len(data) + step_size - 1) // step_size) * step_size
        padding_length = target_length - len(data)
        if padding_length > 0:
            pre_size = len(data)
            result = [data[i:i + self.sample_size] for i in range(0, (target_length - step_size), self.sample_size)]
            data = torch.concat((data[(target_length - step_size):], data[:padding_length]))
            result.extend([data[i:i + self.sample_size] for i in range(0, len(data), self.sample_size)])
            logger.info(f"Continuous data aligned and transformed to {len(result)} samples. "
                        f"Pre-alignment size: {pre_size}, "
                        f"post-alignment size: {target_length}, "
                        f"padding added: {padding_length}")
            return result
        else:
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
                    raise Exception(f"'input_ids' field not found in sample! Available keys: {', '.join(data[idx].keys())}")
                elif isinstance(data[idx]['input_ids'], np.ndarray):
                    data[idx]['input_ids'] = torch.from_numpy(data[idx]['input_ids'])
                if 'target_ids' not in data[idx]:
                    data[idx]['target_ids'] = data[idx]['input_ids'][1:]
                elif isinstance(data[idx]['target_ids'], np.ndarray):
                    data[idx]['target_ids'] = torch.from_numpy(data[idx]['target_ids'])
                
                if self.weighted_loss:
                    if 'target_weights' not in data[idx]:
                        data[idx]['target_weights'] = torch.where(data[idx]['target_ids'] == self.ignore_index, 0, 1)
                    elif isinstance(data[idx]['target_weights'], np.ndarray):
                        data[idx]['target_weights'] = torch.from_numpy(data[idx]['target_weights'])
                elif 'target_weights' in data[idx]:
                    del data[idx]['target_weights']
                    
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
                
                if self.weighted_loss:
                    if len(data[idx]['target_weights']) >= self.sample_size:
                        data[idx]['target_weights'] = data[idx]['target_weights'][:self.sample_size-1]
                    elif self.pad_token_id >= 0 and len(data[idx]['target_weights']) < self.sample_size-1:
                        padding = self.sample_size - 1 - len(data[idx]['target_weights'])
                        data[idx]['target_weights'] = torch.cat([data[idx]['target_weights'], torch.full((padding,), 0)], dim=0)
                
                assert len(data[idx]['input_ids']) == len(data[idx]['target_ids'])
                if self.weighted_loss:
                    assert len(data[idx]['input_ids']) == len(data[idx]['target_weights'])
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
        return list(s for s in samples[self.rank::self.world_size]) if self.world_size > 1 else samples
        
    def has_data(self):
        return self.data and len(self.data) > 0
    
    def prepare_alm_dpo_sample(self, sample):
        result = {
            'chosen_input_ids': torch.from_numpy(sample['chosen_input_ids']),
            'chosen_target_ids': torch.from_numpy(sample['chosen_target_ids']),
            'rejected_input_ids': torch.from_numpy(sample['rejected_input_ids']),
            'rejected_target_ids': torch.from_numpy(sample['rejected_target_ids'])
        }
        if "reference_chosen_logps" in sample and "reference_rejected_logps" in sample:
            result["reference_chosen_logps"] = torch.tensor(sample['reference_chosen_logps'])
            result["reference_rejected_logps"] = torch.tensor(sample['reference_rejected_logps'])
        
        if self.pad_token_id >= 0:
            if len(result['chosen_input_ids']) < self.block_size:
                result['chosen_input_ids'] = F.pad(result['chosen_input_ids'], (0, self.block_size - len(result['chosen_input_ids'])), value=self.pad_token_id)
            if len(result['chosen_target_ids']) < self.block_size:
                result['chosen_target_ids'] = F.pad(result['chosen_target_ids'], (0, self.block_size - len(result['chosen_target_ids'])), value=self.ignore_index)
            if len(result['rejected_input_ids']) < self.block_size:
                result['rejected_input_ids'] = F.pad(result['rejected_input_ids'], (0, self.block_size - len(result['rejected_input_ids'])), value=self.pad_token_id)
            if len(result['rejected_target_ids']) < self.block_size:
                result['rejected_target_ids'] = F.pad(result['rejected_target_ids'], (0, self.block_size - len(result['rejected_target_ids'])), value=self.ignore_index)
        
        return result
    
    def prepare_alm_sample(self, sample):
        """
        Assumes input sample for the SFT training contains at least 'input_ids' and 'target_ids' fields. 
        When the weighted loss is active, 'target_weights' field is required.
        When samples are packed, it is assumed that a list of sequence lengths will be available
        in the "seq_lens" field. This information will be used to create the attention mask.
        If pad_token_id is set in the configuration, it is assumed that the sample list
        did not have padding and samples are of length up to block_size.
        """
        if isinstance(sample, np.ndarray):
            # Use only for testing
            # if len(sample) > self.sample_size:
            #    sample = sample[:self.sample_size]
            assert len(sample) == self.sample_size, "Invalid sample size"
            return torch.from_numpy(sample)

        if self.training_type == 'dpo':
            return self.prepare_alm_dpo_sample(sample)
        
        if len(sample['input_ids']) == self.sample_size and "target_ids" not in sample:
            result = {
                'input_ids': torch.from_numpy(sample['input_ids'][:-1]),
                'target_ids': torch.from_numpy(sample['input_ids'][1:])
            }
        else:
            result = {
                'input_ids': torch.from_numpy(sample['input_ids']),
                'target_ids': torch.from_numpy(sample['target_ids'])
            }
        
        if self.weighted_loss:
            if 'target_weights' in sample:
                result['target_weights'] = torch.from_numpy(sample['target_weights'])
            else:
                result['target_weights'] = torch.where(result['target_ids'] == self.ignore_index, 0, 1)
        
        if self.pad_token_id >= 0:
            if len(result['input_ids']) < self.block_size:
                result['input_ids'] = F.pad(result['input_ids'], (0, self.block_size - len(result['input_ids'])), value=self.pad_token_id)
            if len(result['target_ids']) < self.block_size:
                result['target_ids'] = F.pad(result['target_ids'], (0, self.block_size - len(result['target_ids'])), value=self.ignore_index)
            if 'target_weights' in result and len(result['target_weights']) < self.block_size:
                result['target_weights'] = F.pad(result['target_weights'], (0, self.block_size - len(result['target_weights'])), value=0)
        
        if "seq_lens" in sample:
            if attention_version.version == 4: # xformers does not need materialized masks
                seq_lens = [sl for sl in sample["seq_lens"]]
                sample_input_pos = []
                for seq_len in sample["seq_lens"]:
                    sample_input_pos.extend(list(range(seq_len)))
                padding_seq_len = len(result['input_ids']) - sum(seq_lens)
                if padding_seq_len > 0:
                    seq_lens.append(padding_seq_len)
                    sample_input_pos.extend(list(range(padding_seq_len)))
                result["input_pos"] = torch.tensor(sample_input_pos)
                result["seq_lens"] = seq_lens
            elif attention_version.version == 5: # FlexAttention does not need materialized masks
                document_ids = [torch.full((length,), doc_id, dtype=torch.int) for doc_id, length in enumerate(sample["seq_lens"])]
                sample_input_pos = []
                for seq_len in sample["seq_lens"]:
                    sample_input_pos.extend(list(range(seq_len)))
                padding_seq_len = len(result['input_ids']) - sum(sample["seq_lens"])
                if padding_seq_len > 0:
                    document_ids.append(torch.full((padding_seq_len,), len(sample["seq_lens"]), dtype=torch.int))
                    sample_input_pos.extend(list(range(padding_seq_len)))
                result["input_pos"] = torch.tensor(sample_input_pos)
                result["attn_mask"] = torch.cat(document_ids)
            else:
                total_seq_len = 0
                block_attn_masks = []
                sample_input_pos = []
                for seq_len in sample["seq_lens"]:
                    sample_input_pos.extend(list(range(seq_len)))
                    total_seq_len += seq_len
                    
                    # append lower triangular matrix for causal mask
                    block_attn_masks.append(torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool)))
                    
                if total_seq_len < len(result['input_ids']):
                    new_pos = sample_input_pos[-1] + 1
                    num_pad = len(result['input_ids']) - total_seq_len
                    sample_input_pos.extend(list(range(new_pos, new_pos + num_pad)))
                    block_attn_masks.append(torch.eye(num_pad, num_pad, dtype=torch.bool))
                result['input_pos'] = torch.tensor(sample_input_pos)
                result['attn_mask'] = torch.block_diag(*block_attn_masks)
        return result
    
    def __len__(self):
        """ Size of currently loaded dataset file """
        return len(self.data) if self.data else 0
        
    def __getitem__(self, idx):
        result = None
        if isinstance(idx, slice):
            result = self.data[idx]
            if self.data_in_alm_format:
                result = list(self.prepare_alm_sample(s) for s in result)
        elif idx < self.__len__():
            result = self.data[idx]
            if self.data_in_alm_format:
                result = self.prepare_alm_sample(result)
        return result

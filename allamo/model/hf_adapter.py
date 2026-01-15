import dataclasses
import json
import os
import torch
from transformers import AutoModelForCausalLM

from allamo.logging import logger
from allamo.train_utils import (
    get_model_checkpoint_path,
    get_config_checkpoint_path,
)

class BaseHFAdapter:
    
    def get_model_config_class(self):
        raise Exception("get_model_config_class not implemented")
    
    def get_model_class(self):
        raise Exception("get_model_class not implemented")

    def create_model_config(self, hf_model):
        raise Exception("create_model_config not implemented")

    def create_weight_mapping(self, hf_model_sd, model_sd, config):
        raise Exception("create_weight_mapping not implemented")
    
    def to_hf_model(self, checkpoint_dir_path, checkpoint_name_base, hf_model_path, hf_model_type, hf_model_dtype, hf_model_max_position_embeddings):
        raise Exception("to_hf_model not implemented")

    def load_model_checkpoint(self, checkpoint_name_base, checkpoint_dir_path):
        with open(get_config_checkpoint_path(checkpoint_name_base, checkpoint_dir_path), "r", encoding="utf-8") as f:
            config_checkpoint = json.load(f)
        model_checkpoint = torch.load(get_model_checkpoint_path(checkpoint_name_base, checkpoint_dir_path), map_location='cpu', weights_only=True)
        return config_checkpoint, model_checkpoint

    def check_bias(self, state_dict):
        for k in state_dict.keys():
            if k.endswith(".bias"):
                return True
        return False

    def set_mapping_or_zero(self, state_dicts_map, src_key, dst_key, src_state_dict, dst_state_dict):
        if src_key in src_state_dict:
            state_dicts_map[dst_key] = src_key
        else:
            dst_state_dict[dst_key].zero_()
            logger.warning(f"Reset '{dst_key}' to zero")
    
    def load_hf_model(self, hf_model_path):
        return AutoModelForCausalLM.from_pretrained(hf_model_path, dtype=torch.float32, low_cpu_mem_usage=True)
    
    def check_parameter_coverage(self, hf_model_sd, model_sd, state_dicts_map):
        for k, v in model_sd.items():
            if k not in state_dicts_map:
                logger.info(f"{k} param won't be updated in the new model!")
                
        for k, v in hf_model_sd.items():
            if k not in state_dicts_map.values():
                logger.info(f"{k} param won't be copied to the new model!")
    
    def copy_parameters(self, hf_model_sd, model_sd, state_dicts_map):
        param_count = 0
        for k, v in state_dicts_map.items():
            if not k.endswith('rotary_emb.inv_freq'):
                assert hf_model_sd[v].shape == model_sd[k].shape
                with torch.no_grad():
                    model_sd[k].copy_(hf_model_sd[v])
                param_count += model_sd[k].numel()
        return param_count
    
    def verify_parameters(self, hf_model_sd, model_sd, state_dicts_map):
        for k, _ in model_sd.items():
            if k in state_dicts_map and not torch.all(torch.eq(model_sd[k], hf_model_sd[state_dicts_map[k]])):
                logger.info(f"{k} param in the new model is not the same as {state_dicts_map[k]} param in the source model!")

    def save_model_checkpoint(self, config, model_sd, output_model_path, output_checkpoint_name_base):
        config_checkpoint = {
            'model_args': dataclasses.asdict(config)
        }
        ckpt_file_path = get_config_checkpoint_path(output_checkpoint_name_base, output_model_path)
        logger.info(f"saving config checkpoint to {ckpt_file_path}")
        with open(ckpt_file_path, "w", encoding="utf-8") as f:
            json.dump(config_checkpoint, f, indent=4, ensure_ascii=False)
        ckpt_file_path = get_model_checkpoint_path(output_checkpoint_name_base, output_model_path)
        logger.info(f"saving model checkpoint to {ckpt_file_path}")
        torch.save(model_sd, ckpt_file_path)

    def from_hf_model(self, hf_model_path, output_model_path, output_checkpoint_name_base):
        assert self.get_model_config_class() is not None, "Model config class is not defined"
        assert self.get_model_class() is not None, "Model class is not defined"

        os.makedirs(output_model_path, exist_ok=True)

        logger.info(f"Importing Huggingface model weights")
        hf_model = self.load_hf_model(hf_model_path)
        logger.info(f"Huggingface model model loaded")
        
        config = self.create_model_config(hf_model)
        logger.info(f"Model config created:\n{config}")

        logger.info(f"Initializing vanilla model")
        model = self.get_model_class()(config)
        
        logger.info(f"Preparing weight mapping")
        model_sd = model.state_dict()
        hf_model_sd = hf_model.state_dict()
        state_dicts_map = self.create_weight_mapping(hf_model_sd, model_sd, config)
        
        logger.info(f"Checking parameter coverage")
        self.check_parameter_coverage(hf_model_sd, model_sd, state_dicts_map)
        
        logger.info(f"Copying parameters to the new model")
        param_count = self.copy_parameters(hf_model_sd, model_sd, state_dicts_map)
        logger.info(f"{param_count} params copied to the new model")
        
        self.verify_parameters(hf_model_sd, model_sd, state_dicts_map)
        logger.info(f"Parameters in the new model verified")
        
        self.save_model_checkpoint(config, model_sd, output_model_path, output_checkpoint_name_base)
        logger.info(f"checkpoint files saved in {output_model_path}")

        logger.info("Import from HF format completed")
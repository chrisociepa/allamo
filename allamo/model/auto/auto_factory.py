import json
import torch

from allamo.logging import logger
from allamo.model.modeling_utils import get_model_spec
from allamo.train_utils import (
    get_model_checkpoint_path,
    get_config_checkpoint_path
)

class AutoModel:

    @classmethod
    def from_pretrained(cls, ckpt_file_name, ckpt_dir):
        with open(get_config_checkpoint_path(ckpt_file_name, ckpt_dir), "r", encoding="utf-8") as f:
            config_checkpoint = json.load(f)
        model_checkpoint = torch.load(get_model_checkpoint_path(ckpt_file_name, ckpt_dir), map_location="cpu")

        model_spec = get_model_spec(config_checkpoint['model_args']['model_type'])
        model_config = model_spec.model_config_cls(**config_checkpoint['model_args'])

        model = model_spec.model_cls(model_config)
        incompatible_keys = model.load_state_dict(model_checkpoint, strict=False)
        if incompatible_keys.unexpected_keys:
            logger.warning(f"Unexpected keys in checkpoint: {incompatible_keys.unexpected_keys}")
        if incompatible_keys.missing_keys:
            logger.warning(f"Missing keys in checkpoint: {incompatible_keys.missing_keys}")

        return model, model_spec, config_checkpoint
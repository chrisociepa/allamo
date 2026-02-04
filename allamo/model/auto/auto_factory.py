import json
import torch

from allamo.model.modeling_utils import get_model_spec
from allamo.train_utils import (
    get_model_checkpoint_path,
    get_config_checkpoint_path
)

class AutoModel:

    @classmethod
    def from_pretrained(ckpt_file_name, ckpt_dir):
        with open(get_config_checkpoint_path(ckpt_file_name, ckpt_dir), "r", encoding="utf-8") as f:
            config_checkpoint = json.load(f)
        model_checkpoint = torch.load(get_model_checkpoint_path(ckpt_file_name, ckpt_dir), map_location="cpu")

        model_spec = get_model_spec(config_checkpoint['model_args']['model_type'])
        model_config = model_spec.model_config_cls(**config_checkpoint['model_args'])

        model = model_spec.model_cls(model_config)
        model.load_state_dict(model_checkpoint)

        return model, model_spec, config_checkpoint
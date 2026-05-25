import argparse
import json
import torch
from allamo.logging import configure_logger, logger
from allamo.train_utils import (
    get_model_checkpoint_path,
    get_config_checkpoint_path,
)

from allamo.model.auto.auto_factory import AutoModel

def init_weights_from_target(
    draft_layers: torch.nn.ModuleList,
    target_layers: torch.nn.ModuleList,
    target_layer_ids: list[int],
):
    target_ids = [l + 1 for l in target_layer_ids]
    for draft_idx, layer_id in enumerate(target_ids):
        draft_layer = draft_layers[draft_idx]
        target_layer = target_layers[layer_id]

        draft_layer.attention.q_proj.weight.data.copy_(target_layer.attention.q_proj.weight.data)
        draft_layer.attention.k_proj.weight.data.copy_(target_layer.attention.k_proj.weight.data)
        draft_layer.attention.v_proj.weight.data.copy_(target_layer.attention.v_proj.weight.data)
        draft_layer.attention.c_proj.weight.data.copy_(target_layer.attention.c_proj.weight.data)
        
        draft_layer.attention_norm.weight.data.copy_(target_layer.attention_norm.weight.data)
        draft_layer.ffn_norm.weight.data.copy_(target_layer.ffn_norm.weight.data)
        
        draft_layer.feed_forward.gate_proj.weight.data.copy_(target_layer.feed_forward.gate_proj.weight.data)
        draft_layer.feed_forward.down_proj.weight.data.copy_(target_layer.feed_forward.down_proj.weight.data)
        draft_layer.feed_forward.up_proj.weight.data.copy_(target_layer.feed_forward.up_proj.weight.data)

        if draft_layer.attention.q_norm is not None and target_layer.attention.q_norm is not None:
            draft_layer.attention.q_norm.weight.data.copy_(target_layer.attention.q_norm.weight.data)
        if draft_layer.attention.k_norm is not None and target_layer.attention.k_norm is not None:
            draft_layer.attention.k_norm.weight.data.copy_(target_layer.attention.k_norm.weight.data)

def save_model_checkpoint(config_checkpoint, model_sd, output_model_path, output_checkpoint_name_base):
    ckpt_file_path = get_config_checkpoint_path(output_checkpoint_name_base, output_model_path)
    logger.info(f"saving config checkpoint to {ckpt_file_path}")
    with open(ckpt_file_path, "w", encoding="utf-8") as f:
        json.dump(config_checkpoint, f, indent=4, ensure_ascii=False)
    ckpt_file_path = get_model_checkpoint_path(output_checkpoint_name_base, output_model_path)
    logger.info(f"saving model checkpoint to {ckpt_file_path}")
    torch.save(model_sd, ckpt_file_path)

def main():
    configure_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Location of ALLaMo weights, which contains a checkpoint file of target model",
    )
    parser.add_argument(
        "--checkpoint_name_base",
        default='ckpt',
        help="Checkpoint file name base",
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write initialized model",
    )
    args = parser.parse_args()

    model, model_spec, config_checkpoint = AutoModel.from_pretrained(args.checkpoint_name_base, args.input_dir)
    logger.info(f"Loaded model from {args.input_dir}")

    assert "dflash_config" in config_checkpoint["model_args"], "Model must have dflash_config"
    init_weights_from_target(
        model.dflash.layers,
        model.layers,
        config_checkpoint["model_args"]["dflash_config"]["target_layer_ids"]
    )
    logger.info("Initialized dflash model from target model")
    
    save_model_checkpoint(config_checkpoint, model.state_dict(), args.output_dir, args.checkpoint_name_base)
    logger.info("Procedure completed")

if __name__ == "__main__":
    main()

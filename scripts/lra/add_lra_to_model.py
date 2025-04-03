import argparse
import json
import os
import torch
from allamo.logging import configure_logger, logger
from allamo.model.lra import LRA, get_supported_base_functions
from allamo.train_utils import (
    get_model_checkpoint_path,
    get_config_checkpoint_path,
    remove_unwanted_prefix_from_model_state_dict,
)

def inject_lra(model_checkpoint, layer_i, lra_sd):
    model_checkpoint[f"layers.{layer_i}.feed_forward.act_fn.p_coeff_0"] = lra_sd["p_coeff_0"].clone()
    model_checkpoint[f"layers.{layer_i}.feed_forward.act_fn.p_coeff_1"] = lra_sd["p_coeff_1"].clone()
    model_checkpoint[f"layers.{layer_i}.feed_forward.act_fn.p_coeff_2"] = lra_sd["p_coeff_2"].clone()
    model_checkpoint[f"layers.{layer_i}.feed_forward.act_fn.p_coeff_3"] = lra_sd["p_coeff_3"].clone()
    model_checkpoint[f"layers.{layer_i}.feed_forward.act_fn.p_coeff_4"] = lra_sd["p_coeff_4"].clone()
    model_checkpoint[f"layers.{layer_i}.feed_forward.act_fn.p_coeff_5"] = lra_sd["p_coeff_5"].clone()
    
    model_checkpoint[f"layers.{layer_i}.feed_forward.act_fn.q_coeff_1"] = lra_sd["q_coeff_1"].clone()
    model_checkpoint[f"layers.{layer_i}.feed_forward.act_fn.q_coeff_2"] = lra_sd["q_coeff_2"].clone()
    model_checkpoint[f"layers.{layer_i}.feed_forward.act_fn.q_coeff_3"] = lra_sd["q_coeff_3"].clone()
    model_checkpoint[f"layers.{layer_i}.feed_forward.act_fn.q_coeff_4"] = lra_sd["q_coeff_4"].clone()
    logger.info(f"LRA injected to layer {layer_i}")

def adjust_model(input_dir_path, input_checkpoint_name_base, output_dir_path, output_checkpoint_name_base, num_groups, base_fn):
    os.makedirs(output_dir_path, exist_ok=True)
    
    logger.info(f"loading checkpoint from {input_dir_path}...")
    with open(get_config_checkpoint_path(input_checkpoint_name_base, input_dir_path), "r", encoding="utf-8") as f:
        config_checkpoint = json.load(f)
    model_checkpoint = torch.load(get_model_checkpoint_path(input_checkpoint_name_base, input_dir_path), map_location='cpu', weights_only=True)
    
    remove_unwanted_prefix_from_model_state_dict(model_checkpoint)

    model_intermediate_size = config_checkpoint["model_args"]["intermediate_size"]
    if num_groups < 1 or model_intermediate_size % num_groups != 0:
        raise Exception("Invalid number of LRA groups")
    else:
        group_size = model_intermediate_size // num_groups

    logger.info(f"Start injecting LRA(base_fn={base_fn}, num_groups={num_groups}, group_size={group_size})")
    for layer_i in range(config_checkpoint['model_args']['n_layer']):
        lra = LRA(base_fn=base_fn, dim=model_intermediate_size, group_size=group_size)
        lra_sd = lra.state_dict()
        inject_lra(model_checkpoint, layer_i, lra_sd)
    
    config_checkpoint["model_args"]["act_fn"] = "lra"
    config_checkpoint["model_args"]["act_fn_params"] = {"base_fn": base_fn, "dim": model_intermediate_size, "group_size": group_size}
    
    param_count = 0
    param_bytes = 0
    for _, v in model_checkpoint.items():
        param_count += v.numel()
        param_bytes += v.numel() * v.element_size()
    
    param_count /= 1e6
    param_bytes /= 1024**2
    logger.info(f"New model parameters: {param_count:.2f}M. Est. Size: {param_bytes:.3f}MB")
            
    ckpt_file_path = get_config_checkpoint_path(output_checkpoint_name_base, output_dir_path)
    logger.info(f"saving config checkpoint to {ckpt_file_path}")
    with open(ckpt_file_path, "w", encoding="utf-8") as f:
        json.dump(config_checkpoint, f, indent=4, ensure_ascii=False)
    ckpt_file_path = get_model_checkpoint_path(output_checkpoint_name_base, output_dir_path)
    logger.info(f"saving model checkpoint to {ckpt_file_path}")
    torch.save(model_checkpoint, ckpt_file_path)
    logger.info(f"checkpoint files saved in {output_dir_path}")


if __name__ == "__main__":
    configure_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Location of ALLaMo weights, which contains a checkpoint file",
    )
    parser.add_argument(
        "--input_checkpoint_name_base",
        default='ckpt',
        help="Source checkpoint file name base",
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write up-scaled model",
    )
    parser.add_argument(
        "--output_checkpoint_name_base",
        default='ckpt',
        help="Output checkpoint file name base",
    )
    parser.add_argument(
        "--num_groups",
        type=int,
        default=4,
        help="Number of LRA groups. Defaults to 4",
    )
    parser.add_argument(
        "--base_fn",
        choices=get_supported_base_functions(),
        default='swish', # replaces SiLU
        help="Base activation function",
    )
    args = parser.parse_args()
    adjust_model(
        input_dir_path=args.input_dir,
        input_checkpoint_name_base=args.input_checkpoint_name_base,
        output_dir_path=args.output_dir,
        output_checkpoint_name_base=args.output_checkpoint_name_base,
        num_groups=args.num_groups,
        base_fn=args.base_fn
    )

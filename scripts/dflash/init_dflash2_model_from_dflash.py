import argparse
import copy
import json
import os
import torch
from allamo.logging import configure_logger, logger
from allamo.train_utils import (
    get_model_checkpoint_path,
    get_config_checkpoint_path,
)
from allamo.model.modeling_utils import get_model_spec

from allamo.model.auto.auto_factory import AutoModel


def save_model_checkpoint(config_checkpoint, model_sd, output_model_path, output_checkpoint_name_base):
    os.makedirs(output_model_path, exist_ok=True)
    ckpt_file_path = get_config_checkpoint_path(output_checkpoint_name_base, output_model_path)
    logger.info(f"saving config checkpoint to {ckpt_file_path}")
    with open(ckpt_file_path, "w", encoding="utf-8") as f:
        json.dump(config_checkpoint, f, indent=4, ensure_ascii=False)
    ckpt_file_path = get_model_checkpoint_path(output_checkpoint_name_base, output_model_path)
    logger.info(f"saving model checkpoint to {ckpt_file_path}")
    torch.save(model_sd, ckpt_file_path)


def main():
    configure_logger()
    parser = argparse.ArgumentParser(
        description="Warm-start a DFlash 2 draft model (dynamic conv + candidate selector) "
                     "from an already-trained DFlash 1 checkpoint. The DFlash 1 weights are "
                     "loaded as-is; the new conv/selector parameters are freshly initialized "
                     "as no-ops, so the resulting checkpoint is numerically identical to the "
                     "DFlash 1 source until DFlash 2 training moves it away from that point."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Location of an already-trained ALLaMo DFlash 1 checkpoint (dflash_config present, dflash2 absent/false)",
    )
    parser.add_argument(
        "--checkpoint_name_base",
        default="ckpt",
        help="Checkpoint file name base (used for both reading --input_dir and writing --output_dir)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Location to write the warm-started DFlash 2 model",
    )
    parser.add_argument("--conv_group_size", type=int, default=1,
                         help="Channels per dynamic-conv group (1 = fully depthwise per channel)")
    parser.add_argument("--conv_kernel_size", type=int, default=2,
                         help="Number of taps in the dynamic-conv kernel (2 = self + immediate predecessor)")
    parser.add_argument("--selector_top_k", type=int, default=8,
                         help="Number of top candidates per position the selector scores")
    parser.add_argument("--selector_rank", type=int, default=128,
                         help="Rank of the selector's predecessor/successor token codebooks")
    parser.add_argument("--selector_loss_weight", type=float, default=0.2,
                         help="Weight of the selector CE term relative to the base draft block CE")
    parser.add_argument("--disable_selector", action="store_true",
                         help="Warm-start with the dynamic convolution only (no candidate selector). "
                              "Useful if you want to train/validate the conv in isolation first.")
    args = parser.parse_args()

    model, model_spec, config_checkpoint = AutoModel.from_pretrained(args.checkpoint_name_base, args.input_dir)
    logger.info(f"Loaded DFlash 1 model from {args.input_dir}")

    source_dflash_config = config_checkpoint["model_args"].get("dflash_config")
    assert source_dflash_config, "Source checkpoint must have dflash_config (a plain DFlash 1 draft model)"
    assert not source_dflash_config.get("dflash2", False), (
        "Source checkpoint is already a DFlash 2 model (dflash_config['dflash2'] is true) - "
        "nothing to warm-start from."
    )

    new_config_checkpoint = copy.deepcopy(config_checkpoint)
    new_dflash_config = new_config_checkpoint["model_args"]["dflash_config"]
    new_dflash_config["dflash2"] = True
    new_dflash_config["conv_group_size"] = args.conv_group_size
    new_dflash_config["conv_kernel_size"] = args.conv_kernel_size
    new_dflash_config["selector_enabled"] = not args.disable_selector
    if not args.disable_selector:
        new_dflash_config["selector_top_k"] = args.selector_top_k
        new_dflash_config["selector_rank"] = args.selector_rank
        new_dflash_config["selector_loss_weight"] = args.selector_loss_weight

    # Building the model from the updated config runs the model's normal __init__,
    # which already initializes the new conv/selector parameters as identity no-ops
    # (see DFlash2DynamicConv.init_weights / DFlash2CandidateSelector.init_weights).
    new_model_config = model_spec.model_config_cls(**new_config_checkpoint["model_args"])
    new_model = model_spec.model_cls(new_model_config)
    logger.info("Constructed a fresh DFlash 2 model (new conv/selector params identity-initialized)")

    incompatible_keys = new_model.load_state_dict(model.state_dict(), strict=False)
    if incompatible_keys.unexpected_keys:
        # Anything here means a DFlash 1 parameter didn't find a matching name/shape
        # in the DFlash 2 model - investigate before trusting the warm start.
        logger.warning(f"Unexpected keys while loading DFlash 1 weights: {incompatible_keys.unexpected_keys}")
    expected_new_params = [k for k in incompatible_keys.missing_keys if ("_conv." in k or ".selector." in k)]
    unexplained_missing = [k for k in incompatible_keys.missing_keys if k not in expected_new_params]
    logger.info(f"DFlash 2-only parameters left at their identity init ({len(expected_new_params)} tensors): "
                f"{expected_new_params}")
    if unexplained_missing:
        # Anything here is NOT one of the new conv/selector tensors - investigate
        # before trusting the warm start (likely a naming mismatch in your fork).
        logger.warning(f"Unexpected missing keys (not conv/selector): {unexplained_missing}")

    logger.info("DFlash 1 -> DFlash 2 warm start complete. Until you train it, this checkpoint "
                "produces numerically identical draft logits to the DFlash 1 source.")

    save_model_checkpoint(new_config_checkpoint, new_model.state_dict(), args.output_dir, args.checkpoint_name_base)
    logger.info("Procedure completed")


if __name__ == "__main__":
    main()

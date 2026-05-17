"""
Swap the embedding layer of a HuggingFace model with embeddings from a separate checkpoint.

Usage:
    python swap_embeddings.py \
        -a path/to/source.ckpt \
        -h path/to/hf_model_dir \
        -o path/to/output_dir
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from allamo.logging import configure_logger, logger
from allamo.train_utils import remove_unwanted_prefix_from_model_state_dict

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace embed_tokens.weight in a HuggingFace model with "
                    "tok_embeddings.weight from a separate checkpoint."
    )
    parser.add_argument(
        "-a", "--src-ckpt",
        required=True,
        metavar="PATH",
        help="Path to the Allamo checkpoint (.pt) that contains 'tok_embeddings.weight'",
    )
    parser.add_argument(
        "-h", "--hf-model",
        required=True,
        metavar="PATH",
        help="Path to the HuggingFace model directory",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        metavar="PATH",
        help="Directory where the patched model will be saved",
    )
    return parser.parse_args()


def load_src_embeddings(ckpt_src_path: str) -> torch.Tensor:
    logger.info(f"Loading source checkpoint: {ckpt_src_path}")
    state_dict = torch.load(ckpt_src_path, map_location="cpu", weights_only=True)

    remove_unwanted_prefix_from_model_state_dict(state_dict)

    key = "tok_embeddings.weight"
    if key not in state_dict:
        available = ", ".join(list(state_dict.keys())[:20])
        raise KeyError(f"Key '{key}' not found in source checkpoint. First available keys: {available}")

    embeddings = state_dict[key]
    logger.info(f"Extracted embeddings - shape: {embeddings.shape}, dtype: {embeddings.dtype}")

    del state_dict
    return embeddings

def load_hf_model(hf_model_path: str) -> AutoModelForCausalLM:
    logger.info(f"Loading HuggingFace model from: {hf_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_path,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    logger.info(f"Model loaded - parameter count: {sum(p.numel() for p in model.parameters()):,}")
    return model


def swap_embeddings(model: AutoModelForCausalLM, new_embeddings: torch.Tensor) -> None:
    target_key = "model.embed_tokens.weight"
    current_sd = model.state_dict()
    if target_key not in current_sd:
        available = ", ".join(list(current_sd.keys())[:20])
        raise KeyError(f"Key '{target_key}' not found in HF model state dict. First available keys: {available}")

    current_shape = current_sd[target_key].shape
    if current_shape != new_embeddings.shape:
        raise ValueError(f"Shape mismatch: HF model has {current_shape}, source checkpoint has {new_embeddings.shape}.")

    if new_embeddings.dtype != current_sd[target_key].dtype:
        logger.warning(
            f"dtype mismatch ({new_embeddings.dtype} vs {current_sd[target_key].dtype}) - casting source embeddings to match the model"
        )
        new_embeddings = new_embeddings.to(current_sd[target_key].dtype)

    with torch.no_grad():
        model.model.embed_tokens.weight.copy_(new_embeddings)

    logger.info(f"Successfully swapped '{target_key}'")


def save_model(model: AutoModelForCausalLM, hf_model_path: str, output_path: str) -> None:
    logger.info(f"Saving patched model to: {output_path}")
    model.save_pretrained(output_path)
    logger.warning("Remember to copy tokenier files!")
    logger.info("Done.")


def main() -> None:
    configure_logger()
    args = parse_args()
    new_embeddings = load_src_embeddings(args.src_ckpt)
    model = load_hf_model(args.hf_model)
    swap_embeddings(model, new_embeddings)
    del new_embeddings
    save_model(model, args.hf_model, args.output)


if __name__ == "__main__":
    main()

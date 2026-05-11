import argparse
import logging
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def find_unique_key(state_dict: dict, substring: str) -> str:
    matches = [k for k in state_dict if substring in k]
    if len(matches) == 0:
        raise KeyError(f"No key containing '{substring}' found in state dict")
    if len(matches) > 1:
        raise KeyError(f"Multiple keys containing '{substring}' found: {matches}")
    return matches[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy mask token embedding into tok_embeddings at a given token ID."
    )
    parser.add_argument("-i", "--input_path", type=str, help="Path to the input state dict file")
    parser.add_argument("-m", "--mask_token_id", type=int, help="Token ID to assign the mask embedding to")
    parser.add_argument("-o", "--output_path", type=str, help="Path to save the modified state dict")
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info(f"Loading state dict from '{args.input_path}'")
    state_dict = torch.load(args.input_path, map_location="cpu", weights_only=True)

    tok_embeddings_key = find_unique_key(state_dict, "tok_embeddings.weight")
    logger.info(f"Found tok_embeddings key: '{tok_embeddings_key}'")

    mask_token_key = find_unique_key(state_dict, "mask_token_embd")
    logger.info(f"Found mask token embedding key: '{mask_token_key}'")

    vocab_size = state_dict[tok_embeddings_key].shape[0]
    logger.info(f"Vocab size: {vocab_size}, target mask_token_id: {args.mask_token_id}")

    if args.mask_token_id < 0 or args.mask_token_id >= vocab_size:
        raise ValueError(
            f"mask_token_id {args.mask_token_id} is out of range [0, {vocab_size})"
        )

    mask_embedding = state_dict[mask_token_key]
    state_dict[tok_embeddings_key][args.mask_token_id] = mask_embedding.detach().clone()
    logger.info(
        f"Copied mask token embedding to '{tok_embeddings_key}' at index {args.mask_token_id}"
    )

    logger.info(f"Saving modified state dict to '{args.output_path}'")
    torch.save(state_dict, args.output_path)
    logger.info("Done")


if __name__ == "__main__":
    main()
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
        description="Initialize mask_token_embd Parameter from tok_embeddings, "
        "either from a given token ID or as the mean of all embeddings."
    )
    parser.add_argument("-i", "--input_path", type=str, help="Path to the input state dict file")
    parser.add_argument(
        "-m",
        "--mask_token_id",
        type=int,
        default=None,
        help="Token ID to initialize the mask embedding from. If omitted, the mean of all embeddings is used instead.",
    )
    parser.add_argument("-o", "--output_path", type=str, help="Path to save the modified state dict")
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info(f"Loading state dict from '{args.input_path}'")
    state_dict = torch.load(args.input_path, map_location="cpu", weights_only=True)

    tok_embeddings_key = find_unique_key(state_dict, "tok_embeddings.weight")
    logger.info(f"Found tok_embeddings key: '{tok_embeddings_key}'")

    prefix = tok_embeddings_key[: -len("tok_embeddings.weight")]
    mask_token_key = f"{prefix}dflash.mask_token_embd"
    logger.info(f"Target mask token embedding key: '{mask_token_key}'")

    if mask_token_key in state_dict:
        raise KeyError(f"Key '{mask_token_key}' already exists in state dict")

    tok_embeddings = state_dict[tok_embeddings_key]
    vocab_size = tok_embeddings.shape[0]

    with torch.no_grad():
        if args.mask_token_id is not None:
            if args.mask_token_id < 0 or args.mask_token_id >= vocab_size:
                raise ValueError(f"mask_token_id {args.mask_token_id} is out of range [0, {vocab_size})")
            orig_emb = tok_embeddings[args.mask_token_id]
            logger.info(f"Vocab size: {vocab_size}. Initializing mask token embedding from token ID {args.mask_token_id}.")
        else:
            orig_emb = tok_embeddings.mean(dim=0)
            logger.info(f"Vocab size: {vocab_size}. Initializing mask token embedding as mean of all embeddings.")

    state_dict[mask_token_key] = torch.nn.Parameter(orig_emb.clone())
    logger.info(f"Created mask token embedding Parameter under key '{mask_token_key}'")

    logger.info(f"Saving modified state dict to '{args.output_path}'")
    torch.save(state_dict, args.output_path)
    logger.info("Done")


if __name__ == "__main__":
    main()
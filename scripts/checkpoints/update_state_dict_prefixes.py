import argparse
import torch
from allamo.logging import configure_logger, logger
from allamo.train_utils import remove_unwanted_prefix_from_model_state_dict

def add_prefix_to_model_state_dict(state_dict, prefix):
    for k, _ in list(state_dict.items()):
        state_dict[prefix + k] = state_dict.pop(k)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', type=str, required=True, help="Path to the source model file")
    parser.add_argument('-d', '--dst', type=str, required=True, help="Path to save the modified target model file")
    parser.add_argument('-r', '--remove_prefix', type=str, default='_orig_mod.', help="Prefix to remove from state dict keys")
    parser.add_argument('-a', '--add_prefix', type=str, help="Prefix to add into state dict keys")
    args = parser.parse_args()
    
    configure_logger()
    state_dict = torch.load(args.src, weights_only=True)
    logger.info(f"Loaded checkpoint {args.src}")

    if args.remove_prefix.strip():
        logger.info(f"Removing '{args.remove_prefix}' prefix from state dict keys")
        remove_unwanted_prefix_from_model_state_dict(state_dict, args.remove_prefix)
    
    if args.add_prefix.strip():
        logger.info(f"Adding '{args.add_prefix}' prefix into state dict keys")
        add_prefix_to_model_state_dict(state_dict, args.add_prefix)

    torch.save(state_dict, args.dst)
    logger.info(f"Transformation completed. New model saved in {args.dst}")

import argparse
import torch
from allamo.logging import configure_logger, logger
from allamo.train_utils import remove_unwanted_prefix_from_model_state_dict

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', type=str, required=True, help="Path to the source model file")
    parser.add_argument('-d', '--dst', type=str, required=True, help="Path to save the modified target model file")
    parser.add_argument('-p', '--prefix', type=str, default='_orig_mod.', help="Prefix to remove from state dict keys")
    args = parser.parse_args()
    
    configure_logger()
    logger.info(f"Removing '{args.prefix}' from the checkpoint {args.src}")

    state_dict = torch.load(args.src)
    remove_unwanted_prefix_from_model_state_dict(state_dict, args.prefix)
    torch.save(state_dict, args.dst)

    logger.info(f"Transformation completed. New model saved in {args.dst}")

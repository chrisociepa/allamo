"""
Use this file to import Huggingface model weights to ALLaMo format.   
"""
import argparse
from allamo.logging import configure_logger, logger
from allamo.model.modeling_utils import get_hf_model_adapter

if __name__ == '__main__':
    configure_logger()
    parser = argparse.ArgumentParser(description='Import Huggingface model weights to ALLaMo format')
    parser.add_argument(
        "--huggingface_model",
        help="Huggingface model path",
    )
    parser.add_argument(
        "--model_type",
        help="Output model type",
    )
    parser.add_argument(
        "--output_dir",
        help="Path to output directory",
    )
    parser.add_argument(
        "--output_checkpoint_name_base",
        default='import_ckpt',
        help="Output checkpoint file name base",
    )
    args = parser.parse_args()

    hf_model_adapter = get_hf_model_adapter(args.model_type)

    hf_model_adapter.from_hf_model(
        hf_model_path=args.huggingface_model,
        output_model_path=args.output_dir,
        output_checkpoint_name_base=args.output_checkpoint_name_base,
    )

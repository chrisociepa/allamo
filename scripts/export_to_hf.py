"""
Use this file to export ALLaMo weights to HuggingFace formats 
"""
import argparse
from allamo.logging import configure_logger, logger
from allamo.model.modeling_utils import get_hf_model_adapter

if __name__ == "__main__":
    configure_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Location of ALLaMo weights, which contains a checkpoint file",
    )
    parser.add_argument(
        "--checkpoint_name_base",
        default='ckpt',
        help="Checkpoint file name base",
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write HF model",
    )
    parser.add_argument(
        "--model_type",
        help="ALLaMo model type, e.g. bielik2",
    )
    parser.add_argument(
        "--output_model_type",
        help="HF model type, e.g. llama, mistral",
    )
    parser.add_argument(
        "--output_dtype",
        choices=['float32', 'bfloat16', 'float16'],
        help="Override model dtype and save the model under a specific dtype",
    )
    parser.add_argument(
        "--max_position_embeddings",
        type=int,
        help="Overwrite max_position_embeddings with this value",
    )
    args = parser.parse_args()

    hf_model_adapter = get_hf_model_adapter(args.model_type)

    hf_model_adapter.to_hf_model(
        checkpoint_dir_path=args.input_dir,
        checkpoint_name_base=args.checkpoint_name_base,
        hf_model_path=args.output_dir,
        hf_model_type=args.output_model_type,
        hf_model_dtype=args.output_dtype,
        hf_model_max_position_embeddings=args.max_position_embeddings,
    )

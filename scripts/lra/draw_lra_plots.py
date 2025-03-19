import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from allamo.logging import configure_logger, logger
from allamo.model.model import AllamoTransformerConfig
from allamo.train_utils import (
    get_model_checkpoint_path,
    get_config_checkpoint_path,
    remove_unwanted_prefix_from_model_state_dict,
)
        
def fn(x, a0, a1, a2, a3, a4, a5, b1, b2, b3, b4):
    x = torch.from_numpy(x)
    numerator = a0 + x * (a1 + x * (a2 + x * (a3 + x * (a4 + a5 * x))))
    denominator = 1 + (x * (b1 + x * (b2 + x * (b3 + x * b4)))).abs()
    return (numerator / denominator).numpy()

def analyze(checkpoint_dir_path, checkpoint_name_base, output_file):
    logger.info(f"loading checkpoint from {checkpoint_dir_path}...")
    with open(get_config_checkpoint_path(checkpoint_name_base, checkpoint_dir_path), "r", encoding="utf-8") as f:
        config_checkpoint = json.load(f)
    model_checkpoint = torch.load(get_model_checkpoint_path(checkpoint_name_base, checkpoint_dir_path), map_location='cpu')

    allamo_transformer_config = AllamoTransformerConfig(**config_checkpoint['model_args'])
    n_layers = allamo_transformer_config.n_layer
    head_size = allamo_transformer_config.head_size
    intermediate_size = allamo_transformer_config.intermediate_size
    assert intermediate_size is not None

    remove_unwanted_prefix_from_model_state_dict(model_checkpoint)
            
    with PdfPages(output_file) as pdf:
        x_data = np.linspace(-3, 3, 1000)
            
        for layer_i in range(n_layers):
            logger.info(f"analyzing layer {layer_i}")
            
            a0 = model_checkpoint[f"layers.{layer_i}.feed_forward.act_fn.p_coeff_0"].view(-1)
            a1 = model_checkpoint[f"layers.{layer_i}.feed_forward.act_fn.p_coeff_1"].view(-1)
            a2 = model_checkpoint[f"layers.{layer_i}.feed_forward.act_fn.p_coeff_2"].view(-1)
            a3 = model_checkpoint[f"layers.{layer_i}.feed_forward.act_fn.p_coeff_3"].view(-1)
            a4 = model_checkpoint[f"layers.{layer_i}.feed_forward.act_fn.p_coeff_4"].view(-1)
            a5 = model_checkpoint[f"layers.{layer_i}.feed_forward.act_fn.p_coeff_5"].view(-1)
            
            b1 = model_checkpoint[f"layers.{layer_i}.feed_forward.act_fn.q_coeff_1"].view(-1)
            b2 = model_checkpoint[f"layers.{layer_i}.feed_forward.act_fn.q_coeff_2"].view(-1)
            b3 = model_checkpoint[f"layers.{layer_i}.feed_forward.act_fn.q_coeff_3"].view(-1)
            b4 = model_checkpoint[f"layers.{layer_i}.feed_forward.act_fn.q_coeff_4"].view(-1)
            
            rows = config_checkpoint['model_args']['act_fn_params']['group_size'] // 8
            fig, axs = plt.subplots(rows, 8, figsize=(16, 12))
            for h in range(head_size):
                i = h // 8
                j = h % 8
                y_data = fn(x_data, a0[h], a1[h], a2[h], a3[h], a4[h], a5[h], b1[h], b2[h], b3[h], b4[h])
                ax = axs[i, j]
                ax.plot(x_data, y_data)
                ax.set_title(f"L={layer_i}, H={h} ({i},{j})")
                ax.grid(True)
                ax.axhline(y=0, color='k', linestyle='--')
                ax.axvline(x=0, color='k', linestyle='--')
                
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    logger.info(f"Analysis completed and result saved in {output_file}")

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
        "--output_file",
        help="PDF report file path",
    )
    args = parser.parse_args()
    analyze(
        checkpoint_dir_path=args.input_dir,
        checkpoint_name_base=args.checkpoint_name_base,
        output_file=args.output_file,
    )

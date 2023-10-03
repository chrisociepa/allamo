"""
Use this file to export ALLaMo weights to Huggingface LLaMA model.   
"""
import argparse
import datetime
import gc
import json
import logging
import os
import shutil
import torch
from transformers import LlamaConfig, LlamaForCausalLM

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger('AllamoModelExporter')

def compute_intermediate_size(config):
    return config.multiple_of * ((int(config.n_embd * 8 / 3) + config.multiple_of - 1) // config.multiple_of)

def read_json(path):
    with open(path, "r") as f:
        return json.load(f)

def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)

def write_model(checkpoint_path, hf_model_path):
    os.makedirs(hf_model_path, exist_ok=True)
    tmp_model_path = os.path.join(hf_model_path, "tmp")
    os.makedirs(tmp_model_path, exist_ok=True)
    
    logger.info(f"loading checkpoint from {checkpoint_path}...")
    config_checkpoint = torch.load(os.path.join(checkpoint_path, 'config_ckpt.pt'), map_location='cpu')
    model_checkpoint = torch.load(os.path.join(checkpoint_path, 'model_ckpt.pt'), map_location='cpu')

    allamo_transformer_config = config_checkpoint['model_args']
    n_layers = allamo_transformer_config.n_layer
    n_heads = allamo_transformer_config.n_head
    num_kv_heads = allamo_transformer_config.num_kv_heads
    dim = allamo_transformer_config.n_embd
    dims_per_head = allamo_transformer_config.head_size

    logger.info(f"converting all parameters from the checkpoint model")
    unwanted_prefix = '_orig_mod.'
    for k,v in list(model_checkpoint.items()):
        if k.startswith(unwanted_prefix):
            model_checkpoint[k[len(unwanted_prefix):]] = model_checkpoint.pop(k)
            
    param_count = 0
    index_dict = {"weight_map": {}}
    for layer_i in range(n_layers):
        logger.info(f"converting weights in layer {layer_i}")
        filename = f"pytorch_model-{layer_i + 1}-of-{n_layers + 1}.bin"
        state_dict = {
            f"model.layers.{layer_i}.self_attn.q_proj.weight": model_checkpoint[f"layers.{layer_i}.attention.q_proj.weight"],
            f"model.layers.{layer_i}.self_attn.k_proj.weight": model_checkpoint[f"layers.{layer_i}.attention.k_proj.weight"],
            f"model.layers.{layer_i}.self_attn.v_proj.weight": model_checkpoint[f"layers.{layer_i}.attention.v_proj.weight"],
            f"model.layers.{layer_i}.self_attn.o_proj.weight": model_checkpoint[f"layers.{layer_i}.attention.c_proj.weight"],
            #f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq": model_checkpoint[f"layers.{layer_i}.attention.rotary_emb.inv_freq"],
            f"model.layers.{layer_i}.mlp.gate_proj.weight": model_checkpoint[f"layers.{layer_i}.feed_forward.gate_proj.weight"],
            f"model.layers.{layer_i}.mlp.down_proj.weight": model_checkpoint[f"layers.{layer_i}.feed_forward.down_proj.weight"],
            f"model.layers.{layer_i}.mlp.up_proj.weight": model_checkpoint[f"layers.{layer_i}.feed_forward.up_proj.weight"],
            f"model.layers.{layer_i}.input_layernorm.weight": model_checkpoint[f"layers.{layer_i}.attention_norm.weight"],
            f"model.layers.{layer_i}.post_attention_layernorm.weight": model_checkpoint[f"layers.{layer_i}.ffn_norm.weight"]
        }
        for k, v in state_dict.items():
            index_dict["weight_map"][k] = filename
            param_count += v.numel()
        torch.save(state_dict, os.path.join(tmp_model_path, filename))

    filename = f"pytorch_model-{n_layers + 1}-of-{n_layers + 1}.bin"
    state_dict = {
        "model.embed_tokens.weight": model_checkpoint["tok_embeddings.weight"],
        "model.norm.weight": model_checkpoint["norm.weight"],
        "lm_head.weight": model_checkpoint["lm_head.weight"],
    }
    # Resolve model params dtype, e.g. torch.float16
    torch_dtype = model_checkpoint["lm_head.weight"].dtype

    for k, v in state_dict.items():
        index_dict["weight_map"][k] = filename
        param_count += v.numel()
    torch.save(state_dict, os.path.join(tmp_model_path, filename))
    logger.info(f"{param_count} params converted to HF LLaMA model")

    # Write configs
    index_dict["metadata"] = {"total_size": param_count * 2}
    write_json(index_dict, os.path.join(tmp_model_path, "pytorch_model.bin.index.json"))

    config = LlamaConfig(
        vocab_size=allamo_transformer_config.vocab_size,
        hidden_size=dim,
        intermediate_size=compute_intermediate_size(allamo_transformer_config),
        num_attention_heads=n_heads,
        num_key_value_heads=num_kv_heads,
        num_hidden_layers=n_layers,
        rms_norm_eps=allamo_transformer_config.norm_eps,
    )
    config.save_pretrained(tmp_model_path)
    logger.info(f"configuration for the HF LLaMA model saved")

    # Make space so we can load the model properly now.
    del state_dict
    del config_checkpoint
    del model_checkpoint
    gc.collect()

    logger.info(f"loading the checkpoint in a LLaMA model with {torch_dtype} dtype")
    model = LlamaForCausalLM.from_pretrained(tmp_model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True)
    # Avoid saving this as part of the config.
    del model.config._name_or_path

    logger.info(f"saving in the Transformers format")
    model.save_pretrained(hf_model_path)
    shutil.rmtree(tmp_model_path)
    logger.info(f"conversion completed!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Location of ALLaMo weights, which contains a checkpoint file",
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write HF model",
    )
    args = parser.parse_args()
    write_model(
        checkpoint_path=args.input_dir,
        hf_model_path=args.output_dir,
    )


if __name__ == "__main__":
    main()

"""
Use this file to export ALLaMo weights to HuggingFace formats 
"""
import argparse
import gc
import json
import os
import shutil
import torch
from allamo.logging import configure_logger, logger
from allamo.model.model import AllamoTransformerConfig
from allamo.train_utils import (
    get_model_checkpoint_path,
    get_config_checkpoint_path,
)

SUPPORTED_MODEL_ARCHS = ['llama', 'mistral', 'llama_lra']

def compute_intermediate_size(config):
    return config.multiple_of * ((int(config.n_embd * 8 / 3) + config.multiple_of - 1) // config.multiple_of)

def read_json(path):
    with open(path, "r") as f:
        return json.load(f)

def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)

def write_model(checkpoint_dir_path, checkpoint_name_base, hf_model_path, hf_model_type, hf_model_dtype=None, hf_model_max_position_embeddings=None):
    assert hf_model_type in SUPPORTED_MODEL_ARCHS, f"{hf_model_type} architecture is not supported"
    os.makedirs(hf_model_path, exist_ok=True)
    tmp_model_path = os.path.join(hf_model_path, "tmp")
    os.makedirs(tmp_model_path, exist_ok=True)
    
    logger.info(f"loading checkpoint from {checkpoint_dir_path}...")
    with open(get_config_checkpoint_path(checkpoint_name_base, checkpoint_dir_path), "r", encoding="utf-8") as f:
        config_checkpoint = json.load(f)
    model_checkpoint = torch.load(get_model_checkpoint_path(checkpoint_name_base, checkpoint_dir_path), map_location='cpu', weights_only=True)

    allamo_transformer_config = AllamoTransformerConfig(**config_checkpoint['model_args'])
    n_layers = allamo_transformer_config.n_layer
    intermediate_size = allamo_transformer_config.intermediate_size if hasattr(allamo_transformer_config, 'intermediate_size') else compute_intermediate_size(allamo_transformer_config)
    max_position_embeddings = allamo_transformer_config.block_size if hf_model_max_position_embeddings is None else hf_model_max_position_embeddings

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
            f"model.layers.{layer_i}.mlp.gate_proj.weight": model_checkpoint[f"layers.{layer_i}.feed_forward.gate_proj.weight"],
            f"model.layers.{layer_i}.mlp.down_proj.weight": model_checkpoint[f"layers.{layer_i}.feed_forward.down_proj.weight"],
            f"model.layers.{layer_i}.mlp.up_proj.weight": model_checkpoint[f"layers.{layer_i}.feed_forward.up_proj.weight"],
            f"model.layers.{layer_i}.input_layernorm.weight": model_checkpoint[f"layers.{layer_i}.attention_norm.weight"],
            f"model.layers.{layer_i}.post_attention_layernorm.weight": model_checkpoint[f"layers.{layer_i}.ffn_norm.weight"]
        }
        if  allamo_transformer_config.bias:
                state_dict[f"model.layers.{layer_i}.self_attn.q_proj.bias"] = model_checkpoint[f"layers.{layer_i}.attention.q_proj.bias"]
                state_dict[f"model.layers.{layer_i}.self_attn.k_proj.bias"] = model_checkpoint[f"layers.{layer_i}.attention.k_proj.bias"]
                state_dict[f"model.layers.{layer_i}.self_attn.v_proj.bias"] = model_checkpoint[f"layers.{layer_i}.attention.v_proj.bias"]
                state_dict[f"model.layers.{layer_i}.self_attn.o_proj.bias"] = model_checkpoint[f"layers.{layer_i}.attention.c_proj.bias"]
                state_dict[f"model.layers.{layer_i}.mlp.gate_proj.bias"] = model_checkpoint[f"layers.{layer_i}.feed_forward.gate_proj.bias"]
                state_dict[f"model.layers.{layer_i}.mlp.down_proj.bias"] = model_checkpoint[f"layers.{layer_i}.feed_forward.down_proj.bias"]
                state_dict[f"model.layers.{layer_i}.mlp.up_proj.bias"] = model_checkpoint[f"layers.{layer_i}.feed_forward.up_proj.bias"]
        if allamo_transformer_config.act_fn == "lra":
                state_dict[f"model.layers.{layer_i}.mlp.act_fn.p_coeff_0"] = model_checkpoint[f"layers.{layer_i}.feed_forward.act_fn.p_coeff_0"]
                state_dict[f"model.layers.{layer_i}.mlp.act_fn.p_coeff_1"] = model_checkpoint[f"layers.{layer_i}.feed_forward.act_fn.p_coeff_1"]
                state_dict[f"model.layers.{layer_i}.mlp.act_fn.p_coeff_2"] = model_checkpoint[f"layers.{layer_i}.feed_forward.act_fn.p_coeff_2"]
                state_dict[f"model.layers.{layer_i}.mlp.act_fn.p_coeff_3"] = model_checkpoint[f"layers.{layer_i}.feed_forward.act_fn.p_coeff_3"]
                state_dict[f"model.layers.{layer_i}.mlp.act_fn.p_coeff_4"] = model_checkpoint[f"layers.{layer_i}.feed_forward.act_fn.p_coeff_4"]
                state_dict[f"model.layers.{layer_i}.mlp.act_fn.p_coeff_5"] = model_checkpoint[f"layers.{layer_i}.feed_forward.act_fn.p_coeff_5"]
                state_dict[f"model.layers.{layer_i}.mlp.act_fn.q_coeff_1"] = model_checkpoint[f"layers.{layer_i}.feed_forward.act_fn.q_coeff_1"]
                state_dict[f"model.layers.{layer_i}.mlp.act_fn.q_coeff_2"] = model_checkpoint[f"layers.{layer_i}.feed_forward.act_fn.q_coeff_2"]
                state_dict[f"model.layers.{layer_i}.mlp.act_fn.q_coeff_3"] = model_checkpoint[f"layers.{layer_i}.feed_forward.act_fn.q_coeff_3"]
                state_dict[f"model.layers.{layer_i}.mlp.act_fn.q_coeff_4"] = model_checkpoint[f"layers.{layer_i}.feed_forward.act_fn.q_coeff_4"]
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
    if hf_model_dtype:
        torch_dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[hf_model_dtype]
        param_size_bytes = {'float32': 4, 'bfloat16': 2, 'float16': 2}[hf_model_dtype]
    else:
        # resolve model params dtype, e.g. torch.float16
        torch_dtype = model_checkpoint["lm_head.weight"].dtype
        param_size_bytes = 4 if torch_dtype == torch.float32 else 2

    for k, v in state_dict.items():
        index_dict["weight_map"][k] = filename
        param_count += v.numel()
    torch.save(state_dict, os.path.join(tmp_model_path, filename))
    logger.info(f"{param_count} params converted to HF LLaMA model")

    # Write configs
    index_dict["metadata"] = {"total_size": param_count * param_size_bytes}
    write_json(index_dict, os.path.join(tmp_model_path, "pytorch_model.bin.index.json"))

    if hf_model_type == "llama":
        from transformers import LlamaConfig
        config = LlamaConfig(
            vocab_size=allamo_transformer_config.vocab_size,
            max_position_embeddings=max_position_embeddings,
            hidden_size=allamo_transformer_config.n_embd,
            intermediate_size=intermediate_size,
            num_attention_heads=allamo_transformer_config.n_head,
            num_key_value_heads=allamo_transformer_config.num_kv_heads,
            num_hidden_layers=n_layers,
            rms_norm_eps=allamo_transformer_config.norm_eps,
            rope_theta=allamo_transformer_config.rope_freq_base,
            attention_bias=allamo_transformer_config.bias,
            mlp_bias=allamo_transformer_config.bias,
            hidden_act=allamo_transformer_config.act_fn,
        )
    elif hf_model_type == "mistral":
        from transformers import MistralConfig
        assert not allamo_transformer_config.bias, "Mistral models don't support bias"
        config = MistralConfig(
            vocab_size=allamo_transformer_config.vocab_size,
            max_position_embeddings=max_position_embeddings,
            hidden_size=allamo_transformer_config.n_embd,
            intermediate_size=intermediate_size,
            num_attention_heads=allamo_transformer_config.n_head,
            num_key_value_heads=allamo_transformer_config.num_kv_heads,
            num_hidden_layers=n_layers,
            rms_norm_eps=allamo_transformer_config.norm_eps,
            rope_theta=allamo_transformer_config.rope_freq_base,
            sliding_window=allamo_transformer_config.sliding_window,
            hidden_act=allamo_transformer_config.act_fn,
        )
    elif hf_model_type == "llama_lra":
        from lra.modeling_lra import LlamaLRAConfig
        config = LlamaLRAConfig(
            vocab_size=allamo_transformer_config.vocab_size,
            max_position_embeddings=max_position_embeddings,
            hidden_size=allamo_transformer_config.n_embd,
            intermediate_size=intermediate_size,
            num_attention_heads=allamo_transformer_config.n_head,
            num_key_value_heads=allamo_transformer_config.num_kv_heads,
            num_hidden_layers=n_layers,
            rms_norm_eps=allamo_transformer_config.norm_eps,
            rope_theta=allamo_transformer_config.rope_freq_base,
            attention_bias=allamo_transformer_config.bias,
            mlp_bias=allamo_transformer_config.bias,
            lra_group_size=allamo_transformer_config.act_fn_params["group_size"]
        )
    config.save_pretrained(tmp_model_path)
    logger.info(f"configuration for the HF LLaMA model saved")

    # Make space so we can load the model properly now.
    del state_dict
    del config_checkpoint
    del model_checkpoint
    gc.collect()

    logger.info(f"loading the checkpoint in a LLaMA model with {torch_dtype} dtype")
    if hf_model_type == "llama":
        from transformers import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(tmp_model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True)
    elif hf_model_type == "mistral":
        from transformers import MistralForCausalLM
        model = MistralForCausalLM.from_pretrained(tmp_model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True)
    elif hf_model_type == "llama_lra":
        from lra.modeling_lra import LlamaLRAForCausalLM
        model = LlamaLRAForCausalLM.from_pretrained(tmp_model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True)
    # Avoid saving this as part of the config.
    del model.config._name_or_path

    logger.info(f"saving in the Transformers format")
    model.save_pretrained(hf_model_path)
    shutil.rmtree(tmp_model_path)
    logger.info(f"conversion completed!")


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
        choices=SUPPORTED_MODEL_ARCHS,
        default='llama',
        help="Determine model type",
    )
    parser.add_argument(
        "--output_dtype",
        choices=['float32', 'bfloat16', 'float16'],
        help="Override model dtype and save the model under a specific dtype",
    )
    parser.add_argument(
        "--max_position_embeddings",
        help="Overwrite max_position_embeddings with this value",
    )
    args = parser.parse_args()
    write_model(
        checkpoint_dir_path=args.input_dir,
        checkpoint_name_base=args.checkpoint_name_base,
        hf_model_path=args.output_dir,
        hf_model_type=args.model_type,
        hf_model_dtype=args.output_dtype,
        hf_model_max_position_embeddings=args.max_position_embeddings,
    )

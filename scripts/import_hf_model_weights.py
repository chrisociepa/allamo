"""
Use this file to import Huggingface model weights to ALLaMo format.   
"""
import argparse
import dataclasses
import json
import os
import torch
from transformers import AutoModelForCausalLM
from allamo.logging import configure_logger, logger
from allamo.model.lra import get_supported_base_functions
from allamo.model.model import AllamoTransformerConfig, AllamoTransformer
from allamo.train_utils import (
    get_model_checkpoint_path,
    get_config_checkpoint_path,
)

def check_bias(state_dict):
    for k in state_dict.keys():
        if k.endswith(".bias"):
            return True
    return False

def set_mapping_or_zero(state_dicts_map, src_key, dst_key, src_state_dict, dst_state_dict):
    if src_key in src_state_dict:
        state_dicts_map[dst_key] = src_key
    else:
        dst_state_dict[dst_key].zero_()
        logger.warning(f"Reset '{dst_key}' to zero")

def import_model(hf_model_path, output_model_path, output_checkpoint_name_base):
    logger.info(f"Importing Huggingface model weights")
    hf_model = AutoModelForCausalLM.from_pretrained(hf_model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    sd_hf_model = hf_model.state_dict()
    logger.info(f"Huggingface model model loaded")

    assert hf_model.config.hidden_act in get_supported_base_functions(), f"Unsupported activation function {hf_model.config.hidden_act}"
    
    config = AllamoTransformerConfig()
    config.block_size = hf_model.config.max_position_embeddings
    config.vocab_size = hf_model.config.vocab_size
    config.n_layer = hf_model.config.num_hidden_layers
    config.n_head = hf_model.config.num_attention_heads
    config.n_embd = hf_model.config.hidden_size
    config.intermediate_size = hf_model.config.intermediate_size
    config.head_size = config.n_embd // config.n_head
    config.num_kv_heads = hf_model.config.num_key_value_heads
    config.sliding_window = None # hf_model.config.sliding_window
    config.dropout = 0.0
    config.bias = check_bias(sd_hf_model)
    config.norm_eps = hf_model.config.rms_norm_eps
    config.rope_freq_base = int(hf_model.config.rope_theta)
    config.act_fn = hf_model.config.hidden_act
    if config.act_fn == "lra":
        config.act_fn_params = {
            "base_fn": "swishglu", # required only for resetting params during training init
            "dim": hf_model.config.intermediate_size, 
            "group_size": hf_model.config.lra_group_size
        }

    logger.info(f"initializing vanilla ALLaMo model")
    model = AllamoTransformer(config)
    
    logger.info(f"preparing weights")
    state_dicts_map = {}
    model_sd = model.state_dict()
    for layer_i in range(config.n_layer):
        state_dicts_map[f"layers.{layer_i}.attention.q_proj.weight"] = f"model.layers.{layer_i}.self_attn.q_proj.weight"
        state_dicts_map[f"layers.{layer_i}.attention.k_proj.weight"] = f"model.layers.{layer_i}.self_attn.k_proj.weight"
        state_dicts_map[f"layers.{layer_i}.attention.v_proj.weight"] = f"model.layers.{layer_i}.self_attn.v_proj.weight"
        state_dicts_map[f"layers.{layer_i}.attention.c_proj.weight"] = f"model.layers.{layer_i}.self_attn.o_proj.weight"
        state_dicts_map[f"layers.{layer_i}.feed_forward.gate_proj.weight"] = f"model.layers.{layer_i}.mlp.gate_proj.weight"
        state_dicts_map[f"layers.{layer_i}.feed_forward.down_proj.weight"] = f"model.layers.{layer_i}.mlp.down_proj.weight"
        state_dicts_map[f"layers.{layer_i}.feed_forward.up_proj.weight"] = f"model.layers.{layer_i}.mlp.up_proj.weight"
        state_dicts_map[f"layers.{layer_i}.attention_norm.weight"] = f"model.layers.{layer_i}.input_layernorm.weight"
        state_dicts_map[f"layers.{layer_i}.ffn_norm.weight"] = f"model.layers.{layer_i}.post_attention_layernorm.weight"
        
        if config.bias:
            set_mapping_or_zero(state_dicts_map, f"model.layers.{layer_i}.self_attn.q_proj.bias", f"layers.{layer_i}.attention.q_proj.bias", sd_hf_model, model_sd)
            set_mapping_or_zero(state_dicts_map, f"model.layers.{layer_i}.self_attn.k_proj.bias", f"layers.{layer_i}.attention.k_proj.bias", sd_hf_model, model_sd)
            set_mapping_or_zero(state_dicts_map, f"model.layers.{layer_i}.self_attn.v_proj.bias", f"layers.{layer_i}.attention.v_proj.bias", sd_hf_model, model_sd)
            set_mapping_or_zero(state_dicts_map, f"model.layers.{layer_i}.self_attn.o_proj.bias", f"layers.{layer_i}.attention.c_proj.bias", sd_hf_model, model_sd)
            set_mapping_or_zero(state_dicts_map, f"model.layers.{layer_i}.mlp.gate_proj.bias", f"layers.{layer_i}.feed_forward.gate_proj.bias", sd_hf_model, model_sd)
            set_mapping_or_zero(state_dicts_map, f"model.layers.{layer_i}.mlp.down_proj.bias", f"layers.{layer_i}.feed_forward.down_proj.bias", sd_hf_model, model_sd)
            set_mapping_or_zero(state_dicts_map, f"model.layers.{layer_i}.mlp.up_proj.bias", f"layers.{layer_i}.feed_forward.up_proj.bias", sd_hf_model, model_sd)
        
    state_dicts_map["tok_embeddings.weight"] = "model.embed_tokens.weight"
    state_dicts_map["norm.weight"] = "model.norm.weight"
    state_dicts_map["lm_head.weight"] = "lm_head.weight"
    
    logger.info(f"checking params coverage")
    for k, v in model_sd.items():
        if k not in state_dicts_map:
            logger.info(f"{k} param won't be updated in the ALLaMo model!")
            
    for k, v in sd_hf_model.items():
        if k not in state_dicts_map.values():
            logger.info(f"{k} param won't be copied to the ALLaMo model!")
    
    logger.info(f"copying params to the ALLaMo model")
    param_count = 0
    for k, v in state_dicts_map.items():
        if not k.endswith('rotary_emb.inv_freq'):
            assert sd_hf_model[v].shape == model_sd[k].shape
            with torch.no_grad():
                model_sd[k].copy_(sd_hf_model[v])
            param_count += model_sd[k].numel()
    logger.info(f"{param_count} params copied to the ALLaMo model")
    
    for k, _ in model_sd.items():
        if k in state_dicts_map and not torch.all(torch.eq(model_sd[k], sd_hf_model[state_dicts_map[k]])):
            logger.info(f"{k} param in the ALLaMo model is not the same as {state_dicts_map[k]} param in the source model!")
    logger.info(f"params verified")
    
    config_checkpoint = {
        'model_args': dataclasses.asdict(config)
    }
    ckpt_file_path = get_config_checkpoint_path(output_checkpoint_name_base, output_model_path)
    logger.info(f"saving config checkpoint to {ckpt_file_path}")
    with open(ckpt_file_path, "w", encoding="utf-8") as f:
        json.dump(config_checkpoint, f, indent=4, ensure_ascii=False)
    ckpt_file_path = get_model_checkpoint_path(output_checkpoint_name_base, output_model_path)
    logger.info(f"saving model checkpoint to {ckpt_file_path}")
    torch.save(model_sd, ckpt_file_path)
    logger.info(f"checkpoint files saved in {output_model_path}")
    
if __name__ == '__main__':
    configure_logger()
    parser = argparse.ArgumentParser(description='Import Huggingface model weights to ALLaMo format')
    parser.add_argument(
        "--huggingface_model",
        help="Huggingface model path",
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
    
    os.makedirs(args.output_dir, exist_ok=True)
    import_model(
        hf_model_path=args.huggingface_model,
        output_model_path=args.output_dir,
        output_checkpoint_name_base=args.output_checkpoint_name_base,
    )
    logger.info("import completed")

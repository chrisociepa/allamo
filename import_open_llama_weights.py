"""
Use this file to import Huggingface LlamaForCausalLM weights to ALLaMo model.   
"""
import argparse
import datetime
import json
import os
import torch
import shutil
from model import AllamoTransformerConfig, AllamoTransformer
from transformers import LlamaForCausalLM

def timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def read_json(path):
    with open(path, "r") as f:
        return json.load(f)
        
def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)
        
def calculate_multiple_of(intermediate_size, hidden_size):
    for possible_multiple in range(1, intermediate_size + 1):
        calculated_intermediate_size = possible_multiple * ((int(hidden_size * 8 / 3) + possible_multiple - 1) // possible_multiple)
        if calculated_intermediate_size == intermediate_size:
            return possible_multiple
    return None
    
def compute_intermediate_size(config):
    return config.multiple_of * ((int(config.n_embd * 8 / 3) + config.multiple_of - 1) // config.multiple_of)
    
def import_model(hf_model_path, output_model_path):
    print(f"{timestamp()} - start importing Huggingface LlamaForCausalLM weights")
    hf_model = LlamaForCausalLM.from_pretrained(hf_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    print(f"{timestamp()} - Huggingface LlamaForCausalLM model loaded")
    
    config = AllamoTransformerConfig()
    config.block_size = hf_model.config.max_position_embeddings
    config.vocab_size = hf_model.config.vocab_size
    config.n_layer = hf_model.config.num_hidden_layers
    config.n_head = hf_model.config.num_attention_heads
    config.n_embd = hf_model.config.hidden_size
    config.head_size = config.n_embd // config.n_head
    config.dropout = 0.0
    config.bias = False
    config.multiple_of = calculate_multiple_of(hf_model.config.intermediate_size, hf_model.config.hidden_size)
    config.norm_eps = hf_model.config.rms_norm_eps
    assert hf_model.config.intermediate_size == compute_intermediate_size(config)

    print(f"{timestamp()} - initializing vanilla ALLaMo model")
    # Open LLaMA models are delivered with float16 weights
    torch.set_default_tensor_type(torch.HalfTensor)
    model = AllamoTransformer(config)
    torch.set_default_tensor_type(torch.FloatTensor)
    
    state_dicts_map = {}
    sd_hf_model = hf_model.state_dict()
    model_sd = model.state_dict()
    for layer_i in range(config.n_layer):
        state_dicts_map[f"layers.{layer_i}.attention.q_proj.weight"] = f"model.layers.{layer_i}.self_attn.q_proj.weight"
        state_dicts_map[f"layers.{layer_i}.attention.k_proj.weight"] = f"model.layers.{layer_i}.self_attn.k_proj.weight"
        state_dicts_map[f"layers.{layer_i}.attention.v_proj.weight"] = f"model.layers.{layer_i}.self_attn.v_proj.weight"
        state_dicts_map[f"layers.{layer_i}.attention.c_proj.weight"] = f"model.layers.{layer_i}.self_attn.o_proj.weight"
        state_dicts_map[f"layers.{layer_i}.attention.rotary_emb.inv_freq"] = f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq"
        state_dicts_map[f"layers.{layer_i}.feed_forward.gate_proj.weight"] = f"model.layers.{layer_i}.mlp.gate_proj.weight"
        state_dicts_map[f"layers.{layer_i}.feed_forward.down_proj.weight"] = f"model.layers.{layer_i}.mlp.down_proj.weight"
        state_dicts_map[f"layers.{layer_i}.feed_forward.up_proj.weight"] = f"model.layers.{layer_i}.mlp.up_proj.weight"
        state_dicts_map[f"layers.{layer_i}.attention_norm.weight"] = f"model.layers.{layer_i}.input_layernorm.weight"
        state_dicts_map[f"layers.{layer_i}.ffn_norm.weight"] = f"model.layers.{layer_i}.post_attention_layernorm.weight"
    state_dicts_map["tok_embeddings.weight"] = "model.embed_tokens.weight"
    state_dicts_map["norm.weight"] = "model.norm.weight"
    state_dicts_map["lm_head.weight"] = "lm_head.weight"
    
    print(f"{timestamp()} - checking params coverage")
    for k, v in model_sd.items():
        if k not in state_dicts_map:
            print(f"{k} param won't be updated in the ALLaMo model!")
            
    for k, v in sd_hf_model.items():
        if k not in state_dicts_map.values():
            print(f"{k} param won't be copied to the ALLaMo model!")
    
    print(f"{timestamp()} - copying params to the ALLaMo model")
    param_count = 0
    for k, v in state_dicts_map.items():
        assert sd_hf_model[v].shape == model_sd[k].shape
        with torch.no_grad():
            model_sd[k].copy_(sd_hf_model[v])
        param_count += model_sd[k].numel()
    print(f"{timestamp()} - {param_count} params copied to the ALLaMo model")
    
    for k, _ in model_sd.items():
        if not torch.all(torch.eq(model_sd[k], sd_hf_model[state_dicts_map[k]])):
            print(f"{k} param in the ALLaMo model is not the same as {state_dicts_map[k]} param in the source model!")
    print(f"{timestamp()} - params checked")
    
    ckpt_file_name = 'ckpt.pt' #'import_ckpt.pt'
    config_checkpoint = {
        'model_args': config
    }
    ckpt_file_path = os.path.join(output_model_path, 'config_' + ckpt_file_name)
    print(f"saving config checkpoint to {ckpt_file_path}")
    torch.save(config_checkpoint, ckpt_file_path)
    ckpt_file_path = os.path.join(output_model_path, 'model_' + ckpt_file_name)
    print(f"saving model checkpoint to {ckpt_file_path}")
    torch.save(model_sd, ckpt_file_path)
    print(f"{timestamp()} - checkpoint files saved in {output_model_path}")
    
def main():
    parser = argparse.ArgumentParser(description='Import Huggingface LlamaForCausalLM weights to ALLaMo model')
    parser.add_argument(
        "--huggingface_model",
        help="Huggingface model path",
    )
    parser.add_argument(
        "--output_dir",
        help="Path to output directory",
    )
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    import_model(
        hf_model_path=args.huggingface_model,
        output_model_path=args.output_dir,
    )
    
if __name__ == '__main__':
    main()    
    print(f"{timestamp()} - import completed")
    

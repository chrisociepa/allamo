"""
Use this file to import LLaMA weights to ALLaMo model.   
"""
import argparse
import datetime
import json
import os
import torch
import shutil
from model import AllamoTransformerConfig, AllamoTransformer

def timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def read_json(path):
    with open(path, "r") as f:
        return json.load(f)
        
def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)
        
# permute for sliced rotary
def permute(w, dim, n_heads):
    return w.view(n_heads, dim // n_heads // 2, 2, dim).transpose(1, 2).reshape(dim, dim)
    
def import_model(input_base_path, output_model_path, max_num_layers, max_block_size):
    print(f"{timestamp()} - start importing llama weights")
    params = read_json(os.path.join(input_base_path, "params.json"))
    
    config = AllamoTransformerConfig()
    config.block_size = min(max_block_size, 2048) if max_block_size else 2048
    config.vocab_size = 32000
    config.n_layer = min(max_num_layers, params["n_layers"]) if max_num_layers else params["n_layers"]
    config.n_head = params["n_heads"]
    config.n_embd = params["dim"]
    config.head_size = config.n_embd // config.n_head
    config.dropout = 0.0
    config.bias = False
    config.multiple_of = params["multiple_of"]
    config.norm_eps = params["norm_eps"]
    
    # Switch to half tensors
    torch.set_default_tensor_type(torch.cuda.HalfTensor)

    print(f"{timestamp()} - initializing vanilla model")
    model = AllamoTransformer(config)
    
    print(f"{timestamp()} - loading llama weights")
    # Sharded models are not supported!
    loaded = torch.load(os.path.join(input_base_path, "consolidated.00.pth"), map_location="cpu")

    print(f"{timestamp()} - copying llama weights to the model")
    theta = 10000.0
    inv_freq = 1.0 / (theta ** (torch.arange(0, config.head_size, 2).float() / config.head_size))
    
    # Switch back to full tensors
    torch.set_default_tensor_type(torch.FloatTensor)
    
    param_count = 0
    index_dict = {"weight_map": {}}
    model_sd = model.state_dict()
    for layer_i in range(config.n_layer):
        print(f"{timestamp()} - copying weights in layer {layer_i}")
        state_dict = {
            f"layers.{layer_i}.attention.q_proj.weight": permute(loaded[f"layers.{layer_i}.attention.wq.weight"], config.n_embd, config.n_head),
            f"layers.{layer_i}.attention.k_proj.weight": permute(loaded[f"layers.{layer_i}.attention.wk.weight"], config.n_embd, config.n_head),
            f"layers.{layer_i}.attention.v_proj.weight": loaded[f"layers.{layer_i}.attention.wv.weight"],
            f"layers.{layer_i}.attention.c_proj.weight": loaded[f"layers.{layer_i}.attention.wo.weight"],
            f"layers.{layer_i}.feed_forward.gate_proj.weight": loaded[f"layers.{layer_i}.feed_forward.w1.weight"],
            f"layers.{layer_i}.feed_forward.down_proj.weight": loaded[f"layers.{layer_i}.feed_forward.w2.weight"],
            f"layers.{layer_i}.feed_forward.up_proj.weight": loaded[f"layers.{layer_i}.feed_forward.w3.weight"],
            f"layers.{layer_i}.attention_norm.weight": loaded[f"layers.{layer_i}.attention_norm.weight"],
            f"layers.{layer_i}.ffn_norm.weight": loaded[f"layers.{layer_i}.ffn_norm.weight"],
        }
        state_dict[f"layers.{layer_i}.attention.rotary_emb.inv_freq"] = inv_freq
        for k, v in state_dict.items():
            assert v.shape == model_sd[k].shape
            with torch.no_grad():
                model_sd[k].copy_(v)
            param_count += v.numel()
        
    state_dict = {
        "tok_embeddings.weight": loaded["tok_embeddings.weight"],
        "norm.weight": loaded["norm.weight"],
        "lm_head.weight": loaded["output.weight"],
    }
    for k, v in state_dict.items():
        assert v.shape == model_sd[k].shape
        with torch.no_grad():
            model_sd[k].copy_(v)
        param_count += v.numel()
    print(f"{timestamp()} - {param_count} params imported to the model")
        
    ckpt_file_name = 'import_ckpt.pt'
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
    
def import_tokenizer(input_tokenizer_path, output_model_path, max_block_size):
    print(f"{timestamp()} - start importing tokenizer")
    model_max_length = min(max_block_size, 2048) if max_block_size else 2048
    write_json({}, os.path.join(output_model_path, "special_tokens_map.json"))
    write_json(
        {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "model_max_length": model_max_length,
            "tokenizer_class": "LlamaTokenizer",
            "unk_token": "<unk>",
        },
        os.path.join(output_model_path, "tokenizer_config.json"),
    )
    shutil.copyfile(input_tokenizer_path, os.path.join(output_model_path, "tokenizer.model"))
    print(f"{timestamp()} - tokenizer files saved in {output_model_path}")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Import LLaMA weights to ALLaMo model')
    parser.add_argument('--input_data_dir', type=str, help='Path to a directory with LLaMA model files')
    parser.add_argument('--input_tokenizer_path', type=str, help='Path to LLaMA tokenizer.model file')
    parser.add_argument('--output_data_dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--max_num_layers', type=int, help='Crop layers to make the model smaller')
    parser.add_argument('--max_block_size', type=int, help='Crop block size to make the model smaller')
    args = parser.parse_args()

    os.makedirs(args.output_data_dir, exist_ok=True)
    
    if args.input_tokenizer_path:
        import_tokenizer(args.input_tokenizer_path, args.output_data_dir, args.max_block_size)

    if args.input_data_dir:
        import_model(args.input_data_dir, args.output_data_dir, args.max_num_layers, args.max_block_size)
    
    print(f"{timestamp()} - import completed")

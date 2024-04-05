"""
Use this file to import Huggingface MistralForCausalLM weights to ALLaMo model.   
"""
import argparse
import logging
import os
import sys
import torch
from transformers import MistralForCausalLM

sys.path.append(os.path.abspath('..'))
from model import AllamoTransformerConfig, AllamoTransformer

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger('AllamoModelImporter')

def import_model(hf_model_path, output_model_path):
    logger.info(f"Importing Huggingface MistralForCausalLM weights")
    hf_model = MistralForCausalLM.from_pretrained(hf_model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    logger.info(f"Huggingface MistralForCausalLM model loaded")

    assert hf_model.config.hidden_act == "silu"
    
    config = AllamoTransformerConfig()
    config.block_size = hf_model.config.sliding_window # hf_model.config.max_position_embeddings
    config.vocab_size = hf_model.config.vocab_size
    config.n_layer = hf_model.config.num_hidden_layers
    config.n_head = hf_model.config.num_attention_heads
    config.n_embd = hf_model.config.hidden_size
    config.intermediate_size = hf_model.config.intermediate_size
    config.head_size = config.n_embd // config.n_head
    config.num_kv_heads = hf_model.config.num_key_value_heads
    config.sliding_window = hf_model.config.sliding_window
    config.dropout = 0.0
    config.bias = False
    config.norm_eps = hf_model.config.rms_norm_eps
    config.rope_freq_base = int(hf_model.config.rope_theta)

    logger.info(f"initializing vanilla ALLaMo model")
    model = AllamoTransformer(config)
    
    logger.info(f"preparing weights")
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
        if not torch.all(torch.eq(model_sd[k], sd_hf_model[state_dicts_map[k]])):
            logger.info(f"{k} param in the ALLaMo model is not the same as {state_dicts_map[k]} param in the source model!")
    logger.info(f"params verified")
    
    ckpt_file_name = 'import_ckpt.pt'
    config_checkpoint = {
        'model_args': config
    }
    ckpt_file_path = os.path.join(output_model_path, 'config_' + ckpt_file_name)
    print(f"saving config checkpoint to {ckpt_file_path}")
    with open(ckpt_file_path, "w", encoding="utf-8") as f:
        json.dump(config_checkpoint, f, indent=4, ensure_ascii=False)
    ckpt_file_path = os.path.join(output_model_path, 'model_' + ckpt_file_name)
    print(f"saving model checkpoint to {ckpt_file_path}")
    torch.save(model_sd, ckpt_file_path)
    print(f"checkpoint files saved in {output_model_path}")
    
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
    print(f"import completed")
    

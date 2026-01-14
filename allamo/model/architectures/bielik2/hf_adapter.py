import json
import gc
import os
import shutil
import torch
from allamo.logging import logger
from allamo.model.architectures.bielik2.model import Bielik2Config, Bielik2Model
from allamo.model.hf_adapter import BaseHFAdapter
from allamo.train_utils import remove_unwanted_prefix_from_model_state_dict

class Bielik2HFAdapter(BaseHFAdapter):

    def get_model_config_class(self):
        return Bielik2Config
    
    def get_model_class(self):
        return Bielik2Model

    def create_model_config(self, hf_model):
        config = Bielik2Config()
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
        config.bias = self.check_bias(hf_model.state_dict())
        config.norm_eps = hf_model.config.rms_norm_eps
        config.rope_freq_base = int(hf_model.config.rope_theta)
        config.qk_norm = getattr(hf_model.config, "qk_norm", False)
        config.gated_mlp = hf_model.config.hidden_act != "xielu" # heurystic that works for Apertus models
        config.attn_output_gate = False
        config.act_fn = hf_model.config.hidden_act
        if config.act_fn == "lra":
            config.act_fn_params = {
                "base_fn": "swishglu", # required only for resetting params during training init
                "dim": hf_model.config.intermediate_size, 
                "group_size": hf_model.config.lra_group_size
            }
        return config

    def create_weight_mapping(self, hf_model_sd, model_sd, config):
        state_dicts_map = {}
        for layer_i in range(config.n_layer):
            state_dicts_map[f"layers.{layer_i}.attention.q_proj.weight"] = f"model.layers.{layer_i}.self_attn.q_proj.weight"
            state_dicts_map[f"layers.{layer_i}.attention.k_proj.weight"] = f"model.layers.{layer_i}.self_attn.k_proj.weight"
            state_dicts_map[f"layers.{layer_i}.attention.v_proj.weight"] = f"model.layers.{layer_i}.self_attn.v_proj.weight"
            state_dicts_map[f"layers.{layer_i}.attention.c_proj.weight"] = f"model.layers.{layer_i}.self_attn.o_proj.weight"
            state_dicts_map[f"layers.{layer_i}.feed_forward.down_proj.weight"] = f"model.layers.{layer_i}.mlp.down_proj.weight"
            state_dicts_map[f"layers.{layer_i}.feed_forward.up_proj.weight"] = f"model.layers.{layer_i}.mlp.up_proj.weight"
            state_dicts_map[f"layers.{layer_i}.attention_norm.weight"] = f"model.layers.{layer_i}.input_layernorm.weight"
            state_dicts_map[f"layers.{layer_i}.ffn_norm.weight"] = f"model.layers.{layer_i}.post_attention_layernorm.weight"

            if config.gated_mlp:
                state_dicts_map[f"layers.{layer_i}.feed_forward.gate_proj.weight"] = f"model.layers.{layer_i}.mlp.gate_proj.weight"
            
            if config.bias:
                self.set_mapping_or_zero(state_dicts_map, f"model.layers.{layer_i}.self_attn.q_proj.bias", f"layers.{layer_i}.attention.q_proj.bias", hf_model_sd, model_sd)
                self.set_mapping_or_zero(state_dicts_map, f"model.layers.{layer_i}.self_attn.k_proj.bias", f"layers.{layer_i}.attention.k_proj.bias", hf_model_sd, model_sd)
                self.set_mapping_or_zero(state_dicts_map, f"model.layers.{layer_i}.self_attn.v_proj.bias", f"layers.{layer_i}.attention.v_proj.bias", hf_model_sd, model_sd)
                self.set_mapping_or_zero(state_dicts_map, f"model.layers.{layer_i}.self_attn.o_proj.bias", f"layers.{layer_i}.attention.c_proj.bias", hf_model_sd, model_sd)
                self.set_mapping_or_zero(state_dicts_map, f"model.layers.{layer_i}.mlp.gate_proj.bias", f"layers.{layer_i}.feed_forward.gate_proj.bias", hf_model_sd, model_sd)
                self.set_mapping_or_zero(state_dicts_map, f"model.layers.{layer_i}.mlp.down_proj.bias", f"layers.{layer_i}.feed_forward.down_proj.bias", hf_model_sd, model_sd)
                self.set_mapping_or_zero(state_dicts_map, f"model.layers.{layer_i}.mlp.up_proj.bias", f"layers.{layer_i}.feed_forward.up_proj.bias", hf_model_sd, model_sd)
            
            if config.qk_norm:
                state_dicts_map[f"layers.{layer_i}.q_norm.weight"] = f"model.layers.{layer_i}.self_attn.q_norm.weight"
                state_dicts_map[f"layers.{layer_i}.k_norm.weight"] = f"model.layers.{layer_i}.self_attn.k_norm.weight"
            
            if config.act_fn == "xielu":
                state_dicts_map[f"layers.{layer_i}.feed_forward.act_fn.alpha_p"] = f"model.layers.{layer_i}.mlp.act_fn.alpha_p"
                state_dicts_map[f"layers.{layer_i}.feed_forward.act_fn.alpha_n"] = f"model.layers.{layer_i}.mlp.act_fn.alpha_n"
            
        state_dicts_map["tok_embeddings.weight"] = "model.embed_tokens.weight"
        state_dicts_map["norm.weight"] = "model.norm.weight"
        state_dicts_map["lm_head.weight"] = "lm_head.weight"
        return state_dicts_map

    def to_hf_model(self, checkpoint_dir_path, checkpoint_name_base, hf_model_path, hf_model_type, hf_model_dtype, hf_model_max_position_embeddings):
        SUPPORTED_MODEL_ARCHS = ['llama', 'mistral', 'apertus', 'llama_lra']
        assert hf_model_type in SUPPORTED_MODEL_ARCHS, f"HF model {hf_model_type} architecture is not supported"

        os.makedirs(hf_model_path, exist_ok=True)
        hf_intermadiate_model_path = os.path.join(hf_model_path, "tmp")
        os.makedirs(hf_intermadiate_model_path, exist_ok=True)

        logger.info(f"Loading model checkpoint from {checkpoint_dir_path}...")
        config_checkpoint, model_checkpoint = self.load_model_checkpoint(checkpoint_name_base, checkpoint_dir_path)
        config = self.get_model_config_class()(**config_checkpoint['model_args'])

        remove_unwanted_prefix_from_model_state_dict(model_checkpoint)

        logger.info(f"Converting parameters from the checkpoint model")
        param_count = 0
        index_dict = {"weight_map": {}}
        for layer_i in range(config.n_layer):
            logger.info(f"converting weights in layer {layer_i}")
            filename = f"pytorch_model-{layer_i + 1}-of-{config.n_layer + 1}.bin"
            state_dict = {
                f"model.layers.{layer_i}.self_attn.q_proj.weight": model_checkpoint[f"layers.{layer_i}.attention.q_proj.weight"],
                f"model.layers.{layer_i}.self_attn.k_proj.weight": model_checkpoint[f"layers.{layer_i}.attention.k_proj.weight"],
                f"model.layers.{layer_i}.self_attn.v_proj.weight": model_checkpoint[f"layers.{layer_i}.attention.v_proj.weight"],
                f"model.layers.{layer_i}.self_attn.o_proj.weight": model_checkpoint[f"layers.{layer_i}.attention.c_proj.weight"],
                f"model.layers.{layer_i}.mlp.down_proj.weight": model_checkpoint[f"layers.{layer_i}.feed_forward.down_proj.weight"],
                f"model.layers.{layer_i}.mlp.up_proj.weight": model_checkpoint[f"layers.{layer_i}.feed_forward.up_proj.weight"],
                f"model.layers.{layer_i}.input_layernorm.weight": model_checkpoint[f"layers.{layer_i}.attention_norm.weight"],
                f"model.layers.{layer_i}.post_attention_layernorm.weight": model_checkpoint[f"layers.{layer_i}.ffn_norm.weight"]
            }

            if config.gated_mlp:
                state_dict[f"model.layers.{layer_i}.mlp.gate_proj.weight"] = model_checkpoint[f"layers.{layer_i}.feed_forward.gate_proj.weight"]

            if config.bias:
                state_dict[f"model.layers.{layer_i}.self_attn.q_proj.bias"] = model_checkpoint[f"layers.{layer_i}.attention.q_proj.bias"]
                state_dict[f"model.layers.{layer_i}.self_attn.k_proj.bias"] = model_checkpoint[f"layers.{layer_i}.attention.k_proj.bias"]
                state_dict[f"model.layers.{layer_i}.self_attn.v_proj.bias"] = model_checkpoint[f"layers.{layer_i}.attention.v_proj.bias"]
                state_dict[f"model.layers.{layer_i}.self_attn.o_proj.bias"] = model_checkpoint[f"layers.{layer_i}.attention.c_proj.bias"]
                state_dict[f"model.layers.{layer_i}.mlp.gate_proj.bias"] = model_checkpoint[f"layers.{layer_i}.feed_forward.gate_proj.bias"]
                state_dict[f"model.layers.{layer_i}.mlp.down_proj.bias"] = model_checkpoint[f"layers.{layer_i}.feed_forward.down_proj.bias"]
                state_dict[f"model.layers.{layer_i}.mlp.up_proj.bias"] = model_checkpoint[f"layers.{layer_i}.feed_forward.up_proj.bias"]
            
            if config.qk_norm:
                state_dict[f"model.layers.{layer_i}.self_attn.q_norm.weight"] = model_checkpoint[f"layers.{layer_i}.attention.q_norm.weight"]
                state_dict[f"model.layers.{layer_i}.self_attn.k_norm.weight"] = model_checkpoint[f"layers.{layer_i}.attention.k_norm.weight"]
            
            if config.act_fn == "lra":
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
            elif config.act_fn == "xielu":
                state_dict[f"model.layers.{layer_i}.mlp.act_fn.alpha_p"] = model_checkpoint[f"layers.{layer_i}.feed_forward.act_fn.alpha_p"]
                state_dict[f"model.layers.{layer_i}.mlp.act_fn.alpha_n"] = model_checkpoint[f"layers.{layer_i}.feed_forward.act_fn.alpha_n"]
            
            for k, v in state_dict.items():
                index_dict["weight_map"][k] = filename
                param_count += v.numel()
            torch.save(state_dict, os.path.join(hf_intermadiate_model_path, filename))

        filename = f"pytorch_model-{config.n_layer + 1}-of-{config.n_layer + 1}.bin"
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
        torch.save(state_dict, os.path.join(hf_intermadiate_model_path, filename))

        index_dict["metadata"] = {"total_size": param_count * param_size_bytes}
        with open(os.path.join(hf_intermadiate_model_path, "pytorch_model.bin.index.json"), "w") as f:
            json.dump(index_dict, f)
        
        logger.info(f"{param_count} params converted to new intermadiate HF model and saved in {hf_intermadiate_model_path}")

        max_position_embeddings = config.block_size if hf_model_max_position_embeddings is None else hf_model_max_position_embeddings
        if hf_model_type == "llama":
            from transformers import LlamaConfig
            hf_config = LlamaConfig(
                vocab_size=config.vocab_size,
                max_position_embeddings=max_position_embeddings,
                hidden_size=config.n_embd,
                intermediate_size=config.intermediate_size,
                num_attention_heads=config.n_head,
                num_key_value_heads=config.num_kv_heads,
                num_hidden_layers=config.n_layer,
                rms_norm_eps=config.norm_eps,
                rope_theta=config.rope_freq_base,
                attention_bias=config.bias,
                mlp_bias=config.bias,
                hidden_act=config.act_fn,
            )
        elif hf_model_type == "mistral":
            from transformers import MistralConfig
            assert not config.bias, "Mistral models don't support bias"
            hf_config = MistralConfig(
                vocab_size=config.vocab_size,
                max_position_embeddings=max_position_embeddings,
                hidden_size=config.n_embd,
                intermediate_size=config.intermediate_size,
                num_attention_heads=config.n_head,
                num_key_value_heads=config.num_kv_heads,
                num_hidden_layers=config.n_layer,
                rms_norm_eps=config.norm_eps,
                rope_theta=config.rope_freq_base,
                sliding_window=config.sliding_window,
                hidden_act=config.act_fn,
            )
        elif hf_model_type == "apertus":
            from transformers import ApertusConfig
            hf_config = ApertusConfig(
                vocab_size=config.vocab_size,
                max_position_embeddings=max_position_embeddings,
                hidden_size=config.n_embd,
                intermediate_size=config.intermediate_size,
                num_attention_heads=config.n_head,
                num_key_value_heads=config.num_kv_heads,
                num_hidden_layers=config.n_layer,
                rms_norm_eps=config.norm_eps,
                rope_theta=config.rope_freq_base,
                attention_bias=config.bias,
                mlp_bias=config.bias,
                qk_norm=config.qk_norm,
                hidden_act=config.act_fn,
            )
        elif hf_model_type == "llama_lra":
            from allamo.model.architectures.bielik2.modeling_hf_lra import LlamaLRAConfig
            hf_config = LlamaLRAConfig(
                vocab_size=config.vocab_size,
                max_position_embeddings=max_position_embeddings,
                hidden_size=config.n_embd,
                intermediate_size=config.intermediate_size,
                num_attention_heads=config.n_head,
                num_key_value_heads=config.num_kv_heads,
                num_hidden_layers=config.n_layer,
                rms_norm_eps=config.norm_eps,
                rope_theta=config.rope_freq_base,
                attention_bias=config.bias,
                mlp_bias=config.bias,
                lra_group_size=config.act_fn_params["group_size"]
            )
        hf_config.save_pretrained(hf_intermadiate_model_path)
        logger.info(f"HF model configuration saved in {hf_intermadiate_model_path}")

        # Make space so we can load the model properly now.
        del state_dict
        del config_checkpoint
        del model_checkpoint
        gc.collect()

        logger.info(f"Loading the intermadiate HF checkpoint in {torch_dtype} dtype")
        if hf_model_type == "llama":
            from transformers import LlamaForCausalLM
            hf_model = LlamaForCausalLM.from_pretrained(hf_intermadiate_model_path, dtype=torch_dtype, low_cpu_mem_usage=True)
        elif hf_model_type == "mistral":
            from transformers import MistralForCausalLM
            hf_model = MistralForCausalLM.from_pretrained(hf_intermadiate_model_path, dtype=torch_dtype, low_cpu_mem_usage=True)
        elif hf_model_type == "apertus":
            from transformers import ApertusForCausalLM
            hf_model = ApertusForCausalLM.from_pretrained(hf_intermadiate_model_path, dtype=torch_dtype, low_cpu_mem_usage=True)
        elif hf_model_type == "llama_lra":
            from allamo.model.architectures.bielik2.modeling_hf_lra import LlamaLRAForCausalLM
            hf_model = LlamaLRAForCausalLM.from_pretrained(hf_intermadiate_model_path, dtype=torch_dtype, low_cpu_mem_usage=True)
        
        # Avoid saving this as part of the config.
        del hf_model.config._name_or_path

        logger.info(f"Saving in the final HF format")
        hf_model.save_pretrained(hf_model_path)
        shutil.rmtree(hf_intermadiate_model_path)
        logger.info(f"Export to HF format completed!")


adapter = Bielik2HFAdapter()
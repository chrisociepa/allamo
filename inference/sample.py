"""
Use this file to sample from a trained model.
"""
import dataclasses
import json
import os
import time
import torch
from allamo.logging import configure_logger, logger
from allamo.configuration import AllamoConfiguration
from allamo.model import AllamoTransformerConfig, AllamoTransformer
from allamo.torch_utils import configure_torch
from allamo.train_utils import remove_unwanted_prefix_from_model_state_dict

class AllamoSampler:

    def __init__(self, config: AllamoConfiguration):
        configure_logger(config, with_file_handler=False)
        self.config = config
        configure_torch(config)

        if config.init_from == 'resume_last':
            checkpoint_name = 'last_eval_ckpt'
        else:
            checkpoint_name = 'ckpt'
        ckpt_dir = config.checkpoint_path if config.checkpoint_path else config.out_dir
        model_config_fields = [f.name for f in dataclasses.fields(AllamoTransformerConfig)]
        logger.info(f"Loading '{checkpoint_name}' checkpoint files from {ckpt_dir}...")
        with open(os.path.join(ckpt_dir, f'config_{checkpoint_name}.json'), "r", encoding="utf-8") as f:
            config_checkpoint = json.load(f)
        for k in model_config_fields:
            if hasattr(config, k) and k in config_checkpoint['model_args']:
                setattr(config, k, config_checkpoint['model_args'][k])
                
        model_checkpoint = torch.load(os.path.join(ckpt_dir, f'model_{checkpoint_name}.pt'), map_location='cpu')
        self.__load_model(config, config_checkpoint, model_checkpoint, model_config_fields)
        self.__load_tokenizer(config)
        del model_checkpoint
            
    def __load_model(self, config: AllamoConfiguration, config_checkpoint, model_checkpoint, model_config_fields):
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'bfloat16-true': torch.bfloat16, 'float16': torch.float16}[config.dtype]
        model_args = {k: getattr(config, k) for k in model_config_fields if hasattr(config, k)}
        modelConf = AllamoTransformerConfig(**model_args)
        model = AllamoTransformer(modelConf)
        
        remove_unwanted_prefix_from_model_state_dict(model_checkpoint)
        model.load_state_dict(model_checkpoint)
        model.eval()
        model.to(device=config.device, dtype=ptdtype)
        if config.compile:
            model = torch.compile(model)
        self.model = model
        logger.info(f"Model loaded from checkpoint")
        if 'iter_num' in config_checkpoint:
            logger.info(f"Last model iteration: {config_checkpoint['iter_num']}")
        
    def __load_tokenizer(self, config: AllamoConfiguration):
        tiktoken_tokenizer_name = config.tiktoken_tokenizer_name
        hf_tokenizer_path = config.hf_tokenizer_path
        if hf_tokenizer_path is not None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_path)
            logger.info(f"HuggingFace tokenizer loaded: {hf_tokenizer_path}")
        elif tiktoken_tokenizer_name is not None:
            import tiktoken
            tokenizer = tiktoken.get_encoding(tiktoken_tokenizer_name)
            logger.info(f"Tiktoken tokenizer loaded: {tiktoken_tokenizer_name}")
        else:
            raise Exception('Tokenizer is not provided. Please specify either a Tiktoken tokenizer or a HuggingFace tokenizer')
        # ensure that the tokenizer and model vocabulary sizes are equal
        assert len(tokenizer) == self.model.config.vocab_size
        self.tokenizer = tokenizer
    
    def tokenize_prompt(self, text: str):
        return self.tokenizer.encode(text)
        
    def encode_prompt(self, text: str):
        prompt_tokens = self.tokenize_prompt(text)
        prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=self.config.device)
        prompt_tokens = torch.unsqueeze(prompt_tokens, 0)
        return prompt_tokens
        
    def generate_embeddings(self, text: str):
        if text:
            with torch.no_grad():
                prompt_tokens = self.encode_prompt(text)
                embeddings = self.model.generate_embeddings(prompt_tokens)
                embeddings = torch.squeeze(embeddings[:, [-1], :]) # use only the last position
                return embeddings.tolist()
        return []
                
    def generate_completions(self, text: str, samples: int, new_tokens: int, temperature: float, top_k: int):
        result = []
        timer = time.time()
        with torch.no_grad():
            prompt_tokens = self.encode_prompt(text)
            for _ in range(samples):
                y = self.model.generate(prompt_tokens, new_tokens, temperature=temperature, top_k=top_k)
                result.append(self.tokenizer.decode(y[0].tolist()).strip())
        dt = time.time() - timer
        logger.info(f"{new_tokens*samples} completion tokens generated in {dt:.2f}secs ({new_tokens*samples/dt:.2f} tokens/sec) for {prompt_tokens.shape[1]} input tokens")
        return result


if __name__ == '__main__':
    config = AllamoConfiguration()
    sampler = AllamoSampler(config)

    # encode the beginning of the prompt
    if config.prompt.startswith('FILE:'):
        with open(config.prompt[5:], 'r', encoding='utf-8') as f:
            config.prompt = f.read()
            
    completions = sampler.generate_completions(config.prompt, config.num_samples, config.max_new_tokens, temperature=config.temperature, top_k=config.top_k)
    logger.info("Completions:")
    for completion in completions:
        logger.info(completion)
        logger.info('----------------')


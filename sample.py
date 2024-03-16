"""
Use this file to sample from a trained model.
"""
import json
import logging
import os
import pickle
from contextlib import nullcontext
import time
import torch
from model import AllamoTransformerConfig, AllamoTransformer
from configuration import AllamoConfiguration

class AllamoSampler:

    def __init__(self, config: AllamoConfiguration):
        self.logger = logging.getLogger('AllamoSampler')
        self.config = config
        self.__init_torch(config)

        if config.init_from == 'resume_last':
            checkpoint_name = 'last_eval_ckpt'
        else:
            checkpoint_name = 'ckpt'
        ckpt_dir = config.checkpoint_path if config.checkpoint_path else config.out_dir
        self.logger.info(f"Loading '{checkpoint_name}' checkpoint files from {ckpt_dir}...")
        with open(os.path.join(ckpt_dir, f'config_{checkpoint_name}.json'), "r", encoding="utf-8") as f:
            config_checkpoint = json.load(f)
        model_checkpoint = torch.load(os.path.join(ckpt_dir, f'model_{checkpoint_name}.pt'), map_location='cpu')
        self.__load_model(config, config_checkpoint, model_checkpoint)
        self.__load_tokenizer(config)
        del model_checkpoint
            
    def __init_torch(self, config: AllamoConfiguration):
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        torch.set_float32_matmul_precision("highest") # set to "high" for faster matrix multiplications with bfloat16
        device_type = 'cuda' if 'cuda' in config.device else 'cpu' # for later use in torch.autocast
        
    def __load_model(self, config: AllamoConfiguration, config_checkpoint, model_checkpoint):
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'bfloat16-true': torch.bfloat16, 'float16': torch.float16}[config.dtype]
        model = AllamoTransformer(**config_checkpoint['model_args'])
        unwanted_prefix = '_orig_mod.'
        for k,v in list(model_checkpoint.items()):
            if k.endswith('.rotary_emb.inv_freq'):
                # For backward compatibility, where we had it in the checkpoint
                model_checkpoint.pop(k)
            elif k.startswith(unwanted_prefix):
                model_checkpoint[k[len(unwanted_prefix):]] = model_checkpoint.pop(k)
        model.load_state_dict(model_checkpoint)
        model.eval()
        model.to(device=config.device, dtype=ptdtype)
        if config.compile:
            model = torch.compile(model) # requires PyTorch 2.0 (optional)
        self.model = model
        self.logger.info(f"Model loaded from checkpoint")
        if 'iter_num' in config_checkpoint:
            self.logger.info(f"Last model iteration: {config_checkpoint['iter_num']}")
        
    def __load_tokenizer(self, config: AllamoConfiguration):
        tiktoken_tokenizer_name = config.tiktoken_tokenizer_name
        hf_tokenizer_path = config.hf_tokenizer_path
        if hf_tokenizer_path is not None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_path)
            self.logger.info(f"HuggingFace tokenizer loaded: {hf_tokenizer_path}")
        elif tiktoken_tokenizer_name is not None:
            import tiktoken
            tokenizer = tiktoken.get_encoding(tiktoken_tokenizer_name)
            self.logger.info(f"Tiktoken tokenizer loaded: {tiktoken_tokenizer_name}")
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
            for k in range(samples):
                y = self.model.generate(prompt_tokens, new_tokens, temperature=temperature, top_k=top_k)
                result.append(self.tokenizer.decode(y[0].tolist()).strip())
        dt = time.time() - timer
        self.logger.info(f"{new_tokens*samples} completion tokens generated in {dt:.2f}secs ({new_tokens*samples/dt:.2f} tokens/sec) for {prompt_tokens.shape[1]} input tokens")
        return result


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
    logger = logging.getLogger('AllamoSamplerMain')

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


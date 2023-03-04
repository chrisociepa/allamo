"""
Use this file to sample from a trained model.
"""
import os
import pickle
from contextlib import nullcontext
import torch
from model import AllamoTransformerConfig, AllamoTransformer
from configuration import AllamoConfiguration

config = AllamoConfiguration()

torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in config.device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config.dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# init from a model saved in a specific directory
ckpt_path = os.path.join(config.out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=config.device)
model = AllamoTransformer(checkpoint['model_args'])
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(config.device)
if config.compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

vocab_size = config.vocab_size
tiktoken_tokenizer_name = config.tiktoken_tokenizer_name
custom_tokenizer_path = config.custom_tokenizer_path
# look for the meta pickle in case it is available in the dataset folder
if 'config' in checkpoint and 'dataset' in checkpoint['config']:
    meta_path = os.path.join(config.data_dir, checkpoint['config']['dataset'], 'meta.pkl')
    if os.path.exists(meta_path):
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        vocab_size = meta['vocab_size']
        if meta['tiktoken_tokenizer_name']:
            tiktoken_tokenizer_name = meta['tiktoken_tokenizer_name']
        if meta['custom_tokenizer_path']:
            custom_tokenizer_path = meta['custom_tokenizer_path']
print(f"Vocab_size: {vocab_size}")
if custom_tokenizer_path is not None:
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=custom_tokenizer_path)
    print(f"Custom tokenizer path: {custom_tokenizer_path}")
else:
    import tiktoken
    tokenizer = tiktoken.get_encoding(tiktoken_tokenizer_name)
    print(f"Tiktoken tokenizer name: {tiktoken_tokenizer_name}")

# encode the beginning of the prompt
if config.prompt.startswith('FILE:'):
    with open(config.prompt[5:], 'r', encoding='utf-8') as f:
        config.prompt = f.read()
prompt_ids = tokenizer.encode(config.prompt)
x = (torch.tensor(prompt_ids, dtype=torch.long, device=config.device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        for k in range(config.num_samples):
            y = model.generate(x, config.max_new_tokens, temperature=config.temperature, top_k=config.top_k)
            print(tokenizer.decode(y[0].tolist()))
            print('---------------')

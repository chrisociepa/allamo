# ALLaMo

This repository is intended as a simple, hackable and fast implementation for training/finetuning/inference [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/)-based models ([arXiv](https://arxiv.org/abs/2302.13971v1)).

## install

Dependencies:

- Python 3
- [pytorch](https://pytorch.org)
- [numpy](https://numpy.org/install/)
- [tiktoken](https://github.com/openai/tiktoken)
- [huggingface transformers](https://huggingface.co/docs/transformers/installation)
- [wandb](https://wandb.ai/)

## training

Use the script `train.py` to start your training. It reads a `train.bin` and `val.bin` files from the dataset directory. You can create the both files with a single script like this:

```
import numpy as np
import tiktoken

def encode_file(input_file_path, output_file_path, tokenizer_name):
    tokenizer = tiktoken.get_encoding(tokenizer_name)
    with open(input_file_path, 'r') as f:
        data = f.read()
    enc_data = tokenizer.encode(data)
    enc_data = np.array(enc_data, dtype=np.uint32)
    enc_data.tofile(output_file_path)
    
encode_file('raw/dataset1/train.txt', 'data/dataset1/train.bin', 'cl100k_base')  
```

The training script can be run on both a single node with one or more GPUs, as well as on multiple nodes with Distributed Data Parallel (DDP).

To run on a single node with 1 GPU, example:

```
$ python train.py \
    --config="../config/train_allamo_cl100k_base.json" \
    --wandb_log=True
```

To run on a single node with 8 GPU with DDP, example:

```
$ torchrun --standalone --nproc_per_node=8 train.py \
    --config="../config/train_allamo_cl100k_base.json" \
    --wandb_log=True
```

To run on 2+ nodes with DDP, example:
- Run on the first (master) node with example IP 123.456.123.456:

```
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py \
    --config="../config/train_allamo_cl100k_base.json" \
    --wandb_log=True
```

- Run on the worker node(s):

```
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py \
    --config="../config/train_allamo_cl100k_base.json" \
    --wandb_log=True
```

Note: in case your cluster does not have Infiniband interconnect prepend `NCCL_IB_DISABLE=1`.

## finetuning

The process of finetuning is similar to regular training, but we initialize from a pretrained model and use a smaller learning rate during training. In addition, it is essential to ensure that the model parameters used for finetuning are consistent with those used during pre-training.

## sampling / inference

Use the script `sample.py` to sample from a model you trained. For example:

```
$ python sample.py \
    --config="../config/train_allamo_cl100k_base.json" \
    --max_new_tokens=100 \
    --temperature=0.7 \
    --top_k=200 \
    --num_samples=5 \
    --prompt="Long long time ago"
```

You can also prompt the model with some text from a file prefixing its path with `FILE:`, example:

```
$ python sample.py \
    --config="../config/train_allamo_cl100k_base.json" \
    --max_new_tokens=100 \
    --temperature=0.7 \
    --top_k=200 \
    --num_samples=5 \
    --prompt="FILE:prompt.txt"
```

Default tokenizer is `tiktoken` (`cl100k_base`) but thanks to HuggingFace Transformers you can easily use your own pretrained tokenizer. Use `--custom_tokenizer_path`  to provide your tokenizer json config file.

Use the script 'sample_api.py' to expose 2 API endpoints. Then you will be able to query a pretrained model for text embeddings and completions. 

To run the API with a pretrained model, example:

```
$ python sample_api.py \
    --config="../config/train_allamo_cl100k_base.json" \
    --max_new_tokens=10 \
    --temperature=0.7 \
    --top_k=200 \
    --num_samples=5
```

- Query for text embeddings, example:

```
$ curl -X POST -H "Content-Type: application/json" http://localhost:5000/embeddings -d '{"prompt": "Long long time ago"}'
```

- Query for text completions, example:

```
$ curl -X POST -H "Content-Type: application/json" http://localhost:5000/completions -d '{"prompt": "Long long time ago", "num_samples": 3}'
```

## References:

1. [nanoGPT](https://github.com/karpathy/nanoGPT) - many thanks to Andrej Karpathy for amazing and inspirational work!
2. [LLaMA](https://github.com/facebookresearch/llama)

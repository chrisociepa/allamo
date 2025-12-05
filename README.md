# ALLaMo

<p align="center">
  <img src="./assets/allamo_logo.jpg" width=512>
</p>

This repository is intended as a simple, hackable and fast implementation for training/finetuning/inference LLMs.

If you're interested in seeing how we trained a 1B model for the Polish language using a single RTX 4090, with 60B tokens over 44 days, check out our [blog](https://azurro.pl/apt3-1b-base-en/).

We use this framework to train Polish Language Model - [Bielik](https://bielik.ai).

## Install

You can easily install and `import allamo` into your project:

```
git clone https://github.com/chrisociepa/allamo.git
cd allamo
pip install -e .
```

Dependencies:

- Python 3.8+
- [pytorch](https://pytorch.org)
- [joblib](https://joblib.readthedocs.io)
- [numpy](https://numpy.org/install/)
- [wandb](https://wandb.ai/quickstart/)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) - optional, for FlashAttention 2 and Sliding Window
- [huggingface transformers](https://huggingface.co/docs/transformers/installation) - optional
- [huggingface tokenizers](https://huggingface.co/docs/tokenizers/python/latest/installation/main.html) - optional

## Datasets

Before you start training a new model, you need to create training and testing datasets. By default, training scripts expect two files: `train.bin` and `val.bin`. You can create both files using `prepare_datasets.py` or by implementing a simple script like the one below:

```python
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

There are other options and formats for handling datasets:

- Instead of using a NumPy array, consider using a [PyTorch tensor](https://pytorch.org/docs/stable/tensors.html) and save the tensor in a file with the `.pt` extension.
- You can also use a list of samples (each being a PyTorch tensor) with each sample's size being `block_size + 1`. These samples can then be saved in a file with the `.pt` extension.

## Training

Use the script `train.py` to start your training. It reads a `train.bin` and `val.bin` files from the dataset directory. 

The training script can be run on both a single node with one or more GPUs, as well as on multiple nodes with Distributed Data Parallel (DDP).

To run on a single node with 1 GPU, example:

```bash
$ python train.py \
    --config="./config/train_1B.json" \
    --wandb_log=True
```

To run on a single node with 8 GPUs with DDP, example:

```bash
$ torchrun --standalone --nnodes=1 --nproc-per-node=8 train.py \
    --config="./config/train_1B.json" \
    --wandb_log=True
```

To run on 2+ nodes (with 8 GPUs each) with DDP, example:
- Run on the first (master) node with example IP 123.456.123.456:

```bash
$ torchrun --nnodes=2 --nproc-per-node=8 --node-rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py \
    --config="./config/train_1B.json" \
    --wandb_log=True
```

- Run on the worker node(s):

```bash
$ torchrun --nnodes=2 --nproc-per-node=8 --node-rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py \
    --config="./config/train_1B.json" \
    --wandb_log=True
```

To run on 2+ nodes (with 8 GPUs each) with FSDP, example:
- Run the same command on all nodes (master node IP: 123.456.123.456):

```bash
torchrun --nnodes=2 --nproc-per-node=8 --rdzv-id=123 --rdzv-backend=c10d --rdzv-endpoint=123.456.123.456:29292 fsdp_train.py \
    --config="./config/train_1B.json" \
    --wandb_log=True
```

Note: in case your cluster does not have Infiniband interconnect prepend `NCCL_IB_DISABLE=1`.

### MFU and MTU

During training, it is possible to calculate indicators such as Model Flops Utilization ([MFU](https://arxiv.org/abs/2204.02311)) and Model Time Utilization (MTU). To calculate MFU, you must specify the maximum declared number of FLOPs for the used GPU in the parameter `mfu_flops_peak`. For example, for the A100 40GB GPU with bfloat16/float16 precision, this value is `312e12`, or `165.2e12` for the RTX 4090. 
The MTU indicator describes the percentage of time during an iteration that the model spends on actual training versus the time spent on other tasks such as data loading, hyperparameter updates, etc. The closer to 100%, the more efficiently the training is in terms of resource utilization.

### Tensor/Sequence Parallelism

To enable training with Tensor Parallelism (TP) or Sequence Parallelism (SP), you need to use FSDP2 with [Distributed Checkpoint (DCP)](https://pytorch.org/docs/stable/distributed.checkpoint.html). If you have a standard checkpoint, you must first convert it to a Distributed Checkpoint (DCP). This can be done by modifying the state_dict key prefixes and then performing the actual conversion, as shown below:

```bash
python scripts/update_state_dict_prefixes.py \
    -s ../out_dir/model_ckpt.pt \
    -d ../out_dir/model_ckpt_fixedprefixes.pt \
    -a "model."

mkdir -p ../out_dir/dcp/model_last_eval_ckpt/
python -m torch.distributed.checkpoint.format_utils torch_to_dcp ../out_dir/model_ckpt_fixedprefixes.pt ../out_dir/dcp/model_last_eval_ckpt/
```

Once the checkpoint is converted, you can start training with the desired TP degree, for example:

```bash
torchrun --nnodes=2 --nproc-per-node=8 --rdzv-id=123 --rdzv-backend=c10d --rdzv-endpoint=123.456.123.456:29292 fsdp_train.py \
    --config="./config/train_1B.json" \
    --distributed_checkpoint \
    --tensor_parallel_degree=2
```

If you need to convert a distributed checkpoint back to a standard format (e.g., for exporting to HuggingFace format), you can do so by reversing the conversion and updating the state_dict key prefixes:

```bash
python -m torch.distributed.checkpoint.format_utils dcp_to_torch ../out_dir/dcp/model_last_eval_ckpt/ ../out_dir/model_ckpt_fromdcp.pt

python scripts/update_state_dict_prefixes.py \
    -s ../out_dir/model_ckpt_fromdcp.pt \
    -d ../out_dir/model_ckpt_converted.pt \
    -r "model"
```

## Finetuning

The process of finetuning is similar to regular training, but we initialize from a pretrained model and use a smaller learning rate during training. In addition, it is essential to ensure that the model parameters used for finetuning are consistent with those used during pre-training.

### Extending Context Window

As part of the training or fine-tuning process, you can easily extend the context window (block size). Set `block_size` to the desired value and provide the `rope_scaling` parameter with the appropriate scaling values. Note that model parameters are also stored as part of the model checkpoint. Therefore, you should modify them within the checkpoint.

#### Linear Scaling

Modify the `block_size` and `rope_freq_base` (default value is `10000`) parameters. Then provide the `rope_scaling` parameter with `rope_type` set to `linear` and `factor` set to the desired value. For more information on Position Interpolation, you can refer to this [paper](https://arxiv.org/abs/2306.15595) or this [blog post](https://kaiokendev.github.io/til).

Below are some empirically derived example values for extending the context window. However, we encourage you to experiment and adjust these values to suit the specific needs of your model:

| context scaling factor | rope_freq_base | factor |
|------------------------|----------------|--------|
| 2                      | 20000          | 0.83   |
| 3                      | 40000          | 0.86   |
| 4                      | 57200          | 0.75   |
| 16                     | 26000          | 0.125  |

#### YaRN

Modify the `block_size` and provide the `rope_scaling` parameter with `rope_type` set to `yarn`, `factor` set to the desired value, and `original_max_position_embeddings` set to the original block_size. For more information on YaRN method, you can refer to this [paper](https://arxiv.org/abs/2309.00071).

## Import HF models

Go to `scripts/` and use the script `import_from_hf.py` to import model weights from Hugging Face, and create a checkpoint for further training. Example script execution:

```bash
python import_from_hf.py \
    --huggingface_model="mistralai/Mistral-7B-v0.1" \
    --model_type=bielik2 \
    --output_dir="/data/models/Mistral-7B-v0.1" \
    --output_checkpoint_name_base="last_eval_ckpt"
```

## Export your model to Hugging Face format

When you have trained your model, you may want to run it in the Hugging Face ecosystem. Using the `export_to_hf.py` script, you can easily convert your model to an HF-compatible LLaMA format. Here's an example of how to run it:

```bash
$ python export_to_hf.py \
    --input_dir="/data/models/Bielik-7B-v0.1" \
    --checkpoint_name_base=last_eval_ckpt \
    --output_dir="/data/models/Bielik-7B-v0.1/hf" \
    --model_type=bielik2 \
    --output_model_type=llama \
    --output_dtype=bfloat16 \
    --max_position_embeddings=8192
```

## Citation

Please cite this repo if you use it.
```
@misc{allamo,
  author = {Ociepa, Krzysztof},
  title = {ALLaMo: A Simple, Hackable, and Fast Framework for Training Medium-Sized LLMs},
  year = {2023},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/chrisociepa/allamo}},
}
```

## References:

1. [nanoGPT](https://github.com/karpathy/nanoGPT) - many thanks to Andrej Karpathy for amazing and inspirational work!
2. [LLaMA](https://github.com/facebookresearch/llama)

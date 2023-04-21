"""
This single file is intended to perform some magic for training/finetuning.
"""

import os
import time
import math
import pickle
import random
import datetime
import dataclasses
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import AllamoTransformerConfig, AllamoTransformer
from configuration import AllamoConfiguration

config = AllamoConfiguration()

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=config.backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(config.device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0

if master_process:
    os.makedirs(config.out_dir, exist_ok=True)
torch.manual_seed(config.seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in config.device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config.dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# batch size & gradient_accumulation scheduler
if config.batch_size_schedule: 
    config.batch_size_max = config.batch_size
    config.batch_size = config.batch_size_initial
if config.grad_accum_schedule: 
    config.grad_accum_max = config.gradient_accumulation_steps
    config.gradient_accumulation_steps = config.grad_accum_initial
batch_size = config.batch_size
gradient_accumulation_steps = config.gradient_accumulation_steps

# poor man's data loader
data_dir = os.path.join(config.data_dir, config.dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
dataset_train_x_start = config.dataset_seq_train_start if config.dataset_seq_train_start is not None else random.randint(0, batch_size-1)
def get_batch(split, random_samples=False):
    data = train_data if split == 'train' else val_data
    data_size = len(data)
    if random_samples == False and split == 'train' and config.dataset_seq_train:
        global dataset_train_x_start
        ix = torch.zeros(batch_size, dtype=torch.int)
        max_batch_id = dataset_train_x_start + (batch_size-1) * config.dataset_seq_step_size + config.block_size + 1
        if max_batch_id >= data_size:
            print(f"Batch is exceeding the data size ({max_batch_id} > {data_size}). Reseting cursor to the beginning")
            dataset_train_x_start = random.randint(0, batch_size-1)
        for i in range(batch_size):
            last_x_start = dataset_train_x_start + i * config.dataset_seq_step_size
            ix[i] = last_x_start
        dataset_train_x_start = last_x_start + config.dataset_seq_step_size 
    else:
        ix = torch.randint(data_size - config.block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+config.block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+config.block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(config.device, non_blocking=True), y.pin_memory().to(config.device, non_blocking=True)
    else:
        x, y = x.to(config.device), y.to(config.device)
    return x, y

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_train_loss = 1e9
best_val_loss = 1e9
processed_tokens = 0
transformer_config_fields = [f.name for f in dataclasses.fields(AllamoTransformerConfig)]
def init_model():
    model_args = {k: getattr(config, k) for k in transformer_config_fields}
    modelConf = AllamoTransformerConfig(**model_args)
    model = AllamoTransformer(modelConf)
    return model
    
if config.init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is not None:
        config.vocab_size = meta_vocab_size
    else:
        print(f"defaulting to vocab_size={config.vocab_size}")
    model = init_model()
elif config.init_from == 'resume':
    print(f"Resuming training from {config.out_dir}")
    # resume training from a checkpoint
    ckpt_path = config.checkpoint_path if config.checkpoint_path else os.path.join(config.out_dir, 'ckpt.pt')
    print(f"Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in transformer_config_fields:
        assert hasattr(config, k), f"Config object does not have {k} field"
        if hasattr(checkpoint_model_args, k): # useful only for the backward compatibility
            setattr(config, k, getattr(checkpoint_model_args, k))
    model = init_model()
    state_dict = checkpoint['model']
    # ignore unwanted checkpoint files
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    if 'iter_num' in checkpoint:
        iter_num = checkpoint['iter_num']
    if 'best_train_loss' in checkpoint:
        best_train_loss = checkpoint['best_train_loss']
    if 'best_val_loss' in checkpoint:
        best_val_loss = checkpoint['best_val_loss']
    if 'processed_tokens' in checkpoint:
        processed_tokens = checkpoint['processed_tokens']
model.to(config.device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(config.dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(config.weight_decay, config.learning_rate, (config.beta1, config.beta2), device_type)
# FIXME: when a checkpoint is loaded on cpu and model on gpu, loading state dict doesn't work.
# Moving the checkpoint or its optimizer state on gpu is a waste of memory.
#if config.init_from == 'resume' and 'optimizer' in checkpoint:
#    optimizer.load_state_dict(checkpoint['optimizer'])

# compile the model
if config.compile:
    print("compiling the model... (takes a ~minute)")
    try:
        model = torch.compile(model) # requires PyTorch 2.0
        print("Model compiled and ready to use")
    except Exception as err:
        print(f"Model compile not supported: {err}")

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split, True)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it >= config.lr_decay_iters:
        return config.min_lr
    # 3) in between, use cosine decay down to min learning rate with restarts (optional)
    if config.lr_decay_reset_iters is not None:
        decay_ratio = (it % config.lr_decay_reset_iters) / config.lr_decay_reset_iters
    else:
        decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)

# batch_size scheduler (when enabled) 
def get_batch_size(it):
    return min(batch_size + 1, config.batch_size_max) if it % (config.batch_size_max_iter/100) == 0 else batch_size 
# grad_accum scheduler (when enabled) 
def get_grad_accum(it):
    return min(gradient_accumulation_steps + 1, config.grad_accum_max) if it % (config.grad_accum_max_iter/100) == 0 else gradient_accumulation_steps 

# logging
if config.wandb_log and master_process:
    import wandb
    wandb.init(project=config.wandb_project, name=config.wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
raw_model = model.module if ddp else model # unwrap DDP container if needed

# helps saving checkpoint to a file
def save_checkpoint(ckpt_file_name):
    checkpoint = {
        'model': raw_model.state_dict(),
        # FIXME: since we don't init optimizer with its state from checkpoint, so there is no need to save it
        #'optimizer': optimizer.state_dict(),
        'model_args': model.config,
        'iter_num': iter_num,
        'best_train_loss': best_train_loss,
        'best_val_loss': best_val_loss,
        'processed_tokens': processed_tokens,
        'config': config.__dict__,
    }
    ckpt_file_path = os.path.join(config.out_dir, ckpt_file_name)
    print(f"saving checkpoint to {ckpt_file_path}")
    torch.save(checkpoint, ckpt_file_path)

if config.decay_lr:
    print(f"Cosing decay learning rate enabled. Currect learning rate: {get_lr(iter_num)}")
else:
    print(f"Using constant learning rate: {config.learning_rate}")

while iter_num <= config.max_iters:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if config.decay_lr else config.learning_rate
    # determine and set batch_size and gradient_accumulation_steps for this iteration 
    batch_size = get_batch_size(iter_num) if config.batch_size_schedule else batch_size
    gradient_accumulation_steps = get_grad_accum(iter_num) if config.grad_accum_schedule else gradient_accumulation_steps
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % config.eval_interval == 0 and master_process:
        losses = estimate_loss()
        total_batch_size = config.block_size * batch_size * gradient_accumulation_steps
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"{timestamp} - step {iter_num:,}: train loss {losses['train']:.4f} (best: {best_train_loss:.4f}), val loss {losses['val']:.4f} (best: {best_val_loss:.4f}), BS {total_batch_size:,}, tokens {processed_tokens:,}, DS offset {dataset_train_x_start:,}")
        if config.wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "tokens": processed_tokens,
                "total_batch_size": total_batch_size,
                "train/ds_offset": dataset_train_x_start
            })
        if losses['train'] < best_train_loss:
            best_train_loss = losses['train']
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            if iter_num > 0:
                save_checkpoint('ckpt.pt')
        if config.always_save_checkpoint:
            if iter_num > 0:
                save_checkpoint('last_eval_ckpt.pt')
    if iter_num == 0 and config.eval_only:
        break
    
    timer = time.time()
    # collect the loss values in gradient accumulation steps
    losses = torch.zeros(gradient_accumulation_steps)
    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')

        if loss.isnan():
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"{timestamp} - loss is NaN in iter {iter_num:,} micro_step {micro_step}")
            continue
        
        processed_tokens += X.numel()
        losses[micro_step] = loss.item()
        # normalize the loss value to account for the number of gradient accumulation steps
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if config.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    dt = time.time() - timer
    if iter_num % config.log_interval == 0 and master_process:
        max_lossf = losses.max()
        lossf = torch.where(losses == 0, max_lossf, losses).mean()
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"{timestamp} - iter {iter_num:,}: loss {lossf:.4f}, iter time {dt*1000:.2f}ms, tokens {processed_tokens:,}, lr {lr:.6f}")
    iter_num += 1

if ddp:
    destroy_process_group()

import gc
import dataclasses
import numpy as np
import torch
from model import AllamoTransformerConfig, AllamoTransformer
from configuration import AllamoConfiguration

config = AllamoConfiguration()

transformer_config_fields = [f.name for f in dataclasses.fields(AllamoTransformerConfig)]
model_args = {k: getattr(config, k) for k in transformer_config_fields if hasattr(config, k)}
modelConf = AllamoTransformerConfig(**model_args)

#torch.set_default_tensor_type(torch.HalfTensor)
model = AllamoTransformer(modelConf)
#torch.set_default_tensor_type(torch.FloatTensor)

optimizer = 'Adam'
if 'cuda' in config.device:
    torch.cuda.set_device(config.device)
    gc.collect()
    torch.cuda.empty_cache()
    model_input = torch.randint(0, config.vocab_size, (config.block_size,), dtype=torch.int64).unsqueeze(0).repeat(config.batch_size, 1)
    print(f"Max Sequence size: {(model_input.numel() * model_input.element_size())/(1024*1024)}")
    a = torch.cuda.memory_allocated(config.device)
    model.to(config.device)
    b = torch.cuda.memory_allocated(config.device)
    with torch.no_grad():
        output = model(model_input.to(config.device))[0].sum() # Taking the sum here just to get a scalar output
        c = torch.cuda.memory_allocated(config.device)
    model_memory = b - a
    interference_memory = c - b
    print(f"Memory allocated by the model: {model_memory/(1024*1024)}")
    print(f"Interference Maximum Memory Estimate: {interference_memory/(1024*1024)}")
    
    if optimizer is not None:
        gc.collect()
        torch.cuda.empty_cache()
        b = torch.cuda.memory_allocated(config.device)
        output = model(model_input.to(config.device))[0].sum() # Taking the sum here just to get a scalar output
        c = torch.cuda.memory_allocated(config.device)
        amp_multiplier = .5 if config.dtype == 'float16' or config.dtype == 'bfloat16' else 1
        forward_pass_memory = (c - b)*amp_multiplier
        print(f"Forward pass memory: {forward_pass_memory/(1024*1024)}")
        gradient_memory = model_memory
        if optimizer == 'Adam':
            o = 2
        elif optimizer == 'RMSprop':
            o = 1
        elif optimizer == 'SGD':
            o = 0
        elif optimizer == 'Adagrad':
            o = 1
        else:
            raise ValueError("Unsupported optimizer. Look up how many moments are stored by your optimizer and add a case to the optimizer checker.")
        gradient_moment_memory = o*gradient_memory
        total_memory = model_memory + forward_pass_memory + gradient_memory + gradient_moment_memory
        print(f"Training Maximum Memory Estimate: {total_memory/(1024*1024)}")
        print(f"* model: {model_memory/(1024*1024)}")
        print(f"* forward pass: {forward_pass_memory/(1024*1024)}")
        print(f"* gradient: {gradient_memory/(1024*1024)}")
        print(f"* gradient moments: {gradient_moment_memory/(1024*1024)}")
        
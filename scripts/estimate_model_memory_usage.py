import dataclasses
import gc
import torch
from allamo.logging import configure_logger, logger
from allamo.model.model import AllamoTransformerConfig, AllamoTransformer
from allamo.configuration import AllamoConfiguration

configure_logger()
config = AllamoConfiguration()
if config.dtype == 'bfloat16-true':
    torch.set_default_dtype(torch.bfloat16)

transformer_config_fields = [f.name for f in dataclasses.fields(AllamoTransformerConfig)]
model_args = {k: getattr(config, k) for k in transformer_config_fields if hasattr(config, k)}
modelConf = AllamoTransformerConfig(**model_args)
model = AllamoTransformer(modelConf)

optimizer = 'Adam'
if 'cuda' in config.device:
    torch.cuda.set_device(config.device)
    gc.collect()
    torch.cuda.empty_cache()
    model_input = torch.randint(0, config.vocab_size, (config.block_size,), dtype=torch.int64).unsqueeze(0).repeat(config.batch_size, 1)
    logger.info(f"Max Sequence size: {(model_input.numel() * model_input.element_size())/(1024*1024)}")
    a = torch.cuda.memory_allocated(config.device)
    model.to(device=config.device)
    b = torch.cuda.memory_allocated(config.device)
    with torch.no_grad():
        output = model(model_input.to(config.device))[0].sum() # Taking the sum here just to get a scalar output
        c = torch.cuda.memory_allocated(config.device)
    model_memory = b - a
    inference_memory = c - b
    logger.info(f"Memory allocated by the model (precision: {config.dtype}): {model_memory/(1024*1024)}")
    logger.info(f"Inference Maximum Memory Estimate: {inference_memory/(1024*1024)}")
    
    if optimizer is not None:
        gc.collect()
        torch.cuda.empty_cache()
        b = torch.cuda.memory_allocated(config.device)
        output = model(model_input.to(config.device))[0].sum() # Taking the sum here just to get a scalar output
        c = torch.cuda.memory_allocated(config.device)
        amp_multiplier = .5 if config.dtype == 'float16' or config.dtype == 'bfloat16' else 1
        # More details: https://stackoverflow.com/a/76994670
        activations = 32 * config.n_layer * (34 * config.block_size * config.n_embd + 5 * config.n_head * config.block_size^2) / 2
        if config.dtype == 'bfloat16-true':
            activations /= 2
        forward_pass_memory = activations #(c - b)*amp_multiplier
        logger.info(f"Forward pass memory: {forward_pass_memory/(1024*1024)}")
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
        logger.info(f"Training Maximum Memory Estimate: {total_memory/(1024*1024)}")
        logger.info(f"* model: {model_memory/(1024*1024)}")
        logger.info(f"* forward pass: {forward_pass_memory/(1024*1024)}")
        logger.info(f"* gradient: {gradient_memory/(1024*1024)}")
        logger.info(f"* gradient moments: {gradient_moment_memory/(1024*1024)}")
        

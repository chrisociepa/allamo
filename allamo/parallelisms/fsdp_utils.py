import torch
import functools
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy

from allamo.logging import logger
from allamo.configuration import AllamoConfiguration
from allamo.model.model import SelfAttentionBlock
from allamo.torch_utils import (
    TORCH_DTYPE_MAP,
)

FSDP_SHARDING_STRATEGY_MAP = {
    'FULL_SHARD': ShardingStrategy.FULL_SHARD,
    'HYBRID_SHARD': ShardingStrategy.HYBRID_SHARD,
    '_HYBRID_SHARD_ZERO2': ShardingStrategy._HYBRID_SHARD_ZERO2,
    'SHARD_GRAD_OP': ShardingStrategy.SHARD_GRAD_OP,
    'NO_SHARD': ShardingStrategy.NO_SHARD
}

def enable_activation_checkpointing(model, config):
    non_reentrant_wrapper = functools.partial(
        checkpoint_wrapper,
        offload_to_cpu=False,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    
    if config.gradient_checkpointing_excluded_layers <= 0:
        check_fn = lambda submodule: isinstance(submodule, SelfAttentionBlock)
    else:
        total_layers = config.n_layer
        excluded_count = min(config.gradient_checkpointing_excluded_layers, total_layers)
        
        if excluded_count >= total_layers:
            logger.warning(f"All layers excluded, so activation checkpointing won't be applied")
            return
        else:
            step = total_layers / excluded_count
            excluded_layers = set([int(i * step) for i in range(excluded_count)])
            actual_excluded = len(excluded_layers)
            if actual_excluded < excluded_count:
                additional_needed = excluded_count - actual_excluded
                layer_idx = 0
                while additional_needed > 0 and layer_idx < total_layers:
                    if layer_idx not in excluded_layers:
                        excluded_layers.add(layer_idx)
                        additional_needed -= 1
                    layer_idx += 1
            
            def custom_check_fn(submodule):
                if not isinstance(submodule, SelfAttentionBlock):
                    return False
                return submodule.layer_id not in excluded_layers
            
            check_fn = custom_check_fn
    
    apply_activation_checkpointing(model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)
    logger.info(f"Activation checkpointing applied to the model (excluded {max(config.gradient_checkpointing_excluded_layers, 0)} layers)")

def parallelize_model_with_fsdp1(model, config: AllamoConfiguration, with_activation_checkpointing: bool = False):
    logger.info("Configuring model with FSDP1")
    ptdtype = TORCH_DTYPE_MAP[config.dtype]
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            SelfAttentionBlock,
        },
    )
    sharding_strategy = FSDP_SHARDING_STRATEGY_MAP[config.fsdp_sharding_strategy]
    fsdp_config = dict(
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        device_id=torch.cuda.current_device(),
        mixed_precision=MixedPrecision(
            param_dtype=ptdtype,
            reduce_dtype=ptdtype,
            buffer_dtype=ptdtype,
        ),
        limit_all_gathers=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # will use slightly more memory vs. no prefetch
        use_orig_params=True, # required to use torch.compile()
    )
    
    model = FSDP(model, **fsdp_config)
    logger.info(f"Model configured with FSDP1 and {sharding_strategy=}")
    
    if with_activation_checkpointing:
        enable_activation_checkpointing(model, config)
        
    logger.info(f"Model after parallelization {model=}\n")
    
    if config.compile:
        logger.info("Compiling model")
        try:
            model = torch.compile(model, mode=config.compile_mode)
            logger.info("Model compiled and ready to use")
        except Exception as err:
            logger.warning(f"Unable to compile the model: {err}")
    
    return model

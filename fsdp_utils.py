import functools
import logging
import os
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from train_utils import (
    rename_file_to_prev_version,
    get_config_checkpoint_path,
    get_model_checkpoint_path,
    get_optimizer_checkpoint_path,
)
from model import SelfAttentionBlock

logger = logging.getLogger("AllamoFSDPUtils")

TP_API_AVAILABLE = True

try:
    # Tensor Parallelism API is experimental and subject to change. Requires PyTorch 2.3.0+
    import torch.distributed._functional_collectives as funcol
    from torch.distributed.device_mesh import init_device_mesh, DeviceMesh
    from torch.distributed._tensor import Shard, Replicate
    from torch.distributed.tensor.parallel import (
        parallelize_module,
        ColwiseParallel,
        RowwiseParallel,
        PrepareModuleInput,
        SequenceParallel,
    )
    from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
    from torch.utils.checkpoint import checkpoint
    from torch.distributed.checkpoint.stateful import Stateful
    from torch.distributed.checkpoint.state_dict import (
        get_model_state_dict,
        get_optimizer_state_dict,
        set_model_state_dict,
        set_optimizer_state_dict,
    )
    from torch.distributed.checkpoint.format_utils import DynamicMetaLoadPlanner, BroadcastingTorchSaveReader
    
    class ModelWrapper(Stateful):
        def __init__(self, model: nn.Module) -> None:
            self.model = model

        def state_dict(self) -> None:
            return get_model_state_dict(self.model)

        def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
            set_model_state_dict(self.model, state_dict)


    class OptimizerWrapper(Stateful):
        def __init__(self, model: nn.Module, optim: torch.optim.Optimizer) -> None:
            self.model = model
            self.optim = optim

        def state_dict(self) -> None:
            return get_optimizer_state_dict(self.model, self.optim)

        def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
            set_optimizer_state_dict(self.model, self.optim, optim_state_dict=state_dict)
            
except ImportError:
    TP_API_AVAILABLE = False
    logger.warning(f"Tensor Parallelism API is not available")

def build_world_mesh(world_size, config):
    if config.tensor_parallelism_size <= 1:
        return None
        
    if not TP_API_AVAILABLE:
        raise Exception("Tensor Parallelism API is not available!")
    
    tp_size = max(config.tensor_parallelism_size, 1)
    assert world_size % tp_size == 0, f"World size {world_size} needs to be divisible by TP size {tp_size}"
    dp_size = world_size // tp_size
    device_mesh = init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))
    return device_mesh

def dist_all_reduce(x: torch.Tensor, op: dist.ReduceOp, device_mesh):
    if device_mesh is None:
        dist.all_reduce(x, op=op)
        return x
    else:
        return funcol.all_reduce(x, reduceOp=op.name, group=device_mesh)

def get_sharding_strategy(config):
    return {
        'FULL_SHARD': ShardingStrategy.FULL_SHARD,
        'HYBRID_SHARD': ShardingStrategy.HYBRID_SHARD,
        '_HYBRID_SHARD_ZERO2': ShardingStrategy._HYBRID_SHARD_ZERO2,
        'SHARD_GRAD_OP': ShardingStrategy.SHARD_GRAD_OP,
        'NO_SHARD': ShardingStrategy.NO_SHARD
    }[config.fsdp_sharding_strategy]
    
def enable_activation_checkpointing(model):
    non_reentrant_wrapper = functools.partial(
        checkpoint_wrapper,
        offload_to_cpu=False,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    check_fn = lambda submodule: isinstance(submodule, SelfAttentionBlock)
    apply_activation_checkpointing(model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)
    logger.info(f"Activation checkpointing applied to the model")

def parallelize_with_fsdp(model, config, with_activation_checkpointing):
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config.dtype]
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            SelfAttentionBlock,
        },
    )
    sharding_strategy=get_sharding_strategy(config)
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
    
    if config.compile:
        # convert to desire dtype, to make it work with FSDP and torch.compile
        model.rotary_emb.cos_cached = model.rotary_emb.cos_cached.to(ptdtype)
        model.rotary_emb.sin_cached = model.rotary_emb.sin_cached.to(ptdtype)
    
    model = FSDP(model, **fsdp_config)
    logger.info(f"Model configured with FSDP and {sharding_strategy=}")
    
    if with_activation_checkpointing:
        enable_activation_checkpointing(model)
        
    logger.info(f"Model after parallelization {model=}\n")
    
    # compile the model - requires PyTorch 2.0
    if config.compile:
        logger.info("Compiling model")
        try:
            model = torch.compile(model, mode=config.compile_mode)
            logger.info("Model compiled and ready to use")
        except Exception as err:
            logger.warning(f"Unable to compile the model: {err}")
    
    return model
    
def parallelize_with_fsdp2(model, config, device_mesh, with_activation_checkpointing):
    if with_activation_checkpointing:
        for layer_id, layer in enumerate(model.layers):
            model.layers[layer_id] = checkpoint_wrapper(
                model.layers[layer_id],
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                checkpoint_fn=checkpoint,
                use_reentrant=False,
                preserve_rng_state=False,
            )
    
    if config.compile:
        # torch.compile does not work properly with FSDP, therefore we compile per-layer before FSDP. 
        # Additionally, dynamic shapes have some issues, so we disable them now.
        for layer_id, layer in enumerate(model.layers):
            model.layers[layer_id] = torch.compile(layer, dynamic=False, fullgraph=True)
        logger.info("Compiled each layer with torch.compile")
        
    fsdp_config = {"mesh": device_mesh}
    if config.dtype != 'float32':
        fsdp_config["mp_policy"] = MixedPrecisionPolicy(
            param_dtype={'bfloat16': torch.bfloat16, 'float16': torch.float16}[config.dtype],
            reduce_dtype=torch.float32
        )
    
    for layer_id, layer in enumerate(model.layers):
        # As an optimization, do not reshard after forward for the last layer
        # since FSDP2 would prefetch it immediately
        reshard_after_forward = layer_id < len(model.layers) - 1
        fully_shard(
            layer,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
        model.layers[layer_id] = layer
    model = fully_shard(model, **fsdp_config)
        
    logger.info(f"Model parallelized with FSDP2: {model}\n")
    return model

def parallelize_model(model, world_mesh, config, with_activation_checkpointing):
    if world_mesh is None:
        return parallelize_with_fsdp(model, config, with_activation_checkpointing)
    
    model = parallelize_module(
        model,
        world_mesh["tp"],
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
            ),
            "lm_head": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Replicate(),
                use_local_output=True,
            ),
            "norm": SequenceParallel(),
            "layers.0": PrepareModuleInput(
                input_layouts=(Replicate(), None),
                desired_input_layouts=(Shard(1), None),
                use_local_output=True,
            ),
        },
    )

    for layer in model.layers:
        layer_plan = {
            "attention": PrepareModuleInput(
                input_layouts=(Shard(1), None, None),
                desired_input_layouts=(Replicate(), None, None),
            ),
            "attention.q_proj": ColwiseParallel(),
            "attention.k_proj": ColwiseParallel(),
            "attention.v_proj": ColwiseParallel(),
            "attention.c_proj": RowwiseParallel(output_layouts=Shard(1)),
            "attention_norm": SequenceParallel(),
            "feed_forward": PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "feed_forward.gate_proj": ColwiseParallel(),
            "feed_forward.down_proj": RowwiseParallel(output_layouts=Shard(1)),
            "feed_forward.up_proj": ColwiseParallel(),
            "ffn_norm": SequenceParallel(),
        }

        layer.attention.num_heads //= world_mesh["tp"].size()
        layer.attention.num_kv_heads //= world_mesh["tp"].size()

        parallelize_module(
            module=layer,
            device_mesh=world_mesh["tp"],
            parallelize_plan=layer_plan,
        )
    logger.info(f"Model parallelized with Tensor Parallelism (size: {world_mesh['tp'].size()})")
    return parallelize_with_fsdp2(model, config, world_mesh["dp"], with_activation_checkpointing)

def model_distributed_checkpoint_files_exist(ckpt_file_name, ckpt_dir):
    return os.path.exists(get_config_checkpoint_path(ckpt_file_name, ckpt_dir)) \
            and os.path.exists(os.path.join(ckpt_dir, ckpt_file_name, "model"))

def load_distributed_checkpoint(model, optimizer, ckpt_dir, ckpt_name, config):
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    model_ckpt_path = os.path.join(ckpt_path, "model")
    
    model_state = get_model_state_dict(model)
    if os.path.exists(model_ckpt_path):
        #model_state = {"model": ModelWrapper(model)}
        dcp.load(model_state, checkpoint_id=model_ckpt_path)
        model.load_state_dict(model_state)
    else:
        model_ckpt_path = get_model_checkpoint_path(ckpt_name, ckpt_dir)
        if os.path.exists(model_ckpt_path):
            dcp.load(model_state, storage_reader=BroadcastingTorchSaveReader(), planner=DynamicMetaLoadPlanner(), checkpoint_id=model_ckpt_path)
            model.load_state_dict(model_state)
        else:
            raise Exception("Model checkpoint not found")
    logger.info(f"Finished loading model checkpoint from {model_ckpt_path}")
    if config.log_checkpoint_md5_on_load:
        logger.warning(f"Logging checkpoint MD5 is not supported with distributed checkpoint")
    
    if optimizer is not None and os.path.exists(os.path.join(ckpt_path, "optimizer")):
        optimizer_state = {"optimizer": OptimizerWrapper(model, optimizer)}
        dcp.load(optimizer_state, checkpoint_id=os.path.join(ckpt_path, "optimizer"), planner=DynamicMetaLoadPlanner())
        logger.info(f"Finished loading optimizer checkpoint from {ckpt_path}")
    else:
        logger.warning("Optimizer checkpoint not found")
        
def save_distributed_checkpoint(model, optimizer, ckpt_dir, ckpt_name, config, model_only):
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    
    model_ckpt_path = os.path.join(ckpt_path, "model")
    logger.info(f"saving model checkpoint to {model_ckpt_path}")
    #model_state = {"model": ModelWrapper(model)}
    model_state = get_model_state_dict(model)
    dcp.save(model_state, checkpoint_id=model_ckpt_path)
    logger.info(f"model checkpoint saved in {model_ckpt_path}")
    
    if optimizer is not None and not model_only and config.save_optimizer_checkpoint:
        optimizer_ckpt_path = os.path.join(ckpt_path, "optimizer")
        logger.info(f"saving optimizer checkpoint to {optimizer_ckpt_path}")
        optimizer_state = {"optimizer": OptimizerWrapper(model, optimizer)}
        dcp.save(optimizer_state, checkpoint_id=optimizer_ckpt_path)
        logger.info(f"model optimizer saved in {optimizer_ckpt_path}")
    

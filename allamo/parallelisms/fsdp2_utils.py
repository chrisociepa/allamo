"""
Tensor Parallelism API requires PyTorch 2.3.0+
"""
from allamo.logging import logger
from allamo.configuration import AllamoConfiguration
from allamo.torch_utils import (
    TORCH_DTYPE_MAP,
)
from allamo.training_context import TrainingContext

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
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

def build_world_mesh(train_ctx: TrainingContext, device_type: str = "cuda"):
    dims = (train_ctx.pp, train_ctx.dp, train_ctx.tp)
    dim_names = ("pp", "dp", "tp")
    device_mesh = init_device_mesh(device_type, dims, mesh_dim_names=dim_names)
    logger.info(f"{len(dims)}-D device mesh built: {dim_names} = {dims}")
    return device_mesh

def parallelize_model_with_fsdp2(model: nn.Module, world_mesh: DeviceMesh, config: AllamoConfiguration):
    if world_mesh['tp'].size() > 1:
        apply_tensor_parallelism(model, world_mesh)
    
    if config.gradient_checkpointing:
        apply_activation_checkpointing(model)
    
    apply_fsdp(model, world_mesh, config)
    
    if config.compile:
        logger.info("Compiling model")
        try:
            model = torch.compile(model, mode=config.compile_mode)
            logger.info("Model compiled and ready to use")
        except Exception as err:
            logger.warning(f"Unable to compile the model: {err}")
    return model

def apply_tensor_parallelism(model: nn.Module, world_mesh: DeviceMesh):
    logger.warning(
        "Tensor parallelism is in an early experimental stage. "
        "Strided sharding is required for 2D/3D DCP, but it is only available in nightly builds "
        "newer than 20240809 and in PyTorch version 2.5 or later."
    )    
    parallelize_module(
        model,
        world_mesh["tp"],
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "norm": SequenceParallel(),
            "lm_head": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Replicate(),
                use_local_output=True,
            ),
        },
    )
    
    for layer in model.layers:
        layer_plan = {
            "attention_norm": SequenceParallel(),
            "attention": PrepareModuleInput(
                input_layouts=(Shard(1), None, None),
                desired_input_layouts=(Replicate(), None, None),
            ),
            "attention.q_proj": ColwiseParallel(),
            "attention.k_proj": ColwiseParallel(),
            "attention.v_proj": ColwiseParallel(),
            "attention.c_proj": RowwiseParallel(output_layouts=Shard(1)),
            "ffn_norm": SequenceParallel(),
            "feed_forward": PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "feed_forward.gate_proj": ColwiseParallel(),
            "feed_forward.down_proj": RowwiseParallel(output_layouts=Shard(1)),
            "feed_forward.up_proj": ColwiseParallel(),
        }
        
        layer.attention.num_heads //= world_mesh["tp"].size()
        layer.attention.num_kv_heads //= world_mesh["tp"].size()
        
        parallelize_module(
            module=layer,
            device_mesh=world_mesh["tp"],
            parallelize_plan=layer_plan,
        )
    logger.info(f"Model parallelized with Tensor Parallelism (size: {world_mesh['tp'].size()})")

def apply_activation_checkpointing(model: nn.Module):
    for layer_id in range(len(model.layers)):
        model.layers[layer_id] = checkpoint_wrapper(
            model.layers[layer_id],
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            checkpoint_fn=checkpoint,
            use_reentrant=False,
            preserve_rng_state=False,
        )

def apply_fsdp(model: nn.Module, world_mesh: DeviceMesh, config: AllamoConfiguration):
    fsdp_config = {"mesh": world_mesh["dp"]}
    if config.dtype != 'float32':
        fsdp_config["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=TORCH_DTYPE_MAP[config.dtype],
            reduce_dtype=torch.float32
        )
    pp_enabled = world_mesh['pp'].size() > 1
    
    for layer_id, layer in enumerate(model.layers):
        if pp_enabled:
            # For PP, do not reshard after forward to avoid per-microbatch
            # all-gathers, which can be expensive and non-overlapped
            reshard_after_forward = False
        else:
            # As an optimization, do not reshard after forward for the last
            # layer since FSDP would prefetch it immediately
            reshard_after_forward = int(layer_id) < len(model.layers) - 1
        fully_shard(
            layer,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    fully_shard(model, **fsdp_config, reshard_after_forward=not pp_enabled)
    logger.info(f"Model parallelized with FSDP2: {model}\n")

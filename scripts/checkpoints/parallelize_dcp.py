import argparse
import torch
import torch.distributed as dist

from allamo.checkpoint.checkpoint_manager import CheckpointManager
from allamo.configuration import AllamoConfiguration
from allamo.logging import configure_logger, logger
from allamo.model.model import AllamoTransformer
from allamo.torch_utils import init_torch
from allamo.train_utils import (
    get_config_checkpoint_path,
    create_model_config,
)
from allamo.training_context import TrainingContext
from allamo.parallelisms.fsdp2_utils import build_world_mesh, parallelize_model_with_fsdp2
            
def parallelize_dcp(args):
    train_ctx = TrainingContext(
        tp = args.tp_size,
    )
    if train_ctx.master_process:
        configure_logger()
    
    config = AllamoConfiguration(
        load_configuration = False,
        device=args.device,
        backend=args.backend,
        tensor_parallel_degree = args.tp_size,
        out_dir = args.dst,
        compile = False,
    )
    init_torch(train_ctx, config)
    logger.info(f"Torch initialized")
    
    config_ckpt_file_path = get_config_checkpoint_path(args.checkpoint_name, args.src)
    checkpoint_manager = CheckpointManager(config, train_ctx, None)
    checkpoint_manager.load_config_checkpoint(config_ckpt_file_path)
    logger.info("Config checkpoint loaded")
    
    world_mesh = build_world_mesh(train_ctx, args.device)
    model_config = create_model_config(config)
    with torch.device('meta'):
        model = AllamoTransformer(model_config)
    logger.info("Model initialized on meta device")
    
    model.to_empty(device=args.device)
    model.init_model_weights()
    logger.info(f"Model weights initialized on {args.device} device")
    
    checkpoint_manager.load_distributed_model_checkpoint_from(model, args.src, args.checkpoint_name)
    logger.info("Model checkpoint loaded")
    
    model = parallelize_model_with_fsdp2(model, world_mesh, config, False)
    
    checkpoint_manager.save_distributed_model_checkpoint_to(model, args.dst, args.checkpoint_name)
    
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', type=str, required=True, help="Path to source checkpoint directory")
    parser.add_argument('-d', '--dst', type=str, required=True, help="Path to target checkpoint directory")
    parser.add_argument('-n', '--checkpoint_name', type=str, required=True, help="Checkpoint name")
    parser.add_argument('-t', '--tp_size', type=int, required=True, help="Tensor parallel degree")
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu', help="Device type to run conversion on")
    parser.add_argument('--backend', type=str, choices=['gloo', 'mpi', 'nccl'], default='gloo', help="Specifies one of three built-in backends")
    
    args = parser.parse_args()
    
    parallelize_dcp(args)
    logger.info(f"Distributed checkpoint parallelized with TP={args.tp_size}")

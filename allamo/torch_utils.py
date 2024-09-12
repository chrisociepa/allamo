import math
import os
from typing import Optional

import torch
import torch.distributed as dist

from allamo.logging import logger
from allamo.configuration import AllamoConfiguration
from allamo.training_context import TrainingContext

TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bfloat16-true": torch.bfloat16,
}

def override_numa_affinity(local_rank: int, verbose: Optional[bool] = None) -> None:
    if torch.cuda.is_available():
        try:
            import pynvml as nvml
        except ImportError:
            logger.warning("To set CPU affinity on CUDA GPUs the `pynvml` package must be available. (`pip install pynvml`)")
            return

        # The below code is based on https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow2/LanguageModeling/BERT/gpu_affinity.py
        nvml.nvmlInit()
        num_elements = math.ceil(os.cpu_count() / 64)
        handle = nvml.nvmlDeviceGetHandleByIndex(local_rank)
        affinity_string = ""
        for j in nvml.nvmlDeviceGetCpuAffinity(handle, num_elements):
            # assume nvml returns list of 64 bit ints
            affinity_string = f"{j:064b}{affinity_string}"
        affinity_list = [int(x) for x in affinity_string]
        affinity_list.reverse()  # so core 0 is the 0th element
        affinity_to_set = set([i for i, e in enumerate(affinity_list) if e != 0])
        current_affinity = set(os.sched_getaffinity(0))
        affinity_to_set = affinity_to_set.intersection(current_affinity)
        if affinity_to_set:
            os.sched_setaffinity(0, affinity_to_set)
            if verbose:
                cpu_cores = os.sched_getaffinity(0)
                logger.info(f"Assigning {len(cpu_cores)} cpu cores to process {local_rank}: {cpu_cores}")
        else:
            logger.info("No affinity available to set")

def configure_torch(config: AllamoConfiguration, rank: int = 0):
    torch.manual_seed(config.seed + rank)
    torch.cuda.manual_seed(config.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Use for setting the internal precision of float32 matrix multiplications    
    # torch.set_float32_matmul_precision("highest")

def init_torch(train_ctx: TrainingContext, config: AllamoConfiguration, distributed=True):
    if distributed:
        dist.init_process_group(backend=config.backend)
        config.device = f'cuda:{train_ctx.local_rank}'
        torch.cuda.set_device(config.device)

    if train_ctx.master_process:
        logger.info(
            f"RANK: {train_ctx.rank}, LOCAL_RANK: {train_ctx.local_rank}, "
            f"WORLD_SIZE: {train_ctx.world_size}, LOCAL_WORLD_SIZE: {train_ctx.local_world_size}"
        )
        os.makedirs(config.out_dir, exist_ok=True)
        
    configure_torch(config, train_ctx.rank)
    override_numa_affinity(train_ctx.local_rank)
    
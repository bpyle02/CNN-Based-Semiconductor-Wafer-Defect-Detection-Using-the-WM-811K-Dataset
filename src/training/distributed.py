"""
Distributed training utilities for multi-GPU training.

Supports DataParallel and DistributedDataParallel for scalable training.
"""

import logging
import os
from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

logger = logging.getLogger(__name__)


def setup_distributed(rank: int, world_size: int, backend: str = "nccl") -> None:
    """
    Initialize distributed training environment.

    Args:
        rank: Rank of current process
        world_size: Total number of processes
        backend: Communication backend ('nccl' for GPU, 'gloo' for CPU)
    """
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")

    torch.distributed.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )


def cleanup_distributed() -> None:
    """Cleanup distributed training environment."""
    torch.distributed.destroy_process_group()


def is_distributed() -> bool:
    """Check if distributed training is active."""
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_rank() -> int:
    """Get rank of current process."""
    if is_distributed():
        return torch.distributed.get_rank()
    return 0


def get_world_size() -> int:
    """Get total number of processes."""
    if is_distributed():
        return torch.distributed.get_world_size()
    return 1


def wrap_model_dataparallel(
    model: nn.Module,
    device_ids: Optional[List[int]] = None,
) -> nn.Module:
    """
    Wrap model with DataParallel for multi-GPU training.

    Args:
        model: Model to wrap
        device_ids: GPU IDs to use (default: all available)

    Returns:
        Wrapped model
    """
    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))

    if len(device_ids) > 1:
        logger.info(f"Using DataParallel on GPUs: {device_ids}")
        model = DataParallel(model, device_ids=device_ids)
    else:
        logger.info(f"Using single GPU: {device_ids[0]}")

    return model


def wrap_model_distributed(
    model: nn.Module,
    rank: int,
    world_size: int,
) -> nn.Module:
    """
    Wrap model with DistributedDataParallel for distributed training.

    Args:
        model: Model to wrap
        rank: Rank of current process
        world_size: Total number of processes

    Returns:
        Wrapped model
    """
    model = model.to(rank)
    model = DistributedDataParallel(
        model,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=True,
    )
    logger.info(f"Rank {rank}/{world_size}: Model wrapped with DistributedDataParallel")
    return model


def synchronize() -> None:
    """Synchronize all processes in distributed training."""
    if is_distributed():
        torch.distributed.barrier()


def reduce_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """
    Reduce tensor across all processes (average).

    Args:
        tensor: Tensor to reduce
        world_size: Total number of processes

    Returns:
        Reduced tensor
    """
    if is_distributed():
        torch.distributed.all_reduce(tensor)
        tensor = tensor / world_size
    return tensor

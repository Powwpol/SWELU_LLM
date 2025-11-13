"""
Training script with Distributed Data Parallel (DDP) support.
Utilise tous les GPUs disponibles pour accélérer l'entraînement.
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Import du script d'entraînement original
from train import Trainer, main as original_main
import argparse


def setup_ddp():
    """Initialize DDP environment."""
    # Get environment variables set by torchrun
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Initialize process group
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
    
    # Set device
    torch.cuda.set_device(local_rank)
    
    return rank, local_rank, world_size


def cleanup_ddp():
    """Cleanup DDP."""
    if dist.is_initialized():
        dist.destroy_process_group()


def print_rank0(*args, **kwargs):
    """Print only on rank 0."""
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)


if __name__ == "__main__":
    # Setup DDP
    rank, local_rank, world_size = setup_ddp()
    
    print_rank0("=" * 80)
    print_rank0(f"  DISTRIBUTED TRAINING: {world_size} GPUs")
    print_rank0("=" * 80)
    print_rank0()
    
    # Run training with DDP
    try:
        # Patch prints to only show on rank 0
        import builtins
        original_print = builtins.print
        builtins.print = print_rank0
        
        # Run main training
        original_main()
        
    finally:
        # Restore original print
        builtins.print = original_print
        
        # Cleanup
        cleanup_ddp()
        
        print_rank0()
        print_rank0("=" * 80)
        print_rank0("  DDP Training Complete!")
        print_rank0("=" * 80)


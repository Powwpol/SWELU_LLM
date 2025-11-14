"""
Training script for MambaSWELU language model.

Supports:
- Mixed precision training (FP16/BF16)
- Gradient accumulation
- Checkpointing
- Weights & Biases logging
- Distributed Data Parallel (DDP) for multi-GPU
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import json
from tqdm import tqdm
from typing import Optional
import math
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Logging disabled.")

from model import MambaSWELU
from data_prep import WikipediaDataset, create_dataloader
from slimpajama_dataloader import create_slimpajama_dataloader


class Trainer:
    """
    Trainer for MambaSWELU model.
    
    Args:
        model: MambaSWELU model
        train_dataloader: Training data loader
        val_dataloader: Validation data loader (optional)
        learning_rate: Initial learning rate
        weight_decay: Weight decay for AdamW
        warmup_steps: Number of warmup steps
        max_steps: Maximum training steps
        gradient_accumulation_steps: Gradient accumulation steps
        mixed_precision: Whether to use mixed precision ('fp16', 'bf16', or None)
        checkpoint_dir: Directory to save checkpoints
        checkpoint_every: Save checkpoint every N steps
        log_every: Log metrics every N steps
        eval_every: Evaluate every N steps
        use_wandb: Whether to use Weights & Biases logging
    """
    
    def __init__(
        self,
        model: MambaSWELU,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        warmup_steps: int = 2000,
        max_steps: int = 100000,
        gradient_accumulation_steps: int = 4,
        mixed_precision: Optional[str] = "bf16",
        checkpoint_dir: str = "./checkpoints",
        checkpoint_every: int = 5000,
        log_every: int = 100,
        eval_every: int = 1000,
        use_wandb: bool = False,
        resume_from_checkpoint: Optional[str] = None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_every = checkpoint_every
        self.log_every = log_every
        self.eval_every = eval_every
        
        # Setup DDP
        self.is_distributed = dist.is_initialized()
        if self.is_distributed:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
        
        # Setup device
        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Wrap model with DDP if distributed
        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank])
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )
        
        # Setup mixed precision
        self.mixed_precision = mixed_precision
        self.scaler = GradScaler() if mixed_precision == "fp16" else None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
        # Logging
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project="mamba-swelu",
                config={
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "warmup_steps": warmup_steps,
                    "max_steps": max_steps,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "mixed_precision": mixed_precision,
                }
            )
        
        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
    
    def get_lr(self, step: int) -> float:
        """Get learning rate with warmup and cosine decay."""
        if step < self.warmup_steps:
            # Linear warmup
            return self.learning_rate * step / self.warmup_steps
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            return self.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))
    
    def train_step(self, batch: dict) -> float:
        """Single training step."""
        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        
        # Forward pass with mixed precision
        if self.mixed_precision == "bf16":
            with autocast(dtype=torch.bfloat16):
                outputs = self.model(input_ids, labels=labels)
                loss = outputs["loss"]
        elif self.mixed_precision == "fp16":
            with autocast():
                outputs = self.model(input_ids, labels=labels)
                loss = outputs["loss"]
        else:
            outputs = self.model(input_ids, labels=labels)
            loss = outputs["loss"]
        
        # Scale loss for gradient accumulation
        loss = loss / self.gradient_accumulation_steps
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.item() * self.gradient_accumulation_steps
    
    def train(self):
        """Main training loop."""
        self.model.train()
        
        if self.rank == 0:
            print(f"Starting training for {self.max_steps} steps...")
            print(f"Device: {self.device}")
            if self.is_distributed:
                print(f"Distributed: {self.world_size} GPUs")
            print(f"Mixed precision: {self.mixed_precision}")
            print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        
        # Training loop
        train_iterator = iter(self.train_dataloader)
        pbar = tqdm(total=self.max_steps, desc="Training", disable=self.rank != 0)
        
        total_loss = 0.0
        
        while self.global_step < self.max_steps:
            # Get batch
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(self.train_dataloader)
                batch = next(train_iterator)
                self.epoch += 1
            
            # Training step
            loss = self.train_step(batch)
            total_loss += loss
            
            # Update weights every gradient_accumulation_steps
            if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                # Update learning rate
                lr = self.get_lr(self.global_step)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                
                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            self.global_step += 1
            pbar.update(1)
            
            # Logging
            if self.global_step % self.log_every == 0:
                avg_loss = total_loss / self.log_every
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{lr:.2e}"})
                
                if self.use_wandb:
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/lr": lr,
                        "train/step": self.global_step,
                        "train/epoch": self.epoch,
                    })
                
                total_loss = 0.0
            
            # Evaluation
            if self.val_dataloader and self.global_step % self.eval_every == 0:
                val_loss = self.evaluate()
                if self.rank == 0:
                    print(f"\nStep {self.global_step}: Validation loss = {val_loss:.4f}")
                
                    if self.use_wandb:
                        wandb.log({"val/loss": val_loss, "val/step": self.global_step})
                
                self.model.train()
            
            # Checkpointing (only on rank 0)
            if self.rank == 0 and self.global_step % self.checkpoint_every == 0:
                self.save_checkpoint()
        
        pbar.close()
        if self.rank == 0:
            print("Training complete!")
        
        # Save final model (only on rank 0)
        if self.rank == 0:
            self.save_checkpoint(final=True)
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate on validation set."""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(self.val_dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            outputs = self.model(input_ids, labels=labels)
            total_loss += outputs["loss"].item()
            num_batches += 1
            
            # Limit evaluation batches
            if num_batches >= 100:
                break
        
        return total_loss / num_batches
    
    def save_checkpoint(self, final: bool = False):
        """Save model checkpoint."""
        if final:
            checkpoint_path = self.checkpoint_dir / "final_model.pt"
        else:
            checkpoint_path = self.checkpoint_dir / f"model_step_{self.global_step}.pt"
        
        # Unwrap DDP model if needed
        model_to_save = self.model.module if self.is_distributed else self.model
        
        model_to_save.save_pretrained(
            str(checkpoint_path),
            optimizer_state=self.optimizer.state_dict(),
            global_step=self.global_step,
            epoch=self.epoch,
        )
        
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training state from checkpoint."""
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Restore optimizer state if available
        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            print("✅ Optimizer state restored")
        
        # Restore training state
        if "global_step" in checkpoint:
            self.global_step = checkpoint["global_step"]
            print(f"✅ Resuming from step {self.global_step}")
        
        if "epoch" in checkpoint:
            self.epoch = checkpoint["epoch"]
            print(f"✅ Resuming from epoch {self.epoch}")
        
        print("Checkpoint loaded successfully!")


def main():
    parser = argparse.ArgumentParser(description="Train MambaSWELU model")
    
    # Model args
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    
    # Training args
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["fp16", "bf16", "none"])
    
    # Data args
    parser.add_argument("--dataset", type=str, default="wikipedia", choices=["wikipedia", "c4", "slimpajama"])
    parser.add_argument("--num_workers", type=int, default=4)
    
    # Logging args
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--checkpoint_every", type=int, default=5000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--use_wandb", action="store_true")
    
    # Resume training
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, 
                        help="Path to checkpoint file to resume training from")
    
    args = parser.parse_args()
    
    # Create or load model
    if args.resume_from_checkpoint:
        print(f"Loading model from checkpoint: {args.resume_from_checkpoint}")
        model = MambaSWELU.from_pretrained(
            args.resume_from_checkpoint,
            device="cpu"  # Will be moved to GPU later by trainer
        )
        print("✅ Model loaded from checkpoint!")
    else:
        print("Creating model...")
        model = MambaSWELU(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            max_seq_len=args.max_seq_len,
        )
    
    params = model.count_parameters()
    print(f"\nModel parameters: {params['total']:,}")
    
    # Load data
    print("\nLoading data...")
    if args.dataset == "slimpajama":
        print("Using SlimPajama-627B dataset...")
        train_dataloader = create_slimpajama_dataloader(
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            num_workers=0,  # Must be 0 for streaming
            skip_samples=0,
        )
        # No validation dataloader for SlimPajama streaming
        val_dataloader = None
        print("Note: Validation disabled for SlimPajama streaming mode")
    else:
        print("Using Wikipedia dataset...")
        wiki_dataset = WikipediaDataset(seq_len=args.max_seq_len)
        train_dataset = wiki_dataset.load(split="train[:90%]")
        val_dataset = wiki_dataset.load(split="train[90%:]")
        
        train_dataloader = create_dataloader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
        )
        val_dataloader = create_dataloader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=None if args.mixed_precision == "none" else args.mixed_precision,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        log_every=args.log_every,
        eval_every=args.eval_every,
        use_wandb=args.use_wandb,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()


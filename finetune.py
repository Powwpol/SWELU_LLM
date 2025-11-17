#!/usr/bin/env python3
"""
Fine-tuning conversationnel pour MambaSWELU.

Optimisé pour 6x RTX 4090 avec grande capacité.

Usage:
    # Single GPU (test)
    python finetune.py --train_file data/instruction/train.jsonl
    
    # Multi-GPU (6x RTX 4090)
    torchrun --nproc_per_node=6 finetune.py --train_file data/instruction/train.jsonl
"""

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import os
import sys
import time

sys.path.insert(0, str(Path(__file__).parent / "src"))
from model import MambaSWELU


class InstructionDataset(Dataset):
    """Dataset pour fine-tuning conversationnel."""
    
    def __init__(self, file_path: str, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Charger les données
        print(f"📂 Chargement de {file_path}...")
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item['text'])
        
        print(f"✓ {len(self.data):,} exemples chargés")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Labels = input_ids shifted
        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def setup_distributed():
    """Configure l'entraînement distribué."""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return rank, local_rank, world_size
    else:
        return 0, 0, 1


def cleanup_distributed():
    """Nettoie l'entraînement distribué."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_lr_with_warmup(step, warmup_steps, max_steps, base_lr, min_lr=1e-7):
    """Learning rate avec warmup puis cosine decay."""
    if step < warmup_steps:
        # Linear warmup
        return base_lr * (step / warmup_steps)
    else:
        # Cosine decay
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return min_lr + (base_lr - min_lr) * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    scaler,
    args,
    rank,
    local_rank,
    world_size
):
    """Boucle d'entraînement principale."""
    
    model.train()
    global_step = args.start_step
    best_val_loss = float('inf')
    
    # Progress bar (seulement sur rank 0)
    if rank == 0:
        pbar = tqdm(total=args.max_steps, initial=global_step, desc="Training")
    
    epoch = 0
    accumulated_loss = 0.0
    
    while global_step < args.max_steps:
        epoch += 1
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            input_ids = batch['input_ids'].to(local_rank)
            labels = batch['labels'].to(local_rank)
            
            # Forward pass
            with autocast(enabled=args.mixed_precision):
                outputs = model(input_ids)
                logits = outputs['logits']
                
                # Calcul de la loss
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100
                )
                
                # Gradient accumulation
                loss = loss / args.gradient_accumulation_steps
            
            # Backward pass
            if args.mixed_precision:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            accumulated_loss += loss.item()
            
            # Update weights
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                if args.mixed_precision:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                
                optimizer.zero_grad()
                
                global_step += 1
                
                # Update learning rate
                lr = get_lr_with_warmup(
                    global_step,
                    args.warmup_steps,
                    args.max_steps,
                    args.learning_rate
                )
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                
                # Logging
                if global_step % args.log_every == 0 and rank == 0:
                    avg_loss = accumulated_loss / args.log_every
                    pbar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{lr:.2e}',
                        'epoch': epoch
                    })
                    accumulated_loss = 0.0
                
                if rank == 0:
                    pbar.update(1)
                
                # Validation
                if global_step % args.eval_every == 0 and val_loader is not None:
                    val_loss = validate(model, val_loader, local_rank, args)
                    
                    if rank == 0:
                        print(f"\n📊 Step {global_step}: Val Loss = {val_loss:.4f}")
                        
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            print(f"   🏆 New best validation loss!")
                    
                    model.train()
                
                # Checkpointing
                if global_step % args.checkpoint_every == 0 and rank == 0:
                    save_checkpoint(model, optimizer, global_step, epoch, args)
                
                # Stop si max steps atteint
                if global_step >= args.max_steps:
                    break
        
        if global_step >= args.max_steps:
            break
    
    if rank == 0:
        pbar.close()
        print("\n✅ Training terminé!")
        
        # Sauvegarder le modèle final
        save_checkpoint(model, optimizer, global_step, epoch, args, is_final=True)


def validate(model, val_loader, device, args):
    """Évaluation sur le set de validation."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            with autocast(enabled=args.mixed_precision):
                outputs = model(input_ids)
                logits = outputs['logits']
                
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100
                )
            
            total_loss += loss.item()
            num_batches += 1
            
            # Limiter l'évaluation pour aller plus vite
            if num_batches >= 50:
                break
    
    return total_loss / num_batches


def save_checkpoint(model, optimizer, step, epoch, args, is_final=False):
    """Sauvegarde un checkpoint."""
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Unwrap DDP si nécessaire
    model_to_save = model.module if isinstance(model, DDP) else model
    
    if is_final:
        checkpoint_path = checkpoint_dir / "finetuned_model.pt"
    else:
        checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pt"
    
    # Sauvegarder
    checkpoint = {
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'global_step': step,
        'epoch': epoch,
        'config': {
            'vocab_size': model_to_save.vocab_size,
            'd_model': model_to_save.d_model,
            'n_layers': model_to_save.n_layers,
            'max_seq_len': model_to_save.max_seq_len,
            'tie_embeddings': model_to_save.tie_embeddings,
        }
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 Checkpoint sauvegardé: {checkpoint_path}")
    
    # Nettoyer les vieux checkpoints (garder les 3 derniers)
    if not is_final:
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_step_*.pt"))
        if len(checkpoints) > 3:
            for old_ckpt in checkpoints[:-3]:
                old_ckpt.unlink()
                print(f"🗑️  Supprimé: {old_ckpt.name}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MambaSWELU pour chat")
    
    # Data
    parser.add_argument("--train_file", type=str, required=True, help="Fichier de training (.jsonl)")
    parser.add_argument("--val_file", type=str, default=None, help="Fichier de validation (.jsonl)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model_gpu5/final_model.pt", 
                        help="Checkpoint de base à fine-tuner")
    
    # Training hyperparams
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size par GPU")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps")
    parser.add_argument("--max_steps", type=int, default=25000, help="Maximum training steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, 
                        help="Gradient accumulation (effective batch = batch_size * gpus * this)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping")
    
    # Model
    parser.add_argument("--max_length", type=int, default=1024, help="Max sequence length")
    
    # Logging & checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/finetuned")
    parser.add_argument("--checkpoint_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=2000)
    
    # System
    parser.add_argument("--mixed_precision", action="store_true", help="Use FP16 mixed precision")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--start_step", type=int, default=0, help="Start step (for resuming)")
    
    args = parser.parse_args()
    
    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    
    if rank == 0:
        print("="*70)
        print("  🔥 FINE-TUNING MambaSWELU (Option 2 - Grande Capacité)")
        print("="*70)
        print(f"\n🖥️  Configuration:")
        print(f"   GPUs: {world_size}")
        print(f"   Batch size par GPU: {args.batch_size}")
        print(f"   Gradient accumulation: {args.gradient_accumulation_steps}")
        print(f"   Effective batch size: {args.batch_size * world_size * args.gradient_accumulation_steps}")
        print(f"   Learning rate: {args.learning_rate:.2e}")
        print(f"   Max steps: {args.max_steps:,}")
        print(f"   Warmup steps: {args.warmup_steps:,}")
    
    # Load tokenizer
    if rank == 0:
        print(f"\n📝 Chargement tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load checkpoint
    if rank == 0:
        print(f"\n📦 Chargement du checkpoint: {args.checkpoint}")
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint['config']
    
    if rank == 0:
        print(f"   Step de base: {checkpoint.get('global_step', 'N/A')}")
    
    # Create model
    model = MambaSWELU(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(local_rank)
    
    if rank == 0:
        params = sum(p.numel() for p in model.parameters())
        print(f"   Paramètres: {params:,}")
    
    # Wrap avec DDP si multi-GPU
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    # Load datasets
    train_dataset = InstructionDataset(args.train_file, tokenizer, args.max_length)
    
    if args.val_file:
        val_dataset = InstructionDataset(args.val_file, tokenizer, args.max_length)
    else:
        # Utiliser les 5% du train comme validation
        val_file = Path(args.train_file).parent / "val.jsonl"
        if val_file.exists():
            val_dataset = InstructionDataset(str(val_file), tokenizer, args.max_length)
        else:
            val_dataset = None
    
    # Create dataloaders
    if world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank
        )
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
    else:
        val_loader = None
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if args.mixed_precision else None
    
    if rank == 0:
        print(f"\n🚀 Démarrage du fine-tuning...")
        print("="*70)
    
    # Train
    try:
        train(model, train_loader, val_loader, optimizer, scaler, args, rank, local_rank, world_size)
    except KeyboardInterrupt:
        if rank == 0:
            print("\n⚠️  Training interrompu par l'utilisateur")
            save_checkpoint(
                model, optimizer, 
                args.start_step, 0, args,
                is_final=True
            )
    finally:
        cleanup_distributed()
    
    if rank == 0:
        print("\n" + "="*70)
        print("  ✅ FINE-TUNING TERMINÉ")
        print("="*70)
        print(f"\n💡 Modèle sauvegardé dans: {args.checkpoint_dir}/finetuned_model.pt")
        print(f"\n🧪 Pour tester:")
        print(f"   python demo_chat.py --checkpoint {args.checkpoint_dir}/finetuned_model.pt")
        print()


if __name__ == "__main__":
    main()


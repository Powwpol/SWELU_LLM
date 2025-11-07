"""
Script d'entraînement pour petit modèle de test.
Charge la configuration depuis configs/small_model.yaml.

Usage:
    python scripts/train_small.py --config configs/small_model.yaml
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import yaml
import argparse
from model import MambaSWELU
from data_prep import WikipediaDataset, create_dataloader
from train import Trainer
from transformers import GPT2TokenizerFast

def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Train small MambaSWELU model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from {args.config}...")
    config = load_config(args.config)
    
    # Create model
    print("\nCreating model...")
    model = MambaSWELU(
        vocab_size=config['model']['vocab_size'],
        d_model=config['model']['d_model'],
        n_layers=config['model']['n_layers'],
        max_seq_len=config['model']['max_seq_len'],
    )
    
    params = model.count_parameters()
    print(f"Model parameters: {params['total']:,}")
    print(f"  - Embeddings: {params['embeddings']:,}")
    print(f"  - Mamba stack: {params['mamba_stack']:,}")
    print(f"  - Dense layers: {params['dense_layers']:,}")
    print(f"  - SWELU params: {params['swelu_params']:,}")
    
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    if device.type == "cpu":
        print("⚠️  WARNING: No GPU detected! Training will be slow.")
        print("   Consider using Google Colab or RunPod for GPU access.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    else:
        # Print GPU info
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load data
    print("\nLoading Wikipedia dataset...")
    print(f"Max samples: {config['training'].get('max_samples', 'all')}")
    
    wiki_dataset = WikipediaDataset(
        tokenizer_name=config['training']['tokenizer'],
        seq_len=config['model']['max_seq_len'],
        max_samples=config['training'].get('max_samples'),
    )
    
    train_dataset = wiki_dataset.load(split="train[:90%]")
    val_dataset = wiki_dataset.load(split="train[90%:]")
    
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        shuffle=True,
    )
    val_dataloader = create_dataloader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        shuffle=False,
    )
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        warmup_steps=config['training']['warmup_steps'],
        max_steps=config['training']['max_steps'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        mixed_precision=config['training']['mixed_precision'],
        checkpoint_dir=config['training']['checkpoint_dir'],
        checkpoint_every=config['training']['checkpoint_every'],
        log_every=config['training']['log_every'],
        eval_every=config['training']['eval_every'],
        use_wandb=config['training']['use_wandb'],
    )
    
    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    trainer.train()
    
    # Generate sample text
    print("\n" + "="*60)
    print("Generating sample text...")
    print("="*60)
    
    model.eval()
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    
    prompt = "The future of artificial intelligence"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    print(f"Prompt: {prompt}")
    print(f"Generating...")
    
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_length=100,
            temperature=0.8,
            top_p=0.9,
        )
    
    generated_text = tokenizer.decode(generated[0])
    print(f"\nGenerated text:\n{generated_text}")
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Checkpoints saved to: {config['training']['checkpoint_dir']}")
    print("="*60)

if __name__ == "__main__":
    main()


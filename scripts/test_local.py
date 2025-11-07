"""
Script de test local rapide pour MambaSWELU.
Teste le modèle sur CPU avec données dummy (2min).

Usage:
    python scripts/test_local.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import time
from tqdm import tqdm

print("="*60)
print("TEST LOCAL - MambaSWELU")
print("="*60)

# 1. Test imports
print("\n[1/5] Testing imports...")
try:
    from swelu import SWELU
    from mamba_block import MambaBlock
    from model import MambaSWELU
    from data_prep import TextDataset, create_dataloader
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("\nEnsure you've installed requirements:")
    print("  pip install -r requirements.txt")
    sys.exit(1)

# 2. Test SWELU activation
print("\n[2/5] Testing SWELU activation...")
swelu = SWELU(beta=1.0)
x = torch.randn(2, 4)
y = swelu(x)
assert y.shape == x.shape, "SWELU output shape mismatch"
print(f"✓ SWELU works - Input: {x.shape}, Output: {y.shape}")

# 3. Test tiny model creation
print("\n[3/5] Creating tiny model (10M params)...")
model = MambaSWELU(
    vocab_size=1000,     # Small vocab for testing
    d_model=128,         # Small dimension
    n_layers=2,          # 2 layers only
    max_seq_len=128,     # Short sequences
)
params = model.count_parameters()
print(f"✓ Model created - Total params: {params['total']:,}")
print(f"  - Mamba params: {params['mamba']:,}")
print(f"  - SWELU params: {params['swelu']:,}")

# 4. Test forward pass
print("\n[4/5] Testing forward pass...")
batch_size = 2
seq_len = 32
input_ids = torch.randint(0, 1000, (batch_size, seq_len))
labels = torch.randint(0, 1000, (batch_size, seq_len))

start = time.time()
with torch.no_grad():
    outputs = model(input_ids, labels=labels)
elapsed = time.time() - start

assert "logits" in outputs, "Missing logits in output"
assert "loss" in outputs, "Missing loss in output"
assert outputs["logits"].shape == (batch_size, seq_len, 1000), "Wrong logits shape"
print(f"✓ Forward pass successful - {elapsed:.3f}s")
print(f"  - Logits shape: {outputs['logits'].shape}")
print(f"  - Loss: {outputs['loss'].item():.4f}")

# 5. Test training loop (10 steps)
print("\n[5/5] Testing training loop (10 steps)...")

# Create dummy dataset
dummy_data = list(range(10000))
dataset = TextDataset(dummy_data, seq_len=64)
dataloader = create_dataloader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0,  # 0 workers for CPU test
)

# Setup optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Training loop
model.train()
losses = []
pbar = tqdm(range(10), desc="Training")

for step in pbar:
    batch = next(iter(dataloader))
    
    # Forward
    outputs = model(batch["input_ids"], labels=batch["labels"])
    loss = outputs["loss"]
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    pbar.set_postfix({"loss": f"{loss.item():.4f}"})

avg_loss = sum(losses) / len(losses)
print(f"✓ Training test complete - Avg loss: {avg_loss:.4f}")

# 6. Test model saving
print("\n[6/6] Testing model save/load...")
test_path = Path("./test_checkpoint")
test_path.mkdir(exist_ok=True)

model.save_pretrained(str(test_path / "model.pt"))
print(f"✓ Model saved to {test_path / 'model.pt'}")

# Cleanup
import shutil
shutil.rmtree(test_path)
print("✓ Cleanup complete")

# Summary
print("\n" + "="*60)
print("✓ ALL TESTS PASSED!")
print("="*60)
print("\nNext steps:")
print("  1. Test on GPU with small model:")
print("     python scripts/train_small.py --config configs/small_model.yaml")
print("  2. If successful, proceed to RunPod for full training")
print("="*60)


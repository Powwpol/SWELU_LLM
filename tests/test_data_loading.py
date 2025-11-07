"""
Tests for data loading and preprocessing pipeline.
Critical checks before launching expensive cloud training.
"""

import pytest
import torch
from torch.utils.data import DataLoader
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_imports():
    """Test that all required imports work."""
    try:
        from transformers import GPT2Tokenizer
        from datasets import load_dataset
        import torch
        assert torch.cuda.is_available() or True  # Pass even on CPU
        print("✓ All imports successful")
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


def test_tokenizer_loading():
    """Test GPT-2 tokenizer loads correctly."""
    from transformers import GPT2Tokenizer
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    assert tokenizer.vocab_size == 50257
    
    # Test tokenization
    text = "Hello, world!"
    tokens = tokenizer.encode(text)
    assert len(tokens) > 0
    assert all(0 <= t < 50257 for t in tokens)
    
    print(f"✓ Tokenizer working: '{text}' → {len(tokens)} tokens")


def test_wikipedia_dataset_loading():
    """Test Wikipedia dataset can be loaded (streaming mode for speed)."""
    from datasets import load_dataset
    
    # Load tiny sample
    dataset = load_dataset(
        "wikipedia",
        "20220301.en",
        split="train",
        streaming=True,
    )
    
    # Get first item
    first_item = next(iter(dataset))
    assert "text" in first_item
    assert len(first_item["text"]) > 0
    
    print(f"✓ Wikipedia dataset accessible: {len(first_item['text'])} chars in first item")


def test_dataset_tokenization():
    """Test that dataset can be tokenized properly."""
    from datasets import load_dataset
    from transformers import GPT2Tokenizer
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load tiny sample
    dataset = load_dataset(
        "wikipedia",
        "20220301.en",
        split="train[:100]",  # First 100 examples only
    )
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt",
        )
    
    # Tokenize
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    assert "input_ids" in tokenized.column_names
    assert len(tokenized) == 100
    
    print(f"✓ Dataset tokenization working: {len(tokenized)} samples")


def test_dataloader_creation():
    """Test DataLoader can be created and iterated."""
    from datasets import load_dataset
    from transformers import GPT2Tokenizer
    from torch.utils.data import DataLoader
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load tiny sample
    dataset = load_dataset(
        "wikipedia",
        "20220301.en",
        split="train[:50]",
    )
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=128,
            padding="max_length",
        )
    
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    # Set format for PyTorch
    tokenized.set_format("torch")
    
    # Create DataLoader
    dataloader = DataLoader(
        tokenized,
        batch_size=4,
        shuffle=True,
    )
    
    # Get first batch
    batch = next(iter(dataloader))
    assert "input_ids" in batch
    assert batch["input_ids"].shape == (4, 128)
    
    print(f"✓ DataLoader working: batch shape {batch['input_ids'].shape}")


def test_model_forward_with_real_data():
    """Test model forward pass with real tokenized data."""
    from transformers import GPT2Tokenizer
    import sys
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from model import MambaSWELU
    
    # Create tiny model
    model = MambaSWELU(
        vocab_size=50257,
        d_model=128,
        n_layers=1,
        max_seq_len=128,
        dense_hidden_dim=256,
    )
    
    # Tokenize sample text
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    text = "The quick brown fox jumps over the lazy dog."
    tokens = tokenizer.encode(text, return_tensors="pt")
    
    # Pad to 128
    if tokens.shape[1] < 128:
        padding = torch.zeros((1, 128 - tokens.shape[1]), dtype=torch.long)
        tokens = torch.cat([tokens, padding], dim=1)
    
    # Forward pass
    outputs = model(tokens)
    
    assert "logits" in outputs
    assert outputs["logits"].shape == (1, 128, 50257)
    
    print(f"✓ Model forward pass with real data: {outputs['logits'].shape}")


def test_training_step_simulation():
    """Simulate one training step with real-ish data."""
    from transformers import GPT2Tokenizer
    import sys
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from model import MambaSWELU
    
    # Create tiny model
    model = MambaSWELU(
        vocab_size=50257,
        d_model=128,
        n_layers=1,
        max_seq_len=64,
        dense_hidden_dim=256,
    )
    
    # Create fake batch
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, 50257, (batch_size, seq_len))
    labels = input_ids.clone()
    
    # Forward + loss
    outputs = model(input_ids, labels=labels)
    loss = outputs["loss"]
    
    assert loss is not None
    assert loss.item() > 0
    
    # Backward
    loss.backward()
    
    # Check gradients exist
    has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grads
    
    print(f"✓ Training step simulation: loss = {loss.item():.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("SWELU Data Loading Tests")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Tokenizer", test_tokenizer_loading),
        ("Wikipedia Dataset", test_wikipedia_dataset_loading),
        ("Dataset Tokenization", test_dataset_tokenization),
        ("DataLoader", test_dataloader_creation),
        ("Model Forward Pass", test_model_forward_with_real_data),
        ("Training Step", test_training_step_simulation),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            print(f"\n[TEST] {name}")
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed > 0:
        sys.exit(1)


"""
Unit tests for MambaSWELU model architecture.
"""

import pytest
import torch
from src.model import MambaSWELU
from src.mamba_block import MambaBlock, MambaStack


class TestMambaBlock:
    """Test cases for Mamba block."""
    
    def test_mamba_block_shape(self):
        """Test that Mamba block preserves shape."""
        block = MambaBlock(d_model=256, use_swelu=True)
        x = torch.randn(2, 64, 256)
        y = block(x)
        assert y.shape == x.shape
    
    def test_mamba_block_residual(self):
        """Test that residual connection works."""
        block = MambaBlock(d_model=128, use_swelu=False)
        x = torch.randn(1, 32, 128)
        y = block(x)
        
        # Output should not be identical to input (due to processing)
        assert not torch.allclose(x, y)
    
    def test_mamba_block_gradient(self):
        """Test gradient flow through Mamba block."""
        block = MambaBlock(d_model=128, use_swelu=True)
        x = torch.randn(2, 16, 128, requires_grad=True)
        y = block(x)
        loss = y.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestMambaStack:
    """Test cases for Mamba stack."""
    
    def test_stack_shape(self):
        """Test that stacked blocks preserve shape."""
        stack = MambaStack(num_layers=4, d_model=256, use_swelu=True)
        x = torch.randn(2, 64, 256)
        y = stack(x)
        assert y.shape == x.shape
    
    def test_stack_different_depths(self):
        """Test stacks with different depths."""
        for num_layers in [1, 2, 4, 6]:
            stack = MambaStack(num_layers=num_layers, d_model=128)
            x = torch.randn(1, 32, 128)
            y = stack(x)
            assert y.shape == x.shape


class TestMambaSWELU:
    """Test cases for complete MambaSWELU model."""
    
    def test_model_initialization(self):
        """Test that model initializes correctly."""
        model = MambaSWELU(
            vocab_size=1000,
            d_model=256,
            n_layers=2,
            max_seq_len=128,
        )
        assert model is not None
    
    def test_model_forward(self):
        """Test forward pass."""
        model = MambaSWELU(
            vocab_size=1000,
            d_model=256,
            n_layers=2,
            max_seq_len=128,
        )
        
        input_ids = torch.randint(0, 1000, (2, 64))
        outputs = model(input_ids)
        
        assert "logits" in outputs
        assert outputs["logits"].shape == (2, 64, 1000)
    
    def test_model_with_labels(self):
        """Test forward pass with labels (compute loss)."""
        model = MambaSWELU(
            vocab_size=1000,
            d_model=256,
            n_layers=2,
            max_seq_len=128,
        )
        
        input_ids = torch.randint(0, 1000, (2, 64))
        labels = torch.randint(0, 1000, (2, 64))
        
        outputs = model(input_ids, labels=labels)
        
        assert "logits" in outputs
        assert "loss" in outputs
        assert outputs["loss"] is not None
        assert outputs["loss"].item() > 0
    
    def test_model_generation(self):
        """Test text generation."""
        model = MambaSWELU(
            vocab_size=1000,
            d_model=128,
            n_layers=2,
            max_seq_len=128,
        )
        
        prompt = torch.randint(0, 1000, (1, 10))
        generated = model.generate(prompt, max_new_tokens=20, do_sample=False)
        
        assert generated.shape == (1, 30)  # 10 + 20
    
    def test_model_generation_sampling(self):
        """Test text generation with sampling."""
        model = MambaSWELU(
            vocab_size=1000,
            d_model=128,
            n_layers=2,
            max_seq_len=128,
        )
        
        prompt = torch.randint(0, 1000, (1, 10))
        generated = model.generate(
            prompt,
            max_new_tokens=20,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            do_sample=True,
        )
        
        assert generated.shape == (1, 30)
    
    def test_model_parameter_count(self):
        """Test parameter counting."""
        model = MambaSWELU(
            vocab_size=1000,
            d_model=256,
            n_layers=2,
            max_seq_len=128,
        )
        
        params = model.count_parameters()
        
        assert "total" in params
        assert params["total"] > 0
        assert "embeddings" in params
        assert "mamba_stack" in params
        assert "dense_layers" in params
    
    def test_model_save_load(self, tmp_path):
        """Test saving and loading model."""
        model = MambaSWELU(
            vocab_size=1000,
            d_model=128,
            n_layers=2,
            max_seq_len=64,
        )
        
        # Save model
        checkpoint_path = tmp_path / "test_model.pt"
        model.save_pretrained(str(checkpoint_path))
        
        assert checkpoint_path.exists()
        
        # Load model
        loaded_model = MambaSWELU.from_pretrained(
            str(checkpoint_path),
            device="cpu",
        )
        
        # Test that loaded model works
        input_ids = torch.randint(0, 1000, (1, 32))
        outputs = loaded_model(input_ids)
        assert outputs["logits"].shape == (1, 32, 1000)
    
    def test_model_gradient_flow(self):
        """Test gradient flow through entire model."""
        model = MambaSWELU(
            vocab_size=1000,
            d_model=128,
            n_layers=2,
            max_seq_len=64,
        )
        
        input_ids = torch.randint(0, 1000, (2, 32))
        labels = torch.randint(0, 1000, (2, 32))
        
        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]
        loss.backward()
        
        # Check that all parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
    
    def test_model_different_batch_sizes(self):
        """Test model with different batch sizes."""
        model = MambaSWELU(
            vocab_size=1000,
            d_model=128,
            n_layers=2,
            max_seq_len=128,
        )
        
        for batch_size in [1, 2, 4, 8]:
            input_ids = torch.randint(0, 1000, (batch_size, 64))
            outputs = model(input_ids)
            assert outputs["logits"].shape == (batch_size, 64, 1000)
    
    def test_model_max_seq_len(self):
        """Test that model respects max sequence length."""
        model = MambaSWELU(
            vocab_size=1000,
            d_model=128,
            n_layers=2,
            max_seq_len=128,
        )
        
        # Should work
        input_ids = torch.randint(0, 1000, (1, 128))
        outputs = model(input_ids)
        assert outputs["logits"].shape == (1, 128, 1000)
        
        # Should raise error
        with pytest.raises(ValueError):
            input_ids_too_long = torch.randint(0, 1000, (1, 256))
            model(input_ids_too_long)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


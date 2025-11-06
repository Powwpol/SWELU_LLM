"""
Unit tests for SWELU activation function.
"""

import pytest
import torch
from src.swelu import SWELU, swelu


class TestSWELU:
    """Test cases for SWELU activation."""
    
    def test_swelu_shape(self):
        """Test that output shape matches input shape."""
        swelu_fn = SWELU(k_init=1.0)
        x = torch.randn(4, 8, 16)
        y = swelu_fn(x)
        assert y.shape == x.shape
    
    def test_swelu_zero(self):
        """Test SWELU behavior at zero."""
        swelu_fn = SWELU(k_init=1.0)
        x = torch.zeros(10)
        y = swelu_fn(x)
        assert torch.allclose(y, torch.zeros(10), atol=1e-5)
    
    def test_swelu_symmetry(self):
        """Test that SWELU is odd function: f(-x) = -f(x)."""
        swelu_fn = SWELU(k_init=1.0, learnable=False)
        x = torch.randn(100)
        y_pos = swelu_fn(x)
        y_neg = swelu_fn(-x)
        assert torch.allclose(y_pos, -y_neg, atol=1e-5)
    
    def test_swelu_bounded(self):
        """Test that SWELU output is bounded."""
        swelu_fn = SWELU(k_init=1.0)
        x = torch.linspace(-10, 10, 1000)
        y = swelu_fn(x)
        assert (y >= -1.0).all() and (y <= 1.0).all()
    
    def test_swelu_learnable_k(self):
        """Test that k parameter is learnable."""
        swelu_fn = SWELU(k_init=1.5, learnable=True)
        assert swelu_fn.k.requires_grad
        assert abs(swelu_fn.k.item() - 1.5) < 0.01
    
    def test_swelu_fixed_k(self):
        """Test that k parameter can be fixed."""
        swelu_fn = SWELU(k_init=2.0, learnable=False)
        assert not swelu_fn.k.requires_grad
    
    def test_swelu_gradient_flow(self):
        """Test that gradients flow through SWELU."""
        swelu_fn = SWELU(k_init=1.0)
        x = torch.randn(10, requires_grad=True)
        y = swelu_fn(x)
        loss = y.sum()
        loss.backward()
        
        assert x.grad is not None
        assert swelu_fn.k.grad is not None
        assert not torch.isnan(x.grad).any()
    
    def test_swelu_functional(self):
        """Test functional interface."""
        x = torch.randn(10)
        y = swelu(x, k=1.0)
        assert y.shape == x.shape
    
    def test_swelu_different_k(self):
        """Test SWELU with different k values."""
        x = torch.linspace(-3, 3, 100)
        
        swelu1 = SWELU(k_init=0.5, learnable=False)
        swelu2 = SWELU(k_init=1.0, learnable=False)
        swelu3 = SWELU(k_init=2.0, learnable=False)
        
        y1 = swelu1(x)
        y2 = swelu2(x)
        y3 = swelu3(x)
        
        # Different k values should produce different outputs
        assert not torch.allclose(y1, y2)
        assert not torch.allclose(y2, y3)
    
    def test_swelu_batch_processing(self):
        """Test SWELU with different batch sizes."""
        swelu_fn = SWELU(k_init=1.0)
        
        for batch_size in [1, 8, 32]:
            x = torch.randn(batch_size, 128)
            y = swelu_fn(x)
            assert y.shape == (batch_size, 128)
    
    def test_swelu_device_compatibility(self):
        """Test SWELU on different devices."""
        swelu_fn = SWELU(k_init=1.0)
        x_cpu = torch.randn(10)
        y_cpu = swelu_fn(x_cpu)
        assert y_cpu.device.type == "cpu"
        
        if torch.cuda.is_available():
            swelu_fn_cuda = swelu_fn.cuda()
            x_cuda = x_cpu.cuda()
            y_cuda = swelu_fn_cuda(x_cuda)
            assert y_cuda.device.type == "cuda"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


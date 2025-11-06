"""
SWELU: Smooth Weighted Exponential Linear Unit activation function.

Formula: SWELU(z, k) = sign(z) × (1 - exp(-|z|^k))

Where k is a learnable parameter that controls the shape of the activation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SWELU(nn.Module):
    """
    SWELU activation function with learnable parameter k.
    
    Args:
        k_init (float): Initial value for parameter k. Default: 1.0
        learnable (bool): Whether k should be learnable. Default: True
    
    Shape:
        - Input: (N, *) where * means any number of additional dimensions
        - Output: (N, *), same shape as input
    
    Examples:
        >>> swelu = SWELU(k_init=1.0, learnable=True)
        >>> x = torch.randn(2, 3)
        >>> output = swelu(x)
    """
    
    def __init__(self, k_init: float = 1.0, learnable: bool = True):
        super().__init__()
        
        if learnable:
            self.k = nn.Parameter(torch.tensor(k_init, dtype=torch.float32))
        else:
            self.register_buffer('k', torch.tensor(k_init, dtype=torch.float32))
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SWELU activation.
        
        Args:
            z: Input tensor
            
        Returns:
            Activated tensor with same shape as input
        """
        # Ensure k is positive
        k = torch.abs(self.k) + 1e-6
        
        # Compute |z|^k
        abs_z = torch.abs(z)
        z_power_k = torch.pow(abs_z + 1e-8, k)  # Add small epsilon for stability
        
        # Compute 1 - exp(-|z|^k)
        activation = 1.0 - torch.exp(-z_power_k)
        
        # Apply sign
        output = torch.sign(z) * activation
        
        return output
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f'k={self.k.item():.4f}'


class SWELUFunction(torch.autograd.Function):
    """
    Custom autograd function for SWELU with optimized backward pass.
    
    This is an alternative implementation using torch.autograd.Function
    for more control over the gradient computation.
    """
    
    @staticmethod
    def forward(ctx, z, k):
        """Forward pass."""
        k = torch.abs(k) + 1e-6
        abs_z = torch.abs(z)
        z_power_k = torch.pow(abs_z + 1e-8, k)
        activation = 1.0 - torch.exp(-z_power_k)
        output = torch.sign(z) * activation
        
        # Save for backward
        ctx.save_for_backward(z, k, z_power_k, activation)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass with analytical gradients."""
        z, k, z_power_k, activation = ctx.saved_tensors
        
        abs_z = torch.abs(z)
        sign_z = torch.sign(z)
        
        # Gradient w.r.t. z
        # d/dz [sign(z) * (1 - exp(-|z|^k))]
        # = sign(z) * exp(-|z|^k) * k * |z|^(k-1)
        exp_term = torch.exp(-z_power_k)
        grad_z = sign_z * exp_term * k * torch.pow(abs_z + 1e-8, k - 1)
        grad_z = grad_output * grad_z
        
        # Gradient w.r.t. k (if needed)
        # d/dk [1 - exp(-|z|^k)]
        # = exp(-|z|^k) * |z|^k * log(|z|)
        grad_k = None
        if ctx.needs_input_grad[1]:
            log_abs_z = torch.log(abs_z + 1e-8)
            grad_k = (grad_output * sign_z * exp_term * z_power_k * log_abs_z).sum()
        
        return grad_z, grad_k


def swelu(z: torch.Tensor, k: float = 1.0) -> torch.Tensor:
    """
    Functional interface for SWELU activation.
    
    Args:
        z: Input tensor
        k: Shape parameter (default: 1.0)
        
    Returns:
        Activated tensor
        
    Examples:
        >>> x = torch.randn(2, 3)
        >>> output = swelu(x, k=1.5)
    """
    k_tensor = torch.tensor(k, dtype=z.dtype, device=z.device)
    return SWELUFunction.apply(z, k_tensor)


if __name__ == "__main__":
    # Simple test
    print("Testing SWELU activation...")
    
    # Test basic functionality
    x = torch.linspace(-3, 3, 100)
    
    # Test module version
    swelu_module = SWELU(k_init=1.0, learnable=True)
    y_module = swelu_module(x)
    
    # Test functional version
    y_func = swelu(x, k=1.0)
    
    print(f"Input range: [{x.min():.2f}, {x.max():.2f}]")
    print(f"Output range (module): [{y_module.min():.2f}, {y_module.max():.2f}]")
    print(f"Output range (functional): [{y_func.min():.2f}, {y_func.max():.2f}]")
    print(f"Learnable parameter k: {swelu_module.k.item():.4f}")
    
    # Test gradient flow
    x_grad = torch.randn(10, requires_grad=True)
    swelu_test = SWELU(k_init=1.0)
    y = swelu_test(x_grad)
    loss = y.sum()
    loss.backward()
    
    print(f"Gradient check: {x_grad.grad is not None}")
    print(f"k gradient: {swelu_test.k.grad is not None}")
    print("✓ SWELU tests passed!")


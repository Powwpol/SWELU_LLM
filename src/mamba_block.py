"""
Mamba SSM Block with SWELU activation.

Based on: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
Paper: https://arxiv.org/abs/2312.00752
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba-ssm not installed. Using simplified version.")

from .swelu import SWELU


class MambaBlock(nn.Module):
    """
    Single Mamba block with SWELU activation and residual connection.
    
    Architecture:
        Input → LayerNorm → Mamba → SWELU → Residual Add → Output
    
    Args:
        d_model (int): Model dimension
        d_state (int): SSM state dimension. Default: 16
        d_conv (int): Convolution dimension. Default: 4
        expand (int): Expansion factor. Default: 2
        use_swelu (bool): Whether to use SWELU activation. Default: True
        swelu_k (float): Initial k parameter for SWELU. Default: 1.0
        dropout (float): Dropout probability. Default: 0.0
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_swelu: bool = True,
        swelu_k: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Mamba SSM layer
        if MAMBA_AVAILABLE:
            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            # Fallback: simplified attention-like mechanism
            self.mamba = SimplifiedMamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        
        # Activation function
        if use_swelu:
            self.activation = SWELU(k_init=swelu_k, learnable=True)
        else:
            self.activation = nn.SiLU()  # Fallback to SiLU (used in original Mamba)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Store residual
        residual = x
        
        # Pre-normalization
        x = self.norm(x)
        
        # Mamba SSM
        x = self.mamba(x)
        
        # Activation
        x = self.activation(x)
        
        # Dropout
        x = self.dropout(x)
        
        # Residual connection
        x = x + residual
        
        return x


class SimplifiedMamba(nn.Module):
    """
    Simplified Mamba-like layer when mamba-ssm is not available.
    
    This is a simplified version that captures some aspects of Mamba's
    selective state space behavior using standard PyTorch operations.
    NOT intended for production - install mamba-ssm for real implementation.
    """
    
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int):
        super().__init__()
        
        self.d_model = d_model
        self.d_inner = d_model * expand
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Depthwise convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )
        
        # SSM parameters (simplified)
        self.x_proj = nn.Linear(self.d_inner, d_state, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simplified forward pass.
        
        Args:
            x: (batch, seq_len, d_model)
            
        Returns:
            (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape
        
        # Input projection and split
        xz = self.in_proj(x)  # (batch, seq_len, d_inner * 2)
        x, z = xz.chunk(2, dim=-1)  # Each: (batch, seq_len, d_inner)
        
        # Convolution (requires channel-first format)
        x_conv = x.transpose(1, 2)  # (batch, d_inner, seq_len)
        x_conv = self.conv1d(x_conv)[..., :seq_len]  # (batch, d_inner, seq_len)
        x = x_conv.transpose(1, 2)  # (batch, seq_len, d_inner)
        
        # Simplified SSM (using linear transformations)
        x = F.silu(x)  # Activation
        
        # Gate mechanism
        z = F.sigmoid(z)
        x = x * z
        
        # Output projection
        x = self.out_proj(x)
        
        return x


class MambaStack(nn.Module):
    """
    Stack of multiple Mamba blocks.
    
    Args:
        num_layers (int): Number of Mamba blocks
        d_model (int): Model dimension
        d_state (int): SSM state dimension
        d_conv (int): Convolution dimension
        expand (int): Expansion factor
        use_swelu (bool): Whether to use SWELU activation
        swelu_k (float): Initial k parameter for SWELU
        dropout (float): Dropout probability
    """
    
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_swelu: bool = True,
        swelu_k: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                use_swelu=use_swelu,
                swelu_k=swelu_k,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all Mamba blocks.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x)
        
        # Final normalization
        x = self.norm(x)
        
        return x


if __name__ == "__main__":
    # Simple test
    print("Testing Mamba blocks...")
    
    batch_size = 2
    seq_len = 128
    d_model = 256
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Test single block
    block = MambaBlock(d_model=d_model, d_state=16, use_swelu=True)
    y = block(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    assert y.shape == x.shape, "Shape mismatch!"
    
    # Test stack of blocks
    stack = MambaStack(num_layers=4, d_model=d_model, use_swelu=True)
    y_stack = stack(x)
    
    print(f"Stack output shape: {y_stack.shape}")
    assert y_stack.shape == x.shape, "Shape mismatch!"
    
    # Test gradient flow
    y_stack.sum().backward()
    print(f"Gradient check: {x.grad is not None}")
    
    print("✓ Mamba block tests passed!")


"""
SWELU: Smooth Weighted Exponential Linear Unit activation function.

Attribution: Logic developed by Paul OBARA.

Formula: SWELU(z) = lambda * sign(z) * (1 - exp(-|z|^k))

Where:
- k is a learnable parameter controlling shape
- lambda is a learnable parameter controlling scale
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SWELU(nn.Module):
    """
    SWELU activation function with learnable parameters k and lambda.
    
    Attribution: Paul OBARA
    
    Args:
        k_init (float): Initial value for parameter k. Default: 1.0
        lambda_init (float): Initial value for parameter lambda. Default: 1.0
        learnable (bool): Whether parameters should be learnable. Default: True
    """
    
    def __init__(self, k_init: float = 1.0, lambda_init: float = 1.0, learnable: bool = True):
        super().__init__()
        
        if learnable:
            self.k = nn.Parameter(torch.tensor(k_init, dtype=torch.float32))
            self.lam = nn.Parameter(torch.tensor(lambda_init, dtype=torch.float32))
        else:
            self.register_buffer('k', torch.tensor(k_init, dtype=torch.float32))
            self.register_buffer('lam', torch.tensor(lambda_init, dtype=torch.float32))
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SWELU activation.
        """
        # Ensure k is positive
        k = torch.abs(self.k) + 1e-6
        
        # Compute |z|^k
        abs_z = torch.abs(z)
        z_power_k = torch.pow(abs_z + 1e-8, k)
        
        # Compute 1 - exp(-|z|^k)
        activation = 1.0 - torch.exp(-z_power_k)
        
        # Apply sign and lambda
        output = self.lam * torch.sign(z) * activation
        
        return output
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f'k={self.k.item():.4f}, lambda={self.lam.item():.4f}'


# Functional interface omitted for brevity as Module is preferred for learnable params


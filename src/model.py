"""
Complete MambaSWELU LLM architecture.

Architecture:
- Embedding layer
- 6× Mamba blocks with SWELU
- 3× Dense layers with SWELU
- Output projection

Total parameters: ~350M (optimized for RTX 4090)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import math

from .swelu import SWELU
from .mamba_block import MambaStack


class MambaSWELU(nn.Module):
    """
    Complete MambaSWELU language model.
    
    Args:
        vocab_size (int): Vocabulary size. Default: 50257 (GPT-2 tokenizer)
        d_model (int): Model dimension. Default: 1024
        n_layers (int): Number of Mamba blocks. Default: 6
        d_state (int): SSM state dimension. Default: 16
        d_conv (int): Convolution dimension. Default: 4
        expand (int): Expansion factor for Mamba. Default: 2
        max_seq_len (int): Maximum sequence length. Default: 2048
        dense_hidden_dim (int): Hidden dimension for dense layers. Default: 2048
        swelu_k (float): Initial k parameter for SWELU. Default: 1.0
        dropout (float): Dropout probability. Default: 0.1
        tie_embeddings (bool): Whether to tie input/output embeddings. Default: True
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 1024,
        n_layers: int = 6,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        max_seq_len: int = 2048,
        dense_hidden_dim: int = 2048,
        swelu_k: float = 1.0,
        dropout: float = 0.1,
        tie_embeddings: bool = True,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.tie_embeddings = tie_embeddings
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional embeddings (learnable)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Embedding dropout
        self.embed_dropout = nn.Dropout(dropout)
        
        # Mamba blocks (6 layers)
        self.mamba_stack = MambaStack(
            num_layers=n_layers,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_swelu=True,
            swelu_k=swelu_k,
            dropout=dropout,
        )
        
        # Dense layers after Mamba
        # Layer 1: d_model → dense_hidden_dim
        self.dense1 = nn.Linear(d_model, dense_hidden_dim)
        self.swelu1 = SWELU(k_init=swelu_k, learnable=True)
        self.dropout1 = nn.Dropout(dropout)
        
        # Layer 2: dense_hidden_dim → dense_hidden_dim
        self.dense2 = nn.Linear(dense_hidden_dim, dense_hidden_dim)
        self.swelu2 = SWELU(k_init=swelu_k, learnable=True)
        self.dropout2 = nn.Dropout(dropout)
        
        # Layer 3: dense_hidden_dim → d_model (for output projection)
        self.dense3 = nn.Linear(dense_hidden_dim, d_model)
        self.swelu3 = SWELU(k_init=swelu_k, learnable=True)
        
        # Output projection to vocabulary
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie embeddings if specified
        if tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using scaled initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            attention_mask: Attention mask (not used in Mamba, kept for compatibility)
            labels: Labels for language modeling loss (batch, seq_len)
            
        Returns:
            Dictionary containing:
                - logits: Output logits (batch, seq_len, vocab_size)
                - loss: Language modeling loss (if labels provided)
        """
        batch_size, seq_len = input_ids.shape
        
        # Check sequence length
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}"
            )
        
        # Token embeddings
        token_embeds = self.token_embedding(input_ids)  # (batch, seq_len, d_model)
        
        # Positional embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeds = self.position_embedding(positions)  # (1, seq_len, d_model)
        
        # Combine embeddings
        x = token_embeds + position_embeds
        x = self.embed_dropout(x)
        
        # Mamba blocks
        x = self.mamba_stack(x)  # (batch, seq_len, d_model)
        
        # Dense layers
        x = self.dense1(x)
        x = self.swelu1(x)
        x = self.dropout1(x)
        
        x = self.dense2(x)
        x = self.swelu2(x)
        x = self.dropout2(x)
        
        x = self.dense3(x)
        x = self.swelu3(x)
        
        # Output projection
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten for cross-entropy
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,  # Ignore padding tokens
            )
        
        return {
            "logits": logits,
            "loss": loss,
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Starting tokens (batch, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            do_sample: Whether to sample (True) or use greedy (False)
            
        Returns:
            Generated token IDs (batch, seq_len + max_new_tokens)
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Crop to max sequence length
            input_ids_cond = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]
            
            # Forward pass
            with torch.no_grad():
                outputs = self.forward(input_ids_cond)
                logits = outputs["logits"]
            
            # Get logits for last token
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample or take argmax
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component."""
        params = {
            "embeddings": sum(p.numel() for p in self.token_embedding.parameters()) + 
                         sum(p.numel() for p in self.position_embedding.parameters()),
            "mamba_stack": sum(p.numel() for p in self.mamba_stack.parameters()),
            "dense_layers": sum(p.numel() for p in self.dense1.parameters()) +
                           sum(p.numel() for p in self.dense2.parameters()) +
                           sum(p.numel() for p in self.dense3.parameters()),
            "swelu_params": sum(p.numel() for name, p in self.named_parameters() if 'swelu' in name.lower() and 'weight' not in name),
            "lm_head": sum(p.numel() for p in self.lm_head.parameters()) if not self.tie_embeddings else 0,
        }
        params["total"] = sum(params.values())
        return params
    
    @classmethod
    def from_pretrained(cls, checkpoint_path: str, device: str = "cuda") -> "MambaSWELU":
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Get config from checkpoint
        config = checkpoint.get("config", {})
        
        # Create model
        model = cls(**config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        
        return model
    
    def save_pretrained(self, checkpoint_path: str, **kwargs):
        """Save model checkpoint."""
        config = {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "max_seq_len": self.max_seq_len,
            "tie_embeddings": self.tie_embeddings,
        }
        
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": config,
            **kwargs
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Model saved to {checkpoint_path}")


if __name__ == "__main__":
    # Test model
    print("Testing MambaSWELU model...")
    
    # Create small model for testing
    model = MambaSWELU(
        vocab_size=1000,
        d_model=256,
        n_layers=2,
        max_seq_len=128,
        dense_hidden_dim=512,
    )
    
    # Print parameter count
    params = model.count_parameters()
    print("\nParameter count:")
    for k, v in params.items():
        print(f"  {k}: {v:,}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    outputs = model(input_ids)
    print(f"\nInput shape: {input_ids.shape}")
    print(f"Output logits shape: {outputs['logits'].shape}")
    
    # Test with labels
    labels = torch.randint(0, 1000, (batch_size, seq_len))
    outputs_with_loss = model(input_ids, labels=labels)
    print(f"Loss: {outputs_with_loss['loss'].item():.4f}")
    
    # Test generation
    prompt = torch.randint(0, 1000, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20, do_sample=False)
    print(f"\nGenerated shape: {generated.shape}")
    
    print("\n✓ Model tests passed!")


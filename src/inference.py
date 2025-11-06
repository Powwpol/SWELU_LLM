"""
Inference utilities for MambaSWELU model.

Provides simple interface for text generation.
"""

import torch
from transformers import AutoTokenizer
from typing import Optional

from model import MambaSWELU


def generate_text(
    model: MambaSWELU,
    prompt: str,
    tokenizer,
    max_length: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.95,
    device: str = "cuda",
) -> str:
    """
    Generate text from a prompt.
    
    Args:
        model: MambaSWELU model
        prompt: Input text prompt
        tokenizer: Tokenizer instance
        max_length: Maximum length of generated text
        temperature: Sampling temperature
        top_k: Top-k sampling
        top_p: Nucleus sampling
        device: Device to run on
        
    Returns:
        Generated text string
    """
    model.eval()
    model.to(device)
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
        )
    
    # Decode
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return generated_text


if __name__ == "__main__":
    # Example usage
    print("Loading model for inference...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Create or load model
    model = MambaSWELU(vocab_size=50257, d_model=512, n_layers=2)
    
    # Generate text
    prompt = "Once upon a time"
    generated = generate_text(
        model,
        prompt,
        tokenizer,
        max_length=50,
        temperature=0.8,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated}")


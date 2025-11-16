#!/usr/bin/env python3
"""Quick chat test - 3 simple prompts to validate the model."""

import torch
from transformers import AutoTokenizer
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))
from model import MambaSWELU

print("="*70)
print("  🚀 Quick Chat Test - MambaSWELU")
print("="*70)

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")

# Load checkpoint
checkpoint_path = "checkpoints/model_gpu5/final_model.pt"
print(f"Loading: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location=device)
config = checkpoint["config"]

print(f"Config: {config}")
print(f"Step: {checkpoint['global_step']:,}")

# Load model
model = MambaSWELU(**config)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

print("\n" + "="*70)
print("  💬 Chat Tests")
print("="*70)

# Test prompts
prompts = [
    "Hello! How are you today?",
    "What is artificial intelligence?",
    "Write a haiku about coding:",
]

for i, prompt in enumerate(prompts, 1):
    print(f"\n[{i}/{len(prompts)}] You: {prompt}")
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=60,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            do_sample=True,
        )
    
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Remove prompt if included
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
    
    print(f"Assistant: {response}")
    print("-"*70)

print("\n✅ Test completed!\n")



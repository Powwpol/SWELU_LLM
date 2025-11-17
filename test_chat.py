#!/usr/bin/env python3
"""
Non-interactive chat test for MambaSWELU model.
"""

import torch
from transformers import AutoTokenizer
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))
from model import MambaSWELU


def load_model(checkpoint_path, device="cuda"):
    print(f"Loading: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    
    print(f"\nConfig: {config}")
    print(f"Step: {checkpoint.get('global_step', 'N/A'):,}")
    
    model = MambaSWELU(**config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model, config


def generate(model, tokenizer, prompt, device="cuda", max_tokens=80, temp=0.7):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temp,
            top_k=50,
            top_p=0.95,
            do_sample=True,
        )
    
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def main():
    print("=" * 70)
    print("  CHAT TEST - MambaSWELU")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model, _ = load_model("checkpoints/model_gpu5/final_model.pt", device)
    
    print("\n" + "=" * 70)
    print("  TEST PROMPTS")
    print("=" * 70)
    
    prompts = [
        "Question: What is machine learning?\nAnswer:",
        "Write a story: Once upon a time",
        "Code: def hello_world():",
        "Explain: The difference between AI and ML is",
        "Translate to French: Hello, how are you?",
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}] {prompt}")
        print("-" * 70)
        result = generate(model, tokenizer, prompt, device, max_tokens=60, temp=0.7)
        print(f"-> {result}\n")
    
    print("=" * 70)
    print("  TEMPERATURE TESTS")
    print("=" * 70)
    
    test_prompt = "The future of AI is"
    for temp in [0.3, 0.7, 1.0]:
        print(f"\nTemp={temp}: {test_prompt}")
        print("-" * 70)
        result = generate(model, tokenizer, test_prompt, device, max_tokens=50, temp=temp)
        print(f"-> {result}\n")
    
    print("=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()




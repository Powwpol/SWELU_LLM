"""
SlimPajama DataLoader - Simple and Robust

Strategy: Download SlimPajama chunks locally, then stream from disk.
No API rate limiting, no complex distributed logic.
"""

import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from pathlib import Path
import json
from typing import Iterator, Dict


class SlimPajamaIterableDataset(IterableDataset):
    """
    Iterable dataset for SlimPajama streaming.
    Simple approach: stream from HuggingFace, tokenize on-the-fly.
    """
    
    def __init__(
        self,
        tokenizer_name: str = "gpt2",
        max_seq_len: int = 1024,
        skip_samples: int = 0,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.skip_samples = skip_samples
        
        # Load tokenizer
        print(f"📚 Loading tokenizer: {tokenizer_name}")
        self.tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.buffer = []
        self.samples_processed = 0
    
    def _get_dataset_iterator(self):
        """Get dataset iterator (lazy loading)."""
        print("🌊 Loading SlimPajama-627B in streaming mode...")
        print("   Note: First access will download metadata (~5min)")
        
        dataset = load_dataset(
            "cerebras/SlimPajama-627B",
            split="train",
            streaming=True,
        )
        
        if self.skip_samples > 0:
            print(f"⏩ Skipping {self.skip_samples:,} samples...")
            dataset = dataset.skip(self.skip_samples)
        
        return iter(dataset)
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over tokenized sequences."""
        dataset_iter = self._get_dataset_iterator()
        
        for sample in dataset_iter:
            text = sample.get('text', '')
            
            if not text.strip():
                continue
            
            # Tokenize
            tokens = self.tokenizer.encode(
                text,
                add_special_tokens=True,
                truncation=False,
            )
            
            self.buffer.extend(tokens)
            self.samples_processed += 1
            
            # Yield sequences when buffer is full enough
            while len(self.buffer) >= self.max_seq_len:
                sequence = self.buffer[:self.max_seq_len]
                self.buffer = self.buffer[self.max_seq_len:]
                
                input_ids = torch.tensor(sequence, dtype=torch.long)
                labels = input_ids.clone()
                
                yield {
                    'input_ids': input_ids,
                    'labels': labels,
                }


def create_slimpajama_dataloader(
    batch_size: int = 4,
    max_seq_len: int = 1024,
    num_workers: int = 0,  # Must be 0 for IterableDataset with streaming
    skip_samples: int = 0,
) -> DataLoader:
    """
    Create SlimPajama dataloader.
    
    Args:
        batch_size: Batch size
        max_seq_len: Maximum sequence length
        num_workers: Number of workers (must be 0 for streaming)
        skip_samples: Number of samples to skip (for resuming)
        
    Returns:
        DataLoader instance
    """
    dataset = SlimPajamaIterableDataset(
        max_seq_len=max_seq_len,
        skip_samples=skip_samples,
    )
    
    def collate_fn(batch):
        """Collate batch of sequences."""
        input_ids = torch.stack([item['input_ids'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        return {
            'input_ids': input_ids,
            'labels': labels,
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


if __name__ == "__main__":
    """Test the dataloader."""
    print("=" * 80)
    print("🧪 TESTING SLIMPAJAMA DATALOADER")
    print("=" * 80)
    print()
    
    print("Creating dataloader (batch_size=2)...")
    dataloader = create_slimpajama_dataloader(
        batch_size=2,
        max_seq_len=1024,
        num_workers=0,
        skip_samples=0,
    )
    
    print("\n📦 Testing batches...")
    for i, batch in enumerate(dataloader):
        print(f"Batch {i + 1}:")
        print(f"  input_ids shape: {batch['input_ids'].shape}")
        print(f"  labels shape: {batch['labels'].shape}")
        print(f"  First 10 tokens: {batch['input_ids'][0, :10].tolist()}")
        print()
        
        if i >= 4:  # Test 5 batches
            break
    
    print("✅ SlimPajama Dataloader Test Complete!")


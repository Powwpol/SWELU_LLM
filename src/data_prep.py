"""
Data preparation and loading utilities for MambaSWELU training.

Supports:
- Wikipedia dataset
- C4 (Colossal Clean Crawled Corpus)
- Custom text datasets
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Iterator
import datasets
from transformers import AutoTokenizer
from pathlib import Path


class TextDataset(Dataset):
    """
    Simple text dataset for language modeling.
    
    Args:
        data: List of token IDs or path to preprocessed file
        seq_len: Sequence length for training
        stride: Stride for creating overlapping sequences
    """
    
    def __init__(
        self,
        data: List[int],
        seq_len: int = 2048,
        stride: Optional[int] = None,
    ):
        self.data = data
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        
        # Calculate number of samples
        self.num_samples = max(0, (len(data) - seq_len) // self.stride + 1)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.seq_len + 1  # +1 for target
        
        # Get sequence
        seq = self.data[start:end]
        
        # Pad if necessary
        if len(seq) < self.seq_len + 1:
            seq = seq + [0] * (self.seq_len + 1 - len(seq))
        
        # Input and target
        input_ids = torch.tensor(seq[:-1], dtype=torch.long)
        labels = torch.tensor(seq[1:], dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "labels": labels,
        }


class WikipediaDataset:
    """
    Wikipedia dataset loader.
    
    Args:
        tokenizer_name: Name of tokenizer (e.g., 'gpt2')
        seq_len: Sequence length
        cache_dir: Directory to cache preprocessed data
        max_samples: Maximum number of samples (for testing)
    """
    
    def __init__(
        self,
        tokenizer_name: str = "gpt2",
        seq_len: int = 2048,
        cache_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.seq_len = seq_len
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./data/processed")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_samples = max_samples
    
    def load(self, split: str = "train") -> TextDataset:
        """
        Load and preprocess Wikipedia dataset.
        
        Args:
            split: Dataset split ('train' or 'validation')
            
        Returns:
            TextDataset instance
        """
        cache_file = self.cache_dir / f"wikipedia_{split}.pt"
        
        # Load from cache if exists
        if cache_file.exists():
            print(f"Loading cached dataset from {cache_file}...")
            token_ids = torch.load(cache_file)
        else:
            print(f"Downloading and tokenizing Wikipedia ({split})...")
            
            # Load dataset
            dataset = datasets.load_dataset(
                "wikipedia",
                "20220301.en",
                split=split,
            )
            
            # Limit samples if specified
            if self.max_samples:
                dataset = dataset.select(range(min(self.max_samples, len(dataset))))
            
            # Tokenize
            token_ids = []
            for i, example in enumerate(dataset):
                if i % 1000 == 0:
                    print(f"  Tokenized {i}/{len(dataset)} documents...")
                
                text = example["text"]
                tokens = self.tokenizer.encode(text)
                token_ids.extend(tokens)
            
            # Save cache
            torch.save(token_ids, cache_file)
            print(f"Cached dataset to {cache_file}")
        
        print(f"Dataset size: {len(token_ids):,} tokens")
        
        return TextDataset(token_ids, seq_len=self.seq_len)


class C4Dataset:
    """
    C4 (Colossal Clean Crawled Corpus) dataset loader.
    
    Streaming version for large-scale training.
    """
    
    def __init__(
        self,
        tokenizer_name: str = "gpt2",
        seq_len: int = 2048,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.seq_len = seq_len
    
    def load(self, split: str = "train"):
        """Load C4 dataset in streaming mode."""
        print(f"Loading C4 dataset ({split}) in streaming mode...")
        
        dataset = datasets.load_dataset(
            "c4",
            "en",
            split=split,
            streaming=True,
        )
        
        return StreamingTextDataset(dataset, self.tokenizer, self.seq_len)


class StreamingTextDataset:
    """
    Streaming text dataset for large corpora.
    
    Processes data on-the-fly without loading everything into memory.
    """
    
    def __init__(
        self,
        dataset,
        tokenizer,
        seq_len: int = 2048,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.buffer = []
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over dataset, yielding tokenized sequences."""
        for example in self.dataset:
            # Tokenize text
            text = example["text"]
            tokens = self.tokenizer.encode(text)
            self.buffer.extend(tokens)
            
            # Yield sequences from buffer
            while len(self.buffer) >= self.seq_len + 1:
                seq = self.buffer[:self.seq_len + 1]
                self.buffer = self.buffer[self.seq_len:]
                
                input_ids = torch.tensor(seq[:-1], dtype=torch.long)
                labels = torch.tensor(seq[1:], dtype=torch.long)
                
                yield {
                    "input_ids": input_ids,
                    "labels": labels,
                }


def create_dataloader(
    dataset,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create DataLoader from dataset.
    
    Args:
        dataset: Dataset instance
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


if __name__ == "__main__":
    # Test data loading
    print("Testing data preparation...")
    
    # Test simple dataset
    dummy_data = list(range(10000))
    dataset = TextDataset(dummy_data, seq_len=128)
    
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample input shape: {sample['input_ids'].shape}")
    print(f"Sample label shape: {sample['labels'].shape}")
    
    # Test dataloader
    dataloader = create_dataloader(dataset, batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(dataloader))
    print(f"Batch input shape: {batch['input_ids'].shape}")
    print(f"Batch label shape: {batch['labels'].shape}")
    
    print("✓ Data preparation tests passed!")


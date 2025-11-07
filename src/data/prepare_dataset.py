"""
Script de préparation complète du dataset Wikipedia.
Télécharge, tokenize, et sauvegarde le dataset pour training.

Usage:
    python -m src.data.prepare_dataset --output data/processed --max_samples 100000
"""

import argparse
from pathlib import Path
import torch
from transformers import GPT2TokenizerFast
import datasets
from tqdm import tqdm
import json


def prepare_wikipedia(
    output_dir: str,
    max_samples: int = None,
    tokenizer_name: str = "gpt2",
    dataset_version: str = "20220301.en",
):
    """
    Prépare le dataset Wikipedia pour l'entraînement.
    
    Args:
        output_dir: Répertoire de sortie
        max_samples: Nombre max d'échantillons (None = tous)
        tokenizer_name: Nom du tokenizer HuggingFace
        dataset_version: Version du dataset Wikipedia
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Wikipedia Dataset Preparation")
    print("="*60)
    
    # Load tokenizer
    print("\n[1/4] Loading tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
    print(f"✓ Tokenizer loaded: {tokenizer_name}")
    print(f"  Vocab size: {len(tokenizer)}")
    
    # Load dataset
    print("\n[2/4] Downloading Wikipedia dataset...")
    print(f"  Version: {dataset_version}")
    
    dataset = datasets.load_dataset(
        "wikipedia",
        dataset_version,
        split="train",
    )
    
    total_docs = len(dataset)
    print(f"✓ Dataset loaded: {total_docs:,} documents")
    
    # Limit samples if specified
    if max_samples and max_samples < total_docs:
        print(f"  Limiting to {max_samples:,} documents")
        dataset = dataset.select(range(max_samples))
    
    # Tokenize
    print("\n[3/4] Tokenizing documents...")
    all_token_ids = []
    total_tokens = 0
    
    for i, example in enumerate(tqdm(dataset, desc="Tokenizing")):
        text = example["text"]
        
        # Skip empty documents
        if not text or len(text.strip()) < 10:
            continue
        
        # Tokenize
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_token_ids.extend(tokens)
        total_tokens += len(tokens)
        
        # Progress info every 1000 docs
        if (i + 1) % 1000 == 0:
            avg_tokens = total_tokens / (i + 1)
            print(f"  Processed {i+1:,} docs | {total_tokens:,} tokens | Avg: {avg_tokens:.1f} tokens/doc")
    
    print(f"\n✓ Tokenization complete")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Total documents: {len(dataset):,}")
    print(f"  Average tokens/doc: {total_tokens/len(dataset):.1f}")
    
    # Save
    print("\n[4/4] Saving preprocessed data...")
    
    # Save token IDs
    train_file = output_path / "wikipedia_train.pt"
    torch.save(all_token_ids, train_file)
    print(f"✓ Saved token IDs: {train_file}")
    
    # Save metadata
    metadata = {
        "num_documents": len(dataset),
        "num_tokens": total_tokens,
        "tokenizer": tokenizer_name,
        "dataset_version": dataset_version,
        "vocab_size": len(tokenizer),
        "max_samples": max_samples,
    }
    
    metadata_file = output_path / "wikipedia_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata: {metadata_file}")
    
    # Print file sizes
    train_size_mb = train_file.stat().st_size / (1024 * 1024)
    print(f"\nFile size: {train_size_mb:.1f} MB")
    
    print("\n" + "="*60)
    print("✓ Dataset preparation complete!")
    print("="*60)
    print(f"\nOutput directory: {output_path}")
    print(f"  - {train_file.name} ({train_size_mb:.1f} MB)")
    print(f"  - {metadata_file.name}")
    print("\nYou can now use this in training with:")
    print(f"  WikipediaDataset(cache_dir='{output_dir}')")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Wikipedia dataset for training"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of documents (None = all)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
        help="Tokenizer name"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="20220301.en",
        help="Wikipedia dataset version"
    )
    
    args = parser.parse_args()
    
    prepare_wikipedia(
        output_dir=args.output,
        max_samples=args.max_samples,
        tokenizer_name=args.tokenizer,
        dataset_version=args.version,
    )


if __name__ == "__main__":
    main()


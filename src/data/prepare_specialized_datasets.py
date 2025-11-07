"""
Préparation de datasets spécialisés pour SWELU LLM.

Datasets couverts:
1. MATHS: ArXiv, MathPile, ProofWiki, Khan Academy
2. LEAN: Mathlib, Lean 4 examples, formalized mathematics
3. SUPPLY CHAIN: (PROBLÈME: peu de datasets publics massifs)

Usage:
    python src/data/prepare_specialized_datasets.py --domain all --output data/specialized
"""

import argparse
from pathlib import Path
import torch
from transformers import GPT2TokenizerFast
import datasets
from tqdm import tqdm
import json
from typing import List, Dict, Optional
import requests
import os


class SpecializedDatasetPreparator:
    """Prépare des datasets spécialisés pour entraînement."""
    
    def __init__(
        self,
        output_dir: str = "data/specialized",
        tokenizer_name: str = "gpt2",
    ):
        self.output_path = Path(output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
        print(f"✓ Tokenizer loaded: {tokenizer_name} (vocab: {len(self.tokenizer)})")
    
    def prepare_arxiv_math(self, max_papers: Optional[int] = None):
        """
        Télécharge et prépare ArXiv papers (maths/CS).
        
        ArXiv contient ~2.3M papers dont ~700K en maths.
        """
        print("\n" + "="*60)
        print("ArXiv Mathematics Dataset")
        print("="*60)
        
        try:
            # Load ArXiv dataset from HuggingFace
            print("\n[1/3] Loading ArXiv dataset...")
            dataset = datasets.load_dataset(
                "scientific_papers",
                "arxiv",
                split="train",
            )
            
            print(f"✓ Loaded {len(dataset):,} papers")
            
            if max_papers and max_papers < len(dataset):
                print(f"  Limiting to {max_papers:,} papers")
                dataset = dataset.select(range(max_papers))
            
            # Filter for math categories
            print("\n[2/3] Filtering for math papers...")
            math_papers = []
            for paper in tqdm(dataset, desc="Filtering"):
                # ArXiv categories: math.*, cs.LG, stat.ML
                categories = paper.get("sections", [])
                text = paper.get("article", "")
                
                if "math" in str(categories).lower() or len(text) > 500:
                    math_papers.append(text)
            
            print(f"✓ Found {len(math_papers):,} math-related papers")
            
            # Tokenize
            print("\n[3/3] Tokenizing...")
            all_tokens = self._tokenize_texts(math_papers, "ArXiv Math")
            
            # Save
            output_file = self.output_path / "arxiv_math.pt"
            torch.save(all_tokens, output_file)
            
            self._save_metadata("arxiv_math", {
                "num_papers": len(math_papers),
                "num_tokens": len(all_tokens),
                "source": "ArXiv (math categories)",
            })
            
            print(f"\n✓ Saved: {output_file} ({len(all_tokens):,} tokens)")
            return all_tokens
            
        except Exception as e:
            print(f"✗ Error loading ArXiv: {e}")
            print("  ArXiv dataset requires ~50GB download. Continue? (y/n)")
            return []
    
    def prepare_proof_pile(self, max_samples: Optional[int] = None):
        """
        Télécharge Proof-Pile-2 (maths formelles + proofs).
        
        Contient ~15GB de preuves mathématiques formalisées.
        """
        print("\n" + "="*60)
        print("Proof-Pile Dataset (Formal Math)")
        print("="*60)
        
        try:
            # Proof-Pile-2 from HuggingFace
            print("\n[1/3] Loading Proof-Pile...")
            dataset = datasets.load_dataset(
                "EleutherAI/proof-pile-2",
                split="train",
                streaming=True,  # Too large to load at once
            )
            
            # Collect samples
            print("\n[2/3] Collecting proofs...")
            proofs = []
            count = 0
            for item in tqdm(dataset, desc="Collecting"):
                proofs.append(item["text"])
                count += 1
                if max_samples and count >= max_samples:
                    break
            
            print(f"✓ Collected {len(proofs):,} proofs")
            
            # Tokenize
            print("\n[3/3] Tokenizing...")
            all_tokens = self._tokenize_texts(proofs, "Proof-Pile")
            
            # Save
            output_file = self.output_path / "proof_pile.pt"
            torch.save(all_tokens, output_file)
            
            self._save_metadata("proof_pile", {
                "num_proofs": len(proofs),
                "num_tokens": len(all_tokens),
                "source": "Proof-Pile-2 (EleutherAI)",
            })
            
            print(f"\n✓ Saved: {output_file} ({len(all_tokens):,} tokens)")
            return all_tokens
            
        except Exception as e:
            print(f"✗ Error loading Proof-Pile: {e}")
            return []
    
    def prepare_lean_mathlib(self):
        """
        Télécharge Lean Mathlib (bibliothèque de maths formelles).
        
        Clone le repo Mathlib et extrait les fichiers .lean.
        """
        print("\n" + "="*60)
        print("Lean Mathlib (Formalized Mathematics)")
        print("="*60)
        
        mathlib_dir = self.output_path / "lean_mathlib"
        
        if not mathlib_dir.exists():
            print("\n[1/3] Cloning Mathlib repository...")
            print("  This will download ~2GB...")
            
            os.system(f"git clone --depth 1 https://github.com/leanprover-community/mathlib4.git {mathlib_dir}")
        else:
            print(f"✓ Mathlib already cloned: {mathlib_dir}")
        
        # Extract .lean files
        print("\n[2/3] Extracting Lean source files...")
        lean_files = list(mathlib_dir.rglob("*.lean"))
        print(f"✓ Found {len(lean_files):,} Lean files")
        
        # Read and concatenate
        print("\n[3/3] Reading and tokenizing...")
        lean_texts = []
        for lean_file in tqdm(lean_files, desc="Reading"):
            try:
                with open(lean_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if len(content) > 100:  # Skip empty files
                        lean_texts.append(content)
            except Exception as e:
                continue
        
        print(f"✓ Read {len(lean_texts):,} Lean files")
        
        # Tokenize
        all_tokens = self._tokenize_texts(lean_texts, "Lean Mathlib")
        
        # Save
        output_file = self.output_path / "lean_mathlib.pt"
        torch.save(all_tokens, output_file)
        
        self._save_metadata("lean_mathlib", {
            "num_files": len(lean_texts),
            "num_tokens": len(all_tokens),
            "source": "Lean 4 Mathlib (leanprover-community)",
        })
        
        print(f"\n✓ Saved: {output_file} ({len(all_tokens):,} tokens)")
        return all_tokens
    
    def prepare_supply_chain_data(self):
        """
        ⚠️ PROBLÈME: Pas de dataset public massif pour Supply Chain.
        
        Options:
        1. Kaggle: petits datasets (quelques MB)
        2. Web scraping: actualités, blogs, docs
        3. Générer synthétique: simulations, cas d'usage
        """
        print("\n" + "="*60)
        print("Supply Chain Data (LIMITED)")
        print("="*60)
        print("\n⚠️  WARNING: No large public supply chain dataset!")
        print("\nOptions:")
        print("  1. Kaggle datasets (small, ~10MB total)")
        print("  2. Web scraping (news, blogs, whitepapers)")
        print("  3. Synthetic generation (simulations)")
        print("\nRecommendation:")
        print("  - Use general business/economics data as proxy")
        print("  - Fine-tune later on proprietary supply chain data")
        
        # Try to download some available datasets
        try:
            print("\n[1/2] Searching for business/economics datasets...")
            
            # Example: Financial news, business reports
            dataset = datasets.load_dataset(
                "financial_phrasebank",
                "sentences_allagree",
                split="train",
            )
            
            texts = [item["sentence"] for item in dataset]
            print(f"✓ Loaded {len(texts):,} financial sentences")
            
            # Tokenize
            print("\n[2/2] Tokenizing...")
            all_tokens = self._tokenize_texts(texts, "Business/Finance")
            
            # Save
            output_file = self.output_path / "business_finance.pt"
            torch.save(all_tokens, output_file)
            
            self._save_metadata("business_finance", {
                "num_sentences": len(texts),
                "num_tokens": len(all_tokens),
                "source": "Financial PhraseBank (proxy for supply chain)",
                "note": "NOT true supply chain data - use as baseline only",
            })
            
            print(f"\n✓ Saved: {output_file} ({len(all_tokens):,} tokens)")
            print(f"\n⚠️  This is NOT real supply chain data!")
            print("   Consider web scraping or synthetic generation.")
            
            return all_tokens
            
        except Exception as e:
            print(f"✗ Error: {e}")
            return []
    
    def prepare_math_pile(self, max_samples: Optional[int] = None):
        """
        MathPile: Large-scale math dataset (12.7GB).
        """
        print("\n" + "="*60)
        print("MathPile Dataset")
        print("="*60)
        
        try:
            print("\n[1/3] Loading MathPile...")
            print("  Size: ~12.7GB of math text")
            
            dataset = datasets.load_dataset(
                "GAIR/MathPile",
                split="train",
                streaming=True,
            )
            
            # Collect
            print("\n[2/3] Collecting math samples...")
            texts = []
            count = 0
            for item in tqdm(dataset, desc="Collecting"):
                texts.append(item["text"])
                count += 1
                if max_samples and count >= max_samples:
                    break
            
            print(f"✓ Collected {len(texts):,} math documents")
            
            # Tokenize
            print("\n[3/3] Tokenizing...")
            all_tokens = self._tokenize_texts(texts, "MathPile")
            
            # Save
            output_file = self.output_path / "mathpile.pt"
            torch.save(all_tokens, output_file)
            
            self._save_metadata("mathpile", {
                "num_documents": len(texts),
                "num_tokens": len(all_tokens),
                "source": "GAIR/MathPile",
            })
            
            print(f"\n✓ Saved: {output_file} ({len(all_tokens):,} tokens)")
            return all_tokens
            
        except Exception as e:
            print(f"✗ Error loading MathPile: {e}")
            return []
    
    def _tokenize_texts(self, texts: List[str], dataset_name: str) -> List[int]:
        """Tokenize une liste de textes."""
        all_tokens = []
        total_chars = 0
        
        for text in tqdm(texts, desc=f"Tokenizing {dataset_name}"):
            if not text or len(text.strip()) < 10:
                continue
            
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)
            total_chars += len(text)
        
        print(f"  Tokens: {len(all_tokens):,}")
        print(f"  Chars: {total_chars:,}")
        print(f"  Compression: {total_chars / len(all_tokens):.2f} chars/token")
        
        return all_tokens
    
    def _save_metadata(self, dataset_name: str, metadata: Dict):
        """Sauvegarde metadata."""
        metadata["tokenizer"] = "gpt2"
        metadata["vocab_size"] = len(self.tokenizer)
        
        metadata_file = self.output_path / f"{dataset_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Metadata: {metadata_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare specialized datasets for SWELU LLM"
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=["all", "math", "lean", "supply_chain"],
        default="all",
        help="Which domain to prepare"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/specialized",
        help="Output directory"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples per dataset (for testing)"
    )
    
    args = parser.parse_args()
    
    preparator = SpecializedDatasetPreparator(output_dir=args.output)
    
    print("\n" + "="*80)
    print("SPECIALIZED DATASET PREPARATION")
    print("="*80)
    
    if args.domain in ["all", "math"]:
        print("\n🔢 MATHEMATICS DATASETS")
        # preparator.prepare_arxiv_math(max_papers=args.max_samples)
        preparator.prepare_math_pile(max_samples=args.max_samples)
        preparator.prepare_proof_pile(max_samples=args.max_samples)
    
    if args.domain in ["all", "lean"]:
        print("\n🔬 LEAN FORMALIZATION")
        preparator.prepare_lean_mathlib()
    
    if args.domain in ["all", "supply_chain"]:
        print("\n📦 SUPPLY CHAIN (LIMITED)")
        preparator.prepare_supply_chain_data()
    
    print("\n" + "="*80)
    print("✓ DATASET PREPARATION COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {args.output}")
    print("\nNext steps:")
    print("  1. Check file sizes in", args.output)
    print("  2. Update training config to include these datasets")
    print("  3. Combine datasets for training")


if __name__ == "__main__":
    main()


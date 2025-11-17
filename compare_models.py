#!/usr/bin/env python3
"""
Comparaison avant/après fine-tuning.

Compare le modèle de base avec le modèle fine-tuné sur les mêmes prompts.
"""

import torch
from transformers import AutoTokenizer
from pathlib import Path
import sys
import argparse

sys.path.insert(0, str(Path(__file__).parent / "src"))
from model import MambaSWELU


def load_model(checkpoint_path, device="cuda"):
    """Charge un modèle depuis un checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model = MambaSWELU(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint.get('global_step', 'N/A')


def generate(model, tokenizer, prompt, device="cuda", max_tokens=80, temp=0.7):
    """Génère une réponse."""
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
    parser = argparse.ArgumentParser(description="Comparer modèle base vs fine-tuné")
    parser.add_argument(
        "--base_model",
        type=str,
        default="checkpoints/model_gpu5/final_model.pt",
        help="Checkpoint du modèle de base"
    )
    parser.add_argument(
        "--finetuned_model",
        type=str,
        default="checkpoints/finetuned/finetuned_model.pt",
        help="Checkpoint du modèle fine-tuné"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("  🔬 COMPARAISON MODÈLES - Base vs Fine-Tuned")
    print("="*80)
    
    # Load tokenizer
    print("\n📝 Chargement tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Load base model
    print(f"\n📦 Chargement modèle de base...")
    print(f"   {args.base_model}")
    base_model, base_step = load_model(args.base_model, args.device)
    print(f"   Step: {base_step}")
    
    # Load finetuned model
    print(f"\n📦 Chargement modèle fine-tuné...")
    print(f"   {args.finetuned_model}")
    
    if not Path(args.finetuned_model).exists():
        print(f"\n⚠️  Le modèle fine-tuné n'existe pas encore!")
        print(f"   Lancez d'abord le fine-tuning:")
        print(f"   python finetune.py --train_file data/instruction/train.jsonl")
        return
    
    finetuned_model, ft_step = load_model(args.finetuned_model, args.device)
    print(f"   Step: {ft_step}")
    
    # Test prompts
    test_prompts = [
        {
            "prompt": "What is the capital of France?",
            "description": "Question factuelle simple",
            "expected": "Paris"
        },
        {
            "prompt": "Question: What is 2+2?\nAnswer:",
            "description": "Calcul mathématique basique",
            "expected": "4"
        },
        {
            "prompt": "User: Hello! How are you?\nAssistant:",
            "description": "Salutation conversationnelle",
            "expected": "Réponse polie et cohérente"
        },
        {
            "prompt": "Explain machine learning in simple terms:",
            "description": "Explication technique",
            "expected": "Explication claire"
        },
        {
            "prompt": "Write a haiku about programming:",
            "description": "Tâche créative",
            "expected": "Haiku 5-7-5 syllabes"
        },
        {
            "prompt": "def fibonacci(n):\n    # This function",
            "description": "Complétion de code",
            "expected": "Code Python cohérent"
        },
    ]
    
    print("\n" + "="*80)
    print("  📊 RÉSULTATS COMPARATIFS")
    print("="*80)
    
    results = {
        'base': {'better': 0, 'worse': 0, 'equal': 0},
        'finetuned': {'better': 0, 'worse': 0, 'equal': 0}
    }
    
    for i, test in enumerate(test_prompts, 1):
        print(f"\n{'─'*80}")
        print(f"Test {i}/{len(test_prompts)}: {test['description']}")
        print(f"{'─'*80}")
        print(f"\n📝 Prompt:")
        print(f"   {test['prompt'][:100]}...")
        print(f"\n🎯 Attendu: {test['expected']}")
        
        # Base model
        print(f"\n🔵 MODÈLE DE BASE:")
        base_response = generate(base_model, tokenizer, test['prompt'], args.device)
        # Extraire uniquement la réponse
        if base_response.startswith(test['prompt']):
            base_response = base_response[len(test['prompt']):].strip()
        print(f"   {base_response[:200]}")
        
        # Finetuned model
        print(f"\n🟢 MODÈLE FINE-TUNÉ:")
        ft_response = generate(finetuned_model, tokenizer, test['prompt'], args.device)
        if ft_response.startswith(test['prompt']):
            ft_response = ft_response[len(test['prompt']):].strip()
        print(f"   {ft_response[:200]}")
        
        # Demander évaluation manuelle
        print(f"\n📊 Évaluation:")
        print(f"   1 = Base meilleur")
        print(f"   2 = Fine-tuned meilleur")
        print(f"   3 = Égal")
        
        # Auto-évaluation basique (peut être améliorée)
        # Pour l'instant, on affiche juste les résultats
        print(f"   [Évaluation manuelle requise]")
    
    print("\n" + "="*80)
    print("  ✅ COMPARAISON TERMINÉE")
    print("="*80)
    
    print("\n📊 Analyse:")
    print("   - Vérifiez si le modèle fine-tuné répond mieux aux questions")
    print("   - Le modèle devrait être plus cohérent et suivre les instructions")
    print("   - Attendez-vous à voir moins d'hallucinations")
    
    print("\n💡 Métriques quantitatives:")
    print("   Pour une évaluation objective, lancez:")
    print("   python benchmark.py --base_model <base> --finetuned_model <ft>")
    print()


if __name__ == "__main__":
    main()


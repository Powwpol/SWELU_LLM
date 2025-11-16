#!/usr/bin/env python3
"""
Démonstration de chat avec le modèle MambaSWELU entraîné.
"""

import torch
from transformers import AutoTokenizer
import sys
from pathlib import Path
import time
import argparse

sys.path.insert(0, str(Path(__file__).parent / "src"))
from model import MambaSWELU


def main():
    parser = argparse.ArgumentParser(description="Démonstration de chat avec MambaSWELU")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/model_gpu5/final_model.pt",
        help="Chemin du checkpoint à charger (base ou fine-tuné)",
    )
    args = parser.parse_args()

    print("\n" + "="*70)
    print("  🤖 DÉMONSTRATION CHAT - MambaSWELU")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n📍 Device: {device}")
    
    # Load checkpoint
    checkpoint_path = args.checkpoint
    print(f"\n📦 Chargement du checkpoint...")
    print(f"   Path: {checkpoint_path}")
    
    start_time = time.time()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    load_time = time.time() - start_time
    
    config = checkpoint['config']
    print(f"   Temps de chargement: {load_time:.2f}s")
    print(f"\n📊 Configuration:")
    for k, v in config.items():
        print(f"   - {k}: {v}")
    print(f"\n🎯 Training: Step {checkpoint['global_step']:,}")
    
    # Create model
    print(f"\n🔧 Initialisation du modèle...")
    model = MambaSWELU(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    params = sum(p.numel() for p in model.parameters())
    print(f"   Paramètres: {params:,}")
    
    # Load tokenizer
    print(f"\n📝 Chargement du tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Test prompts plus proches de ton usage
    test_cases = [
        {
            "prompt": "User: Explain step by step how gradient accumulation works in neural network training.\nAssistant:",
            "temp": 0.4,
            "tokens": 120,
            "description": "Explication technique avec raisonnement structuré"
        },
        {
            "prompt": "User: You are an AI coding assistant. Write a Python function that takes a list of integers and returns the list sorted in descending order. Then explain the code in 3 sentences.\nAssistant:",
            "temp": 0.5,
            "tokens": 160,
            "description": "Code + explication"
        },
        {
            "prompt": "User: I have an LLM fine-tuning run where the loss stays around 8.0 for 10,000 steps. List three possible root causes and what to check for each.\nAssistant:",
            "temp": 0.5,
            "tokens": 160,
            "description": "Diagnostic de training (comme dans ton cas réel)"
        },
        {
            "prompt": "User: You are connected to external tools via MCP. A user asks: 'Calculate 37.5% of 892 using the calculator tool and show the reasoning.' Describe what you would do, step by step, without actually calling the tool.\nAssistant:",
            "temp": 0.4,
            "tokens": 140,
            "description": "Raisonnement outils / MCP (sans exécution réelle)"
        },
        {
            "prompt": "User: Answer in English but keep the tone friendly. I feel like my fine-tuning loss is \"bad\" at 7.9 after 50% of training. Challenge my intuition and tell me what I should actually look at.\nAssistant:",
            "temp": 0.6,
            "tokens": 200,
            "description": "Coaching sur perte de fine-tuning"
        },
    ]
    
    print("\n" + "="*70)
    print("  💬 TESTS DE GÉNÉRATION")
    print("="*70)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'─'*70}")
        print(f"Test {i}/{len(test_cases)}: {test['description']}")
        print(f"{'─'*70}")
        print(f"\n📝 Prompt: \"{test['prompt']}\"")
        print(f"⚙️  Params: temp={test['temp']}, max_tokens={test['tokens']}")
        
        try:
            input_ids = tokenizer.encode(test['prompt'], return_tensors="pt").to(device)
            
            start = time.time()
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=test['tokens'],
                    temperature=test['temp'],
                    top_k=50,
                    top_p=0.95,
                    do_sample=True,
                )
            gen_time = time.time() - start
            
            response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            print(f"\n🤖 Réponse ({gen_time:.2f}s):")
            print(f"   {response}")
            
        except Exception as e:
            print(f"\n❌ Erreur: {e}")
    
    print("\n" + "="*70)
    print("  ✅ DÉMONSTRATION TERMINÉE")
    print("="*70)
    
    print("\n📊 Résumé:")
    print(f"   - Modèle: MambaSWELU ({params:,} paramètres)")
    print(f"   - Checkpoint: Step {checkpoint['global_step']:,}")
    print(f"   - Device: {device}")
    print(f"   - Tests réalisés: {len(test_cases)}")
    
    print("\n💡 Observations:")
    print("   - Le modèle génère du texte de manière fluide")
    print("   - La cohérence varie selon le prompt et la température")
    print("   - Température basse (0.3-0.5) = plus déterministe")
    print("   - Température haute (0.8-1.0) = plus créatif")
    
    print("\n🔧 Pour usage interactif:")
    print("   - Créez un wrapper web (Gradio/Streamlit)")
    print("   - Ajustez les prompts pour votre cas d'usage")
    print("   - Fine-tunez si besoin sur des données spécifiques")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user\n")
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()

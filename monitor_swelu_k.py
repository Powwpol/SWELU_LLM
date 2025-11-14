"""
Monitorer l'évolution des paramètres k de SWELU pendant l'entraînement.
Charge un checkpoint et affiche les valeurs de k.
"""

import torch
import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent / "src"))
from model import MambaSWELU

def extract_swelu_k_values(model):
    """Extraire tous les k de SWELU."""
    k_values = {}
    
    for name, module in model.named_modules():
        if hasattr(module, 'k') and hasattr(module.k, 'item'):
            k_values[name] = module.k.item()
    
    return k_values

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint')
    parser.add_argument('--gpu', type=int, default=0, help='GPU number (0-5)')
    args = parser.parse_args()
    
    print("═" * 80)
    print("  MONITORING DES PARAMÈTRES k DE SWELU")
    print("═" * 80)
    print()
    
    if args.checkpoint:
        print(f"📂 Chargement checkpoint: {args.checkpoint}")
        model = MambaSWELU.from_pretrained(args.checkpoint, device='cpu')
        
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        step = checkpoint.get('global_step', 'unknown')
        print(f"   Step: {step}")
    else:
        # Chercher le dernier checkpoint
        checkpoint_dir = Path(f"checkpoints/model_gpu{args.gpu}")
        checkpoints = sorted(checkpoint_dir.glob("model_step_*.pt"))
        
        if not checkpoints:
            print(f"❌ Aucun checkpoint trouvé dans {checkpoint_dir}")
            print()
            print("💡 Utilisation:")
            print("   python monitor_swelu_k.py --checkpoint checkpoints/model_gpu0/model_step_5000.pt")
            print("   python monitor_swelu_k.py --gpu 0  # Dernier checkpoint GPU 0")
            return
        
        latest = checkpoints[-1]
        print(f"📂 Dernier checkpoint GPU {args.gpu}: {latest.name}")
        model = MambaSWELU.from_pretrained(str(latest), device='cpu')
        
        checkpoint = torch.load(latest, map_location='cpu')
        step = checkpoint.get('global_step', 'unknown')
        print(f"   Step: {step}")
    
    print()
    
    # Extraire les k
    k_values = extract_swelu_k_values(model)
    
    print(f"🔍 {len(k_values)} paramètres k trouvés")
    print()
    
    # Grouper par type
    mamba_k = {name: val for name, val in k_values.items() if 'mamba' in name}
    dense_k = {name: val for name, val in k_values.items() if 'swelu' in name and 'mamba' not in name}
    
    print("━" * 80)
    print("📊 MAMBA BLOCKS (12 SWELU)")
    print("━" * 80)
    for name, val in sorted(mamba_k.items()):
        layer_num = name.split('.')[2] if 'layers' in name else '?'
        activation_type = 'block' if 'activation' in name else 'SSM'
        print(f"   Layer {layer_num} ({activation_type:>5}): k = {val:.6f}")
    print()
    
    # Stats Mamba
    if mamba_k:
        mamba_vals = list(mamba_k.values())
        print(f"   Min:  {min(mamba_vals):.6f}")
        print(f"   Max:  {max(mamba_vals):.6f}")
        print(f"   Mean: {sum(mamba_vals)/len(mamba_vals):.6f}")
        print(f"   Std:  {torch.tensor(mamba_vals).std().item():.6f}")
    print()
    
    print("━" * 80)
    print("📊 DENSE LAYERS (3 SWELU)")
    print("━" * 80)
    for name, val in sorted(dense_k.items()):
        print(f"   {name:20} k = {val:.6f}")
    print()
    
    # Stats Dense
    if dense_k:
        dense_vals = list(dense_k.values())
        print(f"   Min:  {min(dense_vals):.6f}")
        print(f"   Max:  {max(dense_vals):.6f}")
        print(f"   Mean: {sum(dense_vals)/len(dense_vals):.6f}")
        print(f"   Std:  {torch.tensor(dense_vals).std().item():.6f}")
    print()
    
    print("━" * 80)
    print("📈 ÉVOLUTION vs INITIAL (k_init = 1.0)")
    print("━" * 80)
    all_vals = list(k_values.values())
    deviations = [abs(v - 1.0) for v in all_vals]
    avg_deviation = sum(deviations) / len(deviations)
    
    print(f"   Écart moyen de k vs 1.0:  {avg_deviation:.6f}")
    print(f"   Max écart:                {max(deviations):.6f}")
    print()
    
    if avg_deviation > 0.01:
        print("   ✅ Les k ÉVOLUENT pendant l'entraînement!")
        print("   → SWELU apprend bien ses paramètres")
    else:
        print("   ⚠️  Les k bougent peu (entraînement précoce?)")
    
    print()
    print("═" * 80)

if __name__ == "__main__":
    main()


"""
Script pour vérifier que les paramètres k de SWELU sont bien appris pendant l'entraînement.
"""

import torch
import sys
from pathlib import Path

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from model import MambaSWELU

def check_swelu_params(model):
    """Extraire tous les paramètres k de SWELU dans le modèle."""
    swelu_params = {}
    
    for name, module in model.named_modules():
        if 'swelu' in name.lower() or 'SWELU' in str(type(module)):
            if hasattr(module, 'k'):
                swelu_params[name] = {
                    'value': module.k.item() if hasattr(module.k, 'item') else module.k,
                    'requires_grad': module.k.requires_grad,
                    'is_parameter': isinstance(module.k, torch.nn.Parameter)
                }
    
    return swelu_params

def main():
    print("═" * 80)
    print("  VÉRIFICATION SWELU - PARAMÈTRES APPRENABLES")
    print("═" * 80)
    print()
    
    # Créer un modèle
    print("📊 Création d'un modèle MambaSWELU...")
    model = MambaSWELU(
        vocab_size=50257,
        d_model=1024,
        n_layers=6,
        max_seq_len=1024,
        swelu_k=1.0,  # Initial k
    )
    
    print("✅ Modèle créé")
    print()
    
    # Compter les paramètres SWELU
    swelu_params = check_swelu_params(model)
    
    print(f"🔍 SWELU trouvés: {len(swelu_params)}")
    print()
    
    # Afficher chaque SWELU
    print("📋 Détails des paramètres k:")
    print("-" * 80)
    print(f"{'Module':<50} {'k value':<12} {'Learnable':<12} {'Parameter'}")
    print("-" * 80)
    
    total_learnable = 0
    for name, info in sorted(swelu_params.items()):
        is_learnable = "✅ OUI" if info['requires_grad'] else "❌ NON"
        is_param = "✅" if info['is_parameter'] else "❌"
        
        print(f"{name:<50} {info['value']:<12.4f} {is_learnable:<12} {is_param}")
        
        if info['requires_grad']:
            total_learnable += 1
    
    print("-" * 80)
    print()
    
    print(f"✅ Total paramètres k apprenables: {total_learnable}/{len(swelu_params)}")
    print()
    
    # Vérifier les gradients
    print("🧪 Test de gradient flow...")
    model.train()
    
    # Forward pass
    dummy_input = torch.randint(0, 50257, (2, 128))
    outputs = model(dummy_input, labels=dummy_input)
    loss = outputs['loss']
    
    # Backward pass
    loss.backward()
    
    # Vérifier les gradients des k
    print("   Gradients des paramètres k:")
    has_grads = 0
    for name, module in model.named_modules():
        if hasattr(module, 'k') and isinstance(module.k, torch.nn.Parameter):
            if module.k.grad is not None:
                print(f"   ✅ {name}: grad = {module.k.grad.item():.6f}")
                has_grads += 1
            else:
                print(f"   ❌ {name}: grad = None")
    
    print()
    print(f"   Total avec gradients: {has_grads}/{total_learnable}")
    print()
    
    # Compte total des paramètres
    print("📊 STATISTIQUES GLOBALES:")
    total_params = sum(p.numel() for p in model.parameters())
    swelu_k_params = sum(1 for p in swelu_params.values() if p['is_parameter'])
    
    print(f"   Total paramètres modèle:     {total_params:,}")
    print(f"   Paramètres SWELU k:          {swelu_k_params}")
    print(f"   Ratio SWELU/Total:           {swelu_k_params/total_params*100:.6f}%")
    print()
    
    # Vérifier dans le count_parameters
    param_count = model.count_parameters()
    print("   Breakdown (count_parameters):")
    for key, val in param_count.items():
        print(f"     {key:<20} {val:,}")
    print()
    
    # Conclusion
    print("═" * 80)
    if total_learnable == len(swelu_params) and has_grads == total_learnable:
        print("✅ SUCCÈS: Tous les paramètres SWELU k sont apprenables et reçoivent des gradients!")
    elif total_learnable > 0:
        print("⚠️  ATTENTION: Certains SWELU ne sont pas apprenables ou sans gradients")
    else:
        print("❌ ERREUR: Aucun paramètre SWELU apprenable trouvé!")
    print("═" * 80)

if __name__ == "__main__":
    main()


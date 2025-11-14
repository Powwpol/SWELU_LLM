#!/bin/bash
# Affiche les dernières loss de tous les GPUs

clear
echo "═══════════════════════════════════════════════════════════════════"
echo "  📊 LOSS DE TOUS LES MODÈLES (6 GPUs)"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
date
echo ""

for i in {0..5}; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "GPU $i - checkpoints/model_gpu$i/"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if [ -f "logs/gpu$i.log" ]; then
        # Dernière ligne avec loss
        last_loss=$(grep "loss" logs/gpu$i.log | tail -1)
        
        if [ -n "$last_loss" ]; then
            echo "$last_loss"
        else
            echo "Pas encore de loss (chargement...)"
        fi
        
        # Stats
        total_lines=$(wc -l < logs/gpu$i.log)
        echo "Lignes de log: $total_lines"
    else
        echo "❌ Log non trouvé"
    fi
    echo ""
done

echo "═══════════════════════════════════════════════════════════════════"
echo "Commandes:"
echo "  tail -f logs/gpu0.log           # Suivre GPU 0"
echo "  grep 'loss' logs/gpu0.log       # Toutes les loss GPU 0"
echo "  watch -n 10 './show_all_losses.sh'  # Auto-refresh"
echo "═══════════════════════════════════════════════════════════════════"


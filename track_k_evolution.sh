#!/bin/bash
# Suivre l'évolution des k de SWELU à travers les checkpoints

echo "═══════════════════════════════════════════════════════════════════"
echo "  📈 ÉVOLUTION DES PARAMÈTRES k DE SWELU"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

GPU=${1:-0}

echo "GPU: $GPU"
echo ""

cd /root/SWELU_LLM

# Checkpoints disponibles
checkpoints=$(ls -1 checkpoints/model_gpu$GPU/model_step_*.pt 2>/dev/null | sort -V)

if [ -z "$checkpoints" ]; then
    echo "❌ Aucun checkpoint trouvé pour GPU $GPU"
    exit 1
fi

echo "Checkpoints trouvés:"
echo "$checkpoints" | wc -l | xargs echo "  Total:"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Analyse de chaque checkpoint:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

for ckpt in $checkpoints; do
    step=$(basename $ckpt | sed 's/model_step_//' | sed 's/.pt//')
    echo "Step $step:"
    python monitor_swelu_k.py --checkpoint $ckpt 2>/dev/null | grep -E "Mean:|Min:|Max:|Std:" | head -8
    echo ""
done

echo "═══════════════════════════════════════════════════════════════════"


#!/bin/bash
# Monitoring simplifié pour 2 modèles SlimPajama

clear
echo "═══════════════════════════════════════════════════════════════════"
echo "  📊 SLIMPAJAMA: LLaMA vs Chinchilla"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
date
echo ""

# Processus
echo "🔍 Processus:"
ps aux | grep -E "torchrun.*slimpajama" | grep -v grep | wc -l | \
    xargs -I {} echo "   {} processus torchrun actifs"
echo ""

# GPUs
echo "🖥️  GPUs:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader,nounits | \
    awk -F', ' 'BEGIN {print "   GPU | Util | Memory |  Temp | Power"} 
                {printf "   %3s | %3s%% | %5sMB | %3s°C | %3sW\n", $1, $2, $3, $4, $5}'
echo ""

# Logs
echo "📝 LLaMA (100x - GPU 0-2):"
tail -3 logs/slimpajama_llama.log 2>/dev/null | grep -E "loss|Step|Training" | tail -1 || echo "   Chargement..."
echo ""

echo "📝 Chinchilla (20x - GPU 3-5):"
tail -3 logs/slimpajama_chinchilla.log 2>/dev/null | grep -E "loss|Step|Training" | tail -1 || echo "   Chargement..."
echo ""

# Checkpoints
echo "💾 Checkpoints:"
llama_count=$(ls checkpoints/slimpajama_llama/*.pt 2>/dev/null | wc -l)
chin_count=$(ls checkpoints/slimpajama_chinchilla/*.pt 2>/dev/null | wc -l)
echo "   LLaMA:      $llama_count checkpoints"
echo "   Chinchilla: $chin_count checkpoints"
echo ""

echo "═══════════════════════════════════════════════════════════════════"
echo "🔄 Auto-refresh: watch -n 10 './monitor_2_models.sh'"
echo "═══════════════════════════════════════════════════════════════════"


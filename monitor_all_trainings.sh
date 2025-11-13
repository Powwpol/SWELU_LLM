#!/bin/bash
# Monitoring des 3 entraînements en parallèle

echo "═══════════════════════════════════════════════════════════════════"
echo "  📊 MONITORING - 3 MODÈLES EN PARALLÈLE"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# Vérifier processus
echo "🔍 Processus actifs:"
ps aux | grep -E "torchrun|train.py" | grep -v grep | \
    awk '{printf "   PID %s: CPU=%s%%, MEM=%s%% - %s\n", $2, $3, $4, substr($0, index($0,$11))}'
echo ""

# GPUs
echo "🖥️  Utilisation GPUs:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader,nounits | \
    awk -F', ' '{printf "   GPU %s: Util=%3s%%, Mem=%5sMB, Temp=%2s°C, Power=%3sW\n", $1, $2, $3, $4, $5}'
echo ""

# Dernières lignes de chaque log
echo "📝 SlimPajama LLaMA (GPU 0-1) - Dernière ligne:"
tail -1 logs/slimpajama_llama.log 2>/dev/null | grep -E "loss|Step" || echo "   En attente..."
echo ""

echo "📝 SlimPajama Chinchilla (GPU 2-3) - Dernière ligne:"
tail -1 logs/slimpajama_chinchilla.log 2>/dev/null | grep -E "loss|Step" || echo "   En attente..."
echo ""

echo "📝 Wikipedia LLaMA (GPU 4-5) - Dernière ligne:"
tail -1 logs/wikipedia_llama.log 2>/dev/null | grep -E "loss|Step" || echo "   En attente..."
echo ""

# Checkpoints
echo "💾 Checkpoints sauvegardés:"
for dir in slimpajama_llama slimpajama_chinchilla wikipedia_llama; do
    count=$(ls checkpoints/$dir/*.pt 2>/dev/null | wc -l)
    if [ $count -gt 0 ]; then
        latest=$(ls -t checkpoints/$dir/*.pt 2>/dev/null | head -1)
        echo "   $dir: $count fichiers - Dernier: $(basename $latest)"
    else
        echo "   $dir: Aucun checkpoint"
    fi
done
echo ""

echo "═══════════════════════════════════════════════════════════════════"
echo "Commandes utiles:"
echo "  tail -f logs/slimpajama_llama.log      # Suivre modèle 1"
echo "  tail -f logs/slimpajama_chinchilla.log # Suivre modèle 2"
echo "  tail -f logs/wikipedia_llama.log       # Suivre modèle 3"
echo "  watch -n 10 './monitor_all_trainings.sh' # Auto-refresh"
echo "═══════════════════════════════════════════════════════════════════"


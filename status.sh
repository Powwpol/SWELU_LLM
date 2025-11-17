#!/bin/bash
# Affichage rapide du status du fine-tuning

clear
echo "════════════════════════════════════════════════════════════════════"
echo "  🔥 FINE-TUNING MambaSWELU - STATUS"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Check si tourne
if pgrep -f "finetune.py" > /dev/null; then
    echo "✅ STATUS : EN COURS"
    
    # Progression
    PROGRESS=$(tail -1 logs/finetune_full.log 2>/dev/null | grep -oP 'Training:\s+\K[0-9]+%')
    STEP=$(tail -1 logs/finetune_full.log 2>/dev/null | grep -oP '\|\s+\K[0-9]+(?=/25000)')
    
    echo "📊 Progression : ${STEP:-?} / 25,000 steps (${PROGRESS:-?})"
    echo ""
    
    # GPU utilization
    echo "🖥️  GPUs :"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
        awk -F',' '{printf "   GPU %s: %3s%% util | %5s / %5s MB\n", $1, $2, $3, $4}'
    
    echo ""
    
    # Checkpoints
    NUM_CKPT=$(ls checkpoints/finetuned/*.pt 2>/dev/null | wc -l)
    echo "💾 Checkpoints sauvegardés : $NUM_CKPT"
    
    if [ $NUM_CKPT -gt 0 ]; then
        echo "   Dernier :"
        ls -t checkpoints/finetuned/*.pt 2>/dev/null | head -1 | xargs basename
    fi
    
else
    echo "⚠️  STATUS : ARRÊTÉ"
    echo ""
    echo "Pour relancer :"
    echo "  ./launch_finetune_6gpu.sh"
fi

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  📝 Commandes"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "  📊 Monitoring détaillé    : ./monitor_finetune.sh"
echo "  📄 Logs en temps réel     : tail -f logs/finetune_full.log"
echo "  🧪 Tester checkpoint      : python demo_chat.py --checkpoint <path>"
echo "  🛑 Arrêter                : pkill -f 'finetune.py'"
echo ""

chmod +x /root/SWELU_LLM/status.sh

#!/bin/bash
# Script de monitoring du fine-tuning

echo "════════════════════════════════════════════════════════════════════"
echo "  📊 MONITORING FINE-TUNING - MambaSWELU"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Vérifier que le processus tourne
if pgrep -f "finetune.py" > /dev/null; then
    echo "✅ Fine-tuning en cours !"
    echo ""
else
    echo "❌ Aucun fine-tuning détecté"
    echo ""
    exit 1
fi

# Progression
echo "📈 Progression :"
tail -1 logs/finetune_full.log 2>/dev/null | grep -oP 'Training:.*'

echo ""
echo "📊 Dernières metrics (loss):"
tail -50 logs/finetune_full.log 2>/dev/null | grep -E "loss=" | tail -5

echo ""
echo "💾 Checkpoints sauvegardés:"
ls -lht checkpoints/finetuned/*.pt 2>/dev/null | head -5

echo ""
echo "🖥️  Utilisation GPUs:"
nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits | head -6

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  📝 Commandes utiles"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "  Suivre les logs en temps réel:"
echo "    tail -f logs/finetune_full.log"
echo ""
echo "  Tester un checkpoint:"
echo "    python demo_chat.py --checkpoint checkpoints/finetuned/checkpoint_step_5000.pt"
echo ""
echo "  Comparer avec modèle de base:"
echo "    python compare_models.py"
echo ""
echo "  Arrêter le fine-tuning:"
echo "    pkill -f 'finetune.py'"
echo ""


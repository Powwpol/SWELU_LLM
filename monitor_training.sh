#!/bin/bash
# Script de monitoring de l'entraînement

echo "═══════════════════════════════════════════════════════════════════"
echo "  MONITORING ENTRAÎNEMENT MAMBASWELU"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# Vérifier si le processus tourne
if ps aux | grep "train.py" | grep -v grep > /dev/null; then
    echo "✅ Entraînement en cours"
    echo ""
    echo "Processus:"
    ps aux | grep "train.py" | grep -v grep | awk '{printf "  PID: %s, CPU: %s%%, MEM: %s%%\n", $2, $3, $4}'
    echo ""
else
    echo "❌ Aucun entraînement en cours"
    echo ""
fi

# GPU
echo "GPU:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
        awk -F', ' '{printf "  Utilisation: %s%%, Mémoire: %s/%s MB\n", $1, $2, $3}'
else
    echo "  nvidia-smi non disponible"
fi
echo ""

# Logs récents
echo "Dernières lignes du log (tail -20):"
echo "────────────────────────────────────────────────────────────────────"
tail -20 /root/SWELU_LLM/training.log
echo "────────────────────────────────────────────────────────────────────"
echo ""

# Checkpoints
echo "Checkpoints sauvegardés:"
ls -lh /root/SWELU_LLM/checkpoints/*.pt 2>/dev/null || echo "  Aucun checkpoint pour l'instant"
echo ""

echo "═══════════════════════════════════════════════════════════════════"
echo "Commandes utiles:"
echo "  tail -f training.log          # Suivre les logs en temps réel"
echo "  watch -n 5 ./monitor_training.sh  # Rafraîchir toutes les 5s"
echo "  kill <PID>                    # Arrêter l'entraînement"
echo "═══════════════════════════════════════════════════════════════════"


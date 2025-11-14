#!/bin/bash
# Configuration LLaMA Style - 100 tokens/param
# Entraînement optimal pour MambaSWELU 124M

echo "═══════════════════════════════════════════════════════════════════"
echo "  🚀 LANCEMENT ENTRAÎNEMENT LLAMA STYLE"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "  Modèle:              MambaSWELU 124M paramètres"
echo "  Dataset:             SlimPajama-627B"
echo "  Ratio:               100 tokens/param (optimal)"
echo "  Total tokens:        12.4B"
echo "  Steps:               757,500"
echo "  Durée estimée:       ~70h (~3 jours)"
echo "  Checkpoints:         Tous les 5,000 steps"
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo ""

cd /root/SWELU_LLM

# Backup ancien log si existe
if [ -f training.log ]; then
    mv training.log training_old_$(date +%Y%m%d_%H%M%S).log
    echo "📝 Ancien log sauvegardé"
fi

# Lancer l'entraînement
nohup python src/train.py \
  --dataset slimpajama \
  --vocab_size 50257 \
  --d_model 1024 \
  --n_layers 6 \
  --max_seq_len 1024 \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --max_steps 757500 \
  --learning_rate 3e-4 \
  --weight_decay 0.1 \
  --warmup_steps 2000 \
  --mixed_precision bf16 \
  --checkpoint_dir ./checkpoints \
  --checkpoint_every 5000 \
  --log_every 100 \
  --eval_every 1000 \
  > training.log 2>&1 &

sleep 3

echo ""
echo "✅ Entraînement LLaMA-style lancé en arrière-plan!"
echo ""
echo "📊 Monitoring:"
echo "   tail -f training.log              # Suivre les logs"
echo "   ./monitor_training.sh             # Status rapide"
echo "   watch -n 10 ./monitor_training.sh # Rafraîchir auto"
echo ""
echo "🔍 Process ID:"
ps aux | grep "train.py" | grep -v grep | awk '{print "   PID: " $2}'
echo ""
echo "═══════════════════════════════════════════════════════════════════"


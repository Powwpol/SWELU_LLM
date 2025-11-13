#!/bin/bash
# Lance l'entraînement avec configuration Chinchilla optimale

echo "═══════════════════════════════════════════════════════════════════"
echo "  LANCEMENT CONFIGURATION CHINCHILLA OPTIMAL"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "  Ratio: 20 tokens/param (minimum recherche)"
echo "  Total tokens: 2.48B"
echo "  Steps: 151,500"
echo "  Durée estimée: ~14h"
echo ""
echo "═══════════════════════════════════════════════════════════════════"

cd /root/SWELU_LLM

nohup python src/train.py \
  --dataset slimpajama \
  --vocab_size 50257 \
  --d_model 1024 \
  --n_layers 6 \
  --max_seq_len 1024 \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --max_steps 151500 \
  --learning_rate 3e-4 \
  --weight_decay 0.1 \
  --warmup_steps 2000 \
  --mixed_precision bf16 \
  --checkpoint_dir ./checkpoints \
  --checkpoint_every 5000 \
  --log_every 100 \
  --eval_every 1000 \
  > training.log 2>&1 &

echo ""
echo "✅ Entraînement lancé en arrière-plan"
echo "   Logs: tail -f training.log"
echo "   Monitoring: ./monitor_training.sh"


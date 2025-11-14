#!/bin/bash
# Script de lancement de l'entraînement MambaSWELU avec SlimPajama
#
# Usage:
#   ./launch_training.sh                          # Nouvel entraînement
#   ./launch_training.sh ./checkpoints/model.pt   # Reprendre depuis checkpoint

CHECKPOINT_PATH="${1:-}"

echo "═══════════════════════════════════════════════════════════════════"
echo "  ENTRAÎNEMENT MAMBASWELU - SLIMPAJAMA"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  - Dataset: SlimPajama-627B (streaming)"
echo "  - Model: 124M paramètres"
echo "  - Batch size: 4"
echo "  - Sequence length: 1024"
echo "  - Mixed precision: BF16"
echo "  - Max steps: 100,000"

if [ -n "$CHECKPOINT_PATH" ]; then
    echo "  - Reprise depuis: $CHECKPOINT_PATH"
    RESUME_ARG="--resume_from_checkpoint $CHECKPOINT_PATH"
else
    echo "  - Mode: Nouvel entraînement"
    RESUME_ARG=""
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo ""

cd /root/SWELU_LLM

python src/train.py \
  --dataset slimpajama \
  --vocab_size 50257 \
  --d_model 1024 \
  --n_layers 6 \
  --max_seq_len 1024 \
  --batch_size 4 \
  --learning_rate 3e-4 \
  --weight_decay 0.1 \
  --warmup_steps 2000 \
  --max_steps 100000 \
  --gradient_accumulation_steps 4 \
  --mixed_precision bf16 \
  --checkpoint_dir ./checkpoints \
  --checkpoint_every 5000 \
  --log_every 100 \
  --eval_every 1000 \
  $RESUME_ARG


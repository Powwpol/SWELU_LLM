#!/bin/bash
# Hyper Powerful SWELU LLM - Single RTX 6000 Launch Script

echo "═══════════════════════════════════════════════════════════════════"
echo "  🚀 SWELU HYPER MODEL - RTX 6000 (Paul OBARA Logic)"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "  Configuration:"
echo "    - Model:    1.5B Parameters (Hyper)"
echo "    - Dataset:  SlimPajama-627B (Streaming)"
echo "    - GPU:      1x RTX 6000 (48GB)"
echo "    - Logic:    SWELU (k + lambda learnable)"
echo "    - LR:       Kelly-Taguchi Adaptive"
echo ""

cd /root/SWELU_LLM

# Load env vars
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

# Check HF Token
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  WARNING: HF_TOKEN not found in environment."
    echo "   Please export HF_TOKEN='your_token' before running."
    echo "   Streaming SlimPajama might require authentication."
fi

# Launch training
echo "🚀 Launching training..."
python src/train.py \
    --vocab_size 50257 \
    --d_model 2048 \
    --n_layers 32 \
    --max_seq_len 2048 \
    --batch_size 8 \
    --gradient_accumulation_steps 8 \
    --max_steps 100000 \
    --learning_rate 3e-4 \
    --dataset slimpajama \
    --mixed_precision bf16 \
    --checkpoint_dir ./checkpoints_hyper \
    --checkpoint_every 2000 \
    --log_every 50 \
    --use_wandb

echo ""
echo "✅ Training finished!"

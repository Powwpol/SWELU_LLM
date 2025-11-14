#!/bin/bash
# Lance SlimPajama LLaMA en single GPU - Configuration qui fonctionnait!

echo "═══════════════════════════════════════════════════════════════════"
echo "  🚀 SLIMPAJAMA LLAMA - SINGLE GPU (comme avant)"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "  Configuration:"
echo "    - Dataset:  SlimPajama-627B"
echo "    - GPU:      1x RTX 4090 (GPU 0)"
echo "    - Ratio:    100 tokens/param"
echo "    - Steps:    757,500"
echo "    - Tokens:   12.4B"
echo "    - Durée:    ~70h"
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo ""

cd /root/SWELU_LLM

# Arrêter tout
pkill -f train.py 2>/dev/null
sleep 2

# Backup ancien log
[ -f training.log ] && mv training.log training_backup_$(date +%Y%m%d_%H%M%S).log

# Lancer en single GPU (comme avant, ça marchait!)
CUDA_VISIBLE_DEVICES=0 \
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
  > training.log 2>&1 &

PID=$!
sleep 3

echo ""
echo "✅ SlimPajama lancé (single GPU)!"
echo ""
echo "🔍 PID: $PID"
echo "📝 Log: tail -f training.log"
echo "📊 Monitoring: ./monitor_training.sh"
echo ""
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | head -1 | \
    awk -F', ' '{printf "GPU 0: Util=%s%%, Mem=%sMB\n", $2, $3}'
echo ""
echo "═══════════════════════════════════════════════════════════════════"


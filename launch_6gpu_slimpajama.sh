#!/bin/bash
# SlimPajama sur 6 GPUs avec token HF - Configuration OPTIMALE

echo "═══════════════════════════════════════════════════════════════════"
echo "  🚀 SLIMPAJAMA - 6x RTX 4090 (CONFIGURATION OPTIMALE)"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "  Configuration:"
echo "    - Dataset:  SlimPajama-627B ✅"
echo "    - GPUs:     6x RTX 4090 (DDP)"
echo "    - Speedup:  ~6x plus rapide"
echo "    - Ratio:    100 tokens/param (LLaMA style)"
echo "    - Steps:    757,500"
echo "    - Tokens:   12.4B"
echo "    - Durée:    ~11.7h (au lieu de 70h!)"
echo "    - HF Token: ✅"
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo ""

cd /root/SWELU_LLM

# Charger token HF
export $(cat .env | xargs)
echo "✅ Token HF: $HF_TOKEN"
echo ""

# Arrêter tout
pkill -f train.py 2>/dev/null
pkill -f torchrun 2>/dev/null
sleep 3

# Backup log
[ -f training.log ] && mv training.log training_single_gpu_$(date +%Y%m%d_%H%M%S).log

echo "🚀 Lancement sur 6 GPUs..."
echo ""

# Lancer avec DDP sur 6 GPUs + Token HF
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
HF_TOKEN=$HF_TOKEN \
nohup torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=6 \
    src/train.py \
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
sleep 5

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "✅ SLIMPAJAMA 6-GPU LANCÉ!"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "🔍 PID: $PID"
echo ""
echo "📊 GPUs:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu --format=csv,noheader,nounits | \
    awk -F', ' '{printf "   GPU %s: Util=%3s%%, Mem=%5sMB, Temp=%2s°C\n", $1, $2, $3, $4}'
echo ""
echo "📝 Monitoring:"
echo "   tail -f training.log"
echo "   ./monitor_training.sh"
echo "   watch -n 10 nvidia-smi"
echo ""
echo "⏱️  Timeline:"
echo "   ~5min:    Métadonnées SlimPajama chargées"
echo "   ~5min:    Premier batch traité"
echo "   ~28min:   Premier checkpoint (step 5,000)"
echo "   ~11.7h:   Entraînement complet! ✨"
echo ""
echo "💾 Checkpoints: ./checkpoints/model_step_XXXXX.pt"
echo ""
echo "═══════════════════════════════════════════════════════════════════"


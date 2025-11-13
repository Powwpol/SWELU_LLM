#!/bin/bash
# Lancement Multi-GPU avec PyTorch DDP
# Utilise les 6 RTX 4090 pour accélérer l'entraînement

NUM_GPUS=6

echo "═══════════════════════════════════════════════════════════════════"
echo "  🚀 LANCEMENT MULTI-GPU (${NUM_GPUS} x RTX 4090)"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "  Configuration:"
echo "    - GPUs:              ${NUM_GPUS}"
echo "    - Speedup:           ~${NUM_GPUS}x"
echo "    - Durée estimée:     ~11.7h (au lieu de 70h)"
echo "    - Économie:          ~58h"
echo ""
echo "  Tokens:"
echo "    - Total:             12.4B"
echo "    - Ratio:             100 tokens/param"
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo ""

cd /root/SWELU_LLM

# Arrêter l'ancien entraînement
pkill -f train.py
sleep 2

# Backup ancien log
if [ -f training.log ]; then
    mv training.log training_single_gpu_$(date +%Y%m%d_%H%M%S).log
    echo "📝 Ancien log sauvegardé"
fi

# Lancer avec torchrun (DDP)
# Important: Set CUDA_VISIBLE_DEVICES to ensure proper GPU distribution
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
nohup torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=${NUM_GPUS} \
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
    --eval_every 1000 \
    > training.log 2>&1 &

sleep 3

echo ""
echo "✅ Entraînement multi-GPU lancé!"
echo ""
echo "📊 Vérification GPUs:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | \
    awk -F', ' '{printf "   GPU %s: Util=%s%%, Mem=%sMB\n", $1, $2, $3}'
echo ""
echo "🔍 Process:"
ps aux | grep "torchrun" | grep -v grep | awk '{print "   PID: " $2}' || echo "   Démarrage..."
echo ""
echo "📝 Logs: tail -f training.log"
echo ""
echo "═══════════════════════════════════════════════════════════════════"


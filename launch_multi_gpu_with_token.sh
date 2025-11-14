#!/bin/bash
# Lancement Multi-GPU avec token HuggingFace

NUM_GPUS=6

echo "═══════════════════════════════════════════════════════════════════"
echo "  🚀 LANCEMENT MULTI-GPU (${NUM_GPUS} x RTX 4090)"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

cd /root/SWELU_LLM

# Charger le token HF depuis .env si existe
if [ -f .env ]; then
    export $(cat .env | xargs)
    echo "✅ Token HF chargé depuis .env"
elif [ -n "$HF_TOKEN" ]; then
    echo "✅ Token HF détecté dans l'environnement"
else
    echo "⚠️  Pas de token HF détecté"
    echo ""
    echo "SlimPajama nécessite un token HuggingFace"
    echo "Options:"
    echo "  1. Exporter: export HF_TOKEN=your_token"
    echo "  2. Créer .env: echo 'HF_TOKEN=your_token' > .env"
    echo "  3. Utiliser: ./setup_hf_token.sh"
    echo ""
    read -p "Continuer quand même? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "  Configuration:"
echo "    - GPUs:              ${NUM_GPUS}"
echo "    - Speedup:           ~${NUM_GPUS}x"
echo "    - Durée estimée:     ~11.7h (au lieu de 70h)"
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# Arrêter ancien entraînement
pkill -f train.py 2>/dev/null
sleep 2

# Backup log
if [ -f training.log ]; then
    mv training.log training_backup_$(date +%Y%m%d_%H%M%S).log
fi

# Lancer avec torchrun
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
HF_TOKEN=${HF_TOKEN} \
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
echo "📊 GPUs:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | \
    awk -F', ' '{printf "   GPU %s: Util=%s%%, Mem=%sMB\n", $1, $2, $3}'
echo ""
echo "📝 Logs: tail -f training.log"
echo "📊 Monitoring: watch -n 5 './monitor_training.sh'"
echo ""
echo "═══════════════════════════════════════════════════════════════════"


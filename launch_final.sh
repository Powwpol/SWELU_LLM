#!/bin/bash
# Lancement FINAL avec token HF - SlimPajama LLaMA optimal

echo "═══════════════════════════════════════════════════════════════════"
echo "  🚀 LANCEMENT FINAL - SLIMPAJAMA LLAMA OPTIMAL"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "  Configuration:"
echo "    - Dataset:  SlimPajama-627B ✅"
echo "    - GPU:      1x RTX 4090"
echo "    - Ratio:    100 tokens/param (LLaMA style)"
echo "    - Steps:    757,500"
echo "    - Tokens:   12.4B"
echo "    - Durée:    ~70h (~3 jours)"
echo "    - HF Token: ✅ Configuré"
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo ""

cd /root/SWELU_LLM

# Charger le token
export $(cat .env | xargs)

# Arrêter tout
pkill -f train.py 2>/dev/null
sleep 2

# Backup ancien log
[ -f training.log ] && mv training.log training_backup_$(date +%Y%m%d_%H%M%S).log

echo "📝 Token HF chargé"
echo "🚀 Démarrage de l'entraînement..."
echo ""

# Lancer avec token HF
CUDA_VISIBLE_DEVICES=0 \
HF_TOKEN=$HF_TOKEN \
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
sleep 5

echo ""
echo "✅ Entraînement SlimPajama LLaMA lancé!"
echo ""
echo "🔍 PID: $PID"
echo ""
echo "📊 Status GPU:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | head -1 | \
    awk -F', ' '{printf "   GPU 0: Util=%s%%, Mem=%sMB\n", $2, $3}'
echo ""
echo "📝 Logs:"
echo "   tail -f training.log"
echo "   ./monitor_training.sh"
echo ""
echo "⏱️  Timeline:"
echo "   ~5min:  Métadonnées SlimPajama chargées"
echo "   ~30min: Premier checkpoint (step 5,000)"
echo "   ~70h:   Entraînement complet (757,500 steps)"
echo ""
echo "═══════════════════════════════════════════════════════════════════"


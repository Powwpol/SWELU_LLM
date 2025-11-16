#!/bin/bash
# Reprendre l'entraînement depuis les derniers checkpoints
# Checkpoints tous les 10k au lieu de 5k pour économiser l'espace

echo "═══════════════════════════════════════════════════════════════════"
echo "  🔄 REPRISE ENTRAÎNEMENT - OPTIMISÉ ESPACE DISQUE"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "  Changements:"
echo "    - Checkpoints: tous les 10,000 steps (au lieu de 5,000)"
echo "    - Cleanup automatique: garde seulement les 3 derniers"
echo "    - Reprise depuis derniers checkpoints disponibles"
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo ""

cd /root/SWELU_LLM

# Charger token HF
export $(cat .env | xargs 2>/dev/null)

# Arrêter tout
pkill -f train.py 2>/dev/null
sleep 3

# Trouver les derniers checkpoints
declare -A last_checkpoints
for gpu in {0..5}; do
    ckpt=$(ls -1t checkpoints/model_gpu$gpu/model_step_*.pt 2>/dev/null | head -1)
    if [ -n "$ckpt" ]; then
        last_checkpoints[$gpu]="$ckpt"
        step=$(basename $ckpt | sed 's/model_step_//' | sed 's/.pt//')
        echo "GPU $gpu: Reprise depuis step $step"
    else
        echo "GPU $gpu: Nouveau (pas de checkpoint)"
    fi
done

echo ""
echo "🚀 Relancement avec checkpoint_every=10000..."
echo ""

# GPU 0
if [ -n "${last_checkpoints[0]}" ]; then
    RESUME_0="--resume_from_checkpoint ${last_checkpoints[0]}"
else
    RESUME_0=""
fi

CUDA_VISIBLE_DEVICES=0 HF_TOKEN=$HF_TOKEN \
nohup python src/train.py \
  --dataset slimpajama --vocab_size 50257 --d_model 1024 --n_layers 6 \
  --max_seq_len 1024 --batch_size 4 --gradient_accumulation_steps 4 \
  --max_steps 757500 --learning_rate 3e-4 --weight_decay 0.1 \
  --warmup_steps 2000 --mixed_precision bf16 \
  --checkpoint_dir ./checkpoints/model_gpu0 \
  --checkpoint_every 10000 --log_every 100 \
  $RESUME_0 \
  > logs/gpu0.log 2>&1 &
echo "   GPU 0: PID $!"

# GPU 1
if [ -n "${last_checkpoints[1]}" ]; then
    RESUME_1="--resume_from_checkpoint ${last_checkpoints[1]}"
else
    RESUME_1=""
fi

CUDA_VISIBLE_DEVICES=1 HF_TOKEN=$HF_TOKEN \
nohup python src/train.py \
  --dataset slimpajama --vocab_size 50257 --d_model 1024 --n_layers 6 \
  --max_seq_len 1024 --batch_size 4 --gradient_accumulation_steps 4 \
  --max_steps 757500 --learning_rate 3e-4 --weight_decay 0.1 \
  --warmup_steps 2000 --mixed_precision bf16 \
  --checkpoint_dir ./checkpoints/model_gpu1 \
  --checkpoint_every 10000 --log_every 100 \
  $RESUME_1 \
  > logs/gpu1.log 2>&1 &
echo "   GPU 1: PID $!"

# GPU 2
if [ -n "${last_checkpoints[2]}" ]; then
    RESUME_2="--resume_from_checkpoint ${last_checkpoints[2]}"
else
    RESUME_2=""
fi

CUDA_VISIBLE_DEVICES=2 HF_TOKEN=$HF_TOKEN \
nohup python src/train.py \
  --dataset slimpajama --vocab_size 50257 --d_model 1024 --n_layers 6 \
  --max_seq_len 1024 --batch_size 4 --gradient_accumulation_steps 4 \
  --max_steps 757500 --learning_rate 3e-4 --weight_decay 0.1 \
  --warmup_steps 2000 --mixed_precision bf16 \
  --checkpoint_dir ./checkpoints/model_gpu2 \
  --checkpoint_every 10000 --log_every 100 \
  $RESUME_2 \
  > logs/gpu2.log 2>&1 &
echo "   GPU 2: PID $!"

# GPU 3
if [ -n "${last_checkpoints[3]}" ]; then
    RESUME_3="--resume_from_checkpoint ${last_checkpoints[3]}"
else
    RESUME_3=""
fi

CUDA_VISIBLE_DEVICES=3 HF_TOKEN=$HF_TOKEN \
nohup python src/train.py \
  --dataset slimpajama --vocab_size 50257 --d_model 1024 --n_layers 6 \
  --max_seq_len 1024 --batch_size 4 --gradient_accumulation_steps 4 \
  --max_steps 757500 --learning_rate 3e-4 --weight_decay 0.1 \
  --warmup_steps 2000 --mixed_precision bf16 \
  --checkpoint_dir ./checkpoints/model_gpu3 \
  --checkpoint_every 10000 --log_every 100 \
  $RESUME_3 \
  > logs/gpu3.log 2>&1 &
echo "   GPU 3: PID $!"

# GPU 4
if [ -n "${last_checkpoints[4]}" ]; then
    RESUME_4="--resume_from_checkpoint ${last_checkpoints[4]}"
else
    RESUME_4=""
fi

CUDA_VISIBLE_DEVICES=4 HF_TOKEN=$HF_TOKEN \
nohup python src/train.py \
  --dataset slimpajama --vocab_size 50257 --d_model 1024 --n_layers 6 \
  --max_seq_len 1024 --batch_size 4 --gradient_accumulation_steps 4 \
  --max_steps 757500 --learning_rate 3e-4 --weight_decay 0.1 \
  --warmup_steps 2000 --mixed_precision bf16 \
  --checkpoint_dir ./checkpoints/model_gpu4 \
  --checkpoint_every 10000 --log_every 100 \
  $RESUME_4 \
  > logs/gpu4.log 2>&1 &
echo "   GPU 4: PID $!"

# GPU 5
if [ -n "${last_checkpoints[5]}" ]; then
    RESUME_5="--resume_from_checkpoint ${last_checkpoints[5]}"
else
    RESUME_5=""
fi

CUDA_VISIBLE_DEVICES=5 HF_TOKEN=$HF_TOKEN \
nohup python src/train.py \
  --dataset slimpajama --vocab_size 50257 --d_model 1024 --n_layers 6 \
  --max_seq_len 1024 --batch_size 4 --gradient_accumulation_steps 4 \
  --max_steps 757500 --learning_rate 3e-4 --weight_decay 0.1 \
  --warmup_steps 2000 --mixed_precision bf16 \
  --checkpoint_dir ./checkpoints/model_gpu5 \
  --checkpoint_every 10000 --log_every 100 \
  $RESUME_5 \
  > logs/gpu5.log 2>&1 &
echo "   GPU 5: PID $!"

sleep 5

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "✅ 6 MODÈLES RELANCÉS!"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "📊 Reprise:"
for gpu in {0..5}; do
    if [ -n "${last_checkpoints[$gpu]}" ]; then
        step=$(basename ${last_checkpoints[$gpu]} | sed 's/model_step_//' | sed 's/.pt//')
        echo "   GPU $gpu: depuis step $step"
    else
        echo "   GPU $gpu: nouveau"
    fi
done
echo ""
echo "💾 Nouveaux checkpoints: tous les 10,000 steps"
echo "🧹 Auto-cleanup: garde les 3 derniers uniquement"
echo ""
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | \
    awk -F', ' '{printf "   GPU %s: Util=%3s%%, Mem=%5sMB\n", $1, $2, $3}'
echo ""
echo "📝 Logs: tail -f logs/gpu0.log"
echo "═══════════════════════════════════════════════════════════════════"


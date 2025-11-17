#!/bin/bash
# Reprendre depuis les checkpoints valides (pas corrompus)

echo "═══════════════════════════════════════════════════════════════════"
echo "  🔄 REPRISE DEPUIS CHECKPOINTS VALIDES"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

cd /root/SWELU_LLM
export $(cat .env | xargs 2>/dev/null)

pkill -f train.py 2>/dev/null
sleep 3

echo "🔍 Recherche des meilleurs checkpoints valides..."
echo ""

# Function pour tester un checkpoint
test_checkpoint() {
    python3 << EOF
import torch
import sys
try:
    torch.load("$1", map_location='cpu')
    sys.exit(0)
except:
    sys.exit(1)
EOF
    return $?
}

# Trouver le meilleur checkpoint pour chaque GPU
declare -A resume_from
declare -A resume_step

for gpu in {0..5}; do
    dir="checkpoints/model_gpu$gpu"
    
    # Chercher checkpoints par ordre décroissant
    for ckpt in $(ls -1t $dir/model_step_*.pt 2>/dev/null); do
        if test_checkpoint "$ckpt"; then
            resume_from[$gpu]="$ckpt"
            step=$(basename $ckpt | sed 's/model_step_//' | sed 's/.pt//')
            resume_step[$gpu]=$step
            echo "   GPU $gpu: ✅ Checkpoint step $step OK"
            break
        else
            echo "   GPU $gpu: ❌ $(basename $ckpt) corrompu, skip..."
        fi
    done
    
    if [ -z "${resume_from[$gpu]}" ]; then
        echo "   GPU $gpu: 🆕 Aucun checkpoint valide, nouveau démarrage"
    fi
done

echo ""
echo "🚀 Relancement avec checkpoint_every=10000..."
echo ""

# GPU 0
RESUME_ARG=""
if [ -n "${resume_from[0]}" ]; then
    RESUME_ARG="--resume_from_checkpoint ${resume_from[0]}"
fi

CUDA_VISIBLE_DEVICES=0 HF_TOKEN=$HF_TOKEN \
nohup python src/train.py \
  --dataset slimpajama --vocab_size 50257 --d_model 1024 --n_layers 6 \
  --max_seq_len 1024 --batch_size 4 --gradient_accumulation_steps 4 \
  --max_steps 757500 --learning_rate 3e-4 --weight_decay 0.1 \
  --warmup_steps 2000 --mixed_precision bf16 \
  --checkpoint_dir ./checkpoints/model_gpu0 \
  --checkpoint_every 10000 --log_every 100 \
  $RESUME_ARG \
  > logs/gpu0.log 2>&1 &
echo "GPU 0: PID $! (step ${resume_step[0]:-0})"

# GPU 1
RESUME_ARG=""
if [ -n "${resume_from[1]}" ]; then
    RESUME_ARG="--resume_from_checkpoint ${resume_from[1]}"
fi

CUDA_VISIBLE_DEVICES=1 HF_TOKEN=$HF_TOKEN \
nohup python src/train.py \
  --dataset slimpajama --vocab_size 50257 --d_model 1024 --n_layers 6 \
  --max_seq_len 1024 --batch_size 4 --gradient_accumulation_steps 4 \
  --max_steps 757500 --learning_rate 3e-4 --weight_decay 0.1 \
  --warmup_steps 2000 --mixed_precision bf16 \
  --checkpoint_dir ./checkpoints/model_gpu1 \
  --checkpoint_every 10000 --log_every 100 \
  $RESUME_ARG \
  > logs/gpu1.log 2>&1 &
echo "GPU 1: PID $! (step ${resume_step[1]:-0})"

# GPU 2
RESUME_ARG=""
if [ -n "${resume_from[2]}" ]; then
    RESUME_ARG="--resume_from_checkpoint ${resume_from[2]}"
fi

CUDA_VISIBLE_DEVICES=2 HF_TOKEN=$HF_TOKEN \
nohup python src/train.py \
  --dataset slimpajama --vocab_size 50257 --d_model 1024 --n_layers 6 \
  --max_seq_len 1024 --batch_size 4 --gradient_accumulation_steps 4 \
  --max_steps 757500 --learning_rate 3e-4 --weight_decay 0.1 \
  --warmup_steps 2000 --mixed_precision bf16 \
  --checkpoint_dir ./checkpoints/model_gpu2 \
  --checkpoint_every 10000 --log_every 100 \
  $RESUME_ARG \
  > logs/gpu2.log 2>&1 &
echo "GPU 2: PID $! (step ${resume_step[2]:-0})"

# GPU 3
RESUME_ARG=""
if [ -n "${resume_from[3]}" ]; then
    RESUME_ARG="--resume_from_checkpoint ${resume_from[3]}"
fi

CUDA_VISIBLE_DEVICES=3 HF_TOKEN=$HF_TOKEN \
nohup python src/train.py \
  --dataset slimpajama --vocab_size 50257 --d_model 1024 --n_layers 6 \
  --max_seq_len 1024 --batch_size 4 --gradient_accumulation_steps 4 \
  --max_steps 757500 --learning_rate 3e-4 --weight_decay 0.1 \
  --warmup_steps 2000 --mixed_precision bf16 \
  --checkpoint_dir ./checkpoints/model_gpu3 \
  --checkpoint_every 10000 --log_every 100 \
  $RESUME_ARG \
  > logs/gpu3.log 2>&1 &
echo "GPU 3: PID $! (step ${resume_step[3]:-0})"

# GPU 4
RESUME_ARG=""
if [ -n "${resume_from[4]}" ]; then
    RESUME_ARG="--resume_from_checkpoint ${resume_from[4]}"
fi

CUDA_VISIBLE_DEVICES=4 HF_TOKEN=$HF_TOKEN \
nohup python src/train.py \
  --dataset slimpajama --vocab_size 50257 --d_model 1024 --n_layers 6 \
  --max_seq_len 1024 --batch_size 4 --gradient_accumulation_steps 4 \
  --max_steps 757500 --learning_rate 3e-4 --weight_decay 0.1 \
  --warmup_steps 2000 --mixed_precision bf16 \
  --checkpoint_dir ./checkpoints/model_gpu4 \
  --checkpoint_every 10000 --log_every 100 \
  $RESUME_ARG \
  > logs/gpu4.log 2>&1 &
echo "GPU 4: PID $! (step ${resume_step[4]:-0})"

# GPU 5
RESUME_ARG=""
if [ -n "${resume_from[5]}" ]; then
    RESUME_ARG="--resume_from_checkpoint ${resume_from[5]}"
fi

CUDA_VISIBLE_DEVICES=5 HF_TOKEN=$HF_TOKEN \
nohup python src/train.py \
  --dataset slimpajama --vocab_size 50257 --d_model 1024 --n_layers 6 \
  --max_seq_len 1024 --batch_size 4 --gradient_accumulation_steps 4 \
  --max_steps 757500 --learning_rate 3e-4 --weight_decay 0.1 \
  --warmup_steps 2000 --mixed_precision bf16 \
  --checkpoint_dir ./checkpoints/model_gpu5 \
  --checkpoint_every 10000 --log_every 100 \
  $RESUME_ARG \
  > logs/gpu5.log 2>&1 &
echo "GPU 5: PID $! (step ${resume_step[5]:-0})"

sleep 5

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "✅ RELANCÉ AVEC OPTIMISATIONS!"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "📊 Configuration:"
echo "   - Checkpoints: tous les 10,000 steps"
echo "   - Auto-cleanup: garde 3 derniers uniquement"
echo "   - Espace max: ~3.6GB par GPU (au lieu de 99GB!)"
echo ""
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
    awk -F', ' '{printf "   GPU %s: %sMB\n", $1, $2}'
echo ""
echo "📝 Monitoring: ./show_all_losses.sh"
echo "═══════════════════════════════════════════════════════════════════"


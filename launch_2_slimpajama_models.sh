#!/bin/bash
# Lancer 2 modèles SlimPajama en parallèle
# GPU 0-2: SlimPajama LLaMA (757k steps - optimal)
# GPU 3-5: SlimPajama Chinchilla (151k steps - rapide)

echo "═══════════════════════════════════════════════════════════════════"
echo "  🚀 COMPARAISON SLIMPAJAMA: LLaMA vs Chinchilla"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "  📊 Modèle 1 - LLaMA Style (GPU 0-2)"
echo "     Ratio:    100 tokens/param (optimal moderne)"
echo "     Steps:    757,500"
echo "     Tokens:   12.4B"
echo "     Durée:    ~23h (3 GPUs)"
echo ""
echo "  📊 Modèle 2 - Chinchilla (GPU 3-5)"
echo "     Ratio:    20 tokens/param (minimum recommandé)"
echo "     Steps:    151,500"
echo "     Tokens:   2.48B"  
echo "     Durée:    ~5h (3 GPUs)"
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo ""

cd /root/SWELU_LLM

# Arrêter tout
pkill -f train.py 2>/dev/null
sleep 3

# Créer répertoires
mkdir -p checkpoints/slimpajama_llama
mkdir -p checkpoints/slimpajama_chinchilla  
mkdir -p logs

echo "🚀 Lancement des 2 modèles SlimPajama..."
echo ""

# ═══════════════════════════════════════════════════════════════════
# MODÈLE 1: SlimPajama LLaMA (GPU 0-2) - 3 GPUs
# ═══════════════════════════════════════════════════════════════════
echo "📊 Modèle 1: LLaMA (GPU 0-2)..."

CUDA_VISIBLE_DEVICES=0,1,2 \
nohup torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=3 \
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
    --checkpoint_dir ./checkpoints/slimpajama_llama \
    --checkpoint_every 5000 \
    --log_every 100 \
    > logs/slimpajama_llama.log 2>&1 &

PID_LLAMA=$!
echo "   PID: $PID_LLAMA"
sleep 3

# ═══════════════════════════════════════════════════════════════════
# MODÈLE 2: SlimPajama Chinchilla (GPU 3-5) - 3 GPUs
# ═══════════════════════════════════════════════════════════════════
echo "📊 Modèle 2: Chinchilla (GPU 3-5)..."

CUDA_VISIBLE_DEVICES=3,4,5 \
nohup torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=3 \
    src/train.py \
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
    --checkpoint_dir ./checkpoints/slimpajama_chinchilla \
    --checkpoint_every 5000 \
    --log_every 100 \
    > logs/slimpajama_chinchilla.log 2>&1 &

PID_CHINCHILLA=$!
echo "   PID: $PID_CHINCHILLA"
sleep 5

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "✅ 2 MODÈLES SLIMPAJAMA LANCÉS!"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "📊 GPUs:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | \
    awk -F', ' '{printf "   GPU %s: Util=%s%%, Mem=%sMB\n", $1, $2, $3}'
echo ""
echo "📝 Logs:"
echo "   tail -f logs/slimpajama_llama.log       # LLaMA (100x)"
echo "   tail -f logs/slimpajama_chinchilla.log  # Chinchilla (20x)"
echo ""
echo "📊 Monitoring:"
echo "   ./monitor_2_models.sh"
echo "   watch -n 10 './monitor_2_models.sh'"
echo ""
echo "🔍 PIDs:"
echo "   LLaMA:      $PID_LLAMA"
echo "   Chinchilla: $PID_CHINCHILLA"
echo ""
echo "═══════════════════════════════════════════════════════════════════"


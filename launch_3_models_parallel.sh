#!/bin/bash
# Lancer 3 entraînements en parallèle sur des GPUs séparés
# GPU 0-1: SlimPajama (LLaMA style - 757k steps)
# GPU 2-3: SlimPajama (Chinchilla - 151k steps)  
# GPU 4-5: Wikipedia (LLaMA style - 757k steps)

echo "═══════════════════════════════════════════════════════════════════"
echo "  🚀 LANCEMENT 3 MODÈLES EN PARALLÈLE - COMPARAISON DATASETS"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "  Stratégie de comparaison:"
echo ""
echo "  📊 Modèle 1 - SlimPajama LLaMA (GPU 0-1)"
echo "     Dataset:  SlimPajama-627B"
echo "     Config:   LLaMA style (100 tokens/param)"
echo "     Steps:    757,500"
echo "     Tokens:   12.4B"
echo "     Durée:    ~35h (2 GPUs)"
echo ""
echo "  📊 Modèle 2 - SlimPajama Chinchilla (GPU 2-3)"
echo "     Dataset:  SlimPajama-627B"
echo "     Config:   Chinchilla (20 tokens/param)"
echo "     Steps:    151,500"
echo "     Tokens:   2.48B"
echo "     Durée:    ~7h (2 GPUs)"
echo ""
echo "  📊 Modèle 3 - Wikipedia LLaMA (GPU 4-5)"
echo "     Dataset:  Wikipedia"
echo "     Config:   LLaMA style (100 tokens/param)"
echo "     Steps:    757,500"
echo "     Tokens:   12.4B"
echo "     Durée:    ~35h (2 GPUs)"
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo ""

cd /root/SWELU_LLM

# Arrêter tout entraînement existant
pkill -f train.py 2>/dev/null
pkill -f torchrun 2>/dev/null
sleep 3

# Créer répertoires pour chaque modèle
mkdir -p checkpoints/slimpajama_llama
mkdir -p checkpoints/slimpajama_chinchilla
mkdir -p checkpoints/wikipedia_llama
mkdir -p logs

echo "📝 Démarrage des 3 entraînements..."
echo ""

# ═══════════════════════════════════════════════════════════════════
# MODÈLE 1: SlimPajama LLaMA (GPU 0-1)
# ═══════════════════════════════════════════════════════════════════
echo "🚀 Lancement Modèle 1: SlimPajama LLaMA (GPU 0-1)..."

CUDA_VISIBLE_DEVICES=0,1 \
nohup torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
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
    --eval_every 1000 \
    > logs/slimpajama_llama.log 2>&1 &

SLIMPAJAMA_LLAMA_PID=$!
echo "   PID: $SLIMPAJAMA_LLAMA_PID"
sleep 2

# ═══════════════════════════════════════════════════════════════════
# MODÈLE 2: SlimPajama Chinchilla (GPU 2-3)
# ═══════════════════════════════════════════════════════════════════
echo "🚀 Lancement Modèle 2: SlimPajama Chinchilla (GPU 2-3)..."

CUDA_VISIBLE_DEVICES=2,3 \
nohup torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
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
    --eval_every 1000 \
    > logs/slimpajama_chinchilla.log 2>&1 &

SLIMPAJAMA_CHINCHILLA_PID=$!
echo "   PID: $SLIMPAJAMA_CHINCHILLA_PID"
sleep 2

# ═══════════════════════════════════════════════════════════════════
# MODÈLE 3: Wikipedia LLaMA (GPU 4-5)
# ═══════════════════════════════════════════════════════════════════
echo "🚀 Lancement Modèle 3: Wikipedia LLaMA (GPU 4-5)..."

CUDA_VISIBLE_DEVICES=4,5 \
nohup torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    src/train.py \
    --dataset wikipedia \
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
    --checkpoint_dir ./checkpoints/wikipedia_llama \
    --checkpoint_every 5000 \
    --log_every 100 \
    --eval_every 1000 \
    > logs/wikipedia_llama.log 2>&1 &

WIKIPEDIA_LLAMA_PID=$!
echo "   PID: $WIKIPEDIA_LLAMA_PID"
sleep 5

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "✅ 3 ENTRAÎNEMENTS LANCÉS EN PARALLÈLE!"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "📊 Vérification GPUs:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | \
    awk -F', ' '{printf "   GPU %s: Util=%s%%, Mem=%sMB\n", $1, $2, $3}'
echo ""
echo "📝 Logs:"
echo "   tail -f logs/slimpajama_llama.log"
echo "   tail -f logs/slimpajama_chinchilla.log"
echo "   tail -f logs/wikipedia_llama.log"
echo ""
echo "📊 Monitoring global:"
echo "   watch -n 10 'nvidia-smi'"
echo "   ./monitor_all_trainings.sh"
echo ""
echo "🔍 PIDs:"
echo "   SlimPajama LLaMA:      $SLIMPAJAMA_LLAMA_PID"
echo "   SlimPajama Chinchilla: $SLIMPAJAMA_CHINCHILLA_PID"
echo "   Wikipedia LLaMA:       $WIKIPEDIA_LLAMA_PID"
echo ""
echo "═══════════════════════════════════════════════════════════════════"


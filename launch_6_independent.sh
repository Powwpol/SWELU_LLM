#!/bin/bash
# Lance 6 entraînements SlimPajama INDÉPENDANTS sur 6 GPUs
# Chaque GPU entraîne son propre modèle → 6 modèles différents à comparer!

echo "═══════════════════════════════════════════════════════════════════"
echo "  🚀 6 MODÈLES SLIMPAJAMA INDÉPENDANTS - 1 PAR GPU"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "  Stratégie: 6 GPUs en parallèle, 6 modèles indépendants"
echo ""
echo "  Avantages:"
echo "    ✅ Pas de complications DDP"
echo "    ✅ 6 modèles à comparer (variations aléatoires)"
echo "    ✅ Robustesse: si 1 crash, les autres continuent"
echo "    ✅ FlexibilGPUité: différents hyperparamètres possibles"
echo ""
echo "  Configuration par modèle:"
echo "    - Steps:   757,500 (LLaMA 100x)"
echo "    - Durée:   ~70h par GPU"
echo "    - Tokens:  12.4B chacun"
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo ""

cd /root/SWELU_LLM

# Charger token
export $(cat .env | xargs)

# Arrêter tout
pkill -f train.py 2>/dev/null
sleep 3

# Créer répertoires
for i in {0..5}; do
    mkdir -p checkpoints/model_gpu$i
    mkdir -p logs
done

echo "🚀 Lancement des 6 modèles..."
echo ""

# GPU 0
CUDA_VISIBLE_DEVICES=0 HF_TOKEN=$HF_TOKEN \
nohup python src/train.py \
  --dataset slimpajama --vocab_size 50257 --d_model 1024 --n_layers 6 \
  --max_seq_len 1024 --batch_size 4 --gradient_accumulation_steps 4 \
  --max_steps 757500 --learning_rate 3e-4 --weight_decay 0.1 \
  --warmup_steps 2000 --mixed_precision bf16 \
  --checkpoint_dir ./checkpoints/model_gpu0 --checkpoint_every 5000 --log_every 100 \
  > logs/gpu0.log 2>&1 &
echo "   GPU 0: PID $! → checkpoints/model_gpu0/"

# GPU 1  
CUDA_VISIBLE_DEVICES=1 HF_TOKEN=$HF_TOKEN \
nohup python src/train.py \
  --dataset slimpajama --vocab_size 50257 --d_model 1024 --n_layers 6 \
  --max_seq_len 1024 --batch_size 4 --gradient_accumulation_steps 4 \
  --max_steps 757500 --learning_rate 3e-4 --weight_decay 0.1 \
  --warmup_steps 2000 --mixed_precision bf16 \
  --checkpoint_dir ./checkpoints/model_gpu1 --checkpoint_every 5000 --log_every 100 \
  > logs/gpu1.log 2>&1 &
echo "   GPU 1: PID $! → checkpoints/model_gpu1/"

# GPU 2
CUDA_VISIBLE_DEVICES=2 HF_TOKEN=$HF_TOKEN \
nohup python src/train.py \
  --dataset slimpajama --vocab_size 50257 --d_model 1024 --n_layers 6 \
  --max_seq_len 1024 --batch_size 4 --gradient_accumulation_steps 4 \
  --max_steps 757500 --learning_rate 3e-4 --weight_decay 0.1 \
  --warmup_steps 2000 --mixed_precision bf16 \
  --checkpoint_dir ./checkpoints/model_gpu2 --checkpoint_every 5000 --log_every 100 \
  > logs/gpu2.log 2>&1 &
echo "   GPU 2: PID $! → checkpoints/model_gpu2/"

# GPU 3
CUDA_VISIBLE_DEVICES=3 HF_TOKEN=$HF_TOKEN \
nohup python src/train.py \
  --dataset slimpajama --vocab_size 50257 --d_model 1024 --n_layers 6 \
  --max_seq_len 1024 --batch_size 4 --gradient_accumulation_steps 4 \
  --max_steps 757500 --learning_rate 3e-4 --weight_decay 0.1 \
  --warmup_steps 2000 --mixed_precision bf16 \
  --checkpoint_dir ./checkpoints/model_gpu3 --checkpoint_every 5000 --log_every 100 \
  > logs/gpu3.log 2>&1 &
echo "   GPU 3: PID $! → checkpoints/model_gpu3/"

# GPU 4
CUDA_VISIBLE_DEVICES=4 HF_TOKEN=$HF_TOKEN \
nohup python src/train.py \
  --dataset slimpajama --vocab_size 50257 --d_model 1024 --n_layers 6 \
  --max_seq_len 1024 --batch_size 4 --gradient_accumulation_steps 4 \
  --max_steps 757500 --learning_rate 3e-4 --weight_decay 0.1 \
  --warmup_steps 2000 --mixed_precision bf16 \
  --checkpoint_dir ./checkpoints/model_gpu4 --checkpoint_every 5000 --log_every 100 \
  > logs/gpu4.log 2>&1 &
echo "   GPU 4: PID $! → checkpoints/model_gpu4/"

# GPU 5
CUDA_VISIBLE_DEVICES=5 HF_TOKEN=$HF_TOKEN \
nohup python src/train.py \
  --dataset slimpajama --vocab_size 50257 --d_model 1024 --n_layers 6 \
  --max_seq_len 1024 --batch_size 4 --gradient_accumulation_steps 4 \
  --max_steps 757500 --learning_rate 3e-4 --weight_decay 0.1 \
  --warmup_steps 2000 --mixed_precision bf16 \
  --checkpoint_dir ./checkpoints/model_gpu5 --checkpoint_every 5000 --log_every 100 \
  > logs/gpu5.log 2>&1 &
echo "   GPU 5: PID $! → checkpoints/model_gpu5/"

sleep 5

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "✅ 6 MODÈLES LANCÉS - 1 PAR GPU!"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | \
    awk -F', ' '{printf "   GPU %s: Util=%3s%%, Mem=%5sMB\n", $1, $2, $3}'
echo ""
echo "📝 Logs individuels:"
echo "   tail -f logs/gpu0.log"
echo "   tail -f logs/gpu1.log"
echo "   ... gpu2, gpu3, gpu4, gpu5 ..."
echo ""
echo "📊 Monitoring global:"
echo "   watch -n 10 'nvidia-smi'"
echo ""
echo "💾 6 sets de checkpoints:"
echo "   checkpoints/model_gpu0/"
echo "   checkpoints/model_gpu1/"
echo "   ... jusqu'à model_gpu5/"
echo ""
echo "═══════════════════════════════════════════════════════════════════"


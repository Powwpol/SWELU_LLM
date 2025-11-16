#!/bin/bash
# Reprendre depuis checkpoint 410k pour GPU 0,3,4,5 et nouveau pour GPU 1,2

echo "═══════════════════════════════════════════════════════════════════"
echo "  🔄 REPRISE OPTIMISÉE"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "  GPU 0,3,4,5: Reprise depuis step 410,000"
echo "  GPU 1,2:     Nouveau démarrage (step 0)"
echo ""
echo "  Config: checkpoint_every=10000 + auto-cleanup (3 derniers)"
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo ""

cd /root/SWELU_LLM
export $(cat .env | xargs 2>/dev/null)

# GPU 0 - Reprise 410k
CUDA_VISIBLE_DEVICES=0 HF_TOKEN=$HF_TOKEN \
nohup python src/train.py \
  --dataset slimpajama --vocab_size 50257 --d_model 1024 --n_layers 6 \
  --max_seq_len 1024 --batch_size 4 --gradient_accumulation_steps 4 \
  --max_steps 757500 --learning_rate 3e-4 --weight_decay 0.1 \
  --warmup_steps 2000 --mixed_precision bf16 \
  --checkpoint_dir ./checkpoints/model_gpu0 \
  --checkpoint_every 10000 --log_every 100 \
  --resume_from_checkpoint ./checkpoints/model_gpu0/model_step_410000.pt \
  > logs/gpu0.log 2>&1 &
echo "GPU 0: PID $! (reprise 410k)"

# GPU 1 - Nouveau
CUDA_VISIBLE_DEVICES=1 HF_TOKEN=$HF_TOKEN \
nohup python src/train.py \
  --dataset slimpajama --vocab_size 50257 --d_model 1024 --n_layers 6 \
  --max_seq_len 1024 --batch_size 4 --gradient_accumulation_steps 4 \
  --max_steps 757500 --learning_rate 3e-4 --weight_decay 0.1 \
  --warmup_steps 2000 --mixed_precision bf16 \
  --checkpoint_dir ./checkpoints/model_gpu1 \
  --checkpoint_every 10000 --log_every 100 \
  > logs/gpu1.log 2>&1 &
echo "GPU 1: PID $! (nouveau)"

# GPU 2 - Nouveau
CUDA_VISIBLE_DEVICES=2 HF_TOKEN=$HF_TOKEN \
nohup python src/train.py \
  --dataset slimpajama --vocab_size 50257 --d_model 1024 --n_layers 6 \
  --max_seq_len 1024 --batch_size 4 --gradient_accumulation_steps 4 \
  --max_steps 757500 --learning_rate 3e-4 --weight_decay 0.1 \
  --warmup_steps 2000 --mixed_precision bf16 \
  --checkpoint_dir ./checkpoints/model_gpu2 \
  --checkpoint_every 10000 --log_every 100 \
  > logs/gpu2.log 2>&1 &
echo "GPU 2: PID $! (nouveau)"

# GPU 3 - Reprise 410k
CUDA_VISIBLE_DEVICES=3 HF_TOKEN=$HF_TOKEN \
nohup python src/train.py \
  --dataset slimpajama --vocab_size 50257 --d_model 1024 --n_layers 6 \
  --max_seq_len 1024 --batch_size 4 --gradient_accumulation_steps 4 \
  --max_steps 757500 --learning_rate 3e-4 --weight_decay 0.1 \
  --warmup_steps 2000 --mixed_precision bf16 \
  --checkpoint_dir ./checkpoints/model_gpu3 \
  --checkpoint_every 10000 --log_every 100 \
  --resume_from_checkpoint ./checkpoints/model_gpu3/model_step_410000.pt \
  > logs/gpu3.log 2>&1 &
echo "GPU 3: PID $! (reprise 410k)"

# GPU 4 - Reprise 410k
CUDA_VISIBLE_DEVICES=4 HF_TOKEN=$HF_TOKEN \
nohup python src/train.py \
  --dataset slimpajama --vocab_size 50257 --d_model 1024 --n_layers 6 \
  --max_seq_len 1024 --batch_size 4 --gradient_accumulation_steps 4 \
  --max_steps 757500 --learning_rate 3e-4 --weight_decay 0.1 \
  --warmup_steps 2000 --mixed_precision bf16 \
  --checkpoint_dir ./checkpoints/model_gpu4 \
  --checkpoint_every 10000 --log_every 100 \
  --resume_from_checkpoint ./checkpoints/model_gpu4/model_step_410000.pt \
  > logs/gpu4.log 2>&1 &
echo "GPU 4: PID $! (reprise 410k)"

# GPU 5 - Reprise 410k
CUDA_VISIBLE_DEVICES=5 HF_TOKEN=$HF_TOKEN \
nohup python src/train.py \
  --dataset slimpajama --vocab_size 50257 --d_model 1024 --n_layers 6 \
  --max_seq_len 1024 --batch_size 4 --gradient_accumulation_steps 4 \
  --max_steps 757500 --learning_rate 3e-4 --weight_decay 0.1 \
  --warmup_steps 2000 --mixed_precision bf16 \
  --checkpoint_dir ./checkpoints/model_gpu5 \
  --checkpoint_every 10000 --log_every 100 \
  --resume_from_checkpoint ./checkpoints/model_gpu5/model_step_410000.pt \
  > logs/gpu5.log 2>&1 &
echo "GPU 5: PID $! (reprise 410k)"

sleep 5

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "✅ 6 MODÈLES RELANCÉS!"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "📊 Configuration:"
echo "   GPU 0: Reprise step 410,000 → 757,500 (347,500 steps restants)"
echo "   GPU 1: Nouveau step 0 → 757,500"
echo "   GPU 2: Nouveau step 0 → 757,500"
echo "   GPU 3: Reprise step 410,000 → 757,500 (347,500 steps restants)"
echo "   GPU 4: Reprise step 410,000 → 757,500 (347,500 steps restants)"
echo "   GPU 5: Reprise step 410,000 → 757,500 (347,500 steps restants)"
echo ""
echo "💾 Checkpoints: tous les 10,000 steps"
echo "🧹 Auto-cleanup: garde 3 derniers"
echo ""
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | \
    awk -F', ' '{printf "   GPU %s: Util=%3s%%, Mem=%5sMB\n", $1, $2, $3}'
echo ""
echo "📝 Monitoring: ./show_all_losses.sh"
echo "═══════════════════════════════════════════════════════════════════"






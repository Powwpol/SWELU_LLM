#!/bin/bash
# GPU 0,3,4,5: Reprendre depuis step 410k
# GPU 1,2: Démarrer de zéro

echo "═══════════════════════════════════════════════════════════════════"
echo "  🚀 REPRISE MIXTE: 4 depuis 410k + 2 nouveaux"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

cd /root/SWELU_LLM
export $(cat .env | xargs 2>/dev/null)

echo "📊 Configuration:"
echo "   GPU 0: Reprise step 410,000"
echo "   GPU 1: Nouveau (step 0)"
echo "   GPU 2: Nouveau (step 0)"
echo "   GPU 3: Reprise step 410,000"
echo "   GPU 4: Reprise step 410,000"
echo "   GPU 5: Reprise step 410,000"
echo ""
echo "💾 Checkpoints: tous les 10,000 steps (auto-cleanup)"
echo ""
echo "🚀 Lancement..."
echo ""

# GPU 0 - REPRISE 410k
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
echo "✅ GPU 0: PID $! (reprise step 410k)"

# GPU 1 - NOUVEAU
CUDA_VISIBLE_DEVICES=1 HF_TOKEN=$HF_TOKEN \
nohup python src/train.py \
  --dataset slimpajama --vocab_size 50257 --d_model 1024 --n_layers 6 \
  --max_seq_len 1024 --batch_size 4 --gradient_accumulation_steps 4 \
  --max_steps 757500 --learning_rate 3e-4 --weight_decay 0.1 \
  --warmup_steps 2000 --mixed_precision bf16 \
  --checkpoint_dir ./checkpoints/model_gpu1 \
  --checkpoint_every 10000 --log_every 100 \
  > logs/gpu1.log 2>&1 &
echo "✅ GPU 1: PID $! (nouveau)"

# GPU 2 - NOUVEAU
CUDA_VISIBLE_DEVICES=2 HF_TOKEN=$HF_TOKEN \
nohup python src/train.py \
  --dataset slimpajama --vocab_size 50257 --d_model 1024 --n_layers 6 \
  --max_seq_len 1024 --batch_size 4 --gradient_accumulation_steps 4 \
  --max_steps 757500 --learning_rate 3e-4 --weight_decay 0.1 \
  --warmup_steps 2000 --mixed_precision bf16 \
  --checkpoint_dir ./checkpoints/model_gpu2 \
  --checkpoint_every 10000 --log_every 100 \
  > logs/gpu2.log 2>&1 &
echo "✅ GPU 2: PID $! (nouveau)"

# GPU 3 - REPRISE 410k
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
echo "✅ GPU 3: PID $! (reprise step 410k)"

# GPU 4 - REPRISE 410k
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
echo "✅ GPU 4: PID $! (reprise step 410k)"

# GPU 5 - REPRISE 410k
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
echo "✅ GPU 5: PID $! (reprise step 410k)"

sleep 5

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "✅ RELANCÉ EN MODE MIXTE!"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "📊 Reprise:"
echo "   GPU 0,3,4,5: Depuis step 410,000 (54% complété!)"
echo "   GPU 1,2:     Nouveaux (step 0)"
echo ""
echo "⏱️  Temps restant:"
echo "   GPU 0,3,4,5: ~21h (347,500 steps restants)"
echo "   GPU 1,2:     ~48h (757,500 steps)"
echo ""
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | \
    awk -F', ' '{printf "   GPU %s: Util=%3s%%, Mem=%5sMB\n", $1, $2, $3}'
echo ""
echo "📝 Monitoring: ./show_all_losses.sh"
echo "═══════════════════════════════════════════════════════════════════"

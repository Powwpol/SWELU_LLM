#!/bin/bash
# Script de lancement training sur RunPod
# Lance en background avec nohup

set -e

echo "======================================"
echo "Starting RunPod Training"
echo "======================================"

# Load environment variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "Warning: .env file not found"
fi

# Check WANDB_API_KEY
if [ -z "$WANDB_API_KEY" ]; then
    echo "⚠️  Warning: WANDB_API_KEY not set"
    echo "   Training will continue but without W&B logging"
fi

# Create logs directory
mkdir -p logs

# Get current timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="logs/training_${TIMESTAMP}.log"

# Launch training in background
echo "Launching training..."
echo "Log file: $LOGFILE"

nohup python src/train.py \
    --vocab_size 50257 \
    --d_model 1024 \
    --n_layers 6 \
    --max_seq_len 2048 \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 3e-4 \
    --weight_decay 0.1 \
    --warmup_steps 2000 \
    --max_steps 100000 \
    --mixed_precision bf16 \
    --dataset wikipedia \
    --checkpoint_dir ./checkpoints \
    --checkpoint_every 5000 \
    --log_every 100 \
    --eval_every 1000 \
    --use_wandb \
    > "$LOGFILE" 2>&1 &

TRAIN_PID=$!
echo "Training started with PID: $TRAIN_PID"
echo "Process ID saved to logs/train.pid"
echo $TRAIN_PID > logs/train.pid

echo ""
echo "======================================"
echo "Training launched successfully!"
echo "======================================"
echo ""
echo "Useful commands:"
echo "  Monitor training:    tail -f $LOGFILE"
echo "  Check GPU usage:     watch -n 1 nvidia-smi"
echo "  Stop training:       kill $TRAIN_PID"
echo "  Resume monitoring:   tail -f $LOGFILE"
echo ""
echo "Checkpoints will be saved to: ./checkpoints"
echo "======================================"

# Show initial output
sleep 2
echo ""
echo "Initial output:"
tail -n 20 "$LOGFILE"


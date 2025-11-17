#!/bin/bash
# Lancement du fine-tuning sur 6x RTX 4090
# Option 2 - Grande Capacité

set -e  # Exit on error

echo "════════════════════════════════════════════════════════════════════"
echo "  🔥 FINE-TUNING MambaSWELU - Option 2 (Grande Capacité)"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "  📊 Configuration:"
echo "     GPUs:                     6x RTX 4090"
echo "     Batch size par GPU:       4"
echo "     Gradient accumulation:    8"
echo "     Effective batch size:     192 (4 × 6 × 8)"
echo "     Learning rate:            5e-6"
echo "     Max steps:                25,000"
echo "     Durée estimée:            ~20h"
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo ""

cd /root/SWELU_LLM

# Vérifier que les données sont prêtes
if [ ! -f "data/instruction/train.jsonl" ]; then
    echo "⚠️  Données d'instruction non trouvées!"
    echo ""
    echo "🔧 Préparation des datasets en cours..."
    python prepare_instruction_data.py
    echo ""
fi

# Vérifier le checkpoint de base
BASE_CHECKPOINT="checkpoints/model_gpu5/final_model.pt"
if [ ! -f "$BASE_CHECKPOINT" ]; then
    echo "❌ Checkpoint de base introuvable: $BASE_CHECKPOINT"
    echo ""
    echo "Checkpoints disponibles:"
    find checkpoints -name "*.pt" -type f | head -10
    exit 1
fi

echo "✓ Checkpoint de base: $BASE_CHECKPOINT"
echo ""

# Créer le répertoire de sortie
mkdir -p checkpoints/finetuned
mkdir -p logs/finetune

# Configuration
TRAIN_FILE="data/instruction/train.jsonl"
VAL_FILE="data/instruction/val.jsonl"
CHECKPOINT_DIR="checkpoints/finetuned"
LOG_FILE="logs/finetune/finetune_$(date +%Y%m%d_%H%M%S).log"

echo "📂 Configuration:"
echo "   Train file:    $TRAIN_FILE"
echo "   Val file:      $VAL_FILE"
echo "   Checkpoint:    $BASE_CHECKPOINT"
echo "   Output dir:    $CHECKPOINT_DIR"
echo "   Log file:      $LOG_FILE"
echo ""

echo "════════════════════════════════════════════════════════════════════"
echo "  🚀 LANCEMENT DU FINE-TUNING"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Lancer avec torchrun (multi-GPU)
torchrun \
    --nproc_per_node=6 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="localhost" \
    --master_port=29500 \
    finetune.py \
    --train_file "$TRAIN_FILE" \
    --val_file "$VAL_FILE" \
    --checkpoint "$BASE_CHECKPOINT" \
    --batch_size 4 \
    --learning_rate 5e-6 \
    --weight_decay 0.05 \
    --warmup_steps 1000 \
    --max_steps 25000 \
    --gradient_accumulation_steps 8 \
    --max_grad_norm 1.0 \
    --max_length 1024 \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --checkpoint_every 1000 \
    --log_every 50 \
    --eval_every 2000 \
    --mixed_precision \
    --num_workers 4 \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  ✅ FINE-TUNING TERMINÉ"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "📁 Modèle fine-tuné sauvegardé:"
echo "   $CHECKPOINT_DIR/finetuned_model.pt"
echo ""
echo "📊 Log sauvegardé:"
echo "   $LOG_FILE"
echo ""
echo "🧪 Pour tester le modèle fine-tuné:"
echo "   python demo_chat.py --checkpoint $CHECKPOINT_DIR/finetuned_model.pt"
echo ""
echo "🔬 Pour comparer avec le modèle de base:"
echo "   python compare_models.py --base_model $BASE_CHECKPOINT --finetuned_model $CHECKPOINT_DIR/finetuned_model.pt"
echo ""


#!/bin/bash
# Test rapide du fine-tuning sur 1 GPU
# Validation avant de lancer les 6 GPUs

set -e

echo "════════════════════════════════════════════════════════════════════"
echo "  🧪 TEST FINE-TUNING (1 GPU) - Validation Rapide"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "  📊 Configuration de test:"
echo "     GPUs:                     1"
echo "     Batch size:               2"
echo "     Gradient accumulation:    4"
echo "     Max steps:                500 (test rapide)"
echo "     Durée estimée:            ~15 minutes"
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo ""

cd /root/SWELU_LLM

# Préparer les données si nécessaire
if [ ! -f "data/instruction/train.jsonl" ]; then
    echo "🔧 Préparation des datasets..."
    python prepare_instruction_data.py --max_samples 10000
    echo ""
fi

BASE_CHECKPOINT="checkpoints/model_gpu5/final_model.pt"

echo "✓ Lancement du test..."
echo ""

# Test sur 1 GPU seulement
CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --train_file data/instruction/train.jsonl \
    --val_file data/instruction/val.jsonl \
    --checkpoint "$BASE_CHECKPOINT" \
    --batch_size 2 \
    --learning_rate 5e-6 \
    --weight_decay 0.05 \
    --warmup_steps 100 \
    --max_steps 500 \
    --gradient_accumulation_steps 4 \
    --max_length 512 \
    --checkpoint_dir checkpoints/test_finetune \
    --checkpoint_every 250 \
    --log_every 10 \
    --eval_every 100 \
    --mixed_precision \
    --num_workers 2

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  ✅ TEST TERMINÉ"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "📊 Vérifications:"
echo "   1. La loss a-t-elle diminué?"
echo "   2. Y a-t-il eu des erreurs GPU/mémoire?"
echo "   3. Le checkpoint a-t-il été sauvegardé?"
echo ""
echo "🧪 Tester le modèle:"
echo "   python demo_chat.py --checkpoint checkpoints/test_finetune/checkpoint_step_500.pt"
echo ""
echo "✅ Si tout fonctionne, lancez le vrai fine-tuning:"
echo "   ./launch_finetune_6gpu.sh"
echo ""


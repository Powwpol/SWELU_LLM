#!/bin/bash
# Configuration optimale pour MambaSWELU 124M
# Basée sur les lois d'échelle Chinchilla (20 tokens/param minimum)

echo "═══════════════════════════════════════════════════════════════════"
echo "  CONFIGURATIONS D'ENTRAÎNEMENT RECOMMANDÉES"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "Modèle: MambaSWELU 124M paramètres"
echo "Dataset: SlimPajama-627B"
echo ""

# Option 1: Chinchilla optimal (minimum recommandé)
cat << 'OPTION1'
─────────────────────────────────────────────────────────────────────
OPTION 1: CHINCHILLA OPTIMAL (Recommandé pour production)
─────────────────────────────────────────────────────────────────────
  Ratio: 20 tokens/param
  Total tokens: 2.48B
  Steps: ~150,000
  Durée estimée: ~14h (RTX 4090)
  
  Commande:
  python src/train.py \
    --dataset slimpajama \
    --vocab_size 50257 \
    --d_model 1024 \
    --n_layers 6 \
    --max_seq_len 1024 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_steps 151500 \
    --learning_rate 3e-4 \
    --mixed_precision bf16 \
    --checkpoint_dir ./checkpoints \
    --checkpoint_every 5000

OPTION1

echo ""

# Option 2: LLaMA style (optimal)
cat << 'OPTION2'
─────────────────────────────────────────────────────────────────────
OPTION 2: LLAMA STYLE (Optimal pour meilleures performances)
─────────────────────────────────────────────────────────────────────
  Ratio: 100 tokens/param
  Total tokens: 12.4B
  Steps: ~757,500
  Durée estimée: ~3 jours (RTX 4090)
  
  Commande:
  python src/train.py \
    --dataset slimpajama \
    --vocab_size 50257 \
    --d_model 1024 \
    --n_layers 6 \
    --max_seq_len 1024 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_steps 757500 \
    --learning_rate 3e-4 \
    --mixed_precision bf16 \
    --checkpoint_dir ./checkpoints \
    --checkpoint_every 5000

OPTION2

echo ""

# Option 3: Batch size augmenté (plus rapide)
cat << 'OPTION3'
─────────────────────────────────────────────────────────────────────
OPTION 3: BATCH SIZE AUGMENTÉ (Même tokens, moins de steps)
─────────────────────────────────────────────────────────────────────
  Ratio: 20 tokens/param
  Total tokens: 2.48B
  Steps: ~75,750 (2x moins grâce au batch size)
  Durée estimée: ~7h (RTX 4090)
  
  Commande:
  python src/train.py \
    --dataset slimpajama \
    --vocab_size 50257 \
    --d_model 1024 \
    --n_layers 6 \
    --max_seq_len 1024 \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --max_steps 75750 \
    --learning_rate 3e-4 \
    --mixed_precision bf16 \
    --checkpoint_dir ./checkpoints \
    --checkpoint_every 5000

OPTION3

echo ""

# Option 4: Séquences plus longues
cat << 'OPTION4'
─────────────────────────────────────────────────────────────────────
OPTION 4: SÉQUENCES LONGUES (Meilleur contexte)
─────────────────────────────────────────────────────────────────────
  Ratio: 20 tokens/param
  Total tokens: 2.48B
  Steps: ~75,750
  Seq length: 2048 (2x plus de contexte)
  Durée estimée: ~10h (RTX 4090)
  
  Commande:
  python src/train.py \
    --dataset slimpajama \
    --vocab_size 50257 \
    --d_model 1024 \
    --n_layers 6 \
    --max_seq_len 2048 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_steps 75750 \
    --learning_rate 3e-4 \
    --mixed_precision bf16 \
    --checkpoint_dir ./checkpoints \
    --checkpoint_every 5000

OPTION4

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "💡 Recommandation:"
echo "   - Développement/Test: Option 1 (Chinchilla minimal)"
echo "   - Production: Option 2 (LLaMA style)"
echo "   - Contrainte GPU: Option 3 (batch augmenté)"
echo "   - Tâches longue portée: Option 4 (seq 2048)"
echo "═══════════════════════════════════════════════════════════════════"


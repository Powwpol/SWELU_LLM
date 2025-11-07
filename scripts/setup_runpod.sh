#!/bin/bash
# Script d'installation automatique pour RunPod
# À exécuter une fois l'instance créée

set -e

echo "======================================"
echo "RunPod Setup - MambaSWELU"
echo "======================================"

# Update system
echo "[1/6] Updating system packages..."
apt-get update
apt-get install -y git wget curl htop nvtop

# Install Python dependencies
echo "[2/6] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Test GPU
echo "[3/6] Testing GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

if ! python -c "import torch; assert torch.cuda.is_available()"; then
    echo "ERROR: No GPU detected!"
    exit 1
fi

# Setup environment variables
echo "[4/6] Setting up environment variables..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "⚠️  Please edit .env file with your API keys!"
    echo "   Especially WANDB_API_KEY for monitoring"
fi

# Create necessary directories
echo "[5/6] Creating directories..."
mkdir -p checkpoints
mkdir -p logs
mkdir -p data/processed

# Test data loading
echo "[6/6] Testing data loading..."
python -c "from src.data_prep import WikipediaDataset; print('✓ Data loading works')"

echo ""
echo "======================================"
echo "✓ Setup complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "  1. Edit .env file with your API keys"
echo "  2. Start training: bash scripts/train_runpod.sh"
echo "  3. Monitor with: tail -f training.log"
echo "  4. Check GPU usage: watch -n 1 nvidia-smi"
echo ""
echo "Estimated cost: ~0.40$/hour (RTX 4090)"
echo "Full training (100k steps): ~16$ total"
echo "======================================"


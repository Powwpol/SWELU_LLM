#!/bin/bash
# Script pour lancer l'entraînement du petit modèle (Linux/Mac)

set -e

echo "======================================"
echo "MambaSWELU - Small Model Training"
echo "======================================"

# Check if config exists
if [ ! -f "configs/small_model.yaml" ]; then
    echo "Error: configs/small_model.yaml not found"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Warning: No virtual environment found"
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Run training
python scripts/train_small.py --config configs/small_model.yaml

echo ""
echo "======================================"
echo "Training completed!"
echo "======================================"


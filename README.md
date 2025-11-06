# SWELU-LLM: Mamba Architecture with SWELU Activation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Overview

Custom LLM based on **Mamba SSM architecture** with **SWELU (Smooth Weighted Exponential Linear Unit)** activation function.

### Key Features

- **Architecture**: 6 Mamba blocks + 3 Dense layers
- **Parameters**: 350M (optimized for RTX 4090)
- **Activation**: SWELU with learnable parameter `k`
- **Context Length**: 2048 tokens
- **Training**: Mixed precision (FP16/BF16) on RunPod

## Architecture

```
Input (2048 tokens)
  ↓
6× Mamba Blocks (d_model=1024, d_state=16, d_conv=4)
  ↓
Dense Layer 1 (1024 → 2048) + SWELU
  ↓
Dense Layer 2 (2048 → 2048) + SWELU  
  ↓
Dense Layer 3 (2048 → vocab_size) + Softmax
  ↓
Output (logits)
```

### SWELU Activation

```python
SWELU(z, k) = sign(z) × (1 - exp(-|z|^k))
```

Where `k` is a learnable parameter initialized at 1.0.

## Installation

### Local Development

```bash
# Clone repo
git clone https://github.com/Powwpol/SWELU_LLM.git
cd SWELU_LLM

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### RunPod Setup

```bash
# On RunPod instance (RTX 4090)
apt update && apt install -y git
git clone https://github.com/Powwpol/SWELU_LLM.git
cd SWELU_LLM
pip install -r requirements.txt
```

## Usage

### Training

```bash
# Full training (100K steps)
python src/train.py --config configs/train_config.yaml

# Resume from checkpoint
python src/train.py --checkpoint checkpoints/model_step_50000.pt
```

### Inference

```python
from src.model import MambaSWELU
from src.inference import generate_text

# Load model
model = MambaSWELU.from_pretrained("checkpoints/final_model.pt")

# Generate text
output = generate_text(
    model,
    prompt="Once upon a time",
    max_length=100,
    temperature=0.8
)
print(output)
```

## Project Structure

```
SWELU_LLM/
├── src/
│   ├── swelu.py          # SWELU activation implementation
│   ├── mamba_block.py    # Mamba SSM block
│   ├── model.py          # Full MambaSWELU model
│   ├── train.py          # Training loop
│   ├── data_prep.py      # Data preprocessing
│   └── inference.py      # Inference utilities
├── tests/
│   ├── test_swelu.py     # SWELU tests
│   └── test_model.py     # Model architecture tests
├── data/                 # Training data (gitignored)
├── checkpoints/          # Model checkpoints (gitignored)
├── .github/
│   └── workflows/
│       └── test.yml      # CI/CD pipeline
├── requirements.txt
└── README.md
```

## Training Details

### Hardware Requirements

- **GPU**: RTX 4090 (24GB VRAM) or A100
- **RAM**: 32GB+
- **Storage**: 500GB+ SSD

### Training Hyperparameters

- **Batch size**: 8 (per GPU)
- **Gradient accumulation**: 4 steps
- **Effective batch size**: 32
- **Learning rate**: 3e-4 (with warmup)
- **Optimizer**: AdamW (β1=0.9, β2=0.95)
- **Weight decay**: 0.1
- **Training steps**: 100,000
- **Mixed precision**: BF16

### Dataset

- **Size**: 10B tokens (Wikipedia, Books, Web)
- **Preprocessing**: BPE tokenization (vocab_size=50,257)
- **Context length**: 2048 tokens

## Performance

| Metric | Target |
|--------|--------|
| Training time | ~40h on RTX 4090 |
| Inference speed | ~50 tokens/sec |
| Perplexity (validation) | <20 |
| Memory usage | ~18GB VRAM |

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# With coverage
pytest --cov=src tests/
```

### Code Quality

```bash
# Linting
ruff check src/

# Formatting
black src/ tests/
```

## Roadmap

- [x] Implement SWELU activation
- [x] Implement Mamba block
- [x] Build full architecture
- [ ] Data collection & preprocessing
- [ ] Initial training run (100K steps)
- [ ] Hyperparameter optimization
- [ ] Model evaluation & benchmarking
- [ ] Deployment pipeline

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@software{swelu_llm_2025,
  author = {Powwpol},
  title = {SWELU-LLM: Mamba Architecture with SWELU Activation},
  year = {2025},
  url = {https://github.com/Powwpol/SWELU_LLM}
}
```

## References

- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- [mamba-ssm GitHub](https://github.com/state-spaces/mamba)


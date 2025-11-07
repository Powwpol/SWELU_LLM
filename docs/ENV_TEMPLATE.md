# Environment Variables Template

Copy this file to `.env` in the project root and fill in your values.

```bash
# Weights & Biases Configuration
WANDB_API_KEY=your_wandb_api_key_here
WANDB_PROJECT=swelu-llm
WANDB_ENTITY=your_username_or_team

# HuggingFace Configuration (optional)
HF_TOKEN=your_huggingface_token_here

# Training Configuration
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Data paths
DATA_DIR=./data
CHECKPOINT_DIR=./checkpoints
LOG_DIR=./logs

# Cloud Storage (for RunPod backup)
# AWS S3
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
S3_BUCKET=

# Google Cloud Storage
GOOGLE_APPLICATION_CREDENTIALS=
GCS_BUCKET=

# RunPod specific
RUNPOD_POD_ID=
RUNPOD_API_KEY=
```

## Getting Your API Keys

### Wandb

1. Create account: https://wandb.ai/signup
2. Get API key: https://wandb.ai/authorize
3. Copy key to `WANDB_API_KEY`

### HuggingFace (optional)

1. Create account: https://huggingface.co/join
2. Get token: https://huggingface.co/settings/tokens
3. Copy to `HF_TOKEN`

### AWS S3 (for checkpoint backup on RunPod)

1. AWS Console → IAM → Users → Create user
2. Attach policy: `AmazonS3FullAccess`
3. Create access key → Copy ID and Secret
4. Create S3 bucket: `s3://swelu-checkpoints`

### Google Cloud Storage (alternative to S3)

1. GCP Console → Cloud Storage → Create bucket
2. IAM → Service Accounts → Create
3. Download JSON key file
4. Set path in `GOOGLE_APPLICATION_CREDENTIALS`

## Usage

```bash
# Create .env file
cp docs/ENV_TEMPLATE.md .env

# Edit with your values
nano .env  # or vi, vim, code, etc.

# Load in Python
from dotenv import load_dotenv
import os

load_dotenv()
wandb_key = os.getenv('WANDB_API_KEY')
```

## Security

⚠️ **NEVER commit `.env` to Git!**

The `.gitignore` already excludes it, but double-check:

```bash
git status  # Should NOT show .env
```


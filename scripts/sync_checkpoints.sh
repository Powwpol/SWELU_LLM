#!/bin/bash
# Script de synchronisation automatique des checkpoints vers cloud storage
# Utilise rclone pour sync vers S3, GCS, ou autre

set -e

CHECKPOINT_DIR="./checkpoints"
BACKUP_INTERVAL=3600  # Sync toutes les heures (3600 secondes)

echo "======================================"
echo "Checkpoint Sync Service"
echo "======================================"

# Check if rclone is installed
if ! command -v rclone &> /dev/null; then
    echo "Installing rclone..."
    curl https://rclone.org/install.sh | sudo bash
fi

# Load environment variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Setup cloud storage based on environment
if [ ! -z "$AWS_BUCKET_NAME" ]; then
    REMOTE_PATH="s3:$AWS_BUCKET_NAME/mamba-swelu/checkpoints"
    echo "Using AWS S3: $REMOTE_PATH"
elif [ ! -z "$GCS_BUCKET_NAME" ]; then
    REMOTE_PATH="gcs:$GCS_BUCKET_NAME/mamba-swelu/checkpoints"
    echo "Using Google Cloud Storage: $REMOTE_PATH"
else
    echo "⚠️  No cloud storage configured in .env"
    echo "   Set AWS_BUCKET_NAME or GCS_BUCKET_NAME"
    exit 1
fi

# Sync function
sync_checkpoints() {
    echo "[$(date)] Syncing checkpoints to cloud..."
    
    if [ -d "$CHECKPOINT_DIR" ]; then
        # Count files
        NUM_FILES=$(find $CHECKPOINT_DIR -type f | wc -l)
        
        if [ $NUM_FILES -gt 0 ]; then
            # Sync to cloud
            rclone sync $CHECKPOINT_DIR $REMOTE_PATH \
                --progress \
                --transfers 4 \
                --checkers 8 \
                --verbose
            
            echo "[$(date)] ✓ Synced $NUM_FILES files to $REMOTE_PATH"
        else
            echo "[$(date)] No checkpoints to sync yet"
        fi
    else
        echo "[$(date)] Checkpoint directory not found: $CHECKPOINT_DIR"
    fi
}

# Run once initially
sync_checkpoints

# Then run in loop
echo ""
echo "Starting automatic sync every $BACKUP_INTERVAL seconds..."
echo "Press Ctrl+C to stop"
echo ""

while true; do
    sleep $BACKUP_INTERVAL
    sync_checkpoints
done


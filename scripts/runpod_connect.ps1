# Script PowerShell pour gérer connexion RunPod
# Usage: .\scripts\runpod_connect.ps1 [action]
# Actions: connect, status, logs, sync_up, sync_down, setup, train

param(
    [Parameter(Position=0)]
    [ValidateSet("connect", "status", "logs", "sync_up", "sync_down", "setup", "train", "stop")]
    [string]$Action = "connect",
    
    [string]$Host = "",
    [string]$Port = "",
    [string]$SshKey = "$env:USERPROFILE\.ssh\id_ed25519"
)

# Colors
$Green = "Green"
$Yellow = "Yellow"
$Red = "Red"
$Cyan = "Cyan"

Write-Host "=" * 60 -ForegroundColor $Cyan
Write-Host "RunPod SSH Manager - SWELU LLM" -ForegroundColor $Cyan
Write-Host "=" * 60 -ForegroundColor $Cyan

# Load config from .env or use defaults
$EnvFile = ".env"
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]*)\s*=\s*(.*)$') {
            $name = $matches[1].Trim()
            $value = $matches[2].Trim()
            Set-Variable -Name $name -Value $value -Scope Script
        }
    }
}

# Override with parameters if provided
if ($Host) { $RUNPOD_HOST = $Host }
if ($Port) { $RUNPOD_PORT = $Port }
if (-not $RUNPOD_HOST) { $RUNPOD_HOST = Read-Host "RunPod Host IP" }
if (-not $RUNPOD_PORT) { $RUNPOD_PORT = Read-Host "RunPod SSH Port" }

$RUNPOD_USER = "root"
$SSH_CMD = "ssh -p $RUNPOD_PORT -i $SshKey ${RUNPOD_USER}@${RUNPOD_HOST}"
$SCP_CMD = "scp -P $RUNPOD_PORT -i $SshKey"
$RSYNC_CMD = "rsync -avz -e `"ssh -p $RUNPOD_PORT -i $SshKey`""

Write-Host "`nConnection Info:" -ForegroundColor $Yellow
Write-Host "  Host: $RUNPOD_HOST" -ForegroundColor $Green
Write-Host "  Port: $RUNPOD_PORT" -ForegroundColor $Green
Write-Host "  User: $RUNPOD_USER" -ForegroundColor $Green
Write-Host ""

switch ($Action) {
    "connect" {
        Write-Host "[ACTION] Connecting to RunPod..." -ForegroundColor $Yellow
        Invoke-Expression $SSH_CMD
    }
    
    "status" {
        Write-Host "[ACTION] Checking RunPod status..." -ForegroundColor $Yellow
        $StatusCmd = "$SSH_CMD 'echo === GPU Status === && nvidia-smi && echo && echo === Training Process === && ps aux | grep train.py | grep -v grep && echo && echo === Disk Usage === && df -h /workspace'"
        Invoke-Expression $StatusCmd
    }
    
    "logs" {
        Write-Host "[ACTION] Following training logs..." -ForegroundColor $Yellow
        Write-Host "Press Ctrl+C to stop" -ForegroundColor $Yellow
        $LogsCmd = "$SSH_CMD 'tail -f /workspace/SWELU_LLM/training.log'"
        Invoke-Expression $LogsCmd
    }
    
    "sync_up" {
        Write-Host "[ACTION] Syncing data TO RunPod..." -ForegroundColor $Yellow
        
        # Check if data exists locally
        if (-not (Test-Path "data\specialized")) {
            Write-Host "ERROR: data\specialized not found!" -ForegroundColor $Red
            Write-Host "Run: python src\data\prepare_specialized_datasets.py first" -ForegroundColor $Yellow
            exit 1
        }
        
        Write-Host "Syncing data\specialized\ -> RunPod:/workspace/SWELU_LLM/data/specialized/" -ForegroundColor $Green
        $SyncCmd = "$RSYNC_CMD data\specialized\ ${RUNPOD_USER}@${RUNPOD_HOST}:/workspace/SWELU_LLM/data/specialized/"
        Invoke-Expression $SyncCmd
        
        Write-Host "`nSync complete!" -ForegroundColor $Green
    }
    
    "sync_down" {
        Write-Host "[ACTION] Syncing checkpoints FROM RunPod..." -ForegroundColor $Yellow
        
        # Create local dir
        New-Item -ItemType Directory -Force -Path "checkpoints_runpod" | Out-Null
        
        Write-Host "Syncing RunPod:/workspace/SWELU_LLM/checkpoints/ -> checkpoints_runpod\" -ForegroundColor $Green
        $SyncCmd = "$RSYNC_CMD ${RUNPOD_USER}@${RUNPOD_HOST}:/workspace/SWELU_LLM/checkpoints/ checkpoints_runpod\"
        Invoke-Expression $SyncCmd
        
        Write-Host "`nCheckpoints downloaded!" -ForegroundColor $Green
    }
    
    "setup" {
        Write-Host "[ACTION] Setting up RunPod environment..." -ForegroundColor $Yellow
        
        $SetupScript = @"
cd /workspace &&
git clone https://github.com/Powwpol/SWELU_LLM.git &&
cd SWELU_LLM &&
bash scripts/setup_runpod.sh &&
export WANDB_API_KEY=dce1f23ec60761cb89913e3f1d8010908fb01048 &&
echo 'export WANDB_API_KEY=dce1f23ec60761cb89913e3f1d8010908fb01048' >> ~/.bashrc &&
python scripts/test_local.py
"@
        
        $SetupCmd = "$SSH_CMD '$SetupScript'"
        Invoke-Expression $SetupCmd
        
        Write-Host "`nSetup complete! RunPod is ready." -ForegroundColor $Green
    }
    
    "train" {
        Write-Host "[ACTION] Launching training on RunPod..." -ForegroundColor $Yellow
        Write-Host "This will start training in background (nohup)" -ForegroundColor $Yellow
        
        $Confirm = Read-Host "Continue? (y/n)"
        if ($Confirm -ne "y") {
            Write-Host "Cancelled." -ForegroundColor $Yellow
            exit 0
        }
        
        $TrainScript = @"
cd /workspace/SWELU_LLM &&
nohup python src/train.py \
  --config configs/full_model_runpod.yaml \
  --use_wandb \
  > training.log 2>&1 &
echo 'Training started in background!'
echo 'Monitor: https://wandb.ai/paul-obara/swelu-llm'
echo 'Logs: tail -f /workspace/SWELU_LLM/training.log'
sleep 2
tail -n 20 training.log
"@
        
        $TrainCmd = "$SSH_CMD '$TrainScript'"
        Invoke-Expression $TrainCmd
        
        Write-Host "`nTraining launched!" -ForegroundColor $Green
        Write-Host "Monitor at: https://wandb.ai/paul-obara/swelu-llm" -ForegroundColor $Cyan
        Write-Host "View logs: .\scripts\runpod_connect.ps1 logs" -ForegroundColor $Cyan
    }
    
    "stop" {
        Write-Host "[ACTION] Stopping RunPod instance..." -ForegroundColor $Yellow
        Write-Host "WARNING: This will shutdown the pod!" -ForegroundColor $Red
        
        $Confirm = Read-Host "Are you sure? (yes/no)"
        if ($Confirm -ne "yes") {
            Write-Host "Cancelled." -ForegroundColor $Yellow
            exit 0
        }
        
        $StopCmd = "$SSH_CMD 'sudo shutdown -h now'"
        Invoke-Expression $StopCmd
        
        Write-Host "`nPod shutdown initiated." -ForegroundColor $Green
        Write-Host "Don't forget to STOP it in RunPod dashboard to avoid charges!" -ForegroundColor $Yellow
    }
    
    default {
        Write-Host "Unknown action: $Action" -ForegroundColor $Red
        Write-Host "`nAvailable actions:" -ForegroundColor $Yellow
        Write-Host "  connect   - SSH into RunPod" -ForegroundColor $Green
        Write-Host "  status    - Check GPU and training status" -ForegroundColor $Green
        Write-Host "  logs      - Follow training logs" -ForegroundColor $Green
        Write-Host "  sync_up   - Upload data to RunPod" -ForegroundColor $Green
        Write-Host "  sync_down - Download checkpoints from RunPod" -ForegroundColor $Green
        Write-Host "  setup     - Initial setup (clone repo, install deps)" -ForegroundColor $Green
        Write-Host "  train     - Launch training in background" -ForegroundColor $Green
        Write-Host "  stop      - Shutdown RunPod instance" -ForegroundColor $Green
    }
}

Write-Host "`n" + "=" * 60 -ForegroundColor $Cyan


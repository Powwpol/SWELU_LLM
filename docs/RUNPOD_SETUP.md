# Configuration RunPod - Entraînement SWELU Complet

Guide pour lancer l'entraînement du modèle SWELU complet (350M params) sur RunPod.

## ⚠️ Avant de Commencer

### Prérequis CRITIQUES

1. **Tests locaux passent:** Vous DEVEZ avoir validé Phase 1 et 2 localement
2. **Budget:** ~16-20$ pour 40h sur RTX 4090 (~0.40$/h)
3. **Wandb configuré:** Compte créé + API key
4. **Code pushed sur GitHub:** Dernière version sur main branch

### Coût estimé

| GPU | Prix/h | 40h | 100h |
|-----|--------|-----|------|
| RTX 4090 | $0.39 | $15.60 | $39.00 |
| RTX A6000 | $0.79 | $31.60 | $79.00 |
| A100 80GB | $1.89 | $75.60 | $189.00 |

**Recommandé:** RTX 4090 (meilleur rapport qualité/prix)

## Configuration RunPod

### 1. Créer un compte RunPod

1. Allez sur https://www.runpod.io/
2. Créez un compte
3. Ajoutez des crédits (minimum $20)

### 2. Créer une instance GPU

1. **Pods** → **+ Deploy**
2. **GPU Type:** RTX 4090 (24GB VRAM)
3. **Container Image:** `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`
4. **Container Disk:** 50GB minimum
5. **Volume:** 100GB (pour checkpoints et dataset)
6. **Cloud Type:** Secure Cloud (plus fiable)
7. Cliquez **Deploy**

**Coût:** ~$0.39/h + $0.10/GB/mois stockage

### 3. Se connecter au Pod

Une fois déployé, cliquez sur **Connect** → **Start Web Terminal**

Ou via SSH:
```bash
ssh root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519
```

## Installation sur RunPod

### 1. Cloner le dépôt

```bash
cd /workspace
git clone https://github.com/Powwpol/SWELU_LLM.git
cd SWELU_LLM
```

### 2. Installer les dépendances

```bash
bash scripts/setup_runpod.sh
```

Ce script:
- Met à jour apt packages
- Installe Python dependencies
- Configure CUDA
- Vérifie que le GPU est détecté

**Vérification GPU:**
```bash
nvidia-smi
```

Vous devriez voir:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 11.8   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0 Off |                  N/A |
| 30%   42C    P0    50W / 450W |      0MiB / 24564MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

### 3. Configurer Wandb

```bash
# Installer wandb si pas déjà fait
pip install wandb

# Login
wandb login
```

Entrez votre API key (trouvable sur https://wandb.ai/authorize)

**Alternative:** Éditez `.env`:
```bash
nano .env
# Ajoutez: WANDB_API_KEY=votre_clé
```

### 4. Configurer le backup des checkpoints (IMPORTANT!)

RunPod peut crasher. **BACKUP OBLIGATOIRE.**

#### Option A: Google Drive (gratuit, simple)

```bash
# Installer rclone
curl https://rclone.org/install.sh | sudo bash

# Configurer Google Drive
rclone config
# Suivre les instructions pour ajouter Google Drive

# Tester
rclone ls gdrive:
```

#### Option B: AWS S3

```bash
pip install boto3
export AWS_ACCESS_KEY_ID=votre_key
export AWS_SECRET_ACCESS_KEY=votre_secret
export S3_BUCKET=swelu-checkpoints
```

Le script `sync_checkpoints.sh` gérera le backup automatique.

## Lancement de l'Entraînement

### 1. Test rapide (5min)

Avant de lancer 40h, faites un test:

```bash
python src/train.py \
  --config configs/small_model.yaml \
  --max_steps 100 \
  --use_wandb
```

Vérifiez:
- ✓ GPU utilisé (nvidia-smi)
- ✓ Loss diminue
- ✓ Wandb logs visibles sur wandb.ai
- ✓ Checkpoints se sauvegardent

### 2. Lancement entraînement complet

```bash
bash scripts/train_runpod.sh
```

Ce script lance:
```bash
nohup python src/train.py \
  --config configs/full_model_runpod.yaml \
  --use_wandb \
  > training.log 2>&1 &
```

- `nohup`: Continue même si la connexion SSH drop
- `&`: Background process
- `training.log`: Tous les logs

### 3. Monitoring

#### Voir les logs en temps réel

```bash
tail -f training.log
```

#### Wandb Dashboard

Ouvrez https://wandb.ai/votre_username/swelu-llm

Vous verrez:
- Loss curve
- Learning rate schedule
- Perplexity
- GPU utilization
- Tokens/sec

#### Vérifier que ça tourne

```bash
ps aux | grep train.py
nvidia-smi  # GPU utilization ~95%+
du -sh checkpoints/  # Taille augmente
```

### 4. Backup automatique des checkpoints

Le script `sync_checkpoints.sh` tourne en parallèle:

```bash
bash scripts/sync_checkpoints.sh &
```

Il sync automatiquement:
- Chaque nouveau checkpoint → Cloud storage
- Toutes les 30 minutes

## Problèmes Fréquents

### Erreur: "CUDA out of memory"

```bash
# Réduire batch size dans configs/full_model_runpod.yaml
training:
  batch_size: 4  # au lieu de 8
  gradient_accumulation_steps: 8  # au lieu de 4
```

### Pod crash / restart

**Solution:** Vos checkpoints sont backupés!

1. Redémarrer le pod
2. Restaurer dernier checkpoint:
```bash
rclone copy gdrive:swelu-checkpoints ./checkpoints/
```
3. Relancer training (reprend automatiquement):
```bash
bash scripts/train_runpod.sh
```

### Dataset download trop lent

**Solution:** Pré-télécharger et cacher:

```bash
# Dans un terminal séparé
python -c "
from datasets import load_dataset
ds = load_dataset('wikipedia', '20220301.en', split='train')
print('Dataset cached!')
"
```

### Training trop lent

Vérifiez:
```bash
nvidia-smi  # GPU util devrait être >90%
```

Si <50%:
- Problème CPU bottleneck → Augmenter num_workers dans DataLoader
- Problème I/O → Dataset sur volume NVMe au lieu de network

### Wandb ne log pas

```bash
# Vérifier API key
echo $WANDB_API_KEY

# Re-login
wandb login --relogin
```

## Télécharger le Modèle Final

### 1. Via rclone (recommandé)

```bash
# Sync tous les checkpoints
rclone sync ./checkpoints/ gdrive:swelu-checkpoints/
```

### 2. Via SCP

```bash
# Sur votre machine locale
scp -r -P <port> root@<pod-ip>:/workspace/SWELU_LLM/checkpoints ./
```

### 3. Via Wandb Artifacts

Le script de training sauvegarde automatiquement sur Wandb:
- Allez sur wandb.ai
- Artifacts → Download

## Arrêter l'Entraînement

### Gracefully (recommandé)

Le training sauvegarde un checkpoint à chaque interruption:

```bash
# Trouver le PID
ps aux | grep train.py
# Tuer proprement
kill -SIGINT <PID>
```

### Forcer

```bash
pkill -9 python
```

**⚠️ Attention:** Peut corrompre le checkpoint en cours d'écriture.

## Coûts Réels

### Exemple d'entraînement complet

- GPU: RTX 4090 @ $0.39/h × 40h = **$15.60**
- Storage: 100GB @ $0.10/GB/mois = **$10.00** (si gardé 1 mois)
- Bande passante: ~gratuit pour download dataset

**Total:** ~$16-26 selon durée stockage

### Optimisation coûts

1. **Stop le pod** dès que training terminé
2. **Sync checkpoints** vers cloud gratuit (Google Drive)
3. **Delete pod** immédiatement après download
4. **Use Spot instances** si OK avec interruptions (-50% prix)

## Checklist Avant de Lancer

- [ ] Tests locaux passent (Phase 1 & 2)
- [ ] Code pushed sur GitHub
- [ ] Wandb configuré et testé
- [ ] Backup checkpoints configuré (rclone ou S3)
- [ ] Budget alloué ($20+ sur RunPod)
- [ ] Test 100 steps réussi sur RunPod
- [ ] Monitoring Wandb fonctionne

**Si tout est ✅ → GO!**

```bash
bash scripts/train_runpod.sh
```

## Support

- RunPod Discord: https://discord.gg/runpod
- Issues GitHub: https://github.com/Powwpol/SWELU_LLM/issues
- Wandb Docs: https://docs.wandb.ai/


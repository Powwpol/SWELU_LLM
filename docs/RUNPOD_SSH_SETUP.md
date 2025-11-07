# Configuration SSH pour RunPod - Contrôle Remote

Guide pour connecter ton environnement local à RunPod via SSH et lancer le training à distance.

## 🎯 Objectif

**Contrôler RunPod depuis ta machine Windows** sans avoir à utiliser leur interface web.

Avantages:
- ✅ Upload code directement depuis local
- ✅ Lancer training via SSH
- ✅ Monitoring en temps réel
- ✅ Sync données bidirectionnel

## 🔑 Prérequis

1. **Compte RunPod créé** avec crédits ($20+)
2. **Clé SSH générée** sur ta machine
3. **Pod déployé** sur RunPod

## Étape 1: Générer Clé SSH (si pas déjà fait)

### Sur Windows (PowerShell)

```powershell
# Vérifier si clé existe déjà
Test-Path ~\.ssh\id_ed25519

# Si FALSE, générer nouvelle clé
ssh-keygen -t ed25519 -C "paulo@swelu"
# Appuie sur Enter 3 fois (pas de passphrase pour automation)

# Afficher la clé publique
Get-Content ~\.ssh\id_ed25519.pub
```

**Copie la clé publique** (commence par `ssh-ed25519 ...`)

## Étape 2: Créer Pod RunPod avec SSH

### 2.1 Via Interface Web

1. Connecte-toi sur https://www.runpod.io/console/pods
2. **Deploy** → **GPU Pod**
3. Configuration:
   - **GPU**: RTX 4090 (24GB)
   - **Container**: `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`
   - **Container Disk**: 50GB
   - **Volume**: 100GB (persistent storage)
   - **Expose HTTP/TCP Ports**: `22` (pour SSH)

4. **Deploy On-Demand**

### 2.2 Ajouter ta clé SSH

Une fois le pod déployé:

1. Clique sur **Connect** → **Start Web Terminal**
2. Dans le terminal web:

```bash
# Créer répertoire SSH
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# Ajouter ta clé publique
echo "ssh-ed25519 AAAA... paulo@swelu" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys

# Vérifier
cat ~/.ssh/authorized_keys
```

3. **Redémarre le service SSH:**

```bash
service ssh restart
```

## Étape 3: Obtenir Info de Connexion

Dans l'interface RunPod, trouve:

```
SSH Command: ssh root@X.X.X.X -p XXXXX -i ~/.ssh/id_ed25519
```

Exemple:
```
SSH: ssh root@194.26.183.45 -p 22456 -i ~/.ssh/id_ed25519
```

**Note les infos:**
- IP: `194.26.183.45`
- Port: `22456`

## Étape 4: Connexion SSH depuis Windows

### 4.1 Premier Test

```powershell
# Remplace par tes valeurs
ssh root@194.26.183.45 -p 22456 -i ~\.ssh\id_ed25519
```

**Si ça demande "Are you sure?"** → tape `yes`

**Si connecté:** Tu verras `root@runpod-...#`

### 4.2 Créer Alias pour Faciliter

Crée un fichier `~\.ssh\config`:

```powershell
# Créer/éditer le config SSH
code ~\.ssh\config
```

Ajoute (remplace par tes valeurs):

```
Host runpod-swelu
    HostName 194.26.183.45
    Port 22456
    User root
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

**Maintenant tu peux connecter avec:**

```powershell
ssh runpod-swelu
```

## Étape 5: Setup Initial sur RunPod

Une fois connecté en SSH:

```bash
# 1. Cloner le repo
cd /workspace
git clone https://github.com/Powwpol/SWELU_LLM.git
cd SWELU_LLM

# 2. Installer dépendances
bash scripts/setup_runpod.sh

# 3. Configurer Wandb
export WANDB_API_KEY=dce1f23ec60761cb89913e3f1d8010908fb01048

# 4. Vérifier GPU
nvidia-smi

# 5. Test rapide
python scripts/test_local.py
```

## Étape 6: Upload Données depuis Local

### Option A: SCP (Small Files)

```powershell
# Depuis ta machine Windows
scp -P 22456 -r data/specialized root@194.26.183.45:/workspace/SWELU_LLM/data/
```

### Option B: Rsync (Recommandé pour gros datasets)

```powershell
# Installer rsync sur Windows (via Chocolatey)
choco install rsync

# Sync données
rsync -avz -e "ssh -p 22456" data/specialized/ root@194.26.183.45:/workspace/SWELU_LLM/data/specialized/
```

### Option C: Télécharger directement sur RunPod

```bash
# Sur RunPod via SSH
cd /workspace/SWELU_LLM
python src/data/prepare_specialized_datasets.py --domain math
python src/data/prepare_specialized_datasets.py --domain lean
```

**⚠️ Attention:** Télécharger sur RunPod = payer le temps machine!  
**Mieux:** Télécharger en local, puis sync

## Étape 7: Lancer Training via SSH

```bash
# Sur RunPod via SSH
cd /workspace/SWELU_LLM

# Lancer training en background
nohup python src/train.py \
  --config configs/full_model_runpod.yaml \
  --use_wandb \
  > training.log 2>&1 &

# Voir le processus
ps aux | grep train.py

# Suivre les logs en temps réel
tail -f training.log

# Pour détacher et revenir plus tard
# Ctrl+C pour arrêter tail
# 'exit' pour déconnecter SSH
# Le training continue en background!
```

## Étape 8: Monitoring depuis Local

### 8.1 Wandb Dashboard

Ouvre dans ton navigateur:
```
https://wandb.ai/paul-obara/swelu-llm
```

### 8.2 SSH + Tail Logs

```powershell
# Depuis Windows
ssh runpod-swelu "tail -f /workspace/SWELU_LLM/training.log"
```

### 8.3 GPU Utilization

```powershell
# Watch GPU en temps réel
ssh runpod-swelu "watch -n 1 nvidia-smi"
```

## Étape 9: Download Checkpoints

```powershell
# Une fois training terminé, récupérer les checkpoints
scp -P 22456 -r root@194.26.183.45:/workspace/SWELU_LLM/checkpoints ./checkpoints_runpod/
```

## 🔧 Scripts Automatisés

### Script: `scripts/runpod_connect.ps1`

Crée ce script PowerShell pour automatiser:

```powershell
# Connexion SSH rapide
param(
    [string]$Action = "connect"
)

$RUNPOD_HOST = "194.26.183.45"
$RUNPOD_PORT = "22456"
$RUNPOD_USER = "root"

switch ($Action) {
    "connect" {
        ssh ${RUNPOD_USER}@${RUNPOD_HOST} -p ${RUNPOD_PORT}
    }
    "status" {
        ssh ${RUNPOD_USER}@${RUNPOD_HOST} -p ${RUNPOD_PORT} "nvidia-smi && ps aux | grep train.py"
    }
    "logs" {
        ssh ${RUNPOD_USER}@${RUNPOD_HOST} -p ${RUNPOD_PORT} "tail -f /workspace/SWELU_LLM/training.log"
    }
    "sync_up" {
        rsync -avz -e "ssh -p ${RUNPOD_PORT}" ./data/specialized/ ${RUNPOD_USER}@${RUNPOD_HOST}:/workspace/SWELU_LLM/data/specialized/
    }
    "sync_down" {
        rsync -avz -e "ssh -p ${RUNPOD_PORT}" ${RUNPOD_USER}@${RUNPOD_HOST}:/workspace/SWELU_LLM/checkpoints/ ./checkpoints_runpod/
    }
}
```

**Usage:**

```powershell
.\scripts\runpod_connect.ps1 connect      # Se connecter
.\scripts\runpod_connect.ps1 status       # Voir statut GPU/training
.\scripts\runpod_connect.ps1 logs         # Suivre logs
.\scripts\runpod_connect.ps1 sync_up      # Upload données
.\scripts\runpod_connect.ps1 sync_down    # Download checkpoints
```

## 🔐 Sécurité

### Protéger ta clé SSH

```powershell
# Vérifier permissions
icacls ~\.ssh\id_ed25519

# Si trop ouvert, restreindre (Windows)
icacls ~\.ssh\id_ed25519 /inheritance:r
icacls ~\.ssh\id_ed25519 /grant:r "$($env:USERNAME):(R)"
```

### Variables d'environnement

**NE JAMAIS** commiter:
- Clés SSH privées
- IPs/Ports RunPod
- API keys

Utilise `.env`:

```bash
# Dans .env (déjà dans .gitignore)
RUNPOD_HOST=194.26.183.45
RUNPOD_PORT=22456
RUNPOD_SSH_KEY=~/.ssh/id_ed25519
```

## ⚠️ Troubleshooting

### "Connection refused"

```bash
# Sur RunPod web terminal
service ssh status
service ssh restart
```

### "Permission denied (publickey)"

```bash
# Vérifier authorized_keys
cat ~/.ssh/authorized_keys

# Permissions
chmod 600 ~/.ssh/authorized_keys
chmod 700 ~/.ssh
```

### "Host key verification failed"

```powershell
# Supprimer ancienne clé
ssh-keygen -R "[194.26.183.45]:22456"

# Reconnecte
ssh runpod-swelu
```

### Pod redémarre (perte IP/Port)

RunPod peut changer IP/Port. **Solution:**

1. Check nouvelle IP dans RunPod dashboard
2. Update `~\.ssh\config` avec nouvelle IP/Port
3. Reconnecte

## 💰 Coûts

| Action | Coût |
|--------|------|
| SSH connecté (idle) | $0.39/h |
| Training running | $0.39/h |
| Transfer data in | Gratuit |
| Transfer data out | Gratuit (<100GB/mois) |

**⚠️ Important:** Arrête le pod quand tu ne l'utilises pas!

```bash
# Avant de déconnecter, si training terminé
sudo shutdown -h now
```

Ou via RunPod dashboard: **Stop Pod**

## 📋 Checklist Complète

- [ ] Clé SSH générée localement
- [ ] Pod RunPod déployé (RTX 4090)
- [ ] Clé publique ajoutée sur RunPod
- [ ] SSH fonctionne (`ssh runpod-swelu`)
- [ ] Repo cloné sur RunPod
- [ ] Dépendances installées
- [ ] Wandb configuré
- [ ] Données sync'ées (ou téléchargées)
- [ ] Test local réussi sur RunPod
- [ ] Script automation créé
- [ ] Training lancé en background
- [ ] Monitoring Wandb OK

## 🚀 Workflow Complet

```powershell
# 1. Développement local
git add -A
git commit -m "New features"
git push origin main

# 2. Sync vers RunPod
ssh runpod-swelu "cd /workspace/SWELU_LLM && git pull"

# 3. Lancer training
ssh runpod-swelu "cd /workspace/SWELU_LLM && bash scripts/train_runpod.sh"

# 4. Monitor
# Ouvrir https://wandb.ai/paul-obara/swelu-llm

# 5. Une fois terminé, récupérer checkpoints
.\scripts\runpod_connect.ps1 sync_down

# 6. Arrêter pod
# Via RunPod dashboard ou
ssh runpod-swelu "sudo shutdown -h now"
```

---

**Prêt à configurer RunPod? Follow ce guide étape par étape!**


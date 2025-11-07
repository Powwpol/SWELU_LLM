# 🚀 SWELU LLM - Quick Start Guide

Guide de démarrage rapide complet du développement local au training sur RunPod.

## ✅ État Actuel du Projet

### Ce qui est FAIT:
- [x] Architecture modèle (Mamba + SWELU)
- [x] Tests locaux fonctionnels
- [x] Wandb configuré (paul-obara/swelu-llm)
- [x] Scripts datasets spécialisés (Maths, Lean, SC)
- [x] Setup SSH RunPod automatisé
- [x] Code sur GitHub: https://github.com/Powwpol/SWELU_LLM

### En COURS:
- [ ] **Téléchargement datasets** (~40GB, 2-3h)
  - MathPile: ~12.7GB
  - Proof-Pile: ~15GB
  - Lean Mathlib: ~2GB

### À FAIRE:
- [ ] Déployer pod RunPod
- [ ] Configurer SSH
- [ ] Sync données vers RunPod
- [ ] Lancer training complet (40h)

---

## 📋 Workflow Complet

### Phase 1: Données (EN COURS)

**Téléchargement Maths + Lean** (2-3h):

```powershell
# Vérifier progression
Get-Job

# Voir logs
Get-Content data_download_math.log -Tail 20 -Wait

# Si pas encore lancé:
python src/data/prepare_specialized_datasets.py --domain math --output data/specialized
python src/data/prepare_specialized_datasets.py --domain lean --output data/specialized
```

**Résultat attendu:**
```
data/specialized/
  ├── mathpile.pt (~12GB)
  ├── proof_pile.pt (~15GB)
  ├── lean_mathlib.pt (~2GB)
  ├── *_metadata.json
  └── Total: ~30-40GB, ~9B tokens
```

### Phase 2: RunPod Setup

#### 2.1 Créer Pod

1. Allez sur https://www.runpod.io/console/pods
2. **Deploy GPU Pod:**
   - GPU: **RTX 4090** (24GB, $0.39/h)
   - Image: `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`
   - Container Disk: **50GB**
   - Volume (persistent): **100GB** ($10/mois)
   - Expose Port: **22** (SSH)

3. **Deploy On-Demand**

**Coût:** ~$0.39/h + $10/mois storage = **~$26 total** pour 40h training

#### 2.2 Configurer SSH

**Voir guide complet:** [docs/RUNPOD_SSH_SETUP.md](docs/RUNPOD_SSH_SETUP.md)

**Quick setup:**

```powershell
# 1. Générer clé SSH (si pas déjà fait)
ssh-keygen -t ed25519 -C "paulo@swelu"

# 2. Copier clé publique
Get-Content ~\.ssh\id_ed25519.pub | clip

# 3. Sur RunPod web terminal, coller la clé:
mkdir -p ~/.ssh && echo "COLLE_TA_CLÉ_ICI" >> ~/.ssh/authorized_keys

# 4. Noter IP et Port du pod
# Exemple: SSH Command: ssh root@194.26.183.45 -p 22456

# 5. Mettre dans .env
Add-Content .env "RUNPOD_HOST=194.26.183.45"
Add-Content .env "RUNPOD_PORT=22456"

# 6. Tester connexion
.\scripts\runpod_connect.ps1 connect
```

#### 2.3 Setup Initial RunPod

```powershell
# Setup automatique (clone repo + install deps + test)
.\scripts\runpod_connect.ps1 setup
```

**Ce que ça fait:**
- Clone le repo GitHub
- Installe requirements.txt
- Configure Wandb
- Teste GPU
- Run test_local.py

**Durée:** ~5-10min

### Phase 3: Sync Données

**Option A: Upload depuis local** (recommandé si déjà téléchargé)

```powershell
# Une fois datasets téléchargés localement
.\scripts\runpod_connect.ps1 sync_up
```

**Durée:** 30min - 2h selon connexion (upload ~40GB)

**Option B: Download directement sur RunPod**

```powershell
# Se connecter
.\scripts\runpod_connect.ps1 connect

# Sur RunPod
cd /workspace/SWELU_LLM
python src/data/prepare_specialized_datasets.py --domain math
python src/data/prepare_specialized_datasets.py --domain lean
```

**⚠️ Attention:** Cette option **coûte plus cher** (tu payes le temps GPU pendant téléchargement)  
**Mieux:** Download en local pendant la nuit, puis sync

### Phase 4: Launch Training

```powershell
# Lancer training en background sur RunPod
.\scripts\runpod_connect.ps1 train
```

**Ce que ça fait:**
- Lance training avec `configs/full_model_runpod.yaml`
- Background process (nohup)
- Logs vers `training.log`
- Wandb monitoring activé

**Durée:** ~40h sur RTX 4090

### Phase 5: Monitoring

#### Wandb Dashboard (Recommandé)

Ouvre dans navigateur:
```
https://wandb.ai/paul-obara/swelu-llm
```

**Métriques visibles:**
- Loss en temps réel
- Perplexité
- Learning rate
- GPU utilization
- Tokens/sec
- ETA completion

#### Logs en temps réel

```powershell
# Suivre logs live
.\scripts\runpod_connect.ps1 logs

# Ou directement
ssh runpod-swelu "tail -f /workspace/SWELU_LLM/training.log"
```

#### Status GPU

```powershell
.\scripts\runpod_connect.ps1 status
```

### Phase 6: Récupération Checkpoints

```powershell
# Download tous les checkpoints
.\scripts\runpod_connect.ps1 sync_down
```

**Checkpoints sauvegardés:**
```
checkpoints_runpod/
  ├── checkpoint_step_10000.pt
  ├── checkpoint_step_20000.pt
  ├── ...
  └── final_model.pt
```

### Phase 7: Stop Pod

**IMPORTANT:** Arrête le pod pour éviter frais!

```powershell
# Via script
.\scripts\runpod_connect.ps1 stop

# OU via RunPod dashboard
# Pods → Stop Pod
```

---

## 🔧 Commandes Utiles

### Gestion Locale

```powershell
# Tests locaux
python scripts/test_local.py

# Vérifier datasets
ls data/specialized/

# Voir taille datasets
(Get-ChildItem data/specialized -Recurse | Measure-Object -Property Length -Sum).Sum / 1GB

# Git sync
git add -A
git commit -m "Updates"
git push origin main
```

### Gestion RunPod

```powershell
# Connexion SSH
.\scripts\runpod_connect.ps1 connect

# Status complet
.\scripts\runpod_connect.ps1 status

# Logs training
.\scripts\runpod_connect.ps1 logs

# Upload données
.\scripts\runpod_connect.ps1 sync_up

# Download checkpoints
.\scripts\runpod_connect.ps1 sync_down

# Setup initial
.\scripts\runpod_connect.ps1 setup

# Lancer training
.\scripts\runpod_connect.ps1 train

# Stop pod
.\scripts\runpod_connect.ps1 stop
```

### Wandb

```powershell
# Login local
wandb login

# Voir runs
start https://wandb.ai/paul-obara/swelu-llm

# Sync offline runs (si besoin)
wandb sync
```

---

## ⏱️ Timeline Complète

| Étape | Durée | État |
|-------|-------|------|
| Tests locaux | 2min | ✅ Fait |
| Download datasets (local) | 2-3h | 🔄 En cours |
| Setup RunPod pod | 5min | ⏳ À faire |
| Configure SSH | 10min | ⏳ À faire |
| Sync données → RunPod | 30min-2h | ⏳ À faire |
| Training complet | 40h | ⏳ À faire |
| Download checkpoints | 30min | ⏳ À faire |
| **TOTAL** | **~43-46h** | |

**Temps actif requis:** ~1-2h (setup + monitoring)  
**Temps passif:** 40h (training automatique)

---

## 💰 Coûts Détaillés

| Item | Coût |
|------|------|
| RunPod RTX 4090 (40h) | $15.60 |
| Storage 100GB (1 mois) | $10.00 |
| Bandwidth (upload/download) | Gratuit |
| **TOTAL** | **~$25-30** |

**⚠️ Optimisation:** Delete pod dès que checkpoints récupérés!

---

## 🐛 Troubleshooting

### Datasets download échoue

```powershell
# Vérifier job
Get-Job

# Voir erreurs
Receive-Job -Id 1

# Relancer manuellement
python src/data/prepare_specialized_datasets.py --domain math
```

### SSH ne fonctionne pas

```powershell
# Vérifier clé SSH
Test-Path ~\.ssh\id_ed25519

# Regénérer
ssh-keygen -t ed25519 -C "paulo@swelu"

# Recopier sur RunPod web terminal
```

### Training plante

```powershell
# Voir logs
.\scripts\runpod_connect.ps1 logs

# Check GPU
.\scripts\runpod_connect.ps1 status

# Relancer
ssh runpod-swelu "cd /workspace/SWELU_LLM && bash scripts/train_runpod.sh"
```

### Out of memory

Édite `configs/full_model_runpod.yaml`:
```yaml
training:
  batch_size: 4  # reduce from 8
  gradient_accumulation_steps: 8  # increase from 4
```

---

## 📚 Documentation Complète

- [SETUP_LOCAL.md](docs/SETUP_LOCAL.md) - Installation locale
- [RUNPOD_SETUP.md](docs/RUNPOD_SETUP.md) - Guide RunPod complet
- [RUNPOD_SSH_SETUP.md](docs/RUNPOD_SSH_SETUP.md) - Configuration SSH
- [DATASETS_OVERVIEW.md](docs/DATASETS_OVERVIEW.md) - Datasets disponibles
- [ENV_TEMPLATE.md](docs/ENV_TEMPLATE.md) - Variables d'environnement

---

## ✅ Checklist Avant Training

- [ ] Tests locaux passent (`python scripts/test_local.py`)
- [ ] Datasets téléchargés (~40GB dans `data/specialized/`)
- [ ] Wandb configuré (`.env` avec API key)
- [ ] Pod RunPod créé et actif
- [ ] SSH fonctionne (`.\scripts\runpod_connect.ps1 connect`)
- [ ] Setup RunPod terminé (`.\scripts\runpod_connect.ps1 setup`)
- [ ] Données sync'ées (`.\scripts\runpod_connect.ps1 sync_up`)
- [ ] Budget alloué ($25-30 sur RunPod)

**Si tout est ✅ → GO!**

```powershell
.\scripts\runpod_connect.ps1 train
```

**Monitoring:** https://wandb.ai/paul-obara/swelu-llm

---

**Questions? Check docs/ ou ouvre une issue sur GitHub!**


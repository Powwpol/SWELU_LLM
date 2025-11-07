<!-- 9ca095de-01e5-442b-83a6-d016679bfb94 28be872e-77b5-416c-aa20-b774eff5488e -->
# Plan: Configuration Entraînement SWELU (Local → Test → RunPod)

## Phase 1: Tests Locaux (CPU)

### 1.1 Script de test rapide

**Fichier**: `scripts/test_local.py`

- Teste imports (torch, mamba-ssm optionnel)
- Teste création modèle tiny (10M params)
- Teste forward pass
- Teste 10 steps training sur données dummy
- Durée: ~2min sur CPU

### 1.2 Guide installation locale

**Fichier**: `docs/SETUP_LOCAL.md`

- Installation Python 3.10+
- Création venv
- Installation requirements.txt
- Commande test: `python scripts/test_local.py`

## Phase 2: Petit Modèle (Test GPU)

### 2.1 Configuration petit modèle

**Fichier**: `configs/small_model.yaml`

```yaml
model:
  vocab_size: 50257
  d_model: 256        # au lieu de 1024
  n_layers: 2         # au lieu de 6
  max_seq_len: 512    # au lieu de 2048
  
training:
  batch_size: 4
  max_steps: 1000     # ~10min sur GPU
  dataset: wikipedia
  max_samples: 10000  # sous-ensemble
```

### 2.2 Script d'entraînement simplifié

**Fichier**: `scripts/train_small.py`

- Charge config small_model.yaml
- Utilise WikipediaDataset avec max_samples
- Sauvegarde checkpoint toutes les 100 steps
- Affiche loss/perplexity
- Génère texte à la fin

### 2.3 Script de lancement

**Fichier**: `scripts/run_small.sh` (Linux/Mac) et `scripts/run_small.bat` (Windows)

```bash
python scripts/train_small.py --config configs/small_model.yaml
```

## Phase 3: Modèle Complet RunPod

### 3.1 Configuration modèle complet

**Fichier**: `configs/full_model_runpod.yaml`

```yaml
model:
  vocab_size: 50257
  d_model: 1024
  n_layers: 6
  max_seq_len: 2048
  
training:
  batch_size: 8
  gradient_accumulation_steps: 4
  max_steps: 100000
  learning_rate: 3e-4
  mixed_precision: bf16
  use_wandb: true
  dataset: wikipedia  # ou c4
```

### 3.2 Guide RunPod

**Fichier**: `docs/RUNPOD_SETUP.md`

- Création instance RTX 4090
- Clone repo GitHub
- Installation dépendances
- Configuration wandb (optionnel)
- Lancement training
- Monitoring et checkpoints
- Téléchargement modèle final

### 3.3 Script RunPod automatisé

**Fichier**: `scripts/setup_runpod.sh`

- Installation système (apt packages)
- Installation Python dependencies
- Test GPU disponible
- Lancement training avec nohup (background)

### 3.4 Script de lancement RunPod

**Fichier**: `scripts/train_runpod.sh`

```bash
nohup python src/train.py \
  --config configs/full_model_runpod.yaml \
  --use_wandb \
  > training.log 2>&1 &
```

## Phase 4: Documentation

### 4.1 Mise à jour README

Ajouter section "Quick Start" avec 3 scénarios:

1. Test local (2min)
2. Small model (10min)
3. Full training RunPod (40h)

### 4.2 Notebook Jupyter

**Fichier**: `notebooks/quickstart.ipynb`

- Test activation SWELU
- Test Mamba block
- Test modèle complet
- Génération de texte
- Visualisations

## Fichiers à créer

```
configs/
  ├── small_model.yaml          # Phase 2
  └── full_model_runpod.yaml    # Phase 3

scripts/
  ├── test_local.py              # Phase 1
  ├── train_small.py             # Phase 2
  ├── run_small.sh/bat           # Phase 2
  ├── setup_runpod.sh            # Phase 3
  └── train_runpod.sh            # Phase 3

docs/
  ├── SETUP_LOCAL.md             # Phase 1
  └── RUNPOD_SETUP.md            # Phase 3

notebooks/
  └── quickstart.ipynb           # Phase 4
```

## Ordre d'exécution

1. **Créer tous les fichiers de configuration**
2. **Tester localement**: `python scripts/test_local.py`
3. **Si OK, tester petit modèle**: `bash scripts/run_small.sh`
4. **Si OK, préparer RunPod**: suivre `docs/RUNPOD_SETUP.md`
5. **Lancer training complet**: `bash scripts/train_runpod.sh`

### To-dos

- [ ] Créer fichiers de configuration (small_model.yaml, full_model_runpod.yaml)
- [ ] Créer script de test local (scripts/test_local.py)
- [ ] Créer script entraînement petit modèle (scripts/train_small.py, run_small.sh/bat)
- [ ] Créer scripts et documentation RunPod (setup_runpod.sh, train_runpod.sh, RUNPOD_SETUP.md)
- [ ] Mettre à jour README avec Quick Start et créer SETUP_LOCAL.md
- [ ] Créer notebook Jupyter quickstart (notebooks/quickstart.ipynb)
- [ ] Tester localement puis commit et push vers GitHub


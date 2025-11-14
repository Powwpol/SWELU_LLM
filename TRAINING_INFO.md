# 🚀 Entraînement MambaSWELU - Configuration Actuelle

## 📊 Configuration Optimale (LLaMA Style)

**Démarré le:** `date`

### Modèle
- **Architecture:** MambaSWELU
- **Paramètres:** 124,104,719 (~124M)
- **Couches:** 6 Mamba blocks + 3 Dense layers
- **Dimension:** 1024
- **Activation:** SWELU (learnable)

### Dataset
- **Source:** SlimPajama-627B
- **Mode:** Streaming (pas de téléchargement complet)
- **Tokenizer:** GPT-2 (vocab_size=50,257)

### Hyperparamètres
```bash
batch_size:                  4
gradient_accumulation_steps: 4
effective_batch_size:        16
sequence_length:             1024
max_steps:                   757,500
learning_rate:               3e-4
weight_decay:                0.1
warmup_steps:                2,000
mixed_precision:             BF16
```

### Tokens d'Entraînement
- **Tokens par step:** 16,384
- **Total tokens:** 12,410,471,900 (12.4B)
- **Ratio tokens/param:** 100x (LLaMA style - optimal)
- **Utilisation SlimPajama:** ~2% du dataset

### Durée Estimée
- **Total:** ~70 heures (~3 jours)
- **Par checkpoint (5k steps):** ~28 minutes
- **Nombre de checkpoints:** 151

### Checkpoints
- **Fréquence:** Tous les 5,000 steps
- **Localisation:** `./checkpoints/`
- **Format:** `model_step_XXXXX.pt`
- **Contenu:** 
  - État du modèle
  - État de l'optimiseur
  - Global step
  - Epoch

### Monitoring

**Commandes utiles:**
```bash
# Suivre les logs en temps réel
tail -f training.log

# Monitoring rapide
./monitor_training.sh

# Auto-refresh toutes les 10 secondes
watch -n 10 ./monitor_training.sh

# Vérifier GPU
nvidia-smi -l 5

# Arrêter l'entraînement
pkill -f train.py
```

**Métriques à surveiller:**
- Loss: devrait diminuer progressivement
- Learning rate: warmup puis cosine decay
- GPU utilization: devrait être >80%
- Memory: ~885MB sur RTX 4090

### Reprendre l'Entraînement

Si l'entraînement s'arrête, reprendre depuis le dernier checkpoint:

```bash
# Trouver le dernier checkpoint
ls -lt checkpoints/*.pt | head -1

# Relancer avec reprise
python src/train.py \
  --dataset slimpajama \
  --resume_from_checkpoint ./checkpoints/model_step_XXXXX.pt \
  [... autres paramètres identiques ...]
```

### Lois d'Échelle Respectées

✅ **Chinchilla (minimum):** 20 tokens/param → 2.48B tokens  
✅ **LLaMA (optimal):** 100 tokens/param → 12.4B tokens ← **Configuration actuelle**  
✅ **GPT-3 (référence):** 300 tokens/param → 37.2B tokens

Notre configuration suit les meilleures pratiques modernes (LLaMA/Pythia).

### Notes Importantes

1. **Premier démarrage:** Téléchargement des métadonnées SlimPajama (~5 min)
2. **Validation:** Désactivée en mode streaming (pas critique)
3. **Mamba-SSM:** Utilise version simplifiée (installer `mamba-ssm` pour optimisation)
4. **WandB:** Désactivé (activer avec `--use_wandb` si installé)

### Fichiers Générés

```
/root/SWELU_LLM/
├── training.log                    # Logs d'entraînement
├── checkpoints/
│   ├── model_step_5000.pt
│   ├── model_step_10000.pt
│   └── ... (151 checkpoints au total)
│   └── final_model.pt             # Modèle final
└── training_old_*.log             # Backups des anciens logs
```

### Prochaines Étapes Après Entraînement

1. **Évaluation:** Tester la perplexité sur un validation set
2. **Génération:** Utiliser `src/inference.py` pour générer du texte
3. **Fine-tuning:** Adapter sur des tâches spécifiques si besoin
4. **Comparaison:** Benchmarker vs modèles de taille similaire

---

**Pour plus d'infos:** Consulter `configs/optimal_training.sh` pour d'autres configurations.


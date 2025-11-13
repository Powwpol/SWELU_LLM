# 🚀 Instructions de Configuration - Entraînement Multi-GPU

## ⚠️ Problème Actuel: Rate Limit HuggingFace

L'entraînement a été bloqué par un **rate limit HuggingFace**. SlimPajama-627B nécessite un token HF pour être téléchargé.

```
Error: 429 Client Error: Too Many Requests
Solution: Créer/utiliser un token HuggingFace
```

---

## 🔧 Solution Rapide

### Option 1: Créer un Token HuggingFace (RECOMMANDÉ)

1. **Créer un compte HF** (si pas encore fait):
   - Aller sur: https://huggingface.co/join
   - S'inscrire gratuitement

2. **Générer un token**:
   - Aller sur: https://huggingface.co/settings/tokens
   - Cliquer "New token"
   - Nom: "swelu-training"
   - Type: "Read"
   - Copier le token

3. **Configurer le token**:
   ```bash
   # Méthode 1: Export direct
   export HF_TOKEN=hf_votre_token_ici
   
   # Méthode 2: Via script interactif
   ./setup_hf_token.sh
   
   # Méthode 3: Créer .env manuellement
   echo "HF_TOKEN=hf_votre_token_ici" > .env
   ```

4. **Lancer l'entraînement**:
   ```bash
   ./launch_multi_gpu_with_token.sh
   ```

---

### Option 2: Login HuggingFace CLI

```bash
# Installer huggingface-cli si nécessaire
pip install huggingface-hub

# Login interactif
huggingface-cli login

# Puis lancer l'entraînement
./launch_multi_gpu_with_token.sh
```

---

### Option 3: Utiliser un Dataset Alternatif (TEMPORAIRE)

Si vous ne pouvez pas obtenir un token HF immédiatement, utilisez Wikipedia à la place:

```bash
# Modifier le script pour utiliser wikipedia
python src/train.py \
  --dataset wikipedia \
  --max_steps 757500 \
  [... autres params ...]
```

**Note:** Wikipedia est beaucoup plus petit (~20GB) que SlimPajama (627B tokens), donc moins optimal.

---

## 📊 Configuration Multi-GPU

Une fois le token configuré, l'entraînement utilisera **6x RTX 4090** en parallèle:

### Avantages:
- **Speedup:** ~6x plus rapide
- **Durée:** ~11.7h au lieu de 70h
- **Économie:** ~58h de temps GPU
- **Tokens:** 12.4B (ratio optimal 100x)

### Répartition:
```
GPU 0: Process rank 0 (master)
GPU 1: Process rank 1
GPU 2: Process rank 2
GPU 3: Process rank 3
GPU 4: Process rank 4
GPU 5: Process rank 5
```

---

## ✅ Vérification Post-Lancement

Après avoir lancé l'entraînement, vérifiez que les 6 GPUs sont utilisés:

```bash
# Vérifier les GPUs
nvidia-smi

# Devrait montrer:
# - 6 processus Python
# - ~885MB par GPU
# - Utilisation >80% sur chaque GPU

# Suivre les logs
tail -f training.log

# Monitoring continu
watch -n 5 './monitor_training.sh'
```

---

## 🐛 Troubleshooting

### Problème: "Rate limit"
- **Solution:** Configurer HF_TOKEN (voir Option 1 ci-dessus)

### Problème: "All processes on GPU 0"
- **Solution:** Déjà corrigé avec `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5`

### Problème: "OOM (Out of Memory)"
- **Solution:** Réduire batch_size à 2 ou 3

### Problème: "Connection timeout"
- **Solution:** Vérifier connexion internet, SlimPajama est volumineux

---

## 📝 Commandes Utiles

```bash
# Configuration initiale
./setup_hf_token.sh                    # Configurer token HF

# Lancement
./launch_multi_gpu_with_token.sh       # Multi-GPU (6x RTX 4090)
./launch_llama_style.sh                # Single GPU (plus lent)

# Monitoring
./monitor_training.sh                  # Status rapide
tail -f training.log                   # Logs en direct
watch -n 10 nvidia-smi                 # GPUs en temps réel

# Contrôle
pkill -f train.py                      # Arrêter entraînement
ps aux | grep train                    # Vérifier processus

# Checkpoints
ls -lh checkpoints/                    # Voir checkpoints sauvegardés
```

---

## 🎯 Prochaines Étapes

1. ✅ Obtenir un token HuggingFace
2. ✅ Configurer le token (`.env` ou `export`)
3. 🚀 Lancer `./launch_multi_gpu_with_token.sh`
4. 📊 Monitorer avec `watch -n 10 ./monitor_training.sh`
5. ⏳ Attendre ~11.7h
6. 🎉 Modèle final dans `checkpoints/final_model.pt`

---

## 💡 Notes Importantes

- Le premier lancement télécharge les métadonnées SlimPajama (~5min)
- Les checkpoints sont sauvegardés tous les 5,000 steps
- Utilisation de ~885MB par GPU
- Total mémoire: ~5.3GB sur les 6 GPUs
- Bande passante: Important pour streaming SlimPajama

---

**Pour toute question, vérifiez les logs: `tail -f training.log`**


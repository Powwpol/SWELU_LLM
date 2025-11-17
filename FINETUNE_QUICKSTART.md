# 🚀 Guide de Fine-Tuning - Démarrage Rapide

## ✅ Scripts Créés

Tous les scripts nécessaires sont prêts :

1. ✅ `prepare_instruction_data.py` - Télécharge et formate les datasets
2. ✅ `finetune.py` - Script de fine-tuning optimisé
3. ✅ `compare_models.py` - Compare avant/après
4. ✅ `launch_finetune_6gpu.sh` - Lance sur 6 GPUs (Option 2)
5. ✅ `test_finetune_1gpu.sh` - Test rapide sur 1 GPU

---

## 🎯 Option 2 - Grande Capacité (Recommandé)

### Configuration
- **6x RTX 4090** (144 GB VRAM total)
- **Batch size effectif**: 192 (4 × 6 GPUs × 8 accumulation)
- **Learning rate**: 5e-6 (très bas pour préserver le modèle)
- **Steps**: 25,000 (~20h d'entraînement)
- **Datasets**: Alpaca + Dolly + OpenAssistant (~200k exemples)

---

## 📋 Étapes Recommandées

### **Étape 1 : Test Rapide (RECOMMANDÉ)** ⚡

Toujours tester avant de lancer un long training !

```bash
# Test de 15 minutes sur 1 GPU pour valider que tout fonctionne
./test_finetune_1gpu.sh
```

**Vérifications** :
- ✅ La loss diminue-t-elle ?
- ✅ Pas d'erreur OOM (Out of Memory) ?
- ✅ Les checkpoints se sauvegardent ?
- ✅ Le modèle génère mieux qu'avant ?

**Test du checkpoint** :
```bash
python demo_chat.py --checkpoint checkpoints/test_finetune/checkpoint_step_500.pt
```

---

### **Étape 2 : Fine-Tuning Complet** 🔥

Si le test fonctionne, lance le vrai fine-tuning :

```bash
# ~20h d'entraînement sur 6 GPUs
./launch_finetune_6gpu.sh
```

**Monitoring pendant le training** :
```bash
# Dans un autre terminal
tail -f logs/finetune/*.log

# Vérifier l'utilisation GPU
watch -n 1 nvidia-smi
```

**Points de contrôle** :
- **@1000 steps (~1h)** : Loss doit avoir baissé de 30-40%
- **@5000 steps (~4h)** : Tester avec demo_chat.py
- **@10000 steps (~8h)** : Comparer avec modèle de base
- **@15000 steps (~12h)** : Vérifier que loss continue de descendre
- **@25000 steps (~20h)** : Fin du training

---

### **Étape 3 : Évaluation** 📊

```bash
# Comparer base vs fine-tuné
python compare_models.py \
    --base_model checkpoints/model_gpu5/final_model.pt \
    --finetuned_model checkpoints/finetuned/finetuned_model.pt

# Tester le modèle final
python demo_chat.py --checkpoint checkpoints/finetuned/finetuned_model.pt
```

---

## 🎛️ Paramètres Personnalisables

Si tu veux ajuster :

### Plus Rapide (mais moins de qualité)
```bash
# 10k steps au lieu de 25k (~8h au lieu de 20h)
torchrun --nproc_per_node=6 finetune.py \
    --max_steps 10000 \
    --learning_rate 1e-5 \
    [autres params...]
```

### Plus Conservateur (moins de risque d'oublier)
```bash
# Learning rate encore plus bas
torchrun --nproc_per_node=6 finetune.py \
    --learning_rate 2e-6 \
    --max_steps 30000 \
    [autres params...]
```

### Plus de Capacité (si tu as >6 GPUs)
```bash
# Exemple pour 8 GPUs
torchrun --nproc_per_node=8 finetune.py \
    --batch_size 6 \
    --gradient_accumulation_steps 6 \
    [autres params...]
# Batch effectif = 6 × 8 × 6 = 288
```

---

## 🔧 Dépannage

### Erreur OOM (Out of Memory)
```bash
# Réduire batch size
--batch_size 2  # au lieu de 4

# Ou réduire max_length
--max_length 512  # au lieu de 1024
```

### Loss qui augmente
```bash
# Learning rate trop élevé, réduire de moitié
--learning_rate 2.5e-6  # au lieu de 5e-6
```

### Training trop lent
```bash
# Vérifier que tous les GPUs sont utilisés
nvidia-smi

# Augmenter num_workers
--num_workers 8  # au lieu de 4
```

### Datasets ne se téléchargent pas
```bash
# Si problème HuggingFace, utiliser seulement Alpaca
python prepare_instruction_data.py --max_samples 52000
# Force l'utilisation d'Alpaca seulement (plus fiable)
```

---

## 📊 Résultats Attendus

### Avant Fine-Tuning ❌
```
Q: What is the capital of France?
A: What are the major areas of the country? [incohérent]

Q: What is 2+2?
A: The first is the number of words... [hors-sujet]
```

### Après Fine-Tuning ✅
```
Q: What is the capital of France?
A: The capital of France is Paris.

Q: What is 2+2?
A: 2+2 equals 4.
```

**Métriques de succès** :
- ✅ Répond correctement aux questions factuelles
- ✅ Suit les instructions (ex: "Write a haiku")
- ✅ Maintient le contexte conversationnel
- ✅ Moins d'hallucinations
- ✅ Code généré cohérent

---

## ⏱️ Timeline Complète

| Temps | Action |
|-------|--------|
| **T+0** | Lancer `test_finetune_1gpu.sh` |
| **T+15min** | Vérifier résultats du test |
| **T+30min** | Si OK, lancer `launch_finetune_6gpu.sh` |
| **T+1h** | Checkpoint @1000 steps - vérifier loss |
| **T+4h** | Checkpoint @5000 steps - tester qualité |
| **T+8h** | Checkpoint @10000 steps - comparer |
| **T+12h** | Checkpoint @15000 steps - validation |
| **T+16h** | Checkpoint @20000 steps - presque fini |
| **T+20h** | Checkpoint @25000 steps - **TERMINÉ** |
| **T+20h30** | Évaluation finale et comparaison |

---

## 🎯 Commandes Essentielles

### Préparation
```bash
# 1. Préparer les données (si pas déjà fait)
python prepare_instruction_data.py

# 2. Test rapide (OBLIGATOIRE)
./test_finetune_1gpu.sh
```

### Fine-Tuning
```bash
# 3. Lancer le vrai fine-tuning
./launch_finetune_6gpu.sh

# Ou manuellement avec contrôle total
torchrun --nproc_per_node=6 finetune.py \
    --train_file data/instruction/train.jsonl \
    --checkpoint checkpoints/model_gpu5/final_model.pt \
    --max_steps 25000 \
    --batch_size 4 \
    --learning_rate 5e-6
```

### Évaluation
```bash
# 4. Tester pendant l'entraînement
python demo_chat.py --checkpoint checkpoints/finetuned/checkpoint_step_5000.pt

# 5. Comparer final
python compare_models.py
```

---

## 💡 Conseils Pratiques

1. **Toujours faire le test 1 GPU d'abord** - Économise du temps si config incorrecte
2. **Monitorer la loss** - Doit descendre graduellement, pas de spike
3. **Tester aux checkpoints** - Qualité observable dès 5000 steps
4. **Garder 3-5 checkpoints** - Au cas où overfitting vers la fin
5. **Patience** - 20h c'est long, mais ça vaut le coup !

---

## 🚨 Erreurs à Éviter

1. ❌ **Ne pas tester avant** → Risque de perdre 20h si erreur config
2. ❌ **Learning rate trop haut** → Catastrophic forgetting
3. ❌ **Pas de monitoring** → Impossible de détecter les problèmes
4. ❌ **Attendre 25k steps sans vérifier** → Peut overfitter
5. ❌ **Oublier de sauvegarder les checkpoints** → Perdu si crash

---

## ✅ Checklist Avant de Lancer

- [ ] Test 1 GPU réussi (`test_finetune_1gpu.sh`)
- [ ] Datasets téléchargés (`data/instruction/train.jsonl` existe)
- [ ] Checkpoint de base existe (`checkpoints/model_gpu5/final_model.pt`)
- [ ] Espace disque suffisant (50GB minimum)
- [ ] Toutes les 6 GPUs disponibles (`nvidia-smi`)
- [ ] Temps disponible (~24h sans interruption recommandé)

---

## 🎉 Prêt à Commencer ?

```bash
# GO GO GO ! 🚀
./test_finetune_1gpu.sh
```

**Good luck! 💪**


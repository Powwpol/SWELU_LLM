# 📊 Étude Comparative - 3 Modèles MambaSWELU

## 🎯 Objectif

Comparer l'impact de **différents datasets** et **stratégies d'entraînement** sur la performance du modèle MambaSWELU 124M.

---

## 🔬 Configuration de l'Expérience

### Architecture Commune (Identique pour tous)
- **Modèle:** MambaSWELU
- **Paramètres:** 124,104,719 (~124M)
- **Couches:** 6 Mamba blocks + 3 Dense layers
- **Dimension:** 1024
- **Activation:** SWELU (learnable)
- **Sequence length:** 1024
- **Batch size:** 4 (effective: 16 avec grad accum)
- **Learning rate:** 3e-4
- **Mixed precision:** BF16

### Variables (Ce qui change)

| Modèle | Dataset | Ratio | Steps | Tokens | GPUs | Durée | Checkpoints |
|--------|---------|-------|-------|--------|------|-------|-------------|
| **1. SlimPajama LLaMA** | SlimPajama-627B | 100x | 757,500 | 12.4B | 0-1 | ~35h | ./checkpoints/slimpajama_llama/ |
| **2. SlimPajama Chinchilla** | SlimPajama-627B | 20x | 151,500 | 2.48B | 2-3 | ~7h | ./checkpoints/slimpajama_chinchilla/ |
| **3. Wikipedia LLaMA** | Wikipedia | 100x | 757,500 | 12.4B | 4-5 | ~35h | ./checkpoints/wikipedia_llama/ |

---

## 📈 Métriques à Comparer

### 1. Loss (Perte d'entraînement)
- Vitesse de convergence
- Loss finale
- Stabilité pendant l'entraînement

### 2. Perplexité
- Sur validation set
- Sur test set
- Par type de texte

### 3. Qualité de Génération
- Cohérence
- Diversité
- Factualité
- Créativité

### 4. Efficacité
- Temps d'entraînement total
- Coût GPU (GPU-heures)
- Ratio performance/coût

---

## 🔍 Hypothèses à Tester

### H1: Impact du Dataset
**Question:** SlimPajama (627B tokens, diversifié) vs Wikipedia (plus petit, plus structuré)

**Attentes:**
- SlimPajama → Meilleure généralisation
- Wikipedia → Meilleur sur texte encyclopédique

### H2: Impact du Ratio Tokens/Paramètres
**Question:** Chinchilla (20x) vs LLaMA (100x)

**Attentes:**
- Chinchilla (20x) → Convergence plus rapide, suffisant pour baseline
- LLaMA (100x) → Meilleure performance finale

### H3: Dataset Quality vs Quantity
**Question:** Est-ce que la qualité de Wikipedia compense sa taille réduite?

**Attentes:**
- À tokens égaux, Wikipedia pourrait être compétitif
- SlimPajama devrait dominer avec 100x ratio

---

## 📊 Timeline

```
Heure 0:    ████ Tous démarrent
Heure 7:    ████ Chinchilla terminé ✓
Heure 35:   ████ LLaMA models terminés ✓
```

**Premier résultat:** Chinchilla (SlimPajama) dans ~7h  
**Résultats finaux:** Dans ~35h

---

## 📝 Logs et Monitoring

### Logs individuels
```bash
tail -f logs/slimpajama_llama.log
tail -f logs/slimpajama_chinchilla.log
tail -f logs/wikipedia_llama.log
```

### Monitoring global
```bash
./monitor_all_trainings.sh
watch -n 10 './monitor_all_trainings.sh'
```

### Checkpoints
```bash
ls -lh checkpoints/slimpajama_llama/
ls -lh checkpoints/slimpajama_chinchilla/
ls -lh checkpoints/wikipedia_llama/
```

---

## 🧪 Protocole d'Évaluation (Post-Entraînement)

### 1. Perplexité Quantitative
```bash
# Évaluer sur même validation set
python eval.py --model checkpoints/slimpajama_llama/final_model.pt
python eval.py --model checkpoints/slimpajama_chinchilla/final_model.pt
python eval.py --model checkpoints/wikipedia_llama/final_model.pt
```

### 2. Génération Qualitative
```bash
# Même prompt pour tous
PROMPT="The future of artificial intelligence is"
python generate.py --model slimpajama_llama --prompt "$PROMPT"
python generate.py --model slimpajama_chinchilla --prompt "$PROMPT"
python generate.py --model wikipedia_llama --prompt "$PROMPT"
```

### 3. Benchmarks Standards
- LAMBADA
- HellaSwag
- PIQA
- WinoGrande

---

## 💡 Questions de Recherche

1. **Le ratio optimal est-il vraiment 20x (Chinchilla)?**
   - Ou 100x apporte-t-il des gains significatifs?

2. **SlimPajama justifie-t-il sa complexité?**
   - Vs Wikipedia qui est plus simple à utiliser

3. **Pour un budget fixe, quelle stratégie?**
   - Chinchilla rapide pour itération
   - LLaMA pour performance maximale

4. **Architecture Mamba + SWELU:**
   - Suit-elle les mêmes lois d'échelle que Transformers?
   - Les gains de SWELU sont-ils dataset-dépendants?

---

## 📊 Résultats Attendus

### Scénario Optimiste
- **Chinchilla (7h):** Baseline acceptable rapidement
- **SlimPajama LLaMA:** Meilleur modèle overall
- **Wikipedia LLaMA:** Compétitif sur texte encyclopédique

### Insights Espérés
1. Courbes de loss comparatives
2. Loi d'échelle pour Mamba+SWELU
3. ROI de chaque stratégie
4. Recommandations pour futurs entraînements

---

## 🎯 Critères de Succès

✅ **Succès** si:
- Les 3 modèles convergent sans erreur
- Loss décroît de manière stable
- Chinchilla termine en <8h
- Modèles génèrent du texte cohérent

⚠️ **Points d'attention:**
- Divergence de loss
- OOM errors
- Checkpoints corrompus
- Variations GPU trop importantes

---

## 📁 Structure des Résultats

```
checkpoints/
├── slimpajama_llama/
│   ├── model_step_5000.pt
│   ├── model_step_10000.pt
│   └── final_model.pt
├── slimpajama_chinchilla/
│   └── final_model.pt (plus rapide)
└── wikipedia_llama/
    └── final_model.pt

logs/
├── slimpajama_llama.log
├── slimpajama_chinchilla.log
└── wikipedia_llama.log
```

---

## 🚀 Prochaines Étapes

1. ✅ Entraînements lancés
2. ⏳ Monitoring continu (7-35h)
3. 📊 Collecte des métriques
4. 🧪 Évaluation comparative
5. 📝 Rapport final avec recommandations

---

**Date de début:** $(date)  
**Statut:** 🟢 En cours  
**Monitoring:** `./monitor_all_trainings.sh`


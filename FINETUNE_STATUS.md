# 🔥 Fine-Tuning Status - MambaSWELU

## ✅ Statut Actuel

**Lancé le** : 15 Nov 2025, 09:49 UTC  
**Configuration** : Option 2 - Grande Capacité (6x RTX 4090)

---

## 📊 Paramètres

| Paramètre | Valeur |
|-----------|--------|
| **GPUs** | 6x RTX 4090 |
| **Batch effectif** | 192 (4 × 6 × 8) |
| **Learning rate** | 5e-6 |
| **Warmup steps** | 1,000 |
| **Max steps** | 25,000 |
| **Dataset** | Alpaca (52k) + Dolly (15k) + OpenAssistant (53k) |
| **Total exemples** | 114k train + 6k val |

---

## ⏱️ Timeline

| Milestone | Steps | Temps estimé | Action |
|-----------|-------|--------------|--------|
| Démarrage | 0 | T+0h | ✅ FAIT |
| Warmup terminé | 1,000 | T+7h | Vérifier loss baisse |
| Premier test | 5,000 | T+35h | **Tester qualité !** |
| Mi-parcours | 12,500 | T+87h (~3.6j) | Comparer vs base |
| Quasi-final | 20,000 | T+139h (~5.8j) | Validation |
| **TERMINÉ** | 25,000 | T+174h (~7.3j) | Évaluation finale |

⚠️ **Vitesse actuelle : ~6.2s/step**  
**Durée totale estimée : ~43 heures (~1.8 jours)**

---

## 📉 Loss Tracking

### Pourquoi loss = 8.1 au lieu de 4.6 ?

**C'EST NORMAL !** Voici pourquoi :

#### Pré-training (SlimPajama)
```
Texte brut: "The capital of France is Paris. The city..."
Loss finale: 4.6 ← Texte continu, facile à prédire
```

#### Fine-tuning (Instructions)
```
Format Q&A: "User: What is the capital?\nAssistant: Paris."  
Loss initiale: 8.1 ← Nouveau format, plus complexe !
```

**Différences clés** :

1. **Distribution shift** : Texte brut → Conversations structurées
2. **Nouveaux tokens** : "User:", "Assistant:", etc.
3. **Complexité intrinsèque** : Q&A moins prédictible que texte continu

**Loss cible à la fin : ~5.0-5.2**  
(JAMAIS 4.6 - c'est impossible et pas souhaitable pour un modèle conversationnel)

### Évolution attendue

| Step | Loss estimée | Commentaire |
|------|--------------|-------------|
| 0 | 8.1 | Démarrage - confusion totale |
| 1,000 | ~7.2 | Warmup terminé |
| 5,000 | ~6.3 | Début d'apprentissage |
| 10,000 | ~5.7 | Amélioration visible |
| 15,000 | ~5.3 | Convergence |
| 20,000 | ~5.1 | Presque optimal |
| 25,000 | ~5.0 | **CIBLE FINALE** ✅ |

---

## 🧪 Points de Contrôle

### @1000 steps (~7h)
- [ ] Loss a baissé à ~7.2 ?
- [ ] Pas de spike de loss ?
- [ ] GPU utilization stable ?

### @5000 steps (~35h)
- [ ] Loss à ~6.3 ?
- [ ] **TESTER LA QUALITÉ** :
  ```bash
  python demo_chat.py --checkpoint checkpoints/finetuned/checkpoint_step_5000.pt
  ```
- [ ] Le modèle répond-il mieux aux questions ?

### @10000 steps (~87h / ~3.6 jours)
- [ ] Loss à ~5.7 ?
- [ ] Comparer avec modèle de base :
  ```bash
  python compare_models.py \
      --base_model checkpoints/model_gpu5/final_model.pt \
      --finetuned_model checkpoints/finetuned/checkpoint_step_10000.pt
  ```

### @20000 steps (~139h / ~5.8 jours)
- [ ] Loss à ~5.1 ?
- [ ] Validation : pas d'overfitting ?

### @25000 steps - FIN (~174h / ~7.3 jours)
- [ ] Loss finale ~5.0 ?
- [ ] **Évaluation complète** !

---

## 📊 Monitoring

### Commandes utiles

```bash
# Suivre en temps réel
tail -f logs/finetune_full.log

# Status rapide
./monitor_finetune.sh

# GPU usage
nvidia-smi

# Tester checkpoint
python demo_chat.py --checkpoint checkpoints/finetuned/checkpoint_step_5000.pt
```

---

## 🚨 Signaux d'Alerte

### ❌ Loss qui augmente
**Cause** : Learning rate trop élevé  
**Action** : Arrêter, réduire LR à 2.5e-6, relancer depuis dernier checkpoint

### ❌ Loss qui stagne
**Cause** : Peut-être déjà optimal ou LR trop bas  
**Action** : Tester qualité, si bonne → terminer, sinon augmenter LR

### ❌ OOM (Out of Memory)
**Cause** : Batch trop grand  
**Action** : Réduire `--batch_size` à 2

### ❌ Loss validation > loss training
**Cause** : Overfitting  
**Action** : Arrêter plus tôt, utiliser checkpoint précédent

---

## 💾 Checkpoints Sauvegardés

Checkpoints créés tous les 1000 steps :
- `checkpoint_step_1000.pt`
- `checkpoint_step_2000.pt`
- ...
- `checkpoint_step_25000.pt`
- `finetuned_model.pt` (final)

**Garde les 3 derniers** pour éviter saturer le disque.

---

## 🎯 Objectif Final

### Avant Fine-Tuning ❌
```
Q: What is the capital of France?
A: What are the major areas of the country? [HORS-SUJET]
```

### Après Fine-Tuning ✅  
```
Q: What is the capital of France?
A: The capital of France is Paris.
```

---

## 📝 Notes

- Loss ~5.0 pour un modèle conversationnel = EXCELLENT
- Ne JAMAIS comparer avec loss pré-training (4.6)
- Dataset instructions ≠ Dataset texte brut
- **Patience** : ~43h de training, mais ça vaut le coup !

---

**Mis à jour** : 15 Nov 2025, 09:50 UTC


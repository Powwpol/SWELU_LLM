# 🚀 DÉMARRAGE FINE-TUNING - Option 2 Grande Capacité

## ✅ INFRASTRUCTURE PRÊTE

Tous les scripts sont créés et opérationnels :

```
✅ prepare_instruction_data.py  (11 KB)  - Télécharge datasets
✅ finetune.py                   (16 KB)  - Fine-tuning multi-GPU  
✅ compare_models.py             (6 KB)   - Compare avant/après
✅ launch_finetune_6gpu.sh       (4 KB)   - Lance 6 GPUs
✅ test_finetune_1gpu.sh         (3 KB)   - Test rapide 1 GPU
```

---

## 🎯 CONFIGURATION OPTION 2

**Objectif** : Fine-tuning conversationnel de haute qualité

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| **GPUs** | 6x RTX 4090 | Maximum de capacité |
| **Batch effectif** | 192 | 4 × 6 × 8 accumulation |
| **Learning rate** | 5e-6 | Très bas (préserve modèle base) |
| **Steps** | 25,000 | ~200k exemples vus |
| **Durée** | ~20h | Sur 6 GPUs |
| **Datasets** | Alpaca + Dolly + OA | ~200-250k exemples |

---

## 🏃 DÉMARRAGE IMMÉDIAT

### Option A : Test Rapide Puis Full (RECOMMANDÉ) ✅

```bash
# 1. Test de 15 minutes sur 1 GPU
./test_finetune_1gpu.sh

# 2. Si OK, lancer le vrai fine-tuning
./launch_finetune_6gpu.sh
```

### Option B : Direct Full Training (Si Confiant) 🔥

```bash
# Tout en un - lance directement les 6 GPUs
./launch_finetune_6gpu.sh
```

---

## 📊 MONITORING

### Pendant l'entraînement

```bash
# Terminal 1 : Suivre les logs
tail -f logs/finetune/*.log

# Terminal 2 : Surveiller GPUs  
watch -n 1 nvidia-smi

# Terminal 3 : Tester aux checkpoints
python demo_chat.py --checkpoint checkpoints/finetuned/checkpoint_step_5000.pt
```

### Checkpoints clés

| Step | Temps | Action |
|------|-------|--------|
| 1,000 | ~1h | Vérifier loss baisse |
| 5,000 | ~4h | **Premier test qualité** |
| 10,000 | ~8h | Comparer vs base |
| 15,000 | ~12h | Validation continue |
| 20,000 | ~16h | Presque fini |
| 25,000 | ~20h | **TERMINÉ** |

---

## 🎯 RÉSULTATS ATTENDUS

### Avant Fine-Tuning ❌

```
Prompt: What is the capital of France?
Base:   What are the major areas of the country? [INCOHÉRENT]

Prompt: User: Hello! How are you?
Base:   J.P. Williams. It was a great time... [HORS-SUJET]
```

### Après Fine-Tuning ✅

```
Prompt: What is the capital of France?
Finetuned: The capital of France is Paris.

Prompt: User: Hello! How are you?
Finetuned: Hello! I'm doing well, thank you for asking. How can I help you today?
```

---

## 🚨 POINTS D'ATTENTION

### Critiques à Surveiller

1. **Loss qui augmente** → Learning rate trop haut
   - Solution : Réduire à 2.5e-6 et relancer

2. **OOM (Out of Memory)** → Batch trop grand
   - Solution : `--batch_size 2` au lieu de 4

3. **Loss qui stagne** → Peut-être déjà optimal
   - Solution : Tester qualité, possiblement arrêter

4. **Génération bizarre** → Catastrophic forgetting
   - Solution : Utiliser checkpoint précédent, LR trop haut

---

## 💡 ASTUCES PRO

1. **Tester dès 5000 steps** - Qualité observable rapidement
2. **Garder 5 checkpoints** - Au cas où overfitting
3. **Comparer régulièrement** - Base vs Finetuné
4. **Patience** - 20h c'est long mais ça vaut le coup !

---

## ✅ CHECKLIST FINALE

Avant de lancer, vérifie :

- [ ] 6x GPUs disponibles (`nvidia-smi`)
- [ ] ~50GB espace disque libre
- [ ] Checkpoint base existe (`checkpoints/model_gpu5/final_model.pt`)
- [ ] Scripts exécutables (`chmod +x *.sh`)
- [ ] Test 1 GPU réussi (si option A)

---

## 🎬 COMMANDE FINALE

```bash
# OPTION RECOMMANDÉE
./test_finetune_1gpu.sh          # 15 min de test
# puis si OK :
./launch_finetune_6gpu.sh        # 20h de fine-tuning

# OU DIRECT
./launch_finetune_6gpu.sh        # YOLO 🚀
```

---

## 📚 DOCUMENTATION

- `FINETUNE_QUICKSTART.md` - Guide détaillé
- `FINE_TUNING_STRATEGY.md` - Stratégie complète
- `finetune.py --help` - Toutes les options

---

## 🆘 BESOIN D'AIDE ?

Si problème, vérifie :

1. Logs : `logs/finetune/*.log`
2. GPU memory : `nvidia-smi`
3. Datasets : `ls data/instruction/`
4. Checkpoints : `ls checkpoints/finetuned/`

---

## 🔥 READY TO GO ?

```bash
cd /root/SWELU_LLM
./test_finetune_1gpu.sh
```

**Let's make this model GREAT! 💪🚀**


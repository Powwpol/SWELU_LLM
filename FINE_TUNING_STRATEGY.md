# 🎯 Stratégie de Fine-Tuning pour MambaSWELU

## 📊 Diagnostic Actuel

**Modèle** : MambaSWELU 124M paramètres
**Training** : 757,500 steps sur SlimPajama
**Problèmes identifiés** :
- ❌ Pas de compréhension des questions
- ❌ Réponses hors-sujet systématiques
- ❌ Hallucinations (capitale de France = "Tijdens")
- ❌ Répétitions en boucle
- ❌ Mélange code/texte incohérent

**Cause racine** : Modèle base sans instruction tuning

---

## 🚀 3 Stratégies Proposées

### **Option 1 : Fine-Tuning Rapide (1-2 jours)** ⚡
*Pour obtenir rapidement un modèle conversationnel basique*

**Dataset** : Alpaca (52k instructions en anglais)
- Format : `instruction` + `input` + `output`
- Taille : ~50MB
- Temps d'entraînement : 5-10k steps (~6h sur 6x RTX 4090)

**Hyperparamètres recommandés** :
```bash
--learning_rate 1e-5          # Plus bas que pré-training
--weight_decay 0.01           # Réduit pour ne pas casser le modèle
--warmup_steps 500            # Court warmup
--max_steps 10000             # 10k steps suffisent
--batch_size 4                # Plus petit pour stabilité
--gradient_accumulation 8     # Total batch = 32
--checkpoint_every 1000
```

**Avantages** :
- ✅ Rapide à mettre en place
- ✅ Dataset propre et testé
- ✅ Résultats visibles en quelques heures

**Inconvénients** :
- ⚠️ Seulement en anglais
- ⚠️ Qualité moyenne (dataset de 2023)

---

### **Option 2 : Fine-Tuning Conversationnel (3-5 jours)** 🎯
*Pour un modèle chat de meilleure qualité*

**Datasets combinés** :
1. **ShareGPT** (~90k conversations)
2. **OpenAssistant** (~160k messages)
3. **Dolly-15k** (instructions diverses)

Total : ~250k exemples de qualité

**Format unifié** :
```
User: [question/instruction]
Assistant: [réponse]
```

**Hyperparamètres** :
```bash
--learning_rate 5e-6          # Très bas pour ne pas oublier
--weight_decay 0.05
--warmup_steps 1000
--max_steps 25000             # ~20h sur 6 GPUs
--batch_size 2
--gradient_accumulation 16    # Total batch = 32
--max_seq_len 1024            # Conversations plus longues
```

**Avantages** :
- ✅ Meilleure qualité conversationnelle
- ✅ Multi-tour (conversations)
- ✅ Datasets variés

**Inconvénients** :
- ⚠️ Plus long à préparer
- ⚠️ Nécessite preprocessing

---

### **Option 3 : Fine-Tuning Spécialisé (1-2 semaines)** 🏆
*Pour un modèle expert dans un domaine*

**Choix du domaine** :
1. **Code** : CodeAlpaca + StackOverflow filtered
2. **Science** : ArXiv papers + PubMed
3. **Français** : Datasets francophones (Fleurs, CulturaX)

**Approche en 2 phases** :
1. **Phase 1** : Instruction tuning général (Option 2)
2. **Phase 2** : Spécialisation domaine (15k-30k steps)

**Avantages** :
- ✅ Expertise de niche
- ✅ Meilleure performance sur cas d'usage ciblé

**Inconvénients** :
- ⚠️ Long à entraîner
- ⚠️ Perte de généralité

---

## 🎯 Recommandation Personnalisée

**JE RECOMMANDE : Option 2 (Conversationnel)**

**Pourquoi ?**
1. Tu as 6x RTX 4090 → capacité suffisante
2. Tu veux un modèle chat fonctionnel
3. Compromis temps/qualité optimal
4. Datasets de qualité disponibles

**Plan d'action concret** :

### 📅 Timeline (5 jours)

**Jour 1** : Préparation données
- Télécharger ShareGPT + OpenAssistant
- Créer script de preprocessing
- Formater en prompt conversationnel

**Jour 2** : Setup fine-tuning
- Adapter train.py pour instruction tuning
- Tester sur 1 GPU (validation)
- Vérifier que ça ne crash pas

**Jour 3-4** : Training
- Lancer sur 6 GPUs
- 25k steps (~20h)
- Monitoring toutes les 2h

**Jour 5** : Évaluation
- Tests qualitatifs (comme demo_chat.py)
- Comparaison avant/après
- Itération si nécessaire

---

## 🛠️ Scripts à Créer

### 1. `prepare_instruction_data.py`
Télécharge et formate les datasets

### 2. `finetune.py`
Script de fine-tuning adapté (learning rate bas, etc.)

### 3. `compare_models.py`
Compare modèle base vs fine-tuné

### 4. `benchmark.py`
Métriques quantitatives (perplexity, BLEU, etc.)

---

## 💡 Conseils Critiques

1. **Learning Rate** : TRÈS IMPORTANT
   - Trop haut → catastrophic forgetting
   - Trop bas → pas d'apprentissage
   - **Optimal : 1e-5 à 5e-6**

2. **Gradient Accumulation**
   - Ton modèle = 124M params → ~500MB en FP16
   - 6x RTX 4090 (24GB chacun)
   - **Tu peux faire batch_size=4 par GPU = 24 total**

3. **Checkpointing**
   - Sauvegarde **TOUS les 1000 steps**
   - Garde les 5 derniers checkpoints
   - Teste régulièrement avec demo_chat.py

4. **Monitoring**
   - Loss doit descendre graduellement
   - Si loss augmente → learning rate trop haut
   - Si loss stagne → peut-être terminé

---

## 🚨 Erreurs à Éviter

1. ❌ **Ne pas partir de zéro** : Utilise ton checkpoint actuel
2. ❌ **Ne pas utiliser adam normal** : Utilise AdamW
3. ❌ **Ne pas oublier warmup** : Sinon instabilité
4. ❌ **Ne pas fine-tuner trop longtemps** : Risque d'overfitting
5. ❌ **Ne pas tester en cours de route** : Vérifie à 5k, 10k, 15k, 20k steps

---

## 📊 Métriques de Succès

**Après fine-tuning, ton modèle devrait** :

✅ Répondre "Paris" à "capitale de France"
✅ Calculer 2+2=4
✅ Maintenir le contexte conversationnel
✅ Ne plus halluciner des noms aléatoires
✅ Suivre les instructions simples

**Si échec** :
- Réduire learning rate ÷ 2
- Augmenter steps (jusqu'à 50k)
- Changer de dataset

---

## 🎬 Prêt à Commencer ?

**Tu veux que je** :
1. 🚀 Crée les scripts de fine-tuning (Option 2)
2. 📥 Prépare le téléchargement des datasets
3. 🧪 Setup un test rapide sur 1 GPU d'abord
4. 📊 Autre chose ?

**Dis-moi et on y va !** 💪


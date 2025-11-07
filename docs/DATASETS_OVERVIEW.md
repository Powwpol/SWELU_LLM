# Datasets Disponibles pour SWELU LLM

## 📊 Tailles Estimées et Sources

### 🔢 MATHÉMATIQUES

| Dataset | Taille | Tokens (estimé) | Qualité | Disponibilité |
|---------|--------|-----------------|---------|---------------|
| **MathPile** | 12.7GB | ~3.2B tokens | ⭐⭐⭐⭐⭐ | ✅ HuggingFace |
| **ArXiv Math** | ~15GB | ~3.8B tokens | ⭐⭐⭐⭐ | ✅ HuggingFace |
| **Proof-Pile-2** | 15GB | ~3.8B tokens | ⭐⭐⭐⭐⭐ | ✅ HuggingFace |
| **ProofWiki** | ~500MB | ~125M tokens | ⭐⭐⭐ | ⚠️ Scraping requis |
| **Khan Academy** | ~2GB | ~500M tokens | ⭐⭐⭐ | ⚠️ API/scraping |
| **TOTAL MATHS** | **~45GB** | **~11.4B tokens** | | |

### 🔬 LEAN & FORMAL MATH

| Dataset | Taille | Tokens (estimé) | Qualité | Disponibilité |
|---------|--------|-----------------|---------|---------------|
| **Lean Mathlib** | ~2GB | ~500M tokens | ⭐⭐⭐⭐⭐ | ✅ GitHub clone |
| **Lean 4 Examples** | ~100MB | ~25M tokens | ⭐⭐⭐⭐ | ✅ GitHub |
| **Coq Standard Lib** | ~500MB | ~125M tokens | ⭐⭐⭐⭐ | ✅ GitHub |
| **Isabelle Archive** | ~1GB | ~250M tokens | ⭐⭐⭐⭐ | ✅ Disponible |
| **TOTAL LEAN** | **~3.6GB** | **~900M tokens** | | |

### 📦 SUPPLY CHAIN (⚠️ PROBLÈME)

| Dataset | Taille | Tokens (estimé) | Qualité | Disponibilité |
|---------|--------|-----------------|---------|---------------|
| **Financial News** | ~500MB | ~125M tokens | ⭐⭐ | ✅ HuggingFace |
| **Business Reports** | ~1GB | ~250M tokens | ⭐⭐ | ⚠️ Scraping |
| **Supply Chain Blogs** | ~200MB | ~50M tokens | ⭐ | ⚠️ Scraping |
| **Kaggle SC Datasets** | ~10MB | ~2.5M tokens | ⭐ | ✅ Kaggle |
| **TOTAL SC** | **~1.7GB** | **~427M tokens** | ⚠️ Faible qualité | |

**⚠️ PROBLÈME CRITIQUE: Pas de grand dataset public Supply Chain!**

Options pour Supply Chain:
1. **Scraping web** (100-500MB possible)
2. **Génération synthétique** (qualité moyenne)
3. **Fine-tuning ultérieur** sur données propriétaires (recommandé)

## 🎯 Stratégie Recommandée

### Phase 1: Base Générale + Maths
```
Wikipedia (6GB)           ~1.5B tokens
MathPile (12.7GB)        ~3.2B tokens
Proof-Pile-2 (15GB)      ~3.8B tokens
TOTAL: 33.7GB            ~8.5B tokens
```

**Durée download:** ~2-3h (selon connexion)  
**Espace disque:** ~70GB (raw + processed)

### Phase 2: Ajout Lean
```
Lean Mathlib (2GB)       ~500M tokens
Lean Examples (100MB)    ~25M tokens
TOTAL: +2.1GB            +525M tokens
```

**Total cumulé:** ~35.8GB, ~9B tokens

### Phase 3: Supply Chain (limité)
```
Financial/Business       ~427M tokens
Web scraping custom      ~50-100M tokens
TOTAL: +1.7GB            +527M tokens
```

**TOTAL FINAL:** ~37.5GB raw, **~9.5B tokens**

## 💾 Espace Disque Requis

| Étape | Espace |
|-------|--------|
| Download raw data | ~40GB |
| Processed tokens | ~30GB |
| Checkpoints (training) | ~20GB |
| **TOTAL** | **~90GB** |

## ⏱️ Temps de Téléchargement

| Connexion | Temps (40GB) |
|-----------|--------------|
| 10 Mbps | ~9h |
| 50 Mbps | ~2h |
| 100 Mbps | ~1h |
| 1 Gbps | ~6min |

## 🚀 Commandes pour Télécharger

### Option 1: Tout télécharger (recommandé pour production)

```bash
# Télécharger tous les datasets spécialisés
python src/data/prepare_specialized_datasets.py \
  --domain all \
  --output data/specialized

# Temps estimé: 2-4h
# Espace requis: ~90GB
```

### Option 2: Test rapide (petit échantillon)

```bash
# Télécharger échantillons uniquement
python src/data/prepare_specialized_datasets.py \
  --domain all \
  --max_samples 1000 \
  --output data/specialized_test

# Temps: ~15min
# Espace: ~1GB
```

### Option 3: Par domaine

```bash
# Seulement maths
python src/data/prepare_specialized_datasets.py --domain math

# Seulement Lean
python src/data/prepare_specialized_datasets.py --domain lean

# Seulement Supply Chain (limité)
python src/data/prepare_specialized_datasets.py --domain supply_chain
```

## 📈 Comparaison avec LLMs Existants

| Modèle | Tokens Training | Notre Target |
|--------|-----------------|--------------|
| GPT-3 | 300B | ❌ Trop ambitieux |
| LLaMA-7B | 1T | ❌ Impossible |
| Pythia-410M | 300B | ❌ Trop |
| **SWELU (350M)** | **~10B** | ✅ Réaliste |

**Notre cible:** 10B tokens = raisonnable pour modèle 350M

## ⚠️ Limitations Actuelles

### Supply Chain
- **Problème:** Pas de dataset public massif
- **Impact:** Modèle sera faible sur SC
- **Solution:** Fine-tuning post-training sur données propriétaires

### Lean
- **Problème:** Syntaxe très spécifique
- **Impact:** Tokenizer GPT-2 pas optimal
- **Solution:** Considérer tokenizer custom pour Lean

### Maths
- **OK:** Beaucoup de données disponibles
- **Qualité:** Excellente (papers académiques)

## 🎯 Recommandation Finale

**Pour training complet (40h sur RTX 4090):**

1. **Télécharge maintenant** (pendant que tu dors):
   ```bash
   nohup python src/data/prepare_specialized_datasets.py --domain all > download.log 2>&1 &
   ```

2. **Utilise 80/20:**
   - 80% Maths (MathPile + Proof-Pile + ArXiv)
   - 15% Lean (Mathlib)
   - 5% Supply Chain (ce qu'on a)

3. **Prévoir fine-tuning ultérieur** sur données SC propriétaires

## 💰 Coût Storage RunPod

| Volume | Coût/mois |
|--------|-----------|
| 50GB | $5/mois |
| 100GB | $10/mois |
| 200GB | $20/mois |

**Recommandation:** Volume 100GB sur RunPod = $10/mois

---

**Tu veux lancer le téléchargement maintenant?**

```bash
# Test rapide (1000 samples, ~15min)
python src/data/prepare_specialized_datasets.py --domain all --max_samples 1000

# OU full download (2-4h)
python src/data/prepare_specialized_datasets.py --domain all
```


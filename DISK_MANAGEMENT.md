# 💾 Gestion de l'Espace Disque - Lessons Learned

## ⚠️ Problème Rencontré

**Saturation de la mémoire** causée par trop de checkpoints:
- **Avant**: 83 checkpoints × 1.2GB = **~99GB par GPU** !
- Checkpoints tous les 5,000 steps
- 6 GPUs = potentiel **600GB** de checkpoints!
- **Résultat**: Saturation disque, checkpoints corrompus

---

## ✅ Solutions Implémentées

### 1. Checkpoints Moins Fréquents
```bash
--checkpoint_every 10000  # Au lieu de 5000
```
- Réduit le nombre de checkpoints de moitié
- Toujours suffisant pour recovery

### 2. Auto-Cleanup dans train.py
```python
def save_checkpoint(self):
    # Sauvegarde normale
    ...
    # Auto-cleanup: garde seulement les 3 derniers
    self._cleanup_old_checkpoints(keep_last=3)
```

**Impact:**
- Max 3 checkpoints par GPU
- 3 × 1.2GB = **3.6GB max par GPU**
- 6 GPUs = **21.6GB max total** (au lieu de 600GB!)

### 3. Script de Nettoyage Manuel
```bash
./cleanup_old_checkpoints.sh
```
- Supprime les vieux checkpoints
- Garde les 3 derniers
- Libère l'espace immédiatement

---

## 📊 Calculs d'Espace Disque

### Configuration Actuelle (OPTIMISÉE)

```
Checkpoints par modèle:
  - Fréquence: 10,000 steps
  - Total pour 757,500 steps: 75 checkpoints générés
  - Auto-cleanup garde: 3 derniers
  - Espace max: 3 × 1.2GB = 3.6GB par GPU

Avec 6 GPUs:
  - Espace total max: 6 × 3.6GB = 21.6GB ✅
```

### Ancien Système (PROBLÉMATIQUE)

```
Checkpoints par modèle:
  - Fréquence: 5,000 steps  
  - Total pour 757,500 steps: 151 checkpoints
  - Pas de cleanup: TOUS gardés
  - Espace: 151 × 1.2GB = 181GB par GPU

Avec 6 GPUs:
  - Espace total: 6 × 181GB = 1,086GB ❌ IMPOSSIBLE!
```

---

## 🎯 Recommandations

### Pour Training Long (>500k steps)

**Option 1: Checkpoints espacés + cleanup**
```bash
--checkpoint_every 10000     # Checkpoint tous les 10k
# + auto-cleanup (déjà implémenté)
```

**Option 2: Checkpoints conditionnels**
```bash
--checkpoint_every 20000     # Encore moins fréquent
# + sauvegarder manuellement si bonne loss
```

**Option 3: Checkpoints externes**
- Sauvegarder sur stockage externe (S3, NAS)
- Garder localement seulement les 2-3 derniers
- Script de backup automatique

### Pour Training Court (<200k steps)

```bash
--checkpoint_every 5000      # OK pour court terme
# + cleanup manuel si nécessaire
```

---

## 🧹 Gestion Quotidienne

### Vérifier l'espace disque

```bash
# Espace total
df -h /

# Espace checkpoints
du -sh checkpoints/model_gpu*/

# Par GPU
for i in {0..5}; do 
    echo "GPU $i: $(du -sh checkpoints/model_gpu$i 2>/dev/null | cut -f1)"
done
```

### Cleanup Préventif

```bash
# Tous les jours ou après gros training
./cleanup_old_checkpoints.sh

# Ou commande directe
find checkpoints/ -name "model_step_*.pt" -type f | \
    sort -V | head -n -3 | xargs rm -f
```

---

## 💡 Leçons Apprises

1. **✅ Checkpoint spacing**: 10k steps optimal pour long training
2. **✅ Auto-cleanup**: Essentiel pour multi-GPU
3. **✅ Monitoring**: Surveiller l'espace disque régulièrement
4. **⚠️  Checkpoints corrompus**: Saturation disque = checkpoints invalides
5. **💾 Planifier l'espace**: 6 GPUs × 3 ckpts × 1.2GB = 22GB minimum

---

## 🔧 Maintenance Scripts

| Script | Usage |
|--------|-------|
| `cleanup_old_checkpoints.sh` | Nettoyage manuel |
| `resume_from_safe_checkpoints.sh` | Reprise avec validation |
| `track_k_evolution.sh` | Suivre SWELU sans charger tout |

---

## 📈 Espace Nécessaire (Guide)

| Training Steps | Ckpt Freq | Keep Last | Espace/GPU | 6 GPUs |
|----------------|-----------|-----------|------------|--------|
| 100k | 5k | 3 | 3.6GB | 22GB ✅ |
| 100k | 5k | all | 24GB | 144GB ⚠️ |
| 757k | 5k | 3 | 3.6GB | 22GB ✅ |
| 757k | 5k | all | 181GB | 1TB ❌ |
| 757k | 10k | 3 | 3.6GB | 22GB ✅ |
| 757k | 10k | all | 90GB | 540GB ❌ |

**Conclusion**: Toujours utiliser auto-cleanup avec multi-GPU!

---

**Statut actuel**: ✅ Optimisé (22GB max)  
**Espace libre**: 583GB  
**Configuration**: checkpoint_every=10000 + keep_last=3


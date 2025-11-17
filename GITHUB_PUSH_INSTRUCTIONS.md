# 🚀 Instructions pour Push GitHub

## ✅ État Actuel

**Commit créé avec succès !**
- Commit ID: `5c61e3d`
- Branche: `pod`
- Fichiers: 36 modifiés (3,641 insertions, 257 suppressions)
- Message: "feat: MambaSWELU with exceptional results - 4.6 loss @ 20% training"

---

## 📋 Options pour Pusher

### Option 1: Token GitHub (RECOMMANDÉ - Plus simple)

```bash
# 1. Créer un Personal Access Token
#    https://github.com/settings/tokens
#    - Click "Generate new token (classic)"
#    - Name: "SWELU_LLM"
#    - Scopes: ✓ repo (full control)
#    - Generate token et copier

# 2. Pusher avec le token
git remote set-url origin https://YOUR_TOKEN@github.com/Powwpol/SWELU_LLM.git
git push origin pod

# Ou one-liner:
git push https://YOUR_TOKEN@github.com/Powwpol/SWELU_LLM.git pod
```

### Option 2: SSH Key

```bash
# 1. Générer une clé SSH (si pas encore fait)
ssh-keygen -t ed25519 -C "powwpol@users.noreply.github.com"
cat ~/.ssh/id_ed25519.pub  # Copier cette clé

# 2. Ajouter la clé à GitHub
#    https://github.com/settings/keys
#    - Click "New SSH key"
#    - Paste la clé publique

# 3. Changer le remote pour SSH
git remote set-url origin git@github.com:Powwpol/SWELU_LLM.git

# 4. Push
git push origin pod
```

### Option 3: GitHub CLI (gh)

```bash
# 1. Installer gh
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update && sudo apt install gh

# 2. Login
gh auth login

# 3. Push
git push origin pod
```

---

## 🎯 Push Rapide (Recommandé)

**Si tu as un token GitHub:**

```bash
# Exporter le token
export GH_TOKEN=ghp_your_token_here

# Push avec le token
git push https://$GH_TOKEN@github.com/Powwpol/SWELU_LLM.git pod
```

---

## 📊 Ce qui sera pushé

### Nouveaux fichiers (36):
- ✅ Code source modifié (`src/train.py`, `src/mamba_block.py`)
- ✅ SlimPajama dataloader (`src/slimpajama_dataloader.py`)
- ✅ Scripts de lancement (15 scripts `.sh`)
- ✅ Outils de monitoring (6 scripts)
- ✅ Documentation complète (README, RESULTS, CHANGELOG, etc.)

### Exclus (.gitignore):
- ❌ Checkpoints (~1.2GB chacun)
- ❌ Logs d'entraînement
- ❌ .env (token HF)
- ❌ Cache et données temporaires

---

## 🔄 Workflow Complet

```bash
# 1. Vérifier le commit
git log --oneline -1

# 2. Push (avec token)
export GH_TOKEN=ghp_your_token
git push https://$GH_TOKEN@github.com/Powwpol/SWELU_LLM.git pod

# 3. Vérifier sur GitHub
# https://github.com/Powwpol/SWELU_LLM

# 4. (Optionnel) Merger pod → main
git checkout main
git merge pod
git push origin main
```

---

## 🎁 Bonus: Créer une Release

Après le push, sur GitHub:

1. Go to: `https://github.com/Powwpol/SWELU_LLM/releases/new`
2. Tag: `v0.1.0-training`
3. Title: "🔥 Initial Training Results - 4.6 Loss @ 20%"
4. Description: Copy from RESULTS.md
5. Attach: Example checkpoint (optional, if < 2GB)

---

## ⚠️ Note Importante

**N'inclus PAS les checkpoints dans le commit !**
- Trop gros (1.2GB chacun)
- Utilise Git LFS ou releases séparées
- Ou partage via HuggingFace Model Hub

---

## 📧 Besoin d'aide ?

Si erreur lors du push:
1. Vérifie le token: https://github.com/settings/tokens
2. Vérifie les permissions: Le token doit avoir scope `repo`
3. Vérifie la connexion: `curl https://github.com`

---

**Ton repo GitHub**: https://github.com/Powwpol/SWELU_LLM 🚀


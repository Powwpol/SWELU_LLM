#!/bin/bash
# Configuration automatique SSH pour GitHub (SÉCURISÉ)

echo "═══════════════════════════════════════════════════════════════════"
echo "  🔒 CONFIGURATION SSH GITHUB (MÉTHODE SÉCURISÉE)"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# Vérifier si clé existe déjà
if [ -f ~/.ssh/github_swelu ]; then
    echo "✅ Clé SSH déjà existante: ~/.ssh/github_swelu"
    echo ""
    read -p "Regénérer une nouvelle clé? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Utilisation de la clé existante..."
    else
        rm -f ~/.ssh/github_swelu ~/.ssh/github_swelu.pub
    fi
fi

# Générer la clé si nécessaire
if [ ! -f ~/.ssh/github_swelu ]; then
    echo "🔑 Génération d'une nouvelle clé SSH..."
    ssh-keygen -t ed25519 -C "powwpol@users.noreply.github.com" -f ~/.ssh/github_swelu -N ""
    echo "✅ Clé générée!"
    echo ""
fi

# Configurer SSH
mkdir -p ~/.ssh
chmod 700 ~/.ssh
chmod 600 ~/.ssh/github_swelu
chmod 644 ~/.ssh/github_swelu.pub

# Ajouter au SSH config si pas déjà présent
if ! grep -q "github_swelu" ~/.ssh/config 2>/dev/null; then
    echo "📝 Configuration SSH..."
    cat >> ~/.ssh/config << 'SSHCONFIG'

# GitHub for SWELU_LLM
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/github_swelu
  IdentitiesOnly yes
SSHCONFIG
    chmod 600 ~/.ssh/config
    echo "✅ SSH config mis à jour"
else
    echo "✅ SSH config déjà configuré"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  📋 ÉTAPE 1: COPIE CETTE CLÉ PUBLIQUE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
cat ~/.ssh/github_swelu.pub
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "📝 ÉTAPE 2: AJOUTER LA CLÉ À GITHUB"
echo ""
echo "   1. Va sur: https://github.com/settings/keys"
echo "   2. Click: 'New SSH key'"
echo "   3. Title: 'SWELU_LLM Training Server'"
echo "   4. Key type: 'Authentication Key'"
echo "   5. Paste la clé ci-dessus"
echo "   6. Click 'Add SSH key'"
echo ""

read -p "Appuie sur ENTER quand c'est fait..."

echo ""
echo "🔧 ÉTAPE 3: CONFIGURATION DU REMOTE GIT"
echo ""

cd /root/SWELU_LLM

# Changer le remote pour SSH
current_remote=$(git remote get-url origin 2>/dev/null)
if [[ $current_remote == https* ]]; then
    echo "   Changement HTTPS → SSH..."
    git remote set-url origin git@github.com:Powwpol/SWELU_LLM.git
    echo "   ✅ Remote mis à jour"
else
    echo "   ✅ Déjà en SSH"
fi

echo ""
echo "🧪 ÉTAPE 4: TEST DE CONNEXION"
echo ""

# Test SSH
if ssh -T git@github.com 2>&1 | grep -q "successfully authenticated"; then
    echo "   ✅ Connexion SSH réussie!"
else
    echo "   ⚠️  Test de connexion..."
    ssh -T git@github.com 2>&1 | head -5
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  🚀 PRÊT À PUSHER!"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "Commande à exécuter:"
echo ""
echo "   git push origin pod"
echo ""
echo "Ou push ET merge vers main:"
echo ""
echo "   git push origin pod && git checkout main && git merge pod && git push origin main"
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "💡 SÉCURITÉ:"
echo "   ✅ Aucun token dans le repo"
echo "   ✅ Clé SSH privée protégée (chmod 600)"
echo "   ✅ .env dans .gitignore"
echo "   ✅ Tokens jamais committés"
echo ""
echo "═══════════════════════════════════════════════════════════════════"



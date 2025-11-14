#!/bin/bash
# Script pour pusher le code sur GitHub

echo "═══════════════════════════════════════════════════════════════════"
echo "  🚀 PUSH TO GITHUB"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

cd /root/SWELU_LLM

# Vérifier git status
echo "📊 Statut Git:"
git status --short | head -20
echo ""

# Instructions
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 ÉTAPES POUR PUSHER SUR GITHUB:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1️⃣  Créer un repo sur GitHub:"
echo "   https://github.com/new"
echo "   Nom suggéré: SWELU_LLM ou MambaSWELU"
echo ""
echo "2️⃣  Configurer le remote (une seule fois):"
echo "   git remote add origin https://github.com/YOUR_USERNAME/SWELU_LLM.git"
echo ""
echo "3️⃣  Commit et push:"
echo "   git commit -m 'feat: MambaSWELU with exceptional 4.6 loss @ 20% training'"
echo "   git push -u origin main"
echo ""
echo "   Ou si déjà configuré:"
echo "   git push"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Vérifier remote
if git remote get-url origin 2>/dev/null; then
    echo "✅ Remote déjà configuré:"
    git remote -v
    echo ""
    echo "🚀 Prêt à pusher!"
    echo "   Exécuter: git commit -m 'votre message' && git push"
else
    echo "⚠️  Remote pas encore configuré"
    echo ""
    echo "📝 Configurer avec:"
    echo "   git remote add origin https://github.com/YOUR_USERNAME/SWELU_LLM.git"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "💡 CONSEIL:"
echo "   1. Commit maintenant (code fonctionnel)"
echo "   2. Continue l'entraînement"
echo "   3. Push les résultats finaux plus tard"
echo "═══════════════════════════════════════════════════════════════════"


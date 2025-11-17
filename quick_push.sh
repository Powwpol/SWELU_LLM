#!/bin/bash
# Push rapide et sécurisé vers GitHub

echo "═══════════════════════════════════════════════════════════════════"
echo "  🚀 PUSH RAPIDE VERS GITHUB"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

cd /root/SWELU_LLM

# Méthode 1: Si SSH configuré
if ssh -T git@github.com 2>&1 | grep -q "successfully authenticated"; then
    echo "✅ SSH configuré - Push avec SSH..."
    echo ""
    git push origin pod
    
elif [ -n "$GH_TOKEN" ]; then
    # Méthode 2: Si token dans l'environnement
    echo "✅ Token trouvé dans environnement..."
    echo ""
    git push https://$GH_TOKEN@github.com/Powwpol/SWELU_LLM.git pod
    
else
    # Méthode 3: Demander le token (temporaire)
    echo "🔑 Configuration requise"
    echo ""
    echo "Choix:"
    echo "  1. Setup SSH (recommandé, permanent)"
    echo "  2. Utiliser un token GitHub (temporaire)"
    echo ""
    read -p "Choix (1 ou 2): " choice
    
    if [ "$choice" = "1" ]; then
        ./setup_github_ssh.sh
else
    echo ""
        echo "📝 Entre ton token GitHub (sera utilisé UNE FOIS):"
        echo "   https://github.com/settings/tokens"
        echo ""
        read -sp "Token: " GH_TOKEN
        echo ""
        echo ""
        
        if [ -n "$GH_TOKEN" ]; then
            echo "🚀 Push en cours..."
            git push https://$GH_TOKEN@github.com/Powwpol/SWELU_LLM.git pod
            
            # Nettoyer
            unset GH_TOKEN
    echo ""
            echo "✅ Token nettoyé de la mémoire"
        else
            echo "❌ Aucun token fourni"
            exit 1
        fi
    fi
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "✅ Push terminé!"
echo ""
echo "Vérifie sur: https://github.com/Powwpol/SWELU_LLM"
echo "═══════════════════════════════════════════════════════════════════"

#!/bin/bash
# Configuration du token HuggingFace pour accéder à SlimPajama

echo "═══════════════════════════════════════════════════════════════════"
echo "  CONFIGURATION HUGGINGFACE TOKEN"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "Pour accéder à SlimPajama-627B, vous avez besoin d'un token HF."
echo ""
echo "1. Créez un compte sur https://huggingface.co/join"
echo "2. Générez un token: https://huggingface.co/settings/tokens"
echo "3. Entrez votre token ci-dessous"
echo ""
read -p "HF Token: " HF_TOKEN

if [ -z "$HF_TOKEN" ]; then
    echo "❌ Aucun token fourni"
    exit 1
fi

# Sauvegarder dans .env
echo "HF_TOKEN=$HF_TOKEN" > /root/SWELU_LLM/.env
echo ""
echo "✅ Token sauvegardé dans .env"
echo ""
echo "Pour vérifier:"
echo "  huggingface-cli whoami --token $HF_TOKEN"
echo ""
echo "═══════════════════════════════════════════════════════════════════"


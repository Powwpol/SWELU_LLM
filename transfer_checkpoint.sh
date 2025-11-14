#!/bin/bash
# Script pour transférer un checkpoint depuis Windows vers Linux
# 
# Usage depuis Windows (avec WSL ou Git Bash):
#   scp "D:\swelu\checkpoints\model_step_5000.pt" root@<SERVER_IP>:/root/SWELU_LLM/checkpoints/
#
# Ou utiliser WinSCP avec interface graphique
#
# Puis lancer l'entraînement avec:
#   python src/train.py --dataset slimpajama --resume_from_checkpoint ./checkpoints/model_step_5000.pt

echo "══════════════════════════════════════════════════════════════════"
echo "   GUIDE DE TRANSFERT DE CHECKPOINT WINDOWS → LINUX"
echo "══════════════════════════════════════════════════════════════════"
echo ""
echo "Option 1: SCP depuis Windows (avec WSL ou Git Bash)"
echo "  scp 'D:\\swelu\\checkpoints\\*.pt' root@$(hostname -I | awk '{print $1}'):/root/SWELU_LLM/checkpoints/"
echo ""
echo "Option 2: WinSCP (GUI)"
echo "  - Télécharger: https://winscp.net/"
echo "  - Connecter à: $(hostname -I | awk '{print $1}')"
echo "  - Copier vers: /root/SWELU_LLM/checkpoints/"
echo ""
echo "Option 3: Monter un partage réseau"
echo "  - Configurer SMB/NFS sur Linux"
echo "  - Accéder depuis Windows"
echo ""
echo "Checkpoints actuels dans ce répertoire:"
ls -lh /root/SWELU_LLM/checkpoints/
echo ""
echo "══════════════════════════════════════════════════════════════════"


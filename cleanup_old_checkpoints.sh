#!/bin/bash
# Nettoyer les vieux checkpoints - Garder seulement les 3 derniers

echo "═══════════════════════════════════════════════════════════════════"
echo "  🧹 NETTOYAGE DES CHECKPOINTS"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

cd /root/SWELU_LLM

for gpu in {0..5}; do
    dir="checkpoints/model_gpu$gpu"
    
    if [ ! -d "$dir" ]; then
        continue
    fi
    
    echo "GPU $gpu: $dir"
    
    # Compter checkpoints
    total=$(ls -1 $dir/model_step_*.pt 2>/dev/null | wc -l)
    echo "  Total checkpoints: $total"
    
    if [ $total -gt 3 ]; then
        # Garder seulement les 3 derniers
        to_delete=$((total - 3))
        echo "  À supprimer: $to_delete checkpoints"
        
        ls -1t $dir/model_step_*.pt | tail -n +4 | while read file; do
            size=$(du -h "$file" | cut -f1)
            echo "    Suppression: $(basename $file) ($size)"
            rm -f "$file"
        done
        
        echo "  ✅ Nettoyé!"
    else
        echo "  ✅ OK (≤3 checkpoints)"
    fi
    
    # Taille après nettoyage
    size=$(du -sh $dir 2>/dev/null | cut -f1)
    echo "  Taille finale: $size"
    echo ""
done

echo "═══════════════════════════════════════════════════════════════════"
echo "Espace libéré:"
df -h /root/SWELU_LLM/checkpoints/ | tail -1
echo "═══════════════════════════════════════════════════════════════════"


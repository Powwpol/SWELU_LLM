#!/usr/bin/env python3
"""
Préparation des datasets d'instruction pour fine-tuning conversationnel.

Datasets utilisés:
1. ShareGPT (~90k conversations)
2. OpenAssistant (~160k messages) 
3. Dolly-15k (instructions diverses)

Total: ~250k exemples de haute qualité
"""

import json
import os
from pathlib import Path
from typing import List, Dict
from datasets import load_dataset
from tqdm import tqdm
import random


class InstructionDatasetPreparator:
    """Prépare et formate les datasets d'instruction."""
    
    def __init__(self, output_dir: str = "data/instruction"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def format_conversation(self, messages: List[Dict]) -> str:
        """
        Formate une conversation au format unifié.
        
        Format:
        User: [message]
        Assistant: [réponse]
        User: [message suivant]
        Assistant: [réponse suivante]
        """
        formatted = []
        for msg in messages:
            role = msg.get('role', msg.get('from', 'unknown'))
            content = msg.get('content', msg.get('value', ''))
            
            if role in ['user', 'human', 'question']:
                formatted.append(f"User: {content}")
            elif role in ['assistant', 'gpt', 'answer']:
                formatted.append(f"Assistant: {content}")
        
        return '\n'.join(formatted)
    
    def prepare_sharegpt(self) -> List[Dict]:
        """Prépare ShareGPT dataset."""
        print("\n📥 Téléchargement ShareGPT...")
        
        try:
            # Note: ShareGPT peut nécessiter un token HuggingFace
            # Alternative: utiliser un subset public
            dataset = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train")
            
            print(f"✓ {len(dataset)} conversations ShareGPT chargées")
            
            formatted_data = []
            for item in tqdm(dataset, desc="Formatage ShareGPT"):
                if 'conversations' in item:
                    text = self.format_conversation(item['conversations'])
                    if len(text) > 50:  # Filtrer conversations trop courtes
                        formatted_data.append({
                            'text': text,
                            'source': 'sharegpt'
                        })
            
            print(f"✓ {len(formatted_data)} conversations ShareGPT formatées")
            return formatted_data
            
        except Exception as e:
            print(f"⚠️  Erreur ShareGPT: {e}")
            print("   Continuons avec les autres datasets...")
            return []
    
    def prepare_openassistant(self) -> List[Dict]:
        """Prépare OpenAssistant dataset."""
        print("\n📥 Téléchargement OpenAssistant...")
        
        try:
            dataset = load_dataset("OpenAssistant/oasst1", split="train")
            print(f"✓ {len(dataset)} messages OpenAssistant chargés")
            
            # OpenAssistant est structuré en arbre de messages
            # On reconstruit les conversations linéaires
            conversations = {}
            
            for item in dataset:
                msg_id = item['message_id']
                parent_id = item['parent_id']
                role = item['role']
                text = item['text']
                
                # Construire des paires question-réponse
                if role == 'assistant' and parent_id:
                    # Trouver la question associée
                    formatted_text = f"User: [question]\nAssistant: {text}"
                    conversations[msg_id] = {
                        'text': formatted_text,
                        'source': 'openassistant'
                    }
            
            formatted_data = list(conversations.values())
            print(f"✓ {len(formatted_data)} conversations OpenAssistant formatées")
            return formatted_data
            
        except Exception as e:
            print(f"⚠️  Erreur OpenAssistant: {e}")
            return []
    
    def prepare_dolly(self) -> List[Dict]:
        """Prépare Dolly-15k dataset."""
        print("\n📥 Téléchargement Dolly-15k...")
        
        try:
            dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
            print(f"✓ {len(dataset)} instructions Dolly chargées")
            
            formatted_data = []
            for item in tqdm(dataset, desc="Formatage Dolly"):
                instruction = item['instruction']
                context = item.get('context', '')
                response = item['response']
                
                # Format avec contexte si disponible
                if context:
                    text = f"User: {instruction}\nContext: {context}\nAssistant: {response}"
                else:
                    text = f"User: {instruction}\nAssistant: {response}"
                
                formatted_data.append({
                    'text': text,
                    'source': 'dolly'
                })
            
            print(f"✓ {len(formatted_data)} instructions Dolly formatées")
            return formatted_data
            
        except Exception as e:
            print(f"⚠️  Erreur Dolly: {e}")
            return []
    
    def prepare_alpaca(self) -> List[Dict]:
        """Prépare Alpaca dataset (fallback si autres échouent)."""
        print("\n📥 Téléchargement Alpaca...")
        
        try:
            dataset = load_dataset("tatsu-lab/alpaca", split="train")
            print(f"✓ {len(dataset)} instructions Alpaca chargées")
            
            formatted_data = []
            for item in tqdm(dataset, desc="Formatage Alpaca"):
                instruction = item['instruction']
                input_text = item.get('input', '')
                output = item['output']
                
                if input_text:
                    text = f"User: {instruction}\nInput: {input_text}\nAssistant: {output}"
                else:
                    text = f"User: {instruction}\nAssistant: {output}"
                
                formatted_data.append({
                    'text': text,
                    'source': 'alpaca'
                })
            
            print(f"✓ {len(formatted_data)} instructions Alpaca formatées")
            return formatted_data
            
        except Exception as e:
            print(f"⚠️  Erreur Alpaca: {e}")
            return []
    
    def prepare_all(self, max_samples: int = None) -> str:
        """
        Prépare tous les datasets et les combine.
        
        Args:
            max_samples: Limite le nombre total d'exemples (None = pas de limite)
            
        Returns:
            Chemin vers le fichier de données combinées
        """
        print("="*70)
        print("  📚 PRÉPARATION DES DATASETS D'INSTRUCTION")
        print("="*70)
        
        all_data = []
        
        # Essayer tous les datasets
        datasets_to_try = [
            ('Alpaca', self.prepare_alpaca),      # Plus fiable, commencer par celui-ci
            ('Dolly', self.prepare_dolly),
            ('OpenAssistant', self.prepare_openassistant),
            ('ShareGPT', self.prepare_sharegpt),
        ]
        
        for name, prepare_func in datasets_to_try:
            try:
                data = prepare_func()
                all_data.extend(data)
            except Exception as e:
                print(f"⚠️  Échec {name}: {e}")
                continue
        
        if not all_data:
            raise ValueError("❌ Aucun dataset n'a pu être chargé !")
        
        # Mélanger pour diversité
        print(f"\n🔀 Mélange de {len(all_data)} exemples...")
        random.shuffle(all_data)
        
        # Limiter si demandé
        if max_samples and len(all_data) > max_samples:
            print(f"✂️  Limitation à {max_samples} exemples")
            all_data = all_data[:max_samples]
        
        # Statistiques par source
        print("\n📊 Statistiques par source:")
        sources = {}
        for item in all_data:
            source = item['source']
            sources[source] = sources.get(source, 0) + 1
        
        for source, count in sorted(sources.items()):
            print(f"   {source}: {count:,} exemples")
        
        # Split train/validation (95/5)
        split_idx = int(len(all_data) * 0.95)
        train_data = all_data[:split_idx]
        val_data = all_data[split_idx:]
        
        print(f"\n📂 Création des fichiers:")
        print(f"   Train: {len(train_data):,} exemples")
        print(f"   Validation: {len(val_data):,} exemples")
        
        # Sauvegarder
        train_file = self.output_dir / "train.jsonl"
        val_file = self.output_dir / "val.jsonl"
        
        with open(train_file, 'w', encoding='utf-8') as f:
            for item in train_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        with open(val_file, 'w', encoding='utf-8') as f:
            for item in val_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"\n✅ Datasets sauvegardés:")
        print(f"   {train_file}")
        print(f"   {val_file}")
        
        # Afficher quelques exemples
        print(f"\n📝 Exemples de données:")
        print("="*70)
        for i, item in enumerate(random.sample(train_data, min(3, len(train_data))), 1):
            print(f"\nExemple {i} ({item['source']}):")
            print("-"*70)
            print(item['text'][:300] + "..." if len(item['text']) > 300 else item['text'])
        print("="*70)
        
        return str(train_file)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Préparer les datasets d'instruction")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/instruction",
        help="Répertoire de sortie"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Nombre maximum d'exemples (None = tous)"
    )
    
    args = parser.parse_args()
    
    preparator = InstructionDatasetPreparator(output_dir=args.output_dir)
    
    try:
        train_file = preparator.prepare_all(max_samples=args.max_samples)
        
        print("\n" + "="*70)
        print("  ✅ PRÉPARATION TERMINÉE")
        print("="*70)
        print(f"\n💡 Prochaine étape:")
        print(f"   python finetune.py --train_file {train_file}")
        print()
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


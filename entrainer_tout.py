"""Entraîne tous les modèles utiles au rapport en une seule commande (connexion requise pour MedMNIST).

    python entrainer_tout.py                      # pneumoniamnist + dermamnist, 12 époques
    python entrainer_tout.py pneumoniamnist 20    # un seul jeu, 20 époques
"""
import sys
from core import train, models, data

datasets = [sys.argv[1]] if len(sys.argv) > 1 else ["pneumoniamnist", "dermamnist"]
epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 12
for ds in datasets:
    assert ds in data.DATASETS, f"jeu inconnu : {ds}"
    for key in models.KEYS:
        print(f"\n=== {ds} · {models.KEYS[key]} ({epochs} époques) ===")
        h = train.train(ds, key, epochs=epochs, progress=lambda p, m: print("  ", m))
        print(f"  -> précision test : {h[-1]['test_acc']:.1%}")

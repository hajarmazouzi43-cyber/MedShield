"""Entraînement (standard ou adversarial) et gestion des modèles sauvegardés."""
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from . import data, models, attacks

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def path(dataset, key):
    return MODELS_DIR / f"{dataset}_{key}.pt"


def exists(dataset, key):
    return path(dataset, key).exists()


@torch.no_grad()
def accuracy(model, X, y, bs=512):
    model.eval()
    ok = 0
    for i in range(0, len(X), bs):
        ok += (model(torch.from_numpy(X[i:i + bs])).argmax(1).numpy() == y[i:i + bs]).sum()
    return float(ok) / len(X)


def train(dataset, key, epochs=8, lr=2e-3, bs=128, progress=None, seed=0, adv_eps=8 / 255):
    """key : 'vit', 'cnn', 'vit_adv' ou 'cnn_adv'.
    Pour les clés *_adv : adversarial training (min-max, éq. 3 de l'article) avec PGD à 3 pas,
    perte = 0.5·perte(propre) + 0.5·perte(adversariale), ε tiré dans [0, adv_eps]."""
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr, classes = data.load(dataset, "train")
    Xte, yte, _ = data.load(dataset, "test")
    model = models.build(key, Xtr.shape[1], len(classes))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * ((len(Xtr) + bs - 1) // bs))
    adv = models.is_adv(key)
    hist, t0 = [], time.time()
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(len(Xtr))
        tot = 0.0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            xb = torch.from_numpy(Xtr[idx]); yb = torch.from_numpy(ytr[idx])
            sh = np.random.randint(-2, 3, 2)  # petit décalage aléatoire (augmentation)
            xb = torch.roll(xb, tuple(int(s) for s in sh), (2, 3))
            if adv:
                e = float(np.random.uniform(0, adv_eps))
                xa = attacks.pgd(model, xb, yb, e, steps=3) if e > 0 else xb
                loss = 0.5 * F.cross_entropy(model(xb), yb) + 0.5 * F.cross_entropy(model(xa), yb)
            else:
                loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(idx)
        acc = accuracy(model, Xte, yte)
        hist.append({"epoch": ep + 1, "loss": tot / len(Xtr), "test_acc": acc})
        if progress:
            progress((ep + 1) / epochs, f"Époque {ep + 1}/{epochs} · perte {tot / len(Xtr):.3f} · précision test {acc:.1%}")
    MODELS_DIR.mkdir(exist_ok=True)
    torch.save({"state": model.state_dict(), "arch": key, "dataset": dataset, "classes": classes,
                "in_ch": Xtr.shape[1], "history": hist, "seconds": time.time() - t0}, path(dataset, key))
    return hist


def load_model(dataset, key):
    ck = torch.load(path(dataset, key), map_location="cpu", weights_only=False)
    m = models.build(key, ck["in_ch"], len(ck["classes"]))
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck

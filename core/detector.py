"""Détecteur d'entrées adversariales (algorithme 2 de l'article).

Trois signaux, calculés sur l'image reçue x' :
  s : score d'anomalie des features (distance de Mahalanobis aux statistiques des images propres)
  c : cohérence de l'explication = corrélation entre la carte XAI de x' et celle de x' purifiée
  p : décalage de prédiction après purification (distance L1 entre probabilités)
Décision :  ADVERSARIAL si  s > τs  ou  c < τe  ou  p > τp.
Les seuils sont calibrés sur des images PROPRES de validation pour viser un FPR cible.
"""
import time
import numpy as np
import torch
from . import xai, defense, data, train


@torch.no_grad()
def features(model, x, bs=256):
    """Représentation interne : token CLS (ViT) ou moyenne spatiale de la dernière conv (CNN)."""
    model.eval()
    store, out = {}, []
    layer = model.norm if model.kind == "vit" else model.last
    h = layer.register_forward_hook(lambda m, i, o: store.__setitem__("o", o))
    try:
        for i in range(0, len(x), bs):
            model(x[i:i + bs])
            o = store["o"]
            out.append((o[:, 0] if model.kind == "vit" else o.mean((2, 3))).cpu())
    finally:
        h.remove()
    return torch.cat(out).numpy().astype(np.float64)


class Detector:
    def __init__(self, purifier="gauss", target_fpr=0.05):
        self.purifier, self.target_fpr = purifier, target_fpr

    # ---- signaux
    def scores(self, model, x):
        f = features(model, x)
        d = f - self.mu
        s = np.sqrt(np.maximum((d @ self.P * d).sum(1), 0))
        xp = defense.purify(x, self.purifier)
        c = xai.corr_batch(xai.explain_batch(model, x), xai.explain_batch(model, xp))
        with torch.no_grad():
            p = (model(x).softmax(1) - model(xp).softmax(1)).abs().sum(1).numpy()
        return s, c, p

    # ---- calibration
    def fit(self, model, Xfit, Xval):
        F = features(model, Xfit)
        self.mu = F.mean(0)
        cov = np.cov(F.T) + 1e-3 * np.eye(F.shape[1])
        self.P = np.linalg.inv(cov)
        sv, cv, pv = self.scores(model, Xval)
        for t in np.linspace(0.05, 0.0005, 100):      # plus petite queue donnant FPR <= cible
            self.ts, self.te, self.tp = np.quantile(sv, 1 - t), np.quantile(cv, t), np.quantile(pv, 1 - t)
            if self._flag(sv, cv, pv).mean() <= self.target_fpr:
                break
        return self

    def _flag(self, s, c, p):
        return (s > self.ts) | (c < self.te) | (p > self.tp)

    # ---- détection
    def detect(self, model, x):
        t0 = time.perf_counter()
        s, c, p = self.scores(model, x)
        flag = self._flag(s, c, p)
        ms = (time.perf_counter() - t0) * 1000 / len(x)
        return {"s": s, "c": c, "p": p, "flag": flag, "ms": ms}


_CACHE = {}


def get(dataset, key, n_fit=2000, n_val=500):
    """Détecteur calibré pour un modèle sauvegardé (mis en cache, invalidé si le modèle change)."""
    stamp = train.path(dataset, key).stat().st_mtime
    k = (dataset, key, stamp)
    if k not in _CACHE:
        model, _ = train.load_model(dataset, key)
        Xtr, _, _ = data.load(dataset, "train")
        Xva, _, _ = data.load(dataset, "val")
        _CACHE[k] = Detector().fit(model, torch.from_numpy(Xtr[:n_fit]), torch.from_numpy(Xva[:n_val]))
    return _CACHE[k]

"""Benchmark : précision sous attaque, ASR, détection (detection rate, FPR, F1, temps) et purification."""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve, roc_auc_score
from . import data, train, attacks, detector, defense, models


def _sample(ds, n):
    X, y, classes = data.load(ds, "test")
    sel = np.random.default_rng(0).choice(len(y), min(n, len(y)), replace=False)
    return torch.from_numpy(X[sel]), torch.from_numpy(y[sel]), classes


def run(ds, keys, atks, eps_list, n=200, steps=10, progress=None):
    """Retourne un DataFrame : une ligne par (modèle, attaque, ε). ε en unités /255."""
    xs, ys, _ = _sample(ds, n)
    rows, total, done = [], len(keys) * len(atks) * len(eps_list), 0
    for key in keys:
        model, _ = train.load_model(ds, key)
        det = detector.get(ds, key)
        with torch.no_grad():
            pc = model(xs).argmax(1)
        ok = pc == ys
        clean = det.detect(model, xs)
        fpr = float(clean["flag"].mean())
        for atk in atks:
            for e in eps_list:
                xa = attacks.run(atk, model, xs, ys, e / 255, steps) if e > 0 else xs
                with torch.no_grad():
                    pa = model(xa).argmax(1)
                    pp = model(defense.purify(xa, det.purifier)).argmax(1)
                r = det.detect(model, xa)
                tp, fp = int(r["flag"].sum()), int(clean["flag"].sum())
                prec = tp / (tp + fp) if tp + fp else 0.0
                rec = tp / len(xs)
                rows.append({
                    "Modèle": models.KEYS[key], "Attaque": atk, "ε (/255)": e,
                    "Précision propre": float(ok.float().mean()),
                    "Précision sous attaque": float((pa == ys).float().mean()),
                    "ASR": float(((pa != ys) & ok).sum() / max(int(ok.sum()), 1)),
                    "Précision après purification": float((pp == ys).float().mean()),
                    "Detection rate": rec, "FPR": fpr,
                    "F1 détection": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                    "Temps détection (ms/img)": r["ms"],
                })
                done += 1
                if progress:
                    progress(done / total, f"{models.KEYS[key]} · {atk} · ε={e}")
    return pd.DataFrame(rows)


def roc(ds, key, atk, eps255, n=300, steps=10):
    """Courbes ROC (propre vs attaqué) pour chaque signal du détecteur + AUC."""
    xs, ys, _ = _sample(ds, n)
    model, _ = train.load_model(ds, key)
    det = detector.get(ds, key)
    xa = attacks.run(atk, model, xs, ys, eps255 / 255, steps)
    c, a = det.detect(model, xs), det.detect(model, xa)
    lab = np.r_[np.zeros(len(xs)), np.ones(len(xs))]
    out = {}
    for name, sign in [("s", 1), ("c", -1), ("p", 1)]:
        sc = np.r_[sign * c[name], sign * a[name]]
        fpr, tpr, _ = roc_curve(lab, sc)
        out[name] = (fpr, tpr, float(roc_auc_score(lab, sc)))
    return out

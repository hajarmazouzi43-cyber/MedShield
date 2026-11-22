"""Chargement des données : MedMNIST (vraies images médicales) ou jeu synthétique de secours."""
import numpy as np

DATASETS = {
    "pneumoniamnist": "Radio thorax : pneumonie (MedMNIST)",
    "dermamnist": "Dermatoscopie : lésions de la peau (MedMNIST)",
    "synthetic": "Démo synthétique (aucun téléchargement)",
}


def _synthetic(n, seed):
    """Fausses radios 28x28 : 2 poumons ; classe 1 = opacités (lésions) plus claires."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:28, 0:28].astype(np.float32)
    X = np.zeros((n, 1, 28, 28), np.float32)
    y = rng.integers(0, 2, n)
    for i in range(n):
        img = 0.15 + 0.05 * rng.standard_normal((28, 28))
        for cx in (9, 19):
            img += 0.35 * np.exp(-(((xx - cx) / 5.0) ** 2 + ((yy - 14) / 9.0) ** 2))
        if y[i] == 1:
            for _ in range(rng.integers(1, 4)):
                cx, cy = rng.choice([8, 9, 10, 18, 19, 20]), rng.integers(8, 20)
                s = rng.uniform(1.6, 3.0)
                img += rng.uniform(0.35, 0.6) * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s)))
        X[i, 0] = np.clip(img + 0.03 * rng.standard_normal((28, 28)), 0, 1)
    return X, y.astype(np.int64)


def load(name, split="train"):
    """Retourne X (N,C,H,W) float32 dans [0,1], y (N,), noms de classes."""
    if name == "synthetic":
        n = {"train": 3000, "val": 500, "test": 600}[split]
        X, y = _synthetic(n, {"train": 1, "val": 2, "test": 3}[split])
        return X, y, ["Normal", "Anomalie (lésion)"]
    import medmnist
    from medmnist import INFO
    info = INFO[name]
    ds = getattr(medmnist, info["python_class"])(split=split, download=True)
    X = ds.imgs.astype(np.float32) / 255.0
    X = X[:, None] if X.ndim == 3 else X.transpose(0, 3, 1, 2)
    y = ds.labels.reshape(-1).astype(np.int64)
    classes = [info["label"][str(i)] for i in range(len(info["label"]))]
    return X, y, classes


def channels(name):
    return 3 if name == "dermamnist" else 1

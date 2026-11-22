"""Explicabilité : attention rollout (ViT) et Grad-CAM (CNN), en version par lots."""
import numpy as np
import torch
import torch.nn.functional as F


def _norm_b(m):
    """Normalise chaque carte (B,H,W) dans [0,1]."""
    lo = m.amin((1, 2), keepdim=True)
    m = m - lo
    return m / (m.amax((1, 2), keepdim=True) + 1e-8)


@torch.no_grad()
def _rollout_b(model, x):
    model.eval(); model(x)
    n = model.blocks[0].attn.attn.shape[-1]
    eye = torch.eye(n)
    r = eye.expand(len(x), n, n)
    for b in model.blocks:
        a = b.attn.attn.mean(1) + eye
        a = a / a.sum(-1, keepdim=True)
        r = a @ r
    g = model.grid
    return r[:, 0, 1:].reshape(-1, g, g)


def _gradcam_b(model, x, target):
    store = {}

    def fwd(_m, _i, o):
        store["a"] = o
        o.register_hook(lambda g: store.__setitem__("g", g))

    h = model.last.register_forward_hook(fwd)
    model.eval()
    out = model(x.clone().requires_grad_(True))
    model.zero_grad()
    out.gather(1, target[:, None]).sum().backward()
    h.remove()
    w = store["g"].mean((2, 3), keepdim=True)
    return F.relu((w * store["a"]).sum(1)).detach()


def explain_batch(model, x, target=None):
    """x : (B,C,H,W) -> cartes (B,H,W) normalisées dans [0,1] (tenseur)."""
    if target is None:
        with torch.no_grad():
            target = model(x).argmax(1)
    m = _rollout_b(model, x) if model.kind == "vit" else _gradcam_b(model, x, target)
    m = F.interpolate(m[:, None], size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    return _norm_b(m)


def explain(model, x, target=None):
    """x : (1,C,H,W). Retourne une carte (H,W) numpy dans [0,1]."""
    t = None if target is None else torch.tensor([int(target)])
    return explain_batch(model, x, t)[0].numpy()


def shift(m1, m2, top=0.2):
    """Écart entre deux cartes : corrélation et recouvrement (IoU) des zones les plus actives."""
    corr = float(np.corrcoef(m1.ravel(), m2.ravel())[0, 1]) if m1.std() > 0 and m2.std() > 0 else 0.0
    k = max(1, int(top * m1.size))
    a, b = set(np.argsort(m1.ravel())[-k:]), set(np.argsort(m2.ravel())[-k:])
    return corr, len(a & b) / len(a | b)


def corr_batch(a, b):
    """Corrélation de Pearson entre deux lots de cartes (B,H,W) -> (B,) numpy."""
    a = a.flatten(1); b = b.flatten(1)
    a = a - a.mean(1, keepdim=True); b = b - b.mean(1, keepdim=True)
    return ((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-8)).numpy()

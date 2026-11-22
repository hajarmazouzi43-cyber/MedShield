"""Atténuation (volet « IPS ») : purification d'entrée avant reclassification."""
import torch
import torch.nn.functional as F

PURIFIERS = {"gauss": "Flou gaussien 3×3", "median": "Filtre médian 3×3", "none": "Aucune"}


def _gauss_kernel(c, sigma=0.8):
    ax = torch.arange(3, dtype=torch.float32) - 1
    k1 = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    k = (k1[:, None] * k1[None, :])
    k = k / k.sum()
    return k.expand(c, 1, 3, 3).contiguous()


def purify(x, method="gauss"):
    """x : (B,C,H,W) dans [0,1]. Supprime une part du bruit adversarial haute fréquence."""
    if method == "none":
        return x
    c = x.shape[1]
    xp = F.pad(x, (1, 1, 1, 1), mode="reflect")
    if method == "gauss":
        return F.conv2d(xp, _gauss_kernel(c), groups=c)
    if method == "median":
        patches = xp.unfold(2, 3, 1).unfold(3, 3, 1)          # (B,C,H,W,3,3)
        return patches.reshape(*patches.shape[:4], 9).median(-1).values
    raise ValueError(method)

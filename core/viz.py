"""Utilitaires d'affichage et de métriques d'image."""
import numpy as np
from PIL import Image
import matplotlib

SIZE = 224


def to_img(x):
    """Tenseur/array (C,H,W) dans [0,1] -> image PIL agrandie."""
    a = np.asarray(x)
    a = a[0] if a.shape[0] == 1 else a.transpose(1, 2, 0)
    im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    return im.resize((SIZE, SIZE), Image.NEAREST)


def overlay(x, cmap_map, alpha=0.5):
    base = np.asarray(to_img(x).convert("RGB")).astype(np.float32) / 255
    cm = Image.fromarray((cmap_map * 255).astype(np.uint8)).resize((SIZE, SIZE), Image.BICUBIC)
    heat = matplotlib.colormaps["inferno"](np.asarray(cm) / 255.0)[..., :3]
    return Image.fromarray((np.clip((1 - alpha) * base + alpha * heat, 0, 1) * 255).astype(np.uint8))


def perturbation_img(x, xa, eps):
    d = (np.asarray(xa) - np.asarray(x)) / (2 * max(eps, 1e-8)) + 0.5
    return to_img(np.clip(d, 0, 1))


def psnr(x, xa):
    mse = float(np.mean((np.asarray(x) - np.asarray(xa)) ** 2))
    return 99.0 if mse < 1e-12 else 10 * np.log10(1 / mse)

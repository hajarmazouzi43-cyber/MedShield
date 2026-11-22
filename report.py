"""Rapport PDF d'une simulation (une page A4 paysage, généré avec matplotlib)."""
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _show(ax, a, title, cmap=None):
    if hasattr(a, "convert"):          # image PIL (H,W) ou (H,W,3)
        a = np.asarray(a).astype(np.float32) / 255
    else:                              # tenseur numpy (C,H,W)
        a = np.asarray(a)
        a = a[0] if a.ndim == 3 and a.shape[0] == 1 else (a.transpose(1, 2, 0) if a.ndim == 3 else a)
    ax.imshow(a, cmap=cmap or ("gray" if a.ndim == 2 else None), vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=9); ax.axis("off")


def make_pdf(info, x0, xa, xp, m0, ma, pert):
    """info : dict de textes/valeurs ; x0, xa, xp : images (C,H,W) ; m0, ma : cartes XAI ; pert : perturbation amplifiée."""
    fig = plt.figure(figsize=(11.7, 8.3))
    fig.suptitle("MedShield Lab — Rapport de simulation", fontsize=15, fontweight="bold")
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.1], hspace=0.45)
    for j, (im, t) in enumerate([(x0, "Image originale"), (xa, "Image attaquée"), (pert, "Perturbation (amplifiée)"), (xp, "Image purifiée")]):
        _show(fig.add_subplot(gs[0, j]), im, t, None)
    for j, (im, t) in enumerate([(m0, f"{info['xai']} — propre"), (ma, f"{info['xai']} — attaquée"), (np.abs(m0 - ma), "Écart entre cartes")]):
        ax = fig.add_subplot(gs[1, j]); ax.imshow(im, cmap="inferno", vmin=0, vmax=1); ax.set_title(t, fontsize=9); ax.axis("off")
    ax = fig.add_subplot(gs[1, 3]); ax.axis("off")
    ax.text(0, 1, "Probabilités", fontsize=9, fontweight="bold", va="top")
    ax.text(0, 0.85, info["probs"], fontsize=8.5, va="top", family="monospace")
    ax = fig.add_subplot(gs[2, :]); ax.axis("off")
    ax.text(0, 1, "\n".join(info["lines"]), fontsize=10, va="top", family="monospace")
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    return buf.getvalue()

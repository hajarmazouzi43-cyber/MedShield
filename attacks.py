"""Attaques adversariales (espace pixel [0,1]) : FGSM, PGD, bruit aléatoire (témoin)."""
import torch
import torch.nn.functional as F


def _grad(model, x, y):
    x = x.clone().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    return torch.autograd.grad(loss, x)[0]


def fgsm(model, x, y, eps):
    return (x + eps * _grad(model, x, y).sign()).clamp(0, 1).detach()


def pgd(model, x, y, eps, alpha=None, steps=10, random_start=True):
    alpha = alpha or 2.5 * eps / steps
    xa = (x + (torch.empty_like(x).uniform_(-eps, eps) if random_start else 0)).clamp(0, 1)
    for _ in range(steps):
        xa = xa + alpha * _grad(model, xa, y).sign()
        xa = torch.min(torch.max(xa, x - eps), x + eps).clamp(0, 1).detach()
    return xa


def noise(model, x, y, eps):
    return (x + eps * torch.empty_like(x).uniform_(-1, 1).sign()).clamp(0, 1)


ATTACKS = {"FGSM": "FGSM (1 pas, rapide)", "PGD": "PGD (itératif, plus fort)", "Bruit": "Bruit aléatoire (témoin)"}


def run(name, model, x, y, eps, steps=10):
    model.eval()
    if name == "FGSM":
        return fgsm(model, x, y, eps)
    if name == "PGD":
        return pgd(model, x, y, eps, steps=steps)
    return noise(model, x, y, eps)

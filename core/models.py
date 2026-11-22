"""Modèles : mini Vision Transformer (ViT) et CNN de référence, en PyTorch pur."""
import math
import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h, self.qkv, self.proj, self.attn = heads, nn.Linear(dim, dim * 3), nn.Linear(dim, dim), None

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        a = ((q @ k.transpose(-2, -1)) * (C // self.h) ** -0.5).softmax(-1)
        self.attn = a  # conservé pour l'attention rollout (XAI)
        return self.proj((a @ v).transpose(1, 2).reshape(B, N, C))


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class TinyViT(nn.Module):
    kind = "vit"

    def __init__(self, in_ch, n_classes, img=28, patch=4, dim=64, depth=4, heads=4):
        super().__init__()
        self.patch, self.grid = patch, img // patch
        self.embed = nn.Conv2d(in_ch, dim, patch, patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, self.grid ** 2 + 1, dim) * 0.02)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm, self.head = nn.LayerNorm(dim), nn.Linear(dim, n_classes)
        self.register_buffer("mean", torch.full((1, in_ch, 1, 1), 0.5))
        self.register_buffer("std", torch.full((1, in_ch, 1, 1), 0.25))

    def forward(self, x):  # x dans [0,1] : la normalisation est interne, donc epsilon est en unités de pixel
        x = self.embed((x - self.mean) / self.std).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], 1) + self.pos
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x)[:, 0])


class SmallCNN(nn.Module):
    kind = "cnn"

    def __init__(self, in_ch, n_classes, **_):
        super().__init__()
        self.f1 = nn.Sequential(nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.f2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.last = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())  # couche cible de Grad-CAM
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, n_classes))
        self.register_buffer("mean", torch.full((1, in_ch, 1, 1), 0.5))
        self.register_buffer("std", torch.full((1, in_ch, 1, 1), 0.25))

    def forward(self, x):
        return self.head(self.last(self.f2(self.f1((x - self.mean) / self.std))))


ARCHS = {"vit": ("Vision Transformer (ViT)", TinyViT), "cnn": ("CNN (référence)", SmallCNN)}

# Clés de modèles sauvegardés : architecture standard ou entraînée avec adversarial training (« _adv »)
KEYS = {"vit": "ViT", "cnn": "CNN", "vit_adv": "ViT + adversarial training", "cnn_adv": "CNN + adversarial training"}


def base(key):
    return key.split("_")[0]


def is_adv(key):
    return key.endswith("_adv")


def build(key, in_ch, n_classes):
    return ARCHS[base(key)][1](in_ch, n_classes)

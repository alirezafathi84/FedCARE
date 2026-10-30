"""Backbones. Every model exposes ``feat_dim`` and ``forward(x, return_features)``,
and keeps its linear classifier in ``self.head`` (needed for class-aware head
aggregation and classifier re-calibration)."""
from __future__ import annotations

import torch
import torch.nn as nn


def _cbr(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class CompactCNN(nn.Module):
    """Three-block CNN for 28x28 inputs (24k parameters at width=16).

    Sized for resource-constrained edge clients; ``width`` scales all layers.
    """

    def __init__(self, in_ch: int = 3, num_classes: int = 9, width: int = 16):
        super().__init__()
        w = width
        self.backbone = nn.Sequential(
            _cbr(in_ch, w), nn.MaxPool2d(2),
            _cbr(w, 2 * w), nn.MaxPool2d(2),
            _cbr(2 * w, 4 * w), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.feat_dim = 4 * w
        self.head = nn.Linear(4 * w, num_classes)

    def forward(self, x, return_features: bool = False):
        h = self.backbone(x)
        z = self.head(h)
        return (h, z) if return_features else z


class _Basic(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.sc = nn.Sequential()
        if stride != 1 or cin != cout:
            self.sc = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = torch.relu(self.b1(self.c1(x)))
        return torch.relu(self.b2(self.c2(out)) + self.sc(x))


class ResNet18(nn.Module):
    """CIFAR-style ResNet-18 (3x3 stem, no max-pool) for 28x28 inputs; for GPU runs."""

    def __init__(self, in_ch: int = 3, num_classes: int = 9, width: int = 64):
        super().__init__()
        w = width
        layers = [nn.Conv2d(in_ch, w, 3, 1, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True)]
        cin = w
        for cout, stride in [(w, 1), (2 * w, 2), (4 * w, 2), (8 * w, 2)]:
            layers += [_Basic(cin, cout, stride), _Basic(cout, cout, 1)]
            cin = cout
        layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
        self.backbone = nn.Sequential(*layers)
        self.feat_dim = 8 * w
        self.head = nn.Linear(8 * w, num_classes)

    def forward(self, x, return_features: bool = False):
        h = self.backbone(x)
        z = self.head(h)
        return (h, z) if return_features else z


def make_model(name: str, in_ch: int, num_classes: int, width: int | None = None) -> nn.Module:
    if name == "cnn":
        return CompactCNN(in_ch, num_classes, width or 16)
    if name == "resnet18":
        return ResNet18(in_ch, num_classes, width or 64)
    raise ValueError(f"Unknown model {name!r}")

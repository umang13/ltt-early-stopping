"""Classifier heads shared across backbones."""

from __future__ import annotations

import torch
import torch.nn as nn


class BinaryHead(nn.Module):
    """Shallow fully-connected head for csPCa classification."""

    def __init__(self, in_features: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple
        return self.net(x).squeeze(-1)


class CSPCaHead(nn.Module):
    """Two-layer MLP head tuned for csPCa logits."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple
        return self.net(x).squeeze(-1)

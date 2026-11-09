"""ResNet3D encoder helpers built on torchvision video backbones."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torchvision.models.video import (
    r3d_18,
    r2plus1d_18,
    mc3_18,
    R3D_18_Weights,
    R2Plus1D_18_Weights,
    MC3_18_Weights,
)


_BACKBONE_REGISTRY = {
    "r3d_18": (r3d_18, R3D_18_Weights.KINETICS400_V1),
    "mc3_18": (mc3_18, MC3_18_Weights.KINETICS400_V1),
    "r2plus1d_18": (r2plus1d_18, R2Plus1D_18_Weights.KINETICS400_V1),
}


class EncoderWrapper(nn.Module):
    """Expose both pooled and feature-map outputs from a torchvision video encoder."""

    def __init__(self, encoder_module: nn.Module, feature_dim: int) -> None:
        super().__init__()
        self.encoder = encoder_module
        self.out_dim = feature_dim

    def forward_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple
        feats = self.forward_feature_map(x)
        return feats.mean(dim=(2, 3, 4))


def build_resnet3d_encoder(backbone_name: str, in_channels: int, *, pretrained: bool = False):
    """Return an encoder module and output dim for the requested backbone."""

    if backbone_name not in _BACKBONE_REGISTRY:
        raise ValueError(f"Unsupported backbone '{backbone_name}'. Choose from {sorted(_BACKBONE_REGISTRY.keys())}")

    constructor, default_weights = _BACKBONE_REGISTRY[backbone_name]
    weights = default_weights if pretrained else None
    model = constructor(weights=weights)
    orig_conv1 = model.stem[0]
    new_conv1 = nn.Conv3d(
        in_channels,
        orig_conv1.out_channels,
        kernel_size=orig_conv1.kernel_size,
        stride=orig_conv1.stride,
        padding=orig_conv1.padding,
        bias=False,
    )
    with torch.no_grad():
        if in_channels == orig_conv1.in_channels:
            new_conv1.weight.copy_(orig_conv1.weight)
        elif in_channels < orig_conv1.in_channels:
            new_conv1.weight[:, :in_channels].copy_(orig_conv1.weight[:, :in_channels])
        else:
            repeat_factor = math.ceil(in_channels / orig_conv1.in_channels)
            expanded = orig_conv1.weight.repeat(1, repeat_factor, 1, 1, 1)
            expanded = expanded[:, :in_channels]
            scale = orig_conv1.in_channels / in_channels
            new_conv1.weight.copy_(expanded * scale)
    model.stem[0] = new_conv1

    encoder = nn.Sequential(
        model.stem,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
    )
    out_dim = model.fc.in_features

    return EncoderWrapper(encoder, out_dim), out_dim

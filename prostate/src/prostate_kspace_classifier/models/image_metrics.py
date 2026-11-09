"""Image-domain classifiers built on ResNet3D encoders."""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn

from .heads import CSPCaHead
from .resnet3d import build_resnet3d_encoder

MODEL_VARIANT_SINGLE_STREAM = "single_stream"
MODEL_VARIANT_T2_RECON_DWI_FUSION = "t2_recon_dwi_fusion"
MODEL_VARIANT_CHOICES = (
    MODEL_VARIANT_SINGLE_STREAM,
    MODEL_VARIANT_T2_RECON_DWI_FUSION,
)
FUSION_ARCHITECTURE_SHARED_BOTTLENECK_V1 = "shared_bottleneck_v1"


class ProstateImageClassifier(nn.Module):
    """Single-stream image classifier for stacked inputs or legacy dual-encoder DWI."""

    def __init__(
        self,
        input_keys: Sequence[str],
        hidden_dim: int = 256,
        dropout: float = 0.3,
        backbone: str = "r3d_18",
        backbone_pretrained: bool = False,
        dual_encoder: bool = False,
        freeze_backbone: bool = False,
        enable_gleason_head: bool = False,
    ) -> None:
        super().__init__()
        self.input_keys = list(input_keys)
        if not self.input_keys:
            raise ValueError("input_keys must include at least one metric name")
        if dual_encoder and len(self.input_keys) != 2:
            raise ValueError("dual_encoder requires exactly two input keys")
        self.dual_encoder = dual_encoder
        num_modalities = len(self.input_keys)

        if self.dual_encoder:
            self.encoder_a, a_dim = build_resnet3d_encoder(
                backbone_name=backbone,
                in_channels=1,
                pretrained=backbone_pretrained,
            )
            self.encoder_b, b_dim = build_resnet3d_encoder(
                backbone_name=backbone,
                in_channels=1,
                pretrained=backbone_pretrained,
            )
            fused_dim = a_dim + b_dim
            if freeze_backbone:
                self._freeze_module(self.encoder_a)
                self._freeze_module(self.encoder_b)
        else:
            self.encoder3d, out_dim = build_resnet3d_encoder(
                backbone_name=backbone,
                in_channels=num_modalities,
                pretrained=backbone_pretrained,
            )
            fused_dim = out_dim
            if freeze_backbone:
                self._freeze_module(self.encoder3d)

        self.head = CSPCaHead(fused_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.enable_gleason_head = enable_gleason_head
        self.gleason_head = (
            CSPCaHead(fused_dim, hidden_dim=hidden_dim, dropout=dropout)
            if enable_gleason_head
            else None
        )

    def _freeze_module(self, module: nn.Module) -> None:
        for param in module.parameters():
            param.requires_grad = False

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        vols = [batch[key] for key in self.input_keys]

        if self.dual_encoder:
            feats = []
            feats.append(self.encoder_a(vols[0]))
            feats.append(self.encoder_b(vols[1]))
            fused = torch.cat(feats, dim=1)
        else:
            x = torch.cat(vols, dim=1)
            fused = self.encoder3d(x)
        pirads_logits = self.head(fused)
        if self.gleason_head is None:
            return pirads_logits
        return {
            "pirads": pirads_logits,
            "gleason": self.gleason_head(fused),
        }


class T2ReconDWIFusionClassifier(nn.Module):
    """Fuse pooled features from DICOM T2 and recon ADC/b1500 branches."""

    def __init__(
        self,
        *,
        t2_input_key: str = "axt2",
        dwi_input_keys: Sequence[str] = ("adc", "b1500"),
        hidden_dim: int = 256,
        dropout: float = 0.3,
        backbone: str = "r3d_18",
        backbone_pretrained: bool = False,
        enable_gleason_head: bool = False,
        fusion_gate: bool = False,
    ) -> None:
        super().__init__()
        self.t2_input_key = t2_input_key
        self.dwi_input_keys = list(dwi_input_keys)
        if len(self.dwi_input_keys) != 2:
            raise ValueError(
                "T2ReconDWIFusionClassifier requires exactly two DWI input keys"
            )

        self.t2_encoder, t2_dim = build_resnet3d_encoder(
            backbone_name=backbone,
            in_channels=1,
            pretrained=backbone_pretrained,
        )
        self.dwi_encoder, dwi_dim = build_resnet3d_encoder(
            backbone_name=backbone,
            in_channels=len(self.dwi_input_keys),
            pretrained=backbone_pretrained,
        )
        fused_dim = t2_dim + dwi_dim
        bottleneck_dim = fused_dim // 2

        self.fusion_architecture = FUSION_ARCHITECTURE_SHARED_BOTTLENECK_V1
        self.fusion_gate = fusion_gate
        self.t2_feature_dim = t2_dim
        self.dwi_feature_dim = dwi_dim
        self.fused_dim = fused_dim
        self.bottleneck_dim = bottleneck_dim
        self.hidden_dim = hidden_dim

        self.gate = (
            nn.Sequential(nn.Linear(fused_dim, fused_dim), nn.Sigmoid())
            if fusion_gate
            else None
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fused_dim, bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.head = CSPCaHead(hidden_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.gleason_head = (
            CSPCaHead(hidden_dim, hidden_dim=hidden_dim, dropout=dropout)
            if enable_gleason_head
            else None
        )

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        t2_feats = self.t2_encoder(batch[self.t2_input_key])
        dwi = torch.cat([batch[key] for key in self.dwi_input_keys], dim=1)
        dwi_feats = self.dwi_encoder(dwi)
        x = torch.cat([t2_feats, dwi_feats], dim=1)
        if self.gate is not None:
            x = self.gate(x) * x
        fused = self.fusion_mlp(x)

        pirads_logits = self.head(fused)
        if self.gleason_head is None:
            return pirads_logits
        return {
            "pirads": pirads_logits,
            "gleason": self.gleason_head(fused),
        }


def build_image_model(
    *,
    model_variant: str,
    input_keys: Sequence[str],
    hidden_dim: int = 256,
    dropout: float = 0.3,
    backbone: str = "r3d_18",
    backbone_pretrained: bool = False,
    dual_encoder: bool = False,
    freeze_backbone: bool = False,
    enable_gleason_head: bool = False,
    fusion_gate: bool = False,
) -> nn.Module:
    """Construct an image model while keeping legacy dual-encoder support isolated."""
    if model_variant == MODEL_VARIANT_SINGLE_STREAM:
        if fusion_gate:
            raise ValueError("fusion_gate is only supported with t2_recon_dwi_fusion")
        return ProstateImageClassifier(
            input_keys=input_keys,
            hidden_dim=hidden_dim,
            dropout=dropout,
            backbone=backbone,
            backbone_pretrained=backbone_pretrained,
            dual_encoder=dual_encoder,
            freeze_backbone=freeze_backbone,
            enable_gleason_head=enable_gleason_head,
        )
    if model_variant == MODEL_VARIANT_T2_RECON_DWI_FUSION:
        if tuple(input_keys) != ("axt2", "adc", "b1500"):
            raise ValueError(
                "t2_recon_dwi_fusion expects input_keys=('axt2', 'adc', 'b1500')"
            )
        if dual_encoder:
            raise ValueError("dual_encoder is not supported with t2_recon_dwi_fusion")
        if freeze_backbone:
            raise ValueError(
                "freeze_backbone is not supported with t2_recon_dwi_fusion"
            )
        return T2ReconDWIFusionClassifier(
            t2_input_key="axt2",
            dwi_input_keys=("adc", "b1500"),
            hidden_dim=hidden_dim,
            dropout=dropout,
            backbone=backbone,
            backbone_pretrained=backbone_pretrained,
            enable_gleason_head=enable_gleason_head,
            fusion_gate=fusion_gate,
        )
    raise ValueError(f"Unsupported model_variant '{model_variant}'")

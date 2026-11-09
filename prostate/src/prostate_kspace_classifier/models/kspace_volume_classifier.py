"""K-space classifier that feeds FFTConv magnitudes into a 3D encoder."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from ..data.transforms_image import StandardImageVolumeTransform
from .heads import CSPCaHead
from .kspacenet_slice import KSpaceSliceLayerMultiChannel, KSpaceSliceLayerSingleFilter
from .resnet3d import build_resnet3d_encoder


class KSpaceVolumeClassifier(nn.Module):
    """Process canonicalized k-space tensors using FFTConv slice layers."""

    def __init__(
        self,
        num_input_channels: int,
        num_kspace_filters: int = 8,
        hidden_dim: int = 256,
        backbone: str = "r3d_18",
        backbone_pretrained: bool = False,
        dropout: float = 0.5,
        filter_mode: str = "single_channel",
        average_direction_groups: bool = True,
        enable_gleason_head: bool = False,
    ) -> None:
        super().__init__()
        if filter_mode not in {"single_channel", "multi_channel"}:
            raise ValueError("filter_mode must be 'single_channel' or 'multi_channel'")
        self.filter_mode = filter_mode
        self.num_input_channels = num_input_channels
        self.average_direction_groups = (
            average_direction_groups if filter_mode == "single_channel" else False
        )
        self._num_direction_groups = 6

        if filter_mode == "single_channel":
            self.slice_layer_single = KSpaceSliceLayerSingleFilter(kernel_size=5)
            if self.average_direction_groups:
                self.num_feat_channels = self._num_direction_groups
            else:
                self.num_feat_channels = num_input_channels
        else:
            self.slice_layer_multi = KSpaceSliceLayerMultiChannel(
                in_channels_complex=num_input_channels,
                out_channels_complex=num_kspace_filters,
            )
            self.num_feat_channels = num_kspace_filters

        self.volume_transform = StandardImageVolumeTransform(target_d=30, target_hw=224)
        self._normalized_shape = (self.volume_transform.target_d, self.volume_transform.target_hw, self.volume_transform.target_hw)
        self.layer_norm = nn.LayerNorm(normalized_shape=self._normalized_shape, elementwise_affine=False)
        self.dropout = nn.Dropout(p=0.2)

        self.encoder3d, out_dim = build_resnet3d_encoder(
            backbone_name=backbone,
            in_channels=self.num_feat_channels,
            pretrained=backbone_pretrained,
        )
        # Keep `head` as PI-RADS/csPCa head for backward-compatible checkpoint keys.
        self.head = CSPCaHead(out_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.gleason_head = (
            CSPCaHead(out_dim, hidden_dim=hidden_dim, dropout=dropout)
            if enable_gleason_head
            else None
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor | Dict[str, torch.Tensor]:
        kspace = batch["kspace"]
        batch_size, c_k, depth, height, width = kspace.shape

        if self.filter_mode == "single_channel":
            x = kspace.permute(0, 2, 1, 3, 4).reshape(batch_size * depth * c_k, 1, height, width)
            mag_2d = self.slice_layer_single(x)
            mag_2d = mag_2d.squeeze(1)
            _, h2, w2 = mag_2d.shape
            mag_2d = mag_2d.view(batch_size, depth, c_k, h2, w2)
            mag_volume = mag_2d.permute(0, 2, 1, 3, 4)
            if self.average_direction_groups:
                direction_counts = batch.get("direction_counts")
                if direction_counts is None:
                    raise ValueError(
                        "direction_counts missing from batch while average_direction_groups=True"
                    )
                mag_volume = self._average_direction_groups(mag_volume, direction_counts)
        else:
            x = kspace.permute(0, 2, 1, 3, 4).reshape(batch_size * depth, c_k, height, width)
            mag_2d = self.slice_layer_multi(x)
            _, fk, h2, w2 = mag_2d.shape
            mag_2d = mag_2d.view(batch_size, depth, fk, h2, w2)
            mag_volume = mag_2d.permute(0, 2, 1, 3, 4)

        mag_volume = torch.stack(
            [self.volume_transform(volume) for volume in mag_volume], dim=0
        )
        mag_volume = self.layer_norm(mag_volume)
        mag_volume = self.dropout(mag_volume)

        feats = self.encoder3d(mag_volume)
        pirads_logits = self.head(feats)
        if self.gleason_head is None:
            return pirads_logits
        return {
            "pirads": pirads_logits,
            "gleason": self.gleason_head(feats),
        }

    def _average_direction_groups(
        self,
        mag_volume: torch.Tensor,
        direction_counts: torch.Tensor,
    ) -> torch.Tensor:
        """Average canonical direction blocks assuming every batch item shares the same layout."""
        if direction_counts.dim() == 1:
            direction_counts = direction_counts.unsqueeze(0)

        counts = direction_counts[0].to(dtype=torch.long, device=mag_volume.device)
        offsets = torch.cat(
            [counts.new_zeros(1), torch.cumsum(counts, dim=0)],
            dim=0,
        )
        per_dir = []
        for start, end in zip(offsets[:-1], offsets[1:]):
            per_dir.append(mag_volume[:, start:end, ...].mean(dim=1, keepdim=True))
        return torch.cat(per_dir, dim=1)

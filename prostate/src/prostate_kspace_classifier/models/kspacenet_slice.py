"""Slice-level FFTConv layers for k-space inputs."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fft_conv import FFTConv2d


class KSpaceSliceLayerSingleFilter(nn.Module):
    """MVP layer applying a shared (1x5x5) FFTConv to each slice/channel."""

    def __init__(self, kernel_size: int = 5, img_target_hw: tuple[int, int] = (224, 224)) -> None:
        super().__init__()
        self.fftconv = FFTConv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, bias=False)
        self.img_target_hw = img_target_hw

    def forward(self, kspace_complex_2d: torch.Tensor) -> torch.Tensor:
        shifted = torch.fft.ifftshift(kspace_complex_2d, dim=(-2, -1))
        ifftd = torch.fft.ifftn(shifted, dim=(-2, -1))
        recentered = torch.fft.fftshift(ifftd, dim=(-2, -1))
        refreq = torch.fft.fftn(recentered, dim=(-2, -1))

        h_complex = self.fftconv(refreq)
        magnitude = h_complex.abs()
        return F.interpolate(
            magnitude,
            size=self.img_target_hw,
            mode="bilinear",
            align_corners=False,
        )


class KSpaceSliceLayerMultiChannel(nn.Module):
    """Full KSPACE-NET layer that mixes channels via FFTConv before decoding."""

    def __init__(
        self,
        in_channels_complex: int,
        out_channels_complex: int = 8,
        img_target_hw: tuple[int, int] = (224, 224),
    ) -> None:
        super().__init__()
        self.fftconv = FFTConv2d(
            in_channels=in_channels_complex,
            out_channels=out_channels_complex,
            kernel_size=5,
            bias=False,
        )
        self.img_target_hw = img_target_hw

    def forward(self, kspace_complex_2d: torch.Tensor) -> torch.Tensor:
        shifted = torch.fft.ifftshift(kspace_complex_2d, dim=(-2, -1))
        ifftd = torch.fft.ifftn(shifted, dim=(-2, -1))
        recentered = torch.fft.fftshift(ifftd, dim=(-2, -1))
        refreq = torch.fft.fftn(recentered, dim=(-2, -1))

        h_complex = self.fftconv(refreq)
        magnitude = h_complex.abs()
        return F.interpolate(
            magnitude,
            size=self.img_target_hw,
            mode="bilinear",
            align_corners=False,
        )

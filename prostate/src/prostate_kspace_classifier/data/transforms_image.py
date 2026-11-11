"""Standardized preprocessing utilities for image-domain volumes."""

from __future__ import annotations

import random
from typing import Any, Dict, Sequence

import torch
import torch.nn.functional as F


ImageSample = Dict[str, Any]
NON_IMAGE_KEYS = {
    "label",
    "accession_id",
    "gleason_target",
    "gleason_mask",
    "grade_group",
    "averaging_tag",
    "recipe_id",
    "recipe_family",
    "direction_mask",
    "counts_b50",
    "counts_b1000",
}


def center_crop_or_pad_slices(x: torch.Tensor, target_d: int = 30) -> torch.Tensor:
    """Center-crop or zero-pad along the depth dimension to ``target_d`` slices."""

    _, depth, _, _ = x.shape
    if depth > target_d:
        start = (depth - target_d) // 2
        end = start + target_d
        return x[:, start:end, :, :]
    if depth < target_d:
        pad_front = (target_d - depth) // 2
        pad_back = target_d - depth - pad_front
        return F.pad(x, (0, 0, 0, 0, pad_front, pad_back))
    return x


def resize_hw_to_224(x: torch.Tensor, target_hw: int = 224) -> torch.Tensor:
    """Resize each slice to ``target_hw`` square using bilinear interpolation."""

    channels, depth, height, width = x.shape
    reshaped = x.reshape(channels * depth, 1, height, width)
    resized = F.interpolate(
        reshaped,
        size=(target_hw, target_hw),
        mode="bilinear",
        align_corners=False,
    )
    return resized.reshape(channels, depth, target_hw, target_hw)


def normalize_volume_zscore(x: torch.Tensor) -> torch.Tensor:
    """Apply global z-score normalization across all voxels."""

    mean = x.mean()
    std = x.std().clamp_min(1e-6)
    return (x - mean) / std


class StandardImageVolumeTransform(torch.nn.Module):
    """Canonical preprocessing for image-domain volumes."""

    def __init__(self, target_d: int = 30, target_hw: int = 224) -> None:
        super().__init__()
        self.target_d = target_d
        self.target_hw = target_hw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = center_crop_or_pad_slices(x, self.target_d)
        x = resize_hw_to_224(x, self.target_hw)
        x = normalize_volume_zscore(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Training augmentations (applied to sample dict after preprocessing)
# ─────────────────────────────────────────────────────────────────────────────


class RandomFlip3D:
    """Randomly flip along H/W axes (not depth - preserves anatomical order)."""

    def __init__(self, p: float = 0.5, keys: Sequence[str] | None = None) -> None:
        self.p = p
        self.keys = list(keys) if keys is not None else None

    def _iter_keys(self, sample: ImageSample) -> Sequence[str]:
        if self.keys is not None:
            return self.keys
        return [
            key
            for key, value in sample.items()
            if key not in NON_IMAGE_KEYS and isinstance(value, torch.Tensor)
        ]

    def __call__(self, sample: ImageSample) -> ImageSample:
        flip_h = random.random() < self.p
        flip_w = random.random() < self.p
        for key in self._iter_keys(sample):
            vol = sample[key]
            if flip_h:
                vol = torch.flip(vol, dims=[2])
            if flip_w:
                vol = torch.flip(vol, dims=[3])
            sample[key] = vol
        return sample


class RandomRotate90:
    """Randomly rotate by 0/90/180/270 degrees in axial plane."""

    def __init__(self, p: float = 0.5, keys: Sequence[str] | None = None) -> None:
        self.p = p
        self.keys = list(keys) if keys is not None else None

    def _iter_keys(self, sample: ImageSample) -> Sequence[str]:
        if self.keys is not None:
            return self.keys
        return [
            key
            for key, value in sample.items()
            if key not in NON_IMAGE_KEYS and isinstance(value, torch.Tensor)
        ]

    def __call__(self, sample: ImageSample) -> ImageSample:
        if random.random() > self.p:
            return sample
        k = random.randint(1, 3)
        for key in self._iter_keys(sample):
            sample[key] = torch.rot90(sample[key], k=k, dims=[2, 3])
        return sample


class RandomIntensityJitter:
    """Random brightness/contrast adjustment."""

    def __init__(
        self,
        brightness: float = 0.1,
        contrast: float = 0.1,
        keys: Sequence[str] | None = None,
    ) -> None:
        self.brightness = brightness
        self.contrast = contrast
        self.keys = list(keys) if keys is not None else None

    def _iter_keys(self, sample: ImageSample) -> Sequence[str]:
        if self.keys is not None:
            return self.keys
        return [
            key
            for key, value in sample.items()
            if key not in NON_IMAGE_KEYS and isinstance(value, torch.Tensor)
        ]

    def __call__(self, sample: ImageSample) -> ImageSample:
        b_shift = (random.random() * 2 - 1) * self.brightness
        c_scale = 1.0 + (random.random() * 2 - 1) * self.contrast
        for key in self._iter_keys(sample):
            sample[key] = sample[key] * c_scale + b_shift
        return sample


class RandomGaussianNoise:
    """Add Gaussian noise."""

    def __init__(self, std: float = 0.05, p: float = 0.3, keys: Sequence[str] | None = None) -> None:
        self.std = std
        self.p = p
        self.keys = list(keys) if keys is not None else None

    def _iter_keys(self, sample: ImageSample) -> Sequence[str]:
        if self.keys is not None:
            return self.keys
        return [
            key
            for key, value in sample.items()
            if key not in NON_IMAGE_KEYS and isinstance(value, torch.Tensor)
        ]

    def __call__(self, sample: ImageSample) -> ImageSample:
        if random.random() > self.p:
            return sample
        for key in self._iter_keys(sample):
            sample[key] = sample[key] + torch.randn_like(sample[key]) * self.std
        return sample


def build_train_augmentations(keys: Sequence[str] | None = None) -> list:
    """Standard augmentation pipeline for training."""
    return [
        RandomFlip3D(p=0.5, keys=keys),
        RandomRotate90(p=0.5, keys=keys),
        RandomIntensityJitter(brightness=0.1, contrast=0.1, keys=keys),
        RandomGaussianNoise(std=0.05, p=0.3, keys=keys),
    ]

"""K-space canonicalization and conversion helpers."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch
import warnings


def get_direction_indices(num_averages: int) -> Dict[str, np.ndarray]:
    """Return canonical diffusion direction indices (fastMRI-prostate style)."""

    if num_averages not in (48, 50):
        warnings.warn(
            f"Unexpected number of averages ({num_averages}); canonical mapping assumes 48 or 50.",
            RuntimeWarning,
            stacklevel=2,
        )

    offset = 2 if num_averages == 48 else 0
    base_indices = {
        "b50x": np.arange(2, 22, 6),
        "b50y": np.arange(3, 23, 6),
        "b50z": np.arange(4, 24, 6),
        "b1000x": np.concatenate([np.arange(5, 25, 6), np.arange(26, 48, 3)]),
        "b1000y": np.concatenate([np.arange(6, 26, 6), np.arange(27, 49, 3)]),
        "b1000z": np.concatenate([np.arange(7, 27, 6), np.arange(28, 50, 3)]),
    }

    direction_map: Dict[str, np.ndarray] = {}
    for name, indices in base_indices.items():
        adjusted = indices - offset
        adjusted = adjusted[(adjusted >= 0) & (adjusted < num_averages)]
        direction_map[name] = adjusted.astype(int)
    return direction_map


def canonicalize_post_grappa(
    kspace: np.ndarray,
    directions: Sequence[str] | None = None,
) -> tuple[np.ndarray, Sequence[int]]:
    """Stack k-space averages grouped by diffusion direction.

    Returns
    -------
    stacked : np.ndarray
        Concatenated k-space tensor ordered by diffusion direction.
    counts : Sequence[int]
        Number of channels that contributed to each direction (in order).
    """

    num_averages = kspace.shape[0]
    direction_indices = get_direction_indices(num_averages)
    if directions is None:
        directions = ["b50x", "b50y", "b50z", "b1000x", "b1000y", "b1000z"]

    channels = []
    counts: list[int] = []
    for dname in directions:
        idx = direction_indices[dname]
        subset = kspace[idx, ...]
        channels.append(subset)
        counts.append(subset.shape[0])
    return np.concatenate(channels, axis=0), counts


def crop_or_pad_slices_kspace(kspace: np.ndarray, target_d: int = 30) -> np.ndarray:
    """Center crop or pad along the depth axis to ``target_d`` slices."""

    _, depth, _, _ = kspace.shape
    if depth > target_d:
        start = (depth - target_d) // 2
        end = start + target_d
        return kspace[:, start:end, :, :]
    if depth < target_d:
        pad_front = (target_d - depth) // 2
        pad_back = target_d - depth - pad_front
        pad_width = ((0, 0), (pad_front, pad_back), (0, 0), (0, 0))
        return np.pad(kspace, pad_width=pad_width, mode="constant")
    return kspace


def kspace_np_to_complex_torch(kspace_np: np.ndarray) -> torch.Tensor:
    """Convert a numpy complex array to a torch.complex64 tensor."""

    tensor = torch.from_numpy(kspace_np.astype(np.complex64))
    return tensor.to(dtype=torch.complex64)

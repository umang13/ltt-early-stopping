"""Datasets that render ADC/b1500 from directional prefix recipes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from prostate_kspace_classifier.data.metadata_index import AccessionLabel
from prostate_kspace_classifier.data.transforms_image import StandardImageVolumeTransform

from .render import render_recipe_metrics
from .sampling import resolve_recipes_for_accession
from .types import RecipeSpec


ImageSample = Dict[str, Any]
ImageTransform = Callable[[ImageSample], ImageSample]


def _build_h5_lookup(roots: Sequence[Path]) -> Dict[str, Path]:
    lookup: Dict[str, Path] = {}
    for root in roots:
        for path in Path(root).glob("*.h5"):
            lookup.setdefault(path.stem, path)
    return lookup


def _resolve_h5_path(label: AccessionLabel, h5_root: Path) -> Path:
    if label.h5_path is not None:
        return Path(h5_root) / Path(label.h5_path).name
    return h5_root / f"{label.accession_id}.h5"


def _label_payload(label: AccessionLabel) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gleason_target = float(label.gleason_label) if label.gleason_label is not None else 0.0
    gleason_mask = 1.0 if label.gleason_label is not None else 0.0
    grade_group = int(label.gleason_grade_group) if label.gleason_grade_group is not None else -1
    return (
        torch.tensor(label.cs_pca_label, dtype=torch.float32),
        torch.tensor(gleason_target, dtype=torch.float32),
        torch.tensor(gleason_mask, dtype=torch.float32),
        torch.tensor(grade_group, dtype=torch.long),
    )


class RecipeReconDataset(Dataset):
    """Render recon ADC/b1500 on the fly from bank-backed recipe specs."""

    def __init__(
        self,
        accession_labels: list[AccessionLabel],
        h5_root: Path,
        *,
        coil_combine: str,
        recipe_mode: str,
        recipes_per_accession: int | None,
        recipe_family_quotas: Optional[Sequence[str]],
        recipe_seed: int,
        recipe_split: str,
        image_transforms: Optional[Sequence[ImageTransform]] = None,
    ) -> None:
        self.h5_root = Path(h5_root)
        self.coil_combine = coil_combine
        self.recipe_mode = recipe_mode
        self.recipes_per_accession = recipes_per_accession
        self.recipe_family_quotas = (
            list(recipe_family_quotas) if recipe_family_quotas is not None else None
        )
        self.recipe_seed = recipe_seed
        self.recipe_split = recipe_split
        self.image_transform = StandardImageVolumeTransform(target_d=30, target_hw=224)
        self._image_transforms = list(image_transforms or [])
        self._base_records = [
            (
                label,
                _resolve_h5_path(label, self.h5_root),
            )
            for label in accession_labels
        ]
        self._records: list[tuple[AccessionLabel, Path, RecipeSpec]] = []
        self.resample_for_epoch(0, recipe_seed)

    def resample_for_epoch(self, epoch: int, recipe_seed: int | None = None) -> None:
        if recipe_seed is not None:
            self.recipe_seed = recipe_seed
        records: list[tuple[AccessionLabel, Path, RecipeSpec]] = []
        recipe_epoch = epoch if self.recipe_mode == "sampled_recipes" and self.recipe_split == "train" else 0
        for label, h5_path in self._base_records:
            specs = resolve_recipes_for_accession(
                label.accession_id,
                recipe_mode=self.recipe_mode,
                recipes_per_accession=self.recipes_per_accession,
                recipe_family_quotas=self.recipe_family_quotas,
                recipe_seed=self.recipe_seed,
                recipe_split=self.recipe_split,
                recipe_epoch=recipe_epoch,
            )
            records.extend((label, h5_path, spec) for spec in specs)
        self._records = records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        label, h5_path, spec = self._records[idx]
        label_tensor, gleason_target, gleason_mask, grade_group = _label_payload(label)
        with h5py.File(h5_path, "r") as handle:
            rendered = render_recipe_metrics(handle, self.coil_combine, spec)

        sample: ImageSample = {
            "accession_id": label.accession_id,
            "averaging_tag": spec.averaging_tag,
            "recipe_id": spec.recipe_id,
            "recipe_family": spec.family,
            "direction_mask": torch.tensor(spec.direction_mask, dtype=torch.float32),
            "counts_b50": torch.tensor(spec.counts_b50, dtype=torch.long),
            "counts_b1000": torch.tensor(spec.counts_b1000, dtype=torch.long),
            "label": label_tensor,
            "gleason_target": gleason_target,
            "gleason_mask": gleason_mask,
            "grade_group": grade_group,
        }
        for output_key, metric_key in (("adc", "adc_map"), ("b1500", "b1500")):
            tensor = torch.from_numpy(rendered[metric_key].astype(np.float32)).unsqueeze(0)
            sample[output_key] = self.image_transform(tensor)
        for transform in self._image_transforms:
            sample = transform(sample)
        return sample


class RecipeHybridT2DWIDataset(Dataset):
    """Pair DICOM T2 with recipe-rendered recon ADC/b1500."""

    def __init__(
        self,
        accession_labels: list[AccessionLabel],
        h5_root: Path,
        *,
        dicom_roots: Sequence[Path],
        coil_combine: str,
        recipe_mode: str,
        recipes_per_accession: int | None,
        recipe_family_quotas: Optional[Sequence[str]],
        recipe_seed: int,
        recipe_split: str,
        image_transforms: Optional[Sequence[ImageTransform]] = None,
    ) -> None:
        self.h5_root = Path(h5_root)
        self.coil_combine = coil_combine
        self.recipe_mode = recipe_mode
        self.recipes_per_accession = recipes_per_accession
        self.recipe_family_quotas = (
            list(recipe_family_quotas) if recipe_family_quotas is not None else None
        )
        self.recipe_seed = recipe_seed
        self.recipe_split = recipe_split
        self.image_transform = StandardImageVolumeTransform(target_d=30, target_hw=224)
        self._image_transforms = list(image_transforms or [])
        self._dicom_lookup = _build_h5_lookup(dicom_roots)
        self._base_records = [
            (
                label,
                self._dicom_lookup[label.accession_id],
                _resolve_h5_path(label, self.h5_root),
            )
            for label in accession_labels
        ]
        self._records: list[tuple[AccessionLabel, Path, Path, RecipeSpec]] = []
        self.resample_for_epoch(0, recipe_seed)

    def resample_for_epoch(self, epoch: int, recipe_seed: int | None = None) -> None:
        if recipe_seed is not None:
            self.recipe_seed = recipe_seed
        records: list[tuple[AccessionLabel, Path, Path, RecipeSpec]] = []
        recipe_epoch = epoch if self.recipe_mode == "sampled_recipes" and self.recipe_split == "train" else 0
        for label, dicom_h5_path, recon_h5_path in self._base_records:
            specs = resolve_recipes_for_accession(
                label.accession_id,
                recipe_mode=self.recipe_mode,
                recipes_per_accession=self.recipes_per_accession,
                recipe_family_quotas=self.recipe_family_quotas,
                recipe_seed=self.recipe_seed,
                recipe_split=self.recipe_split,
                recipe_epoch=recipe_epoch,
            )
            records.extend((label, dicom_h5_path, recon_h5_path, spec) for spec in specs)
        self._records = records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        label, dicom_h5_path, recon_h5_path, spec = self._records[idx]
        label_tensor, gleason_target, gleason_mask, grade_group = _label_payload(label)
        with h5py.File(dicom_h5_path, "r") as dicom_handle, h5py.File(
            recon_h5_path, "r"
        ) as recon_handle:
            rendered = render_recipe_metrics(recon_handle, self.coil_combine, spec)
            t2_np = dicom_handle["axt2"][()]

        sample: ImageSample = {
            "accession_id": label.accession_id,
            "averaging_tag": spec.averaging_tag,
            "recipe_id": spec.recipe_id,
            "recipe_family": spec.family,
            "direction_mask": torch.tensor(spec.direction_mask, dtype=torch.float32),
            "counts_b50": torch.tensor(spec.counts_b50, dtype=torch.long),
            "counts_b1000": torch.tensor(spec.counts_b1000, dtype=torch.long),
            "label": label_tensor,
            "gleason_target": gleason_target,
            "gleason_mask": gleason_mask,
            "grade_group": grade_group,
        }

        t2_tensor = torch.from_numpy(t2_np.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        sample["axt2"] = self.image_transform(t2_tensor)
        for output_key, metric_key in (("adc", "adc_map"), ("b1500", "b1500")):
            tensor = torch.from_numpy(rendered[metric_key].astype(np.float32)).unsqueeze(0)
            sample[output_key] = self.image_transform(tensor)

        for transform in self._image_transforms:
            sample = transform(sample)
        return sample

"""Unified accession-level dataset for image and k-space modalities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .metadata_index import AccessionLabel
from .transforms_image import StandardImageVolumeTransform
from .transforms_kspace import (
    canonicalize_post_grappa,
    crop_or_pad_slices_kspace,
    kspace_np_to_complex_torch,
)

ImageSample = Dict[str, Any]
ImageTransform = Callable[[ImageSample], ImageSample]
KspaceTransform = Callable[[torch.Tensor], torch.Tensor]


logger = logging.getLogger(__name__)

DEFAULT_COIL_COMBINE = "espirit"
DEFAULT_AVERAGING_TAG = "b50_4_b1000_12"
T2_ONLY_AVERAGING_TAG = "t2_only"
KSPACE_DATASET = "kspace/post_grappa_full"
# recon image_metrics -> ((batch_output_key, h5_namespace, h5_leaf_key), ...)
RECON_IMAGE_METRICS = {
    "adc_b1500": (("adc", "metrics", "adc_map"), ("b1500", "metrics", "b1500")),
    "trace_b50_b1000": (
        ("trace_b50", "metrics", "trace_b50"),
        ("trace_b1000", "metrics", "trace_b1000"),
    ),
    "b50x_b1000x": (("b50x", "images", "b50x"), ("b1000x", "images", "b1000x")),
    "b50y_b1000y": (("b50y", "images", "b50y"), ("b1000y", "images", "b1000y")),
    "b50z_b1000z": (("b50z", "images", "b50z"), ("b1000z", "images", "b1000z")),
}
DICOM_IMAGE_METRICS = {
    "adc_b1500": (("adc", "adc"), ("b1500", "b1500")),
    "axt2": (("axt2", "axt2"),),
}
HYBRID_IMAGE_METRICS = {
    "adc_b1500": RECON_IMAGE_METRICS["adc_b1500"],
}
IMAGE_SOURCE_CHOICES = ("recon", "dicom", "hybrid")
IMAGE_METRICS_CHOICES = tuple(
    sorted(
        set(RECON_IMAGE_METRICS) | set(DICOM_IMAGE_METRICS) | set(HYBRID_IMAGE_METRICS)
    )
)


def validate_image_source_metrics(image_source: str, image_metrics: str) -> None:
    """Ensure the requested metric is supported for the selected image source."""
    if image_source == "recon":
        supported = RECON_IMAGE_METRICS
    elif image_source == "dicom":
        supported = DICOM_IMAGE_METRICS
    elif image_source == "hybrid":
        supported = HYBRID_IMAGE_METRICS
    else:
        raise ValueError(f"Unsupported image_source '{image_source}'")

    if image_metrics not in supported:
        raise ValueError(
            f"Unsupported image_metrics '{image_metrics}' for image_source='{image_source}'"
        )


def get_image_metric_input_keys(image_metrics: str) -> tuple[str, ...]:
    """Return batch keys for a recon image metric selection."""
    if image_metrics not in RECON_IMAGE_METRICS:
        raise ValueError(f"Unsupported recon image_metrics '{image_metrics}'")
    return tuple(output_key for output_key, _, _ in RECON_IMAGE_METRICS[image_metrics])


def resolve_image_input_keys(image_source: str, image_metrics: str) -> tuple[str, ...]:
    """Return batch keys emitted by the requested image dataset path."""
    validate_image_source_metrics(image_source, image_metrics)
    if image_source == "recon":
        return get_image_metric_input_keys(image_metrics)
    if image_source == "dicom":
        return tuple(output_key for output_key, _ in DICOM_IMAGE_METRICS[image_metrics])
    return ("axt2",) + get_image_metric_input_keys(image_metrics)


@dataclass
class AlignedCohortStats:
    """Statistics from cohort alignment."""

    labeled_count: int
    recon_h5_count: int  # accessions with recon H5 files (from label matching)
    raw_kspace_count: int  # accessions with raw k-space stored (subset of recon_h5)
    dicom_count: int
    aligned_count: int


def build_h5_lookup(roots: Sequence[Path]) -> Dict[str, Path]:
    """Return a mapping from accession ID (stem) to the first matching H5 path."""
    lookup: Dict[str, Path] = {}
    for root in roots:
        root = Path(root)
        for path in root.glob("*.h5"):
            accession = path.stem
            lookup.setdefault(accession, path)
    return lookup


def get_raw_kspace_accessions(
    accession_labels: List[AccessionLabel],
    h5_root: Path,
) -> set[str]:
    """Return accession IDs that have raw k-space (`kspace/post_grappa_full`) on disk.

    This is a subset of accessions that have recon H5 files - only those where
    the raw k-space was also stored (i.e., `skip_kspace=False` during processing).
    """
    usable: set[str] = set()
    for label in accession_labels:
        if label.h5_path is not None:
            h5_path = Path(label.h5_path)
        else:
            h5_path = Path(h5_root) / f"{label.accession_id}.h5"
        if not h5_path.exists():
            continue
        with h5py.File(h5_path, "r") as handle:
            if KSPACE_DATASET in handle:
                usable.add(label.accession_id)
    return usable


def get_aligned_cohort(
    accession_labels: List[AccessionLabel],
    recon_h5_root: Path,
    dicom_h5_roots: Sequence[Path],
    require_raw_kspace: bool = False,
) -> tuple[List[AccessionLabel], AlignedCohortStats]:
    """Return accessions available for fair cross-baseline comparison.

    This is the single source of truth for cohort alignment across baselines:
      1. DICOM image classifier - needs labels + DICOM H5
      2. Reconstructed image classifier - needs labels + recon H5 (with selected image metrics)
      3. K-space classifier - needs labels + recon H5 with raw k-space stored

    By default (require_raw_kspace=False), aligns to accessions with:
      - Labels (from CSV, matched to recon H5 via MRN/date)
      - DICOM H5 files

    When require_raw_kspace=True, additionally requires:
      - Raw k-space stored in the recon H5 (`kspace/post_grappa_full` dataset)

    Parameters
    ----------
    accession_labels
        Labels loaded from CSV, already matched to recon H5 files via MRN/date.
        Each label has an `h5_path` pointing to the recon H5 file.
    recon_h5_root
        Directory containing reconstruction H5 files (used as fallback if h5_path is None).
    dicom_h5_roots
        Directories containing DICOM-derived H5 files (named `<accession>.h5`).
    require_raw_kspace
        If True, only include accessions where raw k-space is stored.
        Use True for k-space classifier baseline, False for image baselines.

    Returns
    -------
    aligned_labels
        Subset of accession_labels meeting alignment criteria.
    stats
        Counts for logging and debugging.
    """
    # All labeled accessions (already matched to recon H5 files)
    labeled_ids = {lbl.accession_id for lbl in accession_labels}

    # Accessions with recon H5 files (same as labeled since load_accession_labels does the matching)
    recon_h5_ids = {
        lbl.accession_id for lbl in accession_labels if lbl.h5_path is not None
    }

    # Accessions with raw k-space stored (subset of recon_h5)
    raw_kspace_ids = get_raw_kspace_accessions(accession_labels, recon_h5_root)

    # Accessions with DICOM H5 files
    dicom_lookup = build_h5_lookup(dicom_h5_roots)
    dicom_ids = set(dicom_lookup.keys())

    if require_raw_kspace:
        # For k-space baseline: need labels + raw k-space + DICOM
        aligned_ids = labeled_ids & raw_kspace_ids & dicom_ids
    else:
        # For image baselines: need labels + recon H5 + DICOM
        aligned_ids = labeled_ids & dicom_ids

    if not aligned_ids:
        raise ValueError(
            f"No overlapping accessions found. "
            f"labeled={len(labeled_ids)}, recon_h5={len(recon_h5_ids)}, "
            f"raw_kspace={len(raw_kspace_ids)}, dicom={len(dicom_ids)}, "
            f"require_raw_kspace={require_raw_kspace}"
        )

    aligned_labels = [
        lbl for lbl in accession_labels if lbl.accession_id in aligned_ids
    ]

    stats = AlignedCohortStats(
        labeled_count=len(labeled_ids),
        recon_h5_count=len(recon_h5_ids),
        raw_kspace_count=len(raw_kspace_ids),
        dicom_count=len(dicom_ids),
        aligned_count=len(aligned_ids),
    )

    logger.info(
        "[aligned_cohort] labeled=%d recon_h5=%d raw_kspace=%d dicom=%d -> aligned=%d (require_raw_kspace=%s)",
        stats.labeled_count,
        stats.recon_h5_count,
        stats.raw_kspace_count,
        stats.dicom_count,
        stats.aligned_count,
        require_raw_kspace,
    )

    return aligned_labels, stats


class ProstateH5Dataset(Dataset):
    """Dataset that reads per-accession HDF5 files for both modalities."""

    def __init__(
        self,
        accession_labels: List[AccessionLabel],
        h5_root: Path,
        mode: str,
        image_metrics: str = "adc_b1500",
        coil_combine: str = DEFAULT_COIL_COMBINE,
        averaging_tag: str = DEFAULT_AVERAGING_TAG,
        averaging_tags: Optional[Sequence[str]] = None,
        image_transforms: Optional[Sequence[ImageTransform]] = None,
        kspace_transforms: Optional[Sequence[KspaceTransform]] = None,
    ) -> None:
        if mode not in {"image", "kspace_post_grappa"}:
            raise ValueError("mode must be 'image' or 'kspace_post_grappa'")
        if mode == "image" and image_metrics not in RECON_IMAGE_METRICS:
            raise ValueError(f"Unsupported recon image_metrics '{image_metrics}'")

        self.mode = mode
        self.image_metrics = image_metrics
        self.coil_combine = coil_combine
        self.averaging_tags = (
            list(averaging_tags) if averaging_tags is not None else [averaging_tag]
        )
        self.averaging_tag = self.averaging_tags[0]
        self.h5_root = Path(h5_root)
        self.image_transform = StandardImageVolumeTransform(target_d=30, target_hw=224)
        self._image_transforms = list(image_transforms or [])
        self._kspace_transforms = list(kspace_transforms or [])
        self._records = self._build_records(accession_labels)

    def _metric_key(
        self, namespace: str, name: str, averaging_tag: str | None = None
    ) -> str:
        tag = averaging_tag if averaging_tag is not None else self.averaging_tag
        if namespace == "metrics":
            return f"metrics/{self.coil_combine}/{tag}/{name}"
        if namespace == "images":
            return f"images/{self.coil_combine}/{tag}/{name}"
        raise ValueError(f"Unsupported namespace '{namespace}' for image metric key")

    def _build_records(self, accession_labels: List[AccessionLabel]):
        records = []
        skipped: List[str] = []
        missing: List[str] = []
        for label in accession_labels:
            if label.h5_path is not None:
                h5_path = Path(label.h5_path)
            else:
                h5_path = self.h5_root / f"{label.accession_id}.h5"
            with h5py.File(h5_path, "r") as handle:
                if self.mode == "kspace_post_grappa":
                    if KSPACE_DATASET not in handle:
                        missing.append(label.accession_id)
                        continue
                    _, _, height, width = handle[KSPACE_DATASET].shape
                    if (height, width) != (200, 150):
                        skipped.append(label.accession_id)
                        continue
                else:
                    metric_specs = RECON_IMAGE_METRICS[self.image_metrics]
                    for averaging_tag in self.averaging_tags:
                        metric_keys = [
                            self._metric_key(
                                namespace, h5_key, averaging_tag=averaging_tag
                            )
                            for _, namespace, h5_key in metric_specs
                        ]
                        if not all(metric_key in handle for metric_key in metric_keys):
                            missing.append(f"{label.accession_id}:{averaging_tag}")
                            continue
                        records.append(
                            (
                                label.accession_id,
                                label.cs_pca_label,
                                label.gleason_label,
                                label.gleason_grade_group,
                                h5_path,
                                averaging_tag,
                            )
                        )
                    continue

            records.append(
                (
                    label.accession_id,
                    label.cs_pca_label,
                    label.gleason_label,
                    label.gleason_grade_group,
                    h5_path,
                    None,
                )
            )

        if skipped:
            logger.warning(
                "Skipped %d accessions due to non-200x150 k-space grids",
                len(skipped),
            )
        if missing:
            logger.warning(
                "Skipped %d accessions due to missing data for mode=%s",
                len(missing),
                self.mode,
            )

        return records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        (
            accession_id,
            cs_label,
            gleason_label,
            grade_group,
            h5_path,
            averaging_tag,
        ) = self._records[idx]
        gleason_target = float(gleason_label) if gleason_label is not None else 0.0
        gleason_mask = 1.0 if gleason_label is not None else 0.0
        grade_group_value = int(grade_group) if grade_group is not None else -1
        with h5py.File(h5_path, "r") as handle:
            if self.mode == "image":
                metric_specs = RECON_IMAGE_METRICS[self.image_metrics]
                sample: ImageSample = {
                    "accession_id": accession_id,
                    "averaging_tag": averaging_tag,
                    "label": torch.tensor(cs_label, dtype=torch.float32),
                    "gleason_target": torch.tensor(gleason_target, dtype=torch.float32),
                    "gleason_mask": torch.tensor(gleason_mask, dtype=torch.float32),
                    "grade_group": torch.tensor(grade_group_value, dtype=torch.long),
                }
                for output_key, namespace, h5_key in metric_specs:
                    metric_key = self._metric_key(
                        namespace, h5_key, averaging_tag=averaging_tag
                    )
                    if metric_key not in handle:
                        raise KeyError(
                            "Missing requested image metric tensor in recon H5 sample "
                            f"(coil_combine={self.coil_combine}, averaging_tag={averaging_tag})"
                        )
                    metric_np = handle[metric_key][()]
                    metric_tensor = torch.from_numpy(
                        metric_np.astype(np.float32)
                    ).unsqueeze(0)
                    sample[output_key] = self.image_transform(metric_tensor)
                for transform in self._image_transforms:
                    sample = transform(sample)
                return sample

            if KSPACE_DATASET not in handle:
                raise KeyError("Missing k-space dataset in recon H5 sample")
            post_grappa = handle[KSPACE_DATASET][()]
            kspace_canon, direction_counts = canonicalize_post_grappa(post_grappa)
            kspace_canon = crop_or_pad_slices_kspace(kspace_canon, target_d=30)
            kspace_tensor = kspace_np_to_complex_torch(kspace_canon)
            for transform in self._kspace_transforms:
                kspace_tensor = transform(kspace_tensor)
            return {
                "accession_id": accession_id,
                "kspace": kspace_tensor,
                "label": torch.tensor(cs_label, dtype=torch.float32),
                "gleason_target": torch.tensor(gleason_target, dtype=torch.float32),
                "gleason_mask": torch.tensor(gleason_mask, dtype=torch.float32),
                "grade_group": torch.tensor(grade_group_value, dtype=torch.long),
                "direction_counts": torch.tensor(direction_counts, dtype=torch.long),
            }


class DicomImageH5Dataset(Dataset):
    """Dataset for scanner/DICOM-derived image volumes stored directly in H5."""

    def __init__(
        self,
        accession_labels: List[AccessionLabel],
        dicom_roots: Sequence[Path],
        image_metrics: str = "adc_b1500",
        averaging_tag: str | None = None,
        image_transforms: Optional[Sequence[ImageTransform]] = None,
    ) -> None:
        if image_metrics not in DICOM_IMAGE_METRICS:
            raise ValueError(f"Unsupported DICOM image_metrics '{image_metrics}'")

        self.image_metrics = image_metrics
        self.averaging_tag = averaging_tag
        self.image_transform = StandardImageVolumeTransform(target_d=30, target_hw=224)
        self._image_transforms = list(image_transforms or [])

        self._lookup = build_h5_lookup(dicom_roots)
        self._records = self._build_records(accession_labels)

    def _build_records(self, accession_labels: List[AccessionLabel]):
        records = []
        missing: List[str] = []
        for label in accession_labels:
            h5_path = self._lookup.get(label.accession_id)
            if h5_path is None:
                missing.append(label.accession_id)
                continue
            records.append(
                (
                    label.accession_id,
                    label.cs_pca_label,
                    label.gleason_label,
                    label.gleason_grade_group,
                    h5_path,
                )
            )

        if missing:
            logger.warning(
                "DICOM H5 missing for %d accessions",
                len(missing),
            )
        return records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        accession_id, cs_label, gleason_label, grade_group, h5_path = self._records[idx]
        gleason_target = float(gleason_label) if gleason_label is not None else 0.0
        gleason_mask = 1.0 if gleason_label is not None else 0.0
        grade_group_value = int(grade_group) if grade_group is not None else -1
        with h5py.File(h5_path, "r") as handle:
            sample: ImageSample = {
                "accession_id": accession_id,
                "label": torch.tensor(cs_label, dtype=torch.float32),
                "gleason_target": torch.tensor(gleason_target, dtype=torch.float32),
                "gleason_mask": torch.tensor(gleason_mask, dtype=torch.float32),
                "grade_group": torch.tensor(grade_group_value, dtype=torch.long),
            }
            if self.averaging_tag is not None:
                sample["averaging_tag"] = self.averaging_tag

            for output_key, h5_key in DICOM_IMAGE_METRICS[self.image_metrics]:
                if h5_key not in handle:
                    raise KeyError(f"Missing '{h5_key}' dataset in DICOM H5 sample")
                metric_np = handle[h5_key][()]
                metric_tensor = (
                    torch.from_numpy(metric_np.astype(np.float32))
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                )
                sample[output_key] = self.image_transform(metric_tensor)

            for transform in self._image_transforms:
                sample = transform(sample)
            return sample


class HybridT2ReconDWIDataset(Dataset):
    """Dataset that pairs DICOM T2 with recon ADC/b1500 across averaging tags."""

    def __init__(
        self,
        accession_labels: List[AccessionLabel],
        h5_root: Path,
        dicom_roots: Sequence[Path],
        image_metrics: str = "adc_b1500",
        coil_combine: str = DEFAULT_COIL_COMBINE,
        averaging_tag: str = DEFAULT_AVERAGING_TAG,
        averaging_tags: Optional[Sequence[str]] = None,
        image_transforms: Optional[Sequence[ImageTransform]] = None,
    ) -> None:
        if image_metrics not in HYBRID_IMAGE_METRICS:
            raise ValueError(f"Unsupported hybrid image_metrics '{image_metrics}'")

        self.h5_root = Path(h5_root)
        self.image_metrics = image_metrics
        self.coil_combine = coil_combine
        self.averaging_tags = (
            list(averaging_tags) if averaging_tags is not None else [averaging_tag]
        )
        self.image_transform = StandardImageVolumeTransform(target_d=30, target_hw=224)
        self._image_transforms = list(image_transforms or [])
        self._dicom_lookup = build_h5_lookup(dicom_roots)
        self._records = self._build_records(accession_labels)

    def _metric_key(self, name: str, averaging_tag: str) -> str:
        return f"metrics/{self.coil_combine}/{averaging_tag}/{name}"

    def _build_records(self, accession_labels: List[AccessionLabel]):
        records = []
        missing: List[str] = []
        metric_specs = HYBRID_IMAGE_METRICS[self.image_metrics]

        for label in accession_labels:
            dicom_h5_path = self._dicom_lookup.get(label.accession_id)
            if dicom_h5_path is None:
                missing.append(f"{label.accession_id}:dicom")
                continue

            if label.h5_path is not None:
                recon_h5_path = Path(label.h5_path)
            else:
                recon_h5_path = self.h5_root / f"{label.accession_id}.h5"

            with h5py.File(dicom_h5_path, "r") as dicom_handle, h5py.File(
                recon_h5_path, "r"
            ) as recon_handle:
                if "axt2" not in dicom_handle:
                    missing.append(f"{label.accession_id}:axt2")
                    continue

                for averaging_tag in self.averaging_tags:
                    metric_keys = [
                        self._metric_key(h5_key, averaging_tag)
                        for _, _, h5_key in metric_specs
                    ]
                    if not all(
                        metric_key in recon_handle for metric_key in metric_keys
                    ):
                        missing.append(f"{label.accession_id}:{averaging_tag}")
                        continue
                    records.append(
                        (
                            label.accession_id,
                            label.cs_pca_label,
                            label.gleason_label,
                            label.gleason_grade_group,
                            dicom_h5_path,
                            recon_h5_path,
                            averaging_tag,
                        )
                    )

        if missing:
            logger.warning(
                "Skipped %d accessions due to missing data for hybrid image mode",
                len(missing),
            )
        return records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        (
            accession_id,
            cs_label,
            gleason_label,
            grade_group,
            dicom_h5_path,
            recon_h5_path,
            averaging_tag,
        ) = self._records[idx]
        gleason_target = float(gleason_label) if gleason_label is not None else 0.0
        gleason_mask = 1.0 if gleason_label is not None else 0.0
        grade_group_value = int(grade_group) if grade_group is not None else -1

        with h5py.File(dicom_h5_path, "r") as dicom_handle, h5py.File(
            recon_h5_path, "r"
        ) as recon_handle:
            sample: ImageSample = {
                "accession_id": accession_id,
                "averaging_tag": averaging_tag,
                "label": torch.tensor(cs_label, dtype=torch.float32),
                "gleason_target": torch.tensor(gleason_target, dtype=torch.float32),
                "gleason_mask": torch.tensor(gleason_mask, dtype=torch.float32),
                "grade_group": torch.tensor(grade_group_value, dtype=torch.long),
            }

            t2_np = dicom_handle["axt2"][()]
            t2_tensor = (
                torch.from_numpy(t2_np.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
            )
            sample["axt2"] = self.image_transform(t2_tensor)

            for output_key, _, h5_key in HYBRID_IMAGE_METRICS[self.image_metrics]:
                metric_key = self._metric_key(h5_key, averaging_tag)
                metric_np = recon_handle[metric_key][()]
                metric_tensor = torch.from_numpy(
                    metric_np.astype(np.float32)
                ).unsqueeze(0)
                sample[output_key] = self.image_transform(metric_tensor)

            for transform in self._image_transforms:
                sample = transform(sample)
            return sample


def create_image_dataset(
    *,
    accession_labels: List[AccessionLabel],
    h5_root: Path,
    image_source: str,
    image_metrics: str,
    dicom_roots: Optional[Sequence[Path]] = None,
    coil_combine: str = DEFAULT_COIL_COMBINE,
    averaging_tag: str = DEFAULT_AVERAGING_TAG,
    averaging_tags: Optional[Sequence[str]] = None,
    recipe_mode: str | None = None,
    recipes_per_accession: int | None = None,
    recipe_family_quotas: Optional[Sequence[str]] = None,
    recipe_seed: int = 1337,
    recipe_split: str = "train",
    image_transforms: Optional[Sequence[ImageTransform]] = None,
) -> Dataset:
    """Build the requested image dataset without duplicating source-specific branching."""
    validate_image_source_metrics(image_source, image_metrics)

    if recipe_mode is not None:
        from .recipes.datasets import RecipeHybridT2DWIDataset, RecipeReconDataset

        if image_metrics != "adc_b1500":
            raise ValueError("recipe_mode currently supports only image_metrics='adc_b1500'")
        if coil_combine != "espirit":
            raise ValueError("recipe_mode currently supports only coil_combine='espirit'")
        if image_source == "recon":
            return RecipeReconDataset(
                accession_labels=accession_labels,
                h5_root=h5_root,
                coil_combine=coil_combine,
                recipe_mode=recipe_mode,
                recipes_per_accession=recipes_per_accession,
                recipe_family_quotas=recipe_family_quotas,
                recipe_seed=recipe_seed,
                recipe_split=recipe_split,
                image_transforms=image_transforms,
            )
        if image_source == "hybrid":
            if dicom_roots is None:
                raise ValueError("dicom_roots is required for recipe-backed hybrid image datasets")
            return RecipeHybridT2DWIDataset(
                accession_labels=accession_labels,
                h5_root=h5_root,
                dicom_roots=dicom_roots,
                coil_combine=coil_combine,
                recipe_mode=recipe_mode,
                recipes_per_accession=recipes_per_accession,
                recipe_family_quotas=recipe_family_quotas,
                recipe_seed=recipe_seed,
                recipe_split=recipe_split,
                image_transforms=image_transforms,
            )
        raise ValueError(f"recipe_mode is not supported for image_source='{image_source}'")

    if image_source == "recon":
        return ProstateH5Dataset(
            accession_labels=accession_labels,
            h5_root=h5_root,
            mode="image",
            image_metrics=image_metrics,
            coil_combine=coil_combine,
            averaging_tag=averaging_tag,
            averaging_tags=averaging_tags,
            image_transforms=image_transforms,
        )

    if dicom_roots is None:
        raise ValueError(f"dicom_roots is required for image_source='{image_source}'")

    if image_source == "dicom":
        dicom_averaging_tag = T2_ONLY_AVERAGING_TAG if image_metrics == "axt2" else None
        return DicomImageH5Dataset(
            accession_labels=accession_labels,
            dicom_roots=dicom_roots,
            image_metrics=image_metrics,
            averaging_tag=dicom_averaging_tag,
            image_transforms=image_transforms,
        )

    return HybridT2ReconDWIDataset(
        accession_labels=accession_labels,
        h5_root=h5_root,
        dicom_roots=dicom_roots,
        image_metrics=image_metrics,
        coil_combine=coil_combine,
        averaging_tag=averaging_tag,
        averaging_tags=averaging_tags,
        image_transforms=image_transforms,
    )

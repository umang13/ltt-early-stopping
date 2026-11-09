#!/usr/bin/env python
"""Export aligned, recon-only, and dicom-only accession manifests to CSV."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import yaml

from prostate_kspace_classifier.data.h5_dataset import build_h5_lookup, get_aligned_cohort
from prostate_kspace_classifier.data.metadata_index import (
    AccessionLabel,
    load_accession_labels,
    load_accession_labels_from_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write accession manifests for aligned, recon-only, and dicom-only cohorts."
        )
    )
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/gpfs/scratch/td2105/dwi_stream"),
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Optional filename prefix. Defaults to experiment name or config stem.",
    )
    parser.add_argument(
        "--require-raw-kspace",
        action="store_true",
        help="Use the k-space aligned-cohort definition (requires raw k-space).",
    )
    return parser.parse_args()


def load_experiment(config_path: Path, experiment_name: str | None) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    experiments = payload.get("experiments", [])
    if not experiments:
        raise ValueError(f"No experiments found in {config_path}")

    if experiment_name is None:
        experiment = experiments[0]
    else:
        experiment = None
        for candidate in experiments:
            if candidate.get("name") == experiment_name:
                experiment = candidate
                break
        if experiment is None:
            names = [str(candidate.get("name")) for candidate in experiments]
            raise ValueError(
                f"Experiment '{experiment_name}' not found. Known experiments: {names}"
            )

    if "base_args" not in experiment:
        raise ValueError("Selected experiment is missing 'base_args'")
    return dict(experiment)


def sanitize_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    stem = stem.strip("._-")
    return stem or "cohort_manifest"


def _study_date_string(label: AccessionLabel) -> str:
    study_datetime = label.study_datetime
    if study_datetime is None or pd.isna(study_datetime):
        return ""
    return pd.Timestamp(study_datetime).date().isoformat()


def build_rows(
    labels: Sequence[AccessionLabel],
    *,
    cohort_name: str,
    dicom_lookup: dict[str, Path],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for label in labels:
        recon_h5_path = Path(label.h5_path) if label.h5_path is not None else None
        dicom_h5_path = dicom_lookup.get(label.accession_id)
        if cohort_name == "dicom_only":
            primary_h5_path = dicom_h5_path
        else:
            primary_h5_path = recon_h5_path

        rows.append(
            {
                "cohort": cohort_name,
                "accession_id": label.accession_id,
                "mrn": label.patient_id or "",
                "study_date": _study_date_string(label),
                "h5_filename": primary_h5_path.name if primary_h5_path is not None else "",
                "h5_path": str(primary_h5_path) if primary_h5_path is not None else "",
                "recon_h5_filename": recon_h5_path.name if recon_h5_path is not None else "",
                "recon_h5_path": str(recon_h5_path) if recon_h5_path is not None else "",
                "dicom_h5_filename": dicom_h5_path.name if dicom_h5_path is not None else "",
                "dicom_h5_path": str(dicom_h5_path) if dicom_h5_path is not None else "",
            }
        )
    return rows


def write_manifest(
    labels: Sequence[AccessionLabel],
    *,
    cohort_name: str,
    dicom_lookup: dict[str, Path],
    output_path: Path,
) -> None:
    frame = pd.DataFrame.from_records(
        build_rows(labels, cohort_name=cohort_name, dicom_lookup=dicom_lookup)
    )
    if frame.empty:
        frame = pd.DataFrame(
            columns=[
                "cohort",
                "accession_id",
                "mrn",
                "study_date",
                "h5_filename",
                "h5_path",
                "recon_h5_filename",
                "recon_h5_path",
                "dicom_h5_filename",
                "dicom_h5_path",
            ]
        )
    frame.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    experiment = load_experiment(args.experiment_config, args.experiment_name)
    base_args = dict(experiment["base_args"])

    h5_root = Path(base_args["h5_root"])
    labels_csv = Path(base_args["labels_csv"])
    pathology_csv = (
        Path(base_args["pathology_csv"]) if base_args.get("pathology_csv") else None
    )
    dicom_roots = [Path(path) for path in base_args["dicom_h5_roots"]]
    date_buffer_days = int(base_args.get("date_buffer_days", 7))
    gleason_positive_threshold = int(
        base_args.get("gleason_positive_grade_group_threshold", 2)
    )

    recon_labels = load_accession_labels(
        csv_path=labels_csv,
        h5_root=h5_root,
        date_buffer_days=date_buffer_days,
        pathology_csv=pathology_csv,
        gleason_positive_threshold=gleason_positive_threshold,
    )
    dicom_lookup = build_h5_lookup(dicom_roots)
    dicom_labels = [
        label
        for label in load_accession_labels_from_csv(
            labels_csv,
            pathology_csv=pathology_csv,
            gleason_positive_threshold=gleason_positive_threshold,
        )
        if label.accession_id in dicom_lookup
    ]

    aligned_labels, aligned_stats = get_aligned_cohort(
        accession_labels=recon_labels,
        recon_h5_root=h5_root,
        dicom_h5_roots=dicom_roots,
        require_raw_kspace=args.require_raw_kspace,
    )
    recon_only_labels = list(recon_labels)
    dicom_only_labels = list(dicom_labels)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix_source = (
        args.output_prefix
        or experiment.get("name")
        or args.experiment_config.stem
    )
    output_prefix = sanitize_stem(str(prefix_source))

    outputs = {
        "aligned": output_dir / f"{output_prefix}_aligned_manifest.csv",
        "recon_only": output_dir / f"{output_prefix}_recon_only_manifest.csv",
        "dicom_only": output_dir / f"{output_prefix}_dicom_only_manifest.csv",
    }

    write_manifest(
        aligned_labels,
        cohort_name="aligned",
        dicom_lookup=dicom_lookup,
        output_path=outputs["aligned"],
    )
    write_manifest(
        recon_only_labels,
        cohort_name="recon_only",
        dicom_lookup=dicom_lookup,
        output_path=outputs["recon_only"],
    )
    write_manifest(
        dicom_only_labels,
        cohort_name="dicom_only",
        dicom_lookup=dicom_lookup,
        output_path=outputs["dicom_only"],
    )

    print(
        f"[aligned] labeled={aligned_stats.labeled_count} "
        f"recon_h5={aligned_stats.recon_h5_count} "
        f"dicom={aligned_stats.dicom_count} "
        f"aligned={len(aligned_labels)} -> {outputs['aligned']}"
    )
    print(
        f"[recon_only] rows={len(recon_only_labels)} "
        f"overlap_with_aligned={len(aligned_labels)} -> {outputs['recon_only']}"
    )
    print(
        f"[dicom_only] rows={len(dicom_only_labels)} "
        f"overlap_with_aligned={len(aligned_labels)} -> {outputs['dicom_only']}"
    )


if __name__ == "__main__":
    main()

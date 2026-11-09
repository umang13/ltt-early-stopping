"""Run image checkpoint inference and write shared prediction CSVs."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
import yaml
from torch.utils.data import DataLoader

from prostate_kspace_classifier.data.h5_dataset import (
    DEFAULT_AVERAGING_TAG,
    DEFAULT_COIL_COMBINE,
    T2_ONLY_AVERAGING_TAG,
    create_image_dataset,
    get_aligned_cohort,
    resolve_image_input_keys,
)
from prostate_kspace_classifier.data.recipes import resolve_recipe_group_labels
from prostate_kspace_classifier.data.metadata_index import (
    AccessionLabel,
    consecutive_patient_holdout_split,
    load_accession_labels,
)
from prostate_kspace_classifier.models.image_metrics import (
    MODEL_VARIANT_SINGLE_STREAM,
    build_image_model,
)
from prostate_kspace_classifier.training.seed import seed_everything
from prostate_kspace_classifier.utils.logging import setup_logging


PREDICTION_METADATA_COLUMNS = [
    "patient_id",
    "study_datetime",
    "has_pretest_history",
    "num_pretest_accessions",
    "patient_total_accession_count",
    "patient_test_accession_count",
    "test_sequence_index",
    "is_first_test_accession",
    "is_last_test_accession",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference for image checkpoints and write prediction CSVs.",
    )
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--gleason-target-sensitivity", type=float, default=0.95)
    parser.add_argument("--gleason-target-specificity", type=float, default=0.95)
    return parser.parse_args()


def load_base_args(config_path: Path, experiment_name: str | None) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    experiments = payload.get("experiments", [])
    if not experiments:
        raise ValueError(f"No experiments found in {config_path}")

    if experiment_name is None:
        return dict(experiments[0]["base_args"])

    for experiment in experiments:
        if experiment.get("name") == experiment_name:
            return dict(experiment["base_args"])

    names = [str(experiment.get("name")) for experiment in experiments]
    raise ValueError(
        f"Experiment '{experiment_name}' not found. Known experiments: {names}"
    )


def uses_averaging_tags(image_source: str) -> bool:
    return image_source in {"recon", "hybrid"}


def resolved_averaging_tags(base_args: dict[str, Any]) -> list[str]:
    image_source = str(base_args.get("image_source", "recon"))
    image_metrics = str(base_args.get("image_metrics", "adc_b1500"))
    recipe_mode = base_args.get("recipe_mode")
    if recipe_mode:
        return resolve_recipe_group_labels(
            str(recipe_mode),
            base_args.get("recipe_family_quotas"),
        )
    if uses_averaging_tags(image_source):
        tags = base_args.get("averaging_tags")
        if tags:
            return list(tags)
        return [str(base_args.get("averaging_tag", DEFAULT_AVERAGING_TAG))]
    if image_source == "dicom" and image_metrics == "axt2":
        return [T2_ONLY_AVERAGING_TAG]
    return [str(base_args.get("averaging_tag", "none"))]


def patient_key(label: AccessionLabel) -> str:
    if label.patient_id is not None:
        return label.patient_id
    return label.accession_id


def label_time(label: AccessionLabel) -> pd.Timestamp:
    study_datetime = label.study_datetime
    if study_datetime is None or pd.isna(study_datetime):
        return pd.Timestamp.max
    return pd.Timestamp(study_datetime)


def resolve_device(device_arg: str) -> torch.device:
    desired = torch.device(device_arg)
    if desired.type == "cpu":
        return desired
    if torch.cuda.is_available():
        return desired
    return torch.device("cpu")


def load_checkpoint(checkpoint_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        config = payload.get("config", {})
    else:
        state_dict = payload
        config = {}

    cleaned_state_dict: dict[str, Any] = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned_state_dict[key[7:]] = value
            continue
        cleaned_state_dict[key] = value

    return cleaned_state_dict, config


def unpack_outputs(
    outputs: torch.Tensor | dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(outputs, dict):
        return outputs["pirads"], outputs.get("gleason")
    return outputs, None


def compute_split_start(
    aligned_labels: list[AccessionLabel],
    *,
    split_test_frac: float,
    split_test_start_date: str | None,
) -> pd.Timestamp:
    if split_test_start_date is not None:
        return pd.Timestamp(split_test_start_date)

    dated = [
        label
        for label in aligned_labels
        if label.study_datetime is not None and not pd.isna(label.study_datetime)
    ]
    if not dated:
        raise ValueError("No valid study datetimes found.")

    dated.sort(key=lambda label: (label_time(label), label.accession_id))
    n_test_tail = max(1, int(math.ceil(len(dated) * split_test_frac)))
    return label_time(dated[-n_test_tail])


def build_test_metadata(
    aligned_labels: list[AccessionLabel],
    test_labels: list[AccessionLabel],
    *,
    split_start: pd.Timestamp,
) -> pd.DataFrame:
    all_labels_by_patient: dict[str, list[AccessionLabel]] = defaultdict(list)
    for label in aligned_labels:
        all_labels_by_patient[patient_key(label)].append(label)

    test_labels_by_patient: dict[str, list[AccessionLabel]] = defaultdict(list)
    for label in test_labels:
        test_labels_by_patient[patient_key(label)].append(label)

    rows: list[dict[str, Any]] = []
    for patient_id, patient_test_labels in test_labels_by_patient.items():
        patient_all_labels = sorted(
            all_labels_by_patient[patient_id],
            key=lambda label: (label_time(label), label.accession_id),
        )
        patient_test_labels = sorted(
            patient_test_labels,
            key=lambda label: (label_time(label), label.accession_id),
        )

        num_pretest_accessions = 0
        for label in patient_all_labels:
            if label_time(label) < split_start:
                num_pretest_accessions += 1

        for test_index, label in enumerate(patient_test_labels):
            study_datetime = label.study_datetime
            if study_datetime is None or pd.isna(study_datetime):
                study_datetime_text = None
            else:
                study_datetime_text = pd.Timestamp(study_datetime).isoformat()

            rows.append(
                {
                    "accession_id": label.accession_id,
                    "patient_id": patient_id,
                    "study_datetime": study_datetime_text,
                    "has_pretest_history": num_pretest_accessions > 0,
                    "num_pretest_accessions": num_pretest_accessions,
                    "patient_total_accession_count": len(patient_all_labels),
                    "patient_test_accession_count": len(patient_test_labels),
                    "test_sequence_index": test_index,
                    "is_first_test_accession": test_index == 0,
                    "is_last_test_accession": test_index
                    == len(patient_test_labels) - 1,
                }
            )

    return pd.DataFrame(rows)


def build_label_metadata(accession_labels: list[AccessionLabel]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label in accession_labels:
        study_datetime = label.study_datetime
        rows.append(
            {
                "accession_id": label.accession_id,
                "patient_id": patient_key(label),
                "study_datetime": None
                if study_datetime is None or pd.isna(study_datetime)
                else pd.Timestamp(study_datetime).isoformat(),
            }
        )
    return pd.DataFrame(rows)


def build_context(
    base_args: dict[str, Any],
) -> tuple[
    list[AccessionLabel],
    list[AccessionLabel],
    pd.DataFrame,
    dict[str, Any],
    dict[str, int],
]:
    h5_root = Path(base_args["h5_root"])
    labels_csv = Path(base_args["labels_csv"])
    pathology_csv = (
        Path(base_args["pathology_csv"]) if base_args.get("pathology_csv") else None
    )
    dicom_roots = [Path(path) for path in base_args["dicom_h5_roots"]]

    labels = load_accession_labels(
        csv_path=labels_csv,
        h5_root=h5_root,
        date_buffer_days=int(base_args.get("date_buffer_days", 7)),
        pathology_csv=pathology_csv,
        gleason_positive_threshold=int(
            base_args.get("gleason_positive_grade_group_threshold", 2)
        ),
    )

    aligned_labels, cohort_stats = get_aligned_cohort(
        accession_labels=labels,
        recon_h5_root=h5_root,
        dicom_h5_roots=dicom_roots,
        require_raw_kspace=False,
    )

    split_strategy = str(base_args.get("split_strategy", "consecutive_patient_holdout"))
    if split_strategy != "consecutive_patient_holdout":
        raise ValueError(
            "This script expects split_strategy=consecutive_patient_holdout."
        )

    split_test_frac = float(base_args.get("split_test_frac", 0.10))
    split_test_start_date = base_args.get("split_test_start_date")
    split_pool_train_frac = float(base_args.get("split_pool_train_frac", 0.85))

    split_start = compute_split_start(
        aligned_labels,
        split_test_frac=split_test_frac,
        split_test_start_date=split_test_start_date,
    )

    _, val_ids, test_ids = consecutive_patient_holdout_split(
        aligned_labels,
        train_frac=split_pool_train_frac,
        test_frac=split_test_frac,
        test_start_date=split_test_start_date,
    )

    label_by_accession = {label.accession_id: label for label in aligned_labels}
    val_labels = [label_by_accession[accession_id] for accession_id in val_ids]
    test_labels = [label_by_accession[accession_id] for accession_id in test_ids]
    test_metadata = build_test_metadata(
        aligned_labels, test_labels, split_start=split_start
    )

    split_details = {
        "split_strategy": split_strategy,
        "split_test_frac": split_test_frac,
        "split_test_start_date": split_test_start_date,
        "split_pool_train_frac": split_pool_train_frac,
        "computed_split_start": split_start.isoformat(),
    }
    cohort_summary = {
        "aligned_count": int(cohort_stats.aligned_count),
        "labeled_count": int(cohort_stats.labeled_count),
        "recon_h5_count": int(cohort_stats.recon_h5_count),
        "dicom_count": int(cohort_stats.dicom_count),
    }
    return val_labels, test_labels, test_metadata, split_details, cohort_summary


def build_dataloader(
    accession_labels: list[AccessionLabel],
    *,
    base_args: dict[str, Any],
    split: str,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    image_source = str(base_args.get("image_source", "recon"))
    averaging_tags = resolved_averaging_tags(base_args)
    dicom_roots = [Path(path) for path in base_args.get("dicom_h5_roots", [])]
    dataset = create_image_dataset(
        accession_labels=accession_labels,
        h5_root=Path(base_args["h5_root"]),
        image_source=image_source,
        image_metrics=str(base_args["image_metrics"]),
        dicom_roots=dicom_roots or None,
        coil_combine=str(base_args.get("coil_combine", DEFAULT_COIL_COMBINE)),
        averaging_tag=averaging_tags[0],
        averaging_tags=averaging_tags if uses_averaging_tags(image_source) else None,
        recipe_mode=base_args.get("recipe_mode"),
        recipes_per_accession=base_args.get("recipes_per_accession"),
        recipe_family_quotas=base_args.get("recipe_family_quotas"),
        recipe_seed=int(base_args.get("recipe_seed", 1337)),
        recipe_split=split,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


def run_one_checkpoint(
    checkpoint_path: Path,
    dataloader: DataLoader,
    *,
    device: torch.device,
    image_source: str,
    image_metrics: str,
) -> pd.DataFrame:
    state_dict, checkpoint_config = load_checkpoint(checkpoint_path)
    model_config = checkpoint_config.get("model", {})
    model_variant = str(model_config.get("model_variant", MODEL_VARIANT_SINGLE_STREAM))

    model = build_image_model(
        model_variant=model_variant,
        input_keys=list(resolve_image_input_keys(image_source, image_metrics)),
        dropout=float(model_config.get("dropout", 0.3)),
        backbone=str(model_config.get("backbone", "r3d_18")),
        backbone_pretrained=bool(model_config.get("backbone_pretrained", False)),
        dual_encoder=bool(model_config.get("dual_encoder", False)),
        freeze_backbone=bool(model_config.get("freeze_backbone", False)),
        enable_gleason_head=bool(model_config.get("enable_gleason_head", False)),
        fusion_gate=bool(model_config.get("fusion_gate", False)),
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in dataloader:
            batch_on_device: dict[str, Any] = {}
            for key, value in batch.items():
                if torch.is_tensor(value):
                    batch_on_device[key] = value.to(device)
                    continue
                batch_on_device[key] = value

            outputs = model(batch_on_device)
            pirads_logits, gleason_logits = unpack_outputs(outputs)

            pirads_probs = torch.sigmoid(pirads_logits).detach().cpu().view(-1).tolist()
            if gleason_logits is None:
                gleason_probs = [float("nan")] * len(pirads_probs)
            else:
                gleason_probs = (
                    torch.sigmoid(gleason_logits).detach().cpu().view(-1).tolist()
                )

            accession_ids = list(batch["accession_id"])
            batch_averaging_tags = batch.get("averaging_tag")
            if batch_averaging_tags is None:
                averaging_tags = [None] * len(accession_ids)
            else:
                averaging_tags = list(batch_averaging_tags)
            batch_recipe_ids = batch.get("recipe_id")
            recipe_ids = [None] * len(accession_ids) if batch_recipe_ids is None else list(batch_recipe_ids)
            batch_recipe_families = batch.get("recipe_family")
            recipe_families = [None] * len(accession_ids) if batch_recipe_families is None else list(batch_recipe_families)
            pirads_labels = batch["label"].detach().cpu().view(-1).tolist()
            gleason_targets = batch["gleason_target"].detach().cpu().view(-1).tolist()
            gleason_masks = batch["gleason_mask"].detach().cpu().view(-1).tolist()
            grade_groups = batch["grade_group"].detach().cpu().view(-1).tolist()

            for index, accession_id in enumerate(accession_ids):
                gleason_label = None
                if gleason_masks[index] > 0.5:
                    gleason_label = int(gleason_targets[index])

                grade_group = int(grade_groups[index])
                if grade_group < 0:
                    grade_group = None

                rows.append(
                    {
                        "checkpoint_name": checkpoint_path.stem,
                        "checkpoint_path": str(checkpoint_path),
                        "accession_id": accession_id,
                        "averaging_tag": averaging_tags[index],
                        "recipe_id": recipe_ids[index],
                        "recipe_family": recipe_families[index],
                        "pirads_label": int(pirads_labels[index]),
                        "gleason_label": gleason_label,
                        "grade_group": grade_group,
                        "pirads_prob": float(pirads_probs[index]),
                        "gleason_prob": float(gleason_probs[index]),
                    }
                )

    return pd.DataFrame(rows)


def predict_split(
    checkpoint_paths: list[Path],
    dataloader: DataLoader,
    *,
    device: torch.device,
    image_source: str,
    image_metrics: str,
    metadata_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for checkpoint_path in checkpoint_paths:
        frame = run_one_checkpoint(
            checkpoint_path,
            dataloader,
            device=device,
            image_source=image_source,
            image_metrics=image_metrics,
        )
        if metadata_table is not None and not metadata_table.empty:
            frame = frame.merge(
                metadata_table,
                on="accession_id",
                how="left",
                validate="many_to_one",
            )
        frames.append(frame)

    if not frames:
        raise ValueError("No checkpoint predictions were produced.")
    return pd.concat(frames, ignore_index=True)


def build_ensemble_predictions(seed_predictions: pd.DataFrame) -> pd.DataFrame:
    aggregation = {
        "pirads_label": ("pirads_label", "first"),
        "gleason_label": ("gleason_label", "first"),
        "grade_group": ("grade_group", "first"),
        "pirads_prob": ("pirads_prob", "mean"),
        "gleason_prob": ("gleason_prob", "mean"),
        "num_checkpoints": ("checkpoint_name", "nunique"),
    }
    for column in PREDICTION_METADATA_COLUMNS:
        if column in seed_predictions.columns:
            aggregation[column] = (column, "first")

    ensemble_predictions = seed_predictions.groupby(
        ["accession_id", "averaging_tag"],
        as_index=False,
    ).agg(**aggregation)

    ensemble_predictions["checkpoint_name"] = "ensemble_mean"
    ensemble_predictions["checkpoint_path"] = None

    ordered_columns = [
        "checkpoint_name",
        "checkpoint_path",
        "accession_id",
        "averaging_tag",
        "pirads_label",
        "gleason_label",
        "grade_group",
    ]
    for column in PREDICTION_METADATA_COLUMNS:
        if column in ensemble_predictions.columns:
            ordered_columns.append(column)

    ordered_columns.extend(
        [
            "pirads_prob",
            "gleason_prob",
            "num_checkpoints",
        ]
    )

    return ensemble_predictions[ordered_columns]


def compute_gleason_auroc(frame: pd.DataFrame) -> float:
    labeled = frame.dropna(subset=["gleason_label", "gleason_prob"]).copy()
    if labeled.empty:
        return float("nan")

    labels = labeled["gleason_label"].astype(int)
    if labels.nunique() < 2:
        return float("nan")

    probs = labeled["gleason_prob"].astype(float)
    return float(roc_auc_score(labels, probs))


def build_gleason_auroc_rows(
    frame: pd.DataFrame,
    *,
    averaging_tags: list[str],
    checkpoint_name: str | None = None,
    checkpoint_path: str | None = None,
) -> list[dict[str, Any]]:
    labeled = frame.dropna(subset=["gleason_label", "gleason_prob"]).copy()
    rows = [
        {
            "checkpoint_name": checkpoint_name,
            "checkpoint_path": checkpoint_path,
            "averaging_tag": "overall",
            "auroc": compute_gleason_auroc(labeled),
            "n": int(len(labeled)),
        }
    ]

    for averaging_tag in averaging_tags:
        tag_frame = labeled[labeled["averaging_tag"] == averaging_tag].copy()
        rows.append(
            {
                "checkpoint_name": checkpoint_name,
                "checkpoint_path": checkpoint_path,
                "averaging_tag": averaging_tag,
                "auroc": compute_gleason_auroc(tag_frame),
                "n": int(len(tag_frame)),
            }
        )

    return rows


def summarize_seed_gleason_auroc(
    seed_predictions: pd.DataFrame,
    *,
    averaging_tags: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for checkpoint_name, checkpoint_frame in seed_predictions.groupby(
        "checkpoint_name",
        sort=False,
    ):
        checkpoint_paths = checkpoint_frame["checkpoint_path"].dropna().astype(str).unique()
        checkpoint_path = checkpoint_paths[0] if len(checkpoint_paths) > 0 else None
        rows.extend(
            build_gleason_auroc_rows(
                checkpoint_frame,
                averaging_tags=averaging_tags,
                checkpoint_name=str(checkpoint_name),
                checkpoint_path=checkpoint_path,
            )
        )

    return pd.DataFrame(rows)


def summarize_ensemble_gleason_auroc(
    ensemble_predictions: pd.DataFrame,
    *,
    averaging_tags: list[str],
) -> pd.DataFrame:
    return pd.DataFrame(
        build_gleason_auroc_rows(
            ensemble_predictions,
            averaging_tags=averaging_tags,
            checkpoint_name="ensemble_mean",
            checkpoint_path=None,
        )
    )


def build_main_gleason_auroc_table(
    seed_gleason_auroc: pd.DataFrame,
    ensemble_gleason_auroc: pd.DataFrame,
    *,
    averaging_tags: list[str],
) -> pd.DataFrame:
    seed_summary = (
        seed_gleason_auroc.groupby("averaging_tag", as_index=False)
        .agg(
            seed_mean_auroc=("auroc", "mean"),
            seed_std_auroc=("auroc", "std"),
        )
    )
    ensemble_summary = ensemble_gleason_auroc[
        ["averaging_tag", "auroc", "n"]
    ].rename(
        columns={
            "auroc": "ensemble_auroc",
        }
    )

    main_table = seed_summary.merge(
        ensemble_summary,
        on="averaging_tag",
        how="left",
    )
    tag_order = ["overall", *averaging_tags]
    main_table["averaging_tag"] = pd.Categorical(
        main_table["averaging_tag"],
        categories=tag_order,
        ordered=True,
    )
    return main_table.sort_values("averaging_tag").reset_index(drop=True)


def compute_threshold_stats(
    labeled_rows: pd.DataFrame, threshold: float
) -> dict[str, Any]:
    labels = labeled_rows["gleason_label"].astype(int)
    probs = labeled_rows["gleason_prob"].astype(float)
    predicted_positive = probs >= threshold
    actual_positive = labels == 1
    actual_negative = labels == 0

    tp = int((predicted_positive & actual_positive).sum())
    fp = int((predicted_positive & actual_negative).sum())
    tn = int((~predicted_positive & actual_negative).sum())
    fn = int((~predicted_positive & actual_positive).sum())

    positive_count = tp + fn
    negative_count = tn + fp

    sensitivity = float("nan")
    specificity = float("nan")
    if positive_count > 0:
        sensitivity = tp / positive_count
    if negative_count > 0:
        specificity = tn / negative_count

    return {
        "threshold": float(threshold),
        "labeled_count": int(len(labeled_rows)),
        "positive_count": int(positive_count),
        "negative_count": int(negative_count),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def pick_threshold(
    labeled_rows: pd.DataFrame,
    *,
    selection_metric: str,
    target_value: float,
    threshold_scope: str,
    threshold_tag: str,
) -> dict[str, Any]:
    result = {
        "threshold_scope": threshold_scope,
        "threshold_tag": threshold_tag,
        "selection_metric": selection_metric,
        "target_value": float(target_value),
        "threshold": float("nan"),
        "target_met": False,
        "labeled_count": 0,
        "positive_count": 0,
        "negative_count": 0,
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0,
        "sensitivity": float("nan"),
        "specificity": float("nan"),
    }
    if labeled_rows.empty:
        return result

    labels = labeled_rows["gleason_label"].astype(int)
    result["labeled_count"] = int(len(labels))
    result["positive_count"] = int((labels == 1).sum())
    result["negative_count"] = int((labels == 0).sum())
    if result["positive_count"] == 0 or result["negative_count"] == 0:
        return result

    thresholds = sorted(set(labeled_rows["gleason_prob"].astype(float)))
    if thresholds[0] > 0.0:
        thresholds = [0.0] + thresholds
    if thresholds[-1] < 1.0:
        thresholds.append(1.0)

    best_stats = None

    if selection_metric == "sensitivity":
        for threshold in thresholds:
            stats = compute_threshold_stats(labeled_rows, threshold)
            if pd.isna(stats["sensitivity"]) or stats["sensitivity"] < target_value:
                continue
            if best_stats is None or threshold > best_stats["threshold"]:
                best_stats = stats

    elif selection_metric == "specificity":
        for threshold in thresholds:
            stats = compute_threshold_stats(labeled_rows, threshold)
            if pd.isna(stats["specificity"]) or stats["specificity"] < target_value:
                continue
            if best_stats is None or threshold < best_stats["threshold"]:
                best_stats = stats

    else:
        raise ValueError(f"Unsupported selection_metric '{selection_metric}'")

    if best_stats is None:
        return result

    result.update(best_stats)
    result["target_met"] = True
    return result


def select_gleason_thresholds(
    val_ensemble_predictions: pd.DataFrame,
    *,
    averaging_tags: list[str],
    target_sensitivity: float,
    target_specificity: float,
) -> pd.DataFrame:
    labeled = val_ensemble_predictions.dropna(
        subset=["gleason_label", "gleason_prob"]
    ).copy()
    labeled["gleason_label"] = labeled["gleason_label"].astype(int)
    labeled["gleason_prob"] = labeled["gleason_prob"].astype(float)
    threshold_specs = [
        ("sensitivity", target_sensitivity),
        ("specificity", target_specificity),
    ]

    rows: list[dict[str, Any]] = []
    for selection_metric, target_value in threshold_specs:
        rows.append(
            pick_threshold(
                labeled,
                selection_metric=selection_metric,
                target_value=target_value,
                threshold_scope="overall",
                threshold_tag="overall",
            )
        )

    for averaging_tag in averaging_tags:
        tag_rows = labeled[labeled["averaging_tag"] == averaging_tag].copy()
        for selection_metric, target_value in threshold_specs:
            rows.append(
                pick_threshold(
                    tag_rows,
                    selection_metric=selection_metric,
                    target_value=target_value,
                    threshold_scope="per_tag",
                    threshold_tag=averaging_tag,
                )
            )

    threshold_table = pd.DataFrame(rows)
    if threshold_table.empty:
        return threshold_table

    threshold_table["threshold_id"] = (
        threshold_table["threshold_scope"].astype(str)
        + "__"
        + threshold_table["threshold_tag"].astype(str)
        + "__"
        + threshold_table["selection_metric"].astype(str)
        + "__target_"
        + threshold_table["target_value"].map(lambda value: f"{value:.3f}")
    )
    ordered_columns = [
        "threshold_id",
        "threshold_scope",
        "threshold_tag",
        "selection_metric",
        "target_value",
        "threshold",
        "target_met",
        "labeled_count",
        "positive_count",
        "negative_count",
        "tp",
        "fp",
        "tn",
        "fn",
        "sensitivity",
        "specificity",
    ]
    return threshold_table[ordered_columns]


def apply_thresholds_to_test(
    test_ensemble_predictions: pd.DataFrame,
    threshold_table: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if test_ensemble_predictions.empty or threshold_table.empty:
        return pd.DataFrame(rows)

    for threshold_row in threshold_table.itertuples(index=False):
        if pd.isna(threshold_row.threshold):
            continue

        if threshold_row.threshold_scope == "overall":
            applicable_rows = test_ensemble_predictions
        else:
            applicable_rows = test_ensemble_predictions[
                test_ensemble_predictions["averaging_tag"]
                == threshold_row.threshold_tag
            ]

        for row in applicable_rows.itertuples(index=False):
            predicted_positive = None
            if not pd.isna(row.gleason_prob):
                predicted_positive = int(
                    float(row.gleason_prob) >= float(threshold_row.threshold)
                )

            rows.append(
                {
                    "threshold_id": threshold_row.threshold_id,
                    "threshold_scope": threshold_row.threshold_scope,
                    "threshold_tag": threshold_row.threshold_tag,
                    "selection_metric": threshold_row.selection_metric,
                    "target_value": threshold_row.target_value,
                    "threshold": threshold_row.threshold,
                    "accession_id": row.accession_id,
                    "averaging_tag": row.averaging_tag,
                    "gleason_label": row.gleason_label,
                    "gleason_prob": row.gleason_prob,
                    "predicted_positive": predicted_positive,
                }
            )

    return pd.DataFrame(rows)


def write_outputs(
    output_dir: Path,
    *,
    base_args: dict[str, Any],
    config_path: Path,
    experiment_name: str | None,
    checkpoint_paths: list[Path],
    device: torch.device,
    cohort_summary: dict[str, int],
    split_details: dict[str, Any],
    target_sensitivity: float,
    target_specificity: float,
    val_labels: list[AccessionLabel],
    test_labels: list[AccessionLabel],
    averaging_tags: list[str],
    val_seed_predictions: pd.DataFrame,
    val_ensemble_predictions: pd.DataFrame,
    test_seed_predictions: pd.DataFrame,
    test_ensemble_predictions: pd.DataFrame,
    val_ensemble_gleason_auroc: pd.DataFrame,
    test_seed_gleason_auroc: pd.DataFrame,
    test_ensemble_gleason_auroc: pd.DataFrame,
    test_main_gleason_auroc_table: pd.DataFrame,
    threshold_table: pd.DataFrame,
    thresholded_test_predictions: pd.DataFrame,
) -> None:
    val_seed_path = output_dir / "val_seed_predictions.csv"
    val_ensemble_path = output_dir / "val_ensemble_predictions.csv"
    test_seed_path = output_dir / "test_seed_predictions.csv"
    test_ensemble_path = output_dir / "test_ensemble_predictions.csv"
    val_ensemble_gleason_auroc_path = output_dir / "val_ensemble_gleason_auroc.csv"
    test_seed_gleason_auroc_path = output_dir / "test_seed_gleason_auroc.csv"
    test_ensemble_gleason_auroc_path = output_dir / "test_ensemble_gleason_auroc.csv"
    test_main_gleason_auroc_table_path = (
        output_dir / "test_main_gleason_auroc_table.csv"
    )
    legacy_test_seed_path = output_dir / "seed_predictions.csv"
    legacy_test_ensemble_path = output_dir / "ensemble_predictions.csv"
    threshold_path = output_dir / "gleason_thresholds.csv"
    applied_threshold_path = output_dir / "test_thresholded_predictions.csv"

    val_seed_predictions.to_csv(val_seed_path, index=False)
    val_ensemble_predictions.to_csv(val_ensemble_path, index=False)
    test_seed_predictions.to_csv(test_seed_path, index=False)
    test_ensemble_predictions.to_csv(test_ensemble_path, index=False)
    val_ensemble_gleason_auroc.to_csv(val_ensemble_gleason_auroc_path, index=False)
    test_seed_gleason_auroc.to_csv(test_seed_gleason_auroc_path, index=False)
    test_ensemble_gleason_auroc.to_csv(
        test_ensemble_gleason_auroc_path,
        index=False,
    )
    test_main_gleason_auroc_table.to_csv(
        test_main_gleason_auroc_table_path,
        index=False,
    )
    test_seed_predictions.to_csv(legacy_test_seed_path, index=False)
    test_ensemble_predictions.to_csv(legacy_test_ensemble_path, index=False)
    threshold_table.to_csv(threshold_path, index=False)
    thresholded_test_predictions.to_csv(applied_threshold_path, index=False)

    run_metadata = {
        "cohort": base_args.get("cohort"),
        "validation_policy": base_args.get("validation_policy"),
        "coil_combine": str(base_args.get("coil_combine", DEFAULT_COIL_COMBINE)),
        "image_source": str(base_args.get("image_source", "recon")),
        "image_metrics": str(base_args["image_metrics"]),
        "model_variant": str(base_args.get("model_variant", MODEL_VARIANT_SINGLE_STREAM)),
        "fusion_gate": bool(base_args.get("fusion_gate", False)),
    }

    summary = {
        "config_path": str(config_path),
        "experiment_name": experiment_name,
        "checkpoints": [str(path) for path in checkpoint_paths],
        "checkpoint_count": len(checkpoint_paths),
        "device_used": str(device),
        "run_metadata": run_metadata,
        "cohort_stats": cohort_summary,
        "split_details": split_details,
        "gleason_threshold_targets": {
            "sensitivity": target_sensitivity,
            "specificity": target_specificity,
        },
        "val_accession_count": len(val_labels),
        "val_seed_row_count": int(len(val_seed_predictions)),
        "val_row_count": int(len(val_ensemble_predictions)),
        "test_accession_count": len(test_labels),
        "test_seed_row_count": int(len(test_seed_predictions)),
        "test_row_count": int(len(test_ensemble_predictions)),
        "averaging_tags": averaging_tags,
        "outputs": {
            "val_seed_predictions": str(val_seed_path),
            "val_ensemble_predictions": str(val_ensemble_path),
            "test_seed_predictions": str(test_seed_path),
            "test_ensemble_predictions": str(test_ensemble_path),
            "val_ensemble_gleason_auroc": str(val_ensemble_gleason_auroc_path),
            "test_seed_gleason_auroc": str(test_seed_gleason_auroc_path),
            "test_ensemble_gleason_auroc": str(test_ensemble_gleason_auroc_path),
            "test_main_gleason_auroc_table": str(
                test_main_gleason_auroc_table_path
            ),
            "seed_predictions": str(legacy_test_seed_path),
            "ensemble_predictions": str(legacy_test_ensemble_path),
            "gleason_thresholds": str(threshold_path),
            "test_thresholded_predictions": str(applied_threshold_path),
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.gleason_target_sensitivity <= 1.0:
        raise ValueError(
            f"gleason_target_sensitivity must be in (0, 1], got {args.gleason_target_sensitivity}"
        )
    if not 0.0 < args.gleason_target_specificity <= 1.0:
        raise ValueError(
            f"gleason_target_specificity must be in (0, 1], got {args.gleason_target_specificity}"
        )

    setup_logging()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_args = load_base_args(args.experiment_config, args.experiment_name)
    (
        val_labels,
        test_labels,
        test_metadata,
        split_details,
        cohort_summary,
    ) = build_context(base_args)
    val_metadata = build_label_metadata(val_labels)

    val_dataloader = build_dataloader(
        val_labels,
        base_args=base_args,
        split="val",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    test_dataloader = build_dataloader(
        test_labels,
        base_args=base_args,
        split="test",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    device = resolve_device(args.device)
    image_source = str(base_args.get("image_source", "recon"))
    image_metrics = str(base_args["image_metrics"])
    averaging_tags = resolved_averaging_tags(base_args)

    val_seed_predictions = predict_split(
        args.checkpoint,
        val_dataloader,
        device=device,
        image_source=image_source,
        image_metrics=image_metrics,
        metadata_table=val_metadata,
    )
    test_seed_predictions = predict_split(
        args.checkpoint,
        test_dataloader,
        device=device,
        image_source=image_source,
        image_metrics=image_metrics,
        metadata_table=test_metadata,
    )

    val_ensemble_predictions = build_ensemble_predictions(val_seed_predictions)
    test_ensemble_predictions = build_ensemble_predictions(test_seed_predictions)
    val_ensemble_gleason_auroc = summarize_ensemble_gleason_auroc(
        val_ensemble_predictions,
        averaging_tags=averaging_tags,
    )
    test_seed_gleason_auroc = summarize_seed_gleason_auroc(
        test_seed_predictions,
        averaging_tags=averaging_tags,
    )
    test_ensemble_gleason_auroc = summarize_ensemble_gleason_auroc(
        test_ensemble_predictions,
        averaging_tags=averaging_tags,
    )
    test_main_gleason_auroc_table = build_main_gleason_auroc_table(
        test_seed_gleason_auroc,
        test_ensemble_gleason_auroc,
        averaging_tags=averaging_tags,
    )
    threshold_table = select_gleason_thresholds(
        val_ensemble_predictions,
        averaging_tags=averaging_tags,
        target_sensitivity=args.gleason_target_sensitivity,
        target_specificity=args.gleason_target_specificity,
    )
    thresholded_test_predictions = apply_thresholds_to_test(
        test_ensemble_predictions,
        threshold_table,
    )

    write_outputs(
        args.output_dir,
        base_args=base_args,
        config_path=args.experiment_config,
        experiment_name=args.experiment_name,
        checkpoint_paths=args.checkpoint,
        device=device,
        cohort_summary=cohort_summary,
        split_details=split_details,
        target_sensitivity=args.gleason_target_sensitivity,
        target_specificity=args.gleason_target_specificity,
        val_labels=val_labels,
        test_labels=test_labels,
        averaging_tags=averaging_tags,
        val_seed_predictions=val_seed_predictions,
        val_ensemble_predictions=val_ensemble_predictions,
        test_seed_predictions=test_seed_predictions,
        test_ensemble_predictions=test_ensemble_predictions,
        val_ensemble_gleason_auroc=val_ensemble_gleason_auroc,
        test_seed_gleason_auroc=test_seed_gleason_auroc,
        test_ensemble_gleason_auroc=test_ensemble_gleason_auroc,
        test_main_gleason_auroc_table=test_main_gleason_auroc_table,
        threshold_table=threshold_table,
        thresholded_test_predictions=thresholded_test_predictions,
    )

    print(
        f"Analyzed {len(args.checkpoint)} checkpoints on device={device} "
        f"val_accessions={len(val_labels)} val_seed_rows={len(val_seed_predictions)} "
        f"val_rows={len(val_ensemble_predictions)} "
        f"test_accessions={len(test_labels)} test_seed_rows={len(test_seed_predictions)} "
        f"test_rows={len(test_ensemble_predictions)}"
    )
    print(f"Wrote {args.output_dir / 'val_seed_predictions.csv'}")
    print(f"Wrote {args.output_dir / 'val_ensemble_predictions.csv'}")
    print(f"Wrote {args.output_dir / 'test_seed_predictions.csv'}")
    print(f"Wrote {args.output_dir / 'test_ensemble_predictions.csv'}")
    print(f"Wrote {args.output_dir / 'val_ensemble_gleason_auroc.csv'}")
    print(f"Wrote {args.output_dir / 'test_seed_gleason_auroc.csv'}")
    print(f"Wrote {args.output_dir / 'test_ensemble_gleason_auroc.csv'}")
    print(f"Wrote {args.output_dir / 'test_main_gleason_auroc_table.csv'}")
    print(f"[compat] Wrote {args.output_dir / 'seed_predictions.csv'}")
    print(f"[compat] Wrote {args.output_dir / 'ensemble_predictions.csv'}")
    print(f"Wrote {args.output_dir / 'gleason_thresholds.csv'}")
    print(f"Wrote {args.output_dir / 'test_thresholded_predictions.csv'}")
    print(f"Wrote {args.output_dir / 'summary.json'}")
    print("Test Gleason AUROC summary:")
    print(test_main_gleason_auroc_table.to_string(index=False))


if __name__ == "__main__":
    main()

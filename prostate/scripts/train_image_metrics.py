"""Train the paired image-metrics classifier end-to-end."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd
import torch
from torch.utils.data import DataLoader

from prostate_kspace_classifier.data.h5_dataset import (
    DEFAULT_AVERAGING_TAG,
    DEFAULT_COIL_COMBINE,
    IMAGE_SOURCE_CHOICES,
    IMAGE_METRICS_CHOICES,
    create_image_dataset,
    get_aligned_cohort,
    resolve_image_input_keys,
    validate_image_source_metrics,
)
from prostate_kspace_classifier.data.metadata_index import (
    AccessionLabel,
    consecutive_patient_holdout_split,
    list_dicom_accessions,
    load_accession_labels,
    load_accession_labels_from_csv,
    summarize_label_distribution,
    temporal_split,
)
from prostate_kspace_classifier.data.recipes import (
    DEFAULT_RECIPE_FAMILY_QUOTAS,
    DEFAULT_RECIPE_SEED,
    DEFAULT_RECIPES_PER_ACCESSION,
    RECIPE_MODE_CHOICES,
    RECIPE_MODE_LEGACY_TAGS,
    RECIPE_MODE_SAMPLED_RECIPES,
    RECIPE_MODE_XYZ_GRID48,
    parse_recipe_family_quotas,
)
from prostate_kspace_classifier.data.recipes.manifests import (
    build_recipe_manifest_rows,
    write_recipe_manifest,
)
from prostate_kspace_classifier.data.transforms_image import build_train_augmentations
from prostate_kspace_classifier.models.image_metrics import (
    MODEL_VARIANT_CHOICES,
    MODEL_VARIANT_SINGLE_STREAM,
    MODEL_VARIANT_T2_RECON_DWI_FUSION,
    ProstateImageClassifier,
    T2ReconDWIFusionClassifier,
    FUSION_ARCHITECTURE_SHARED_BOTTLENECK_V1,
    build_image_model,
)
from prostate_kspace_classifier.training.engine import evaluate, train_one_epoch
from prostate_kspace_classifier.training.schedulers import create_epoch_scheduler
from prostate_kspace_classifier.training.seed import seed_everything
from prostate_kspace_classifier.training.wandb_logging import WandbLogger
from prostate_kspace_classifier.utils.logging import setup_logging


DEFAULT_WANDB_PROJECT = "prostate-kspace-classifier"
COHORT_CHOICES = ("recon_only", "dicom_only", "aligned")
SPLIT_STRATEGY_CHOICES = ("temporal", "consecutive_patient_holdout")
VALIDATION_POLICY_CHOICES = ("aligned_holdout", "source_internal")
AMP_DTYPE_CHOICES = ("none", "bf16")
SOURCE_VAL_FRAC = 0.15
HYBRID_MODEL_VARIANTS = (
    MODEL_VARIANT_T2_RECON_DWI_FUSION,
)


def _uses_averaging_tags(args: argparse.Namespace) -> bool:
    return args.image_source in {"recon", "hybrid"} and getattr(args, "recipe_mode", None) is None


def _uses_recipe_mode(args: argparse.Namespace) -> bool:
    return getattr(args, "recipe_mode", None) is not None


def _resolved_averaging_tags(args: argparse.Namespace) -> List[str]:
    if not _uses_averaging_tags(args):
        return []
    averaging_tags = getattr(args, "averaging_tags", None)
    return list(averaging_tags) if averaging_tags else [args.averaging_tag]


def _averaging_tag_run_name(args: argparse.Namespace) -> str:
    averaging_tags = _resolved_averaging_tags(args)
    if len(averaging_tags) == 1:
        return averaging_tags[0]
    return f"avgset{len(averaging_tags)}"


def _averaging_tag_config_values(args: argparse.Namespace) -> Dict[str, Any]:
    averaging_tags = _resolved_averaging_tags(args)
    values: Dict[str, Any] = {"averaging_tag": averaging_tags[0]}
    if len(averaging_tags) > 1:
        values["averaging_tags"] = averaging_tags
    return values


def _resolved_recipe_family_quotas(args: argparse.Namespace) -> list[str]:
    if args.recipe_mode != RECIPE_MODE_SAMPLED_RECIPES:
        return []
    quotas = list(args.recipe_family_quotas) if args.recipe_family_quotas else list(DEFAULT_RECIPE_FAMILY_QUOTAS)
    return quotas


def _resolved_recipes_per_accession(args: argparse.Namespace) -> int | None:
    if args.recipe_mode != RECIPE_MODE_SAMPLED_RECIPES:
        return None
    return (
        int(args.recipes_per_accession)
        if args.recipes_per_accession is not None
        else DEFAULT_RECIPES_PER_ACCESSION
    )


def _recipe_mode_run_name(args: argparse.Namespace) -> str | None:
    if not _uses_recipe_mode(args):
        return None
    if args.recipe_mode == RECIPE_MODE_LEGACY_TAGS:
        return "legacy15"
    if args.recipe_mode == RECIPE_MODE_XYZ_GRID48:
        return "xyz48"
    if args.recipe_mode == RECIPE_MODE_SAMPLED_RECIPES:
        return f"sampled{_resolved_recipes_per_accession(args)}"
    return args.recipe_mode


def _recipe_config_values(args: argparse.Namespace) -> Dict[str, Any]:
    values: Dict[str, Any] = {"recipe_mode": args.recipe_mode}
    if args.recipe_mode == RECIPE_MODE_SAMPLED_RECIPES:
        values["recipes_per_accession"] = _resolved_recipes_per_accession(args)
        values["recipe_family_quotas"] = _resolved_recipe_family_quotas(args)
        values["recipe_seed"] = int(args.recipe_seed)
    return values


def _is_hybrid_model_variant(model_variant: str) -> bool:
    return model_variant in HYBRID_MODEL_VARIANTS


def _model_variant_run_name(args: argparse.Namespace) -> str:
    if args.model_variant == MODEL_VARIANT_T2_RECON_DWI_FUSION:
        gate_tag = "gate" if getattr(args, "fusion_gate", False) else "nogate"
        return f"{MODEL_VARIANT_T2_RECON_DWI_FUSION}_{gate_tag}"
    if args.model_variant != MODEL_VARIANT_SINGLE_STREAM:
        return args.model_variant
    if getattr(args, "dual_encoder", False):
        return "legacy_dual_encoder"
    return MODEL_VARIANT_SINGLE_STREAM


def _initialized_fusion_branches(args: argparse.Namespace) -> tuple[str, ...]:
    branches: List[str] = []
    if getattr(args, "t2_branch_init_from_checkpoint", None) is not None:
        branches.append("t2")
    if getattr(args, "dwi_branch_init_from_checkpoint", None) is not None:
        branches.append("dwi")
    return tuple(branches)


def _initialization_run_name(args: argparse.Namespace) -> str:
    if _is_hybrid_model_variant(args.model_variant):
        initialized_branches = _initialized_fusion_branches(args)
        if not initialized_branches:
            return "noinit"
        branch_tag = (
            "both" if len(initialized_branches) == 2 else initialized_branches[0]
        )
        return f"init{branch_tag}_fz{int(args.freeze_initialized_branch_epochs)}"

    if getattr(args, "init_from_checkpoint", None):
        return (
            f"initdicom_{getattr(args, 'init_scope', 'full')}_"
            f"fz{int(getattr(args, 'init_freeze_backbone_epochs', 0))}"
        )
    return "noinit"


def _lr_run_name(args: argparse.Namespace) -> str:
    return f"lr{args.lr:.0e}"


def _validate_image_args(args: argparse.Namespace) -> None:
    validate_image_source_metrics(args.image_source, args.image_metrics)
    initialized_fusion_branches = _initialized_fusion_branches(args)

    if _uses_recipe_mode(args):
        if args.image_source not in {"recon", "hybrid"}:
            raise ValueError("recipe_mode is only supported for recon and hybrid image sources")
        if args.coil_combine != "espirit":
            raise ValueError("recipe_mode currently supports only --coil-combine=espirit")
        if args.image_metrics != "adc_b1500":
            raise ValueError("recipe_mode currently supports only --image-metrics=adc_b1500")
        if args.recipe_mode == RECIPE_MODE_SAMPLED_RECIPES:
            quotas = parse_recipe_family_quotas(_resolved_recipe_family_quotas(args))
            recipes_per_accession = _resolved_recipes_per_accession(args)
            if sum(count for _, count in quotas) != recipes_per_accession:
                raise ValueError(
                    "recipes_per_accession must equal the sum of recipe_family_quotas"
                )

    if args.validation_policy == "source_internal":
        if args.cohort == "aligned":
            raise ValueError(
                "validation_policy=source_internal is only supported for source cohorts."
            )
        if args.model_variant != MODEL_VARIANT_SINGLE_STREAM:
            raise ValueError(
                "validation_policy=source_internal is only supported for single-stream source runs."
            )

    if args.fusion_gate and args.model_variant != MODEL_VARIANT_T2_RECON_DWI_FUSION:
        raise ValueError("fusion_gate is only supported with t2_recon_dwi_fusion")

    if initialized_fusion_branches and not _is_hybrid_model_variant(args.model_variant):
        raise ValueError(
            "Fusion branch checkpoint initialization requires a hybrid fusion model variant."
        )
    if initialized_fusion_branches and args.init_from_checkpoint is not None:
        raise ValueError(
            "Use fusion branch init fields or legacy init_from_checkpoint, not both."
        )
    if not initialized_fusion_branches and args.freeze_initialized_branch_epochs != 0:
        raise ValueError(
            "freeze_initialized_branch_epochs requires at least one fusion branch init checkpoint."
        )

    if _is_hybrid_model_variant(args.model_variant):
        if args.image_source != "hybrid":
            raise ValueError(
                f"model_variant={args.model_variant} requires --image-source=hybrid"
            )
        if args.image_metrics != "adc_b1500":
            raise ValueError(
                f"{args.model_variant} currently supports only --image-metrics=adc_b1500"
            )
        if args.dual_encoder:
            raise ValueError(
                f"dual_encoder is not supported with model_variant={args.model_variant}"
            )
        if args.init_from_checkpoint is not None:
            raise ValueError(
                f"Legacy init_from_checkpoint is not supported with {args.model_variant}"
            )
        if args.init_scope != "full":
            raise ValueError(
                "init_scope is only supported for legacy single-stream transfer runs"
            )
        if args.init_freeze_backbone_epochs != 0:
            raise ValueError(
                "init_freeze_backbone_epochs is only supported for legacy single-stream transfer runs"
            )
        if args.freeze_backbone:
            raise ValueError(
                f"freeze_backbone is not supported with {args.model_variant}"
            )
        if args.backbone_pretrained:
            raise ValueError(
                f"{args.model_variant} should train from scratch without pretrained backbones"
            )
    elif args.image_source == "hybrid":
        raise ValueError(
            "--image-source=hybrid requires a hybrid fusion model variant"
        )

    if args.image_source == "dicom" and args.image_metrics == "axt2":
        if args.model_variant != MODEL_VARIANT_SINGLE_STREAM:
            raise ValueError(
                "T2-only DICOM training requires --model-variant=single_stream"
            )
        if args.dual_encoder:
            raise ValueError("dual_encoder is not supported for T2-only runs")
        if args.init_from_checkpoint is not None:
            raise ValueError(
                "T2-only source and baseline runs should train from scratch."
            )
        if args.init_scope != "full":
            raise ValueError(
                "init_scope is only supported for legacy single-stream transfer runs"
            )
        if args.init_freeze_backbone_epochs != 0:
            raise ValueError(
                "init_freeze_backbone_epochs is not supported for T2-only runs"
            )
        if args.freeze_backbone:
            raise ValueError("freeze_backbone is not supported for T2-only runs")
        if args.backbone_pretrained:
            raise ValueError(
                "T2-only runs should train from scratch without pretrained backbones"
            )


def default_run_name(args: argparse.Namespace) -> str:
    init_tag = _initialization_run_name(args)
    group_run_name = _recipe_mode_run_name(args) if _uses_recipe_mode(args) else _averaging_tag_run_name(args) if _uses_averaging_tags(args) else None
    suffix = [
        args.image_source,
        "image",
        args.cohort,
        args.split_strategy if args.split_strategy != "temporal" else None,
        "mtl" if args.enable_gleason_task else "pirads",
        args.coil_combine if args.image_source in {"recon", "hybrid"} else None,
        group_run_name,
        args.image_metrics,
        _model_variant_run_name(args),
        args.backbone,
        "pt" if args.backbone_pretrained else "scratch",
        _lr_run_name(args),
        f"wd{args.weight_decay:.0e}",
        f"bs{args.batch_size}",
        f"seed{args.seed}",
        init_tag,
    ]
    return "-".join([item for item in suffix if item])


def legacy_single_stream_run_name(args: argparse.Namespace) -> str | None:
    if args.model_variant != MODEL_VARIANT_SINGLE_STREAM:
        return None
    init_tag = _initialization_run_name(args)
    group_run_name = _recipe_mode_run_name(args) if _uses_recipe_mode(args) else _averaging_tag_run_name(args) if _uses_averaging_tags(args) else None
    suffix = [
        args.image_source,
        "image",
        args.cohort,
        args.split_strategy if args.split_strategy != "temporal" else None,
        "mtl" if args.enable_gleason_task else "pirads",
        args.coil_combine if args.image_source in {"recon", "hybrid"} else None,
        group_run_name,
        args.image_metrics,
        "dual" if getattr(args, "dual_encoder", False) else "stack",
        args.backbone,
        "pt" if args.backbone_pretrained else "scratch",
        f"lr{args.lr:.0e}",
        f"wd{args.weight_decay:.0e}",
        f"bs{args.batch_size}",
        f"seed{args.seed}",
        init_tag,
    ]
    return "-".join([item for item in suffix if item])


def _flatten_split_stats(prefix: str, summary: Dict[str, object]) -> Dict[str, float]:
    flat: Dict[str, float] = {}
    for key, value in summary.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat[f"{prefix}/{sub_key}"] = float(sub_value)
        else:
            flat[f"{prefix}/{key}"] = float(value)
    return flat


def _print_split_stats(split_name: str, summary: Dict[str, object]) -> None:
    pirads_dist = summary["pirads_distribution"]
    gg_dist = summary["grade_group_distribution"]
    print(
        f"[labels:{split_name}] n={summary['count']} "
        f"pirads_pos={summary['pirads_positive']} pirads_neg={summary['pirads_negative']} "
        f"gleason_labeled={summary['gleason_labeled_count']} "
        f"gleason_pos={summary['gleason_positive']} gleason_neg={summary['gleason_negative']} "
        f"pirads_dist={pirads_dist} grade_group_dist={gg_dist}"
    )


def _patient_key(label: AccessionLabel) -> str:
    return label.patient_id if label.patient_id is not None else label.accession_id


def _patient_first_study_dates(
    labels: Sequence[AccessionLabel],
) -> Dict[str, pd.Timestamp]:
    first_dates: Dict[str, pd.Timestamp] = {}
    for label in labels:
        study_dt = label.study_datetime
        if study_dt is None or pd.isna(study_dt):
            continue
        patient_id = _patient_key(label)
        prev = first_dates.get(patient_id)
        if prev is None or study_dt < prev:
            first_dates[patient_id] = study_dt
    return first_dates


def _split_accession_ids(
    labels: Sequence[AccessionLabel],
    args: argparse.Namespace,
) -> tuple[List[str], List[str], List[str]]:
    if args.split_strategy == "temporal":
        return temporal_split(labels, train_frac=0.75, val_frac=0.15)
    if args.split_strategy == "consecutive_patient_holdout":
        return consecutive_patient_holdout_split(
            labels,
            train_frac=args.split_pool_train_frac,
            test_frac=args.split_test_frac,
            test_start_date=args.split_test_start_date,
        )
    raise ValueError(f"Unsupported split strategy '{args.split_strategy}'")


def _build_fair_train_labels(
    source_labels: Sequence[AccessionLabel],
    aligned_val_labels: Sequence[AccessionLabel],
    aligned_test_labels: Sequence[AccessionLabel],
) -> tuple[List[AccessionLabel], pd.Timestamp]:
    """Filter source-cohort labels so training remains fair against aligned holdout."""
    holdout_patient_ids = {
        _patient_key(label) for label in [*aligned_val_labels, *aligned_test_labels]
    }
    aligned_val_first_dates = _patient_first_study_dates(aligned_val_labels)
    if not aligned_val_first_dates:
        raise ValueError(
            "Aligned validation split has no valid study dates; cannot enforce fair temporal boundary."
        )
    aligned_val_start = min(aligned_val_first_dates.values())

    source_patient_first_dates = _patient_first_study_dates(source_labels)
    train_labels: List[AccessionLabel] = []
    excluded_holdout = 0
    excluded_temporal = 0
    excluded_missing_date = 0

    for label in source_labels:
        patient_id = _patient_key(label)
        if patient_id in holdout_patient_ids:
            excluded_holdout += 1
            continue
        patient_first_dt = source_patient_first_dates.get(patient_id)
        if patient_first_dt is None:
            excluded_missing_date += 1
            continue
        if patient_first_dt >= aligned_val_start:
            excluded_temporal += 1
            continue
        train_labels.append(label)

    if not train_labels:
        raise ValueError(
            "Fair split filtering removed all source-cohort training examples. "
            "Check labels, dates, and aligned split construction."
        )

    train_patient_ids = {_patient_key(label) for label in train_labels}
    overlap_patients = train_patient_ids & holdout_patient_ids
    if overlap_patients:
        raise ValueError(
            f"Leakage detected: {len(overlap_patients)} train patients overlap aligned val/test holdout."
        )

    holdout_accessions = {
        label.accession_id for label in [*aligned_val_labels, *aligned_test_labels]
    }
    overlap_accessions = {
        label.accession_id for label in train_labels
    } & holdout_accessions
    if overlap_accessions:
        raise ValueError(
            f"Leakage detected: {len(overlap_accessions)} train accessions overlap aligned val/test holdout."
        )

    print(
        "[fair_split] policy=aligned_reference_holdout "
        f"aligned_val_start={aligned_val_start.date()} "
        f"excluded_holdout_accessions={excluded_holdout} "
        f"excluded_temporal_accessions={excluded_temporal} "
        f"excluded_missing_date_accessions={excluded_missing_date} "
        f"train_after_filter={len(train_labels)}"
    )
    return train_labels, aligned_val_start


def _split_source_internal_labels(
    labels: Sequence[AccessionLabel],
) -> tuple[List[AccessionLabel], List[AccessionLabel]]:
    patient_first_dates = _patient_first_study_dates(labels)
    patient_order = sorted(
        patient_first_dates.items(), key=lambda item: (item[1], item[0])
    )
    if len(patient_order) < 2:
        raise ValueError(
            "source_internal requires at least two dated patients after fair filtering."
        )

    n_val_patients = max(1, int(math.ceil(len(patient_order) * SOURCE_VAL_FRAC)))
    n_val_patients = min(n_val_patients, len(patient_order) - 1)
    val_patient_ids = {patient_id for patient_id, _ in patient_order[-n_val_patients:]}

    train_labels: List[AccessionLabel] = []
    val_labels: List[AccessionLabel] = []
    excluded_missing_date = 0
    for label in labels:
        patient_id = _patient_key(label)
        if patient_id not in patient_first_dates:
            excluded_missing_date += 1
            continue
        if patient_id in val_patient_ids:
            val_labels.append(label)
        else:
            train_labels.append(label)

    if not train_labels or not val_labels:
        raise ValueError(
            "source_internal produced an empty train or validation split after fair filtering."
        )

    train_patient_ids = {_patient_key(label) for label in train_labels}
    print(
        "[source_internal_split] "
        f"train_patients={len(train_patient_ids)} "
        f"val_patients={len(val_patient_ids)} "
        f"train_accessions={len(train_labels)} "
        f"val_accessions={len(val_labels)} "
        f"excluded_missing_date_accessions={excluded_missing_date}"
    )
    return train_labels, val_labels


def _backbone_prefixes(*, model_variant: str, dual_encoder: bool) -> tuple[str, ...]:
    if _is_hybrid_model_variant(model_variant):
        return ("t2_encoder.", "dwi_encoder.")
    return ("encoder_a.", "encoder_b.") if dual_encoder else ("encoder3d.",)


def _set_backbone_trainable(
    model: ProstateImageClassifier | T2ReconDWIFusionClassifier,
    *,
    model_variant: str,
    dual_encoder: bool,
    trainable: bool,
) -> None:
    if _is_hybrid_model_variant(model_variant):
        modules = (model.t2_encoder, model.dwi_encoder)
    elif dual_encoder:
        modules = (model.encoder_a, model.encoder_b)
    else:
        modules = (model.encoder3d,)
    for module in modules:
        for param in module.parameters():
            param.requires_grad = trainable


def _set_fusion_branches_trainable(
    model: T2ReconDWIFusionClassifier,
    *,
    branches: Sequence[str],
    trainable: bool,
) -> None:
    branch_modules = {"t2": model.t2_encoder, "dwi": model.dwi_encoder}
    for branch in branches:
        for param in branch_modules[branch].parameters():
            param.requires_grad = trainable


def _load_checkpoint_payload(path: Path) -> tuple[dict, dict]:
    payload = torch.load(path, map_location="cpu")
    state_dict = (
        payload["state_dict"]
        if isinstance(payload, dict) and "state_dict" in payload
        else payload
    )
    config = payload.get("config", {}) if isinstance(payload, dict) else {}
    normalized_state_dict = {
        (key[7:] if key.startswith("module.") else key): tensor
        for key, tensor in state_dict.items()
    }
    return normalized_state_dict, config


def _validate_init_checkpoint_config(
    source_config: dict,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if not source_config:
        return {"source_has_config": False}

    source_mode = source_config.get("mode")
    if source_mode not in {None, "image"}:
        raise ValueError(
            f"Transfer init requires an image checkpoint, got mode={source_mode!r}"
        )

    source_model = source_config.get("model", {})
    source_data = source_config.get("data", {})
    source_variant = source_model.get("model_variant")
    source_dual_encoder = bool(source_model.get("dual_encoder", False))
    source_backbone = source_model.get("backbone")
    if source_variant is not None and source_variant != args.model_variant:
        raise ValueError(
            "Transfer init checkpoint model_variant does not match the target run: "
            f"{source_variant!r} vs {args.model_variant!r}"
        )
    if source_variant is None and source_dual_encoder != bool(args.dual_encoder):
        raise ValueError(
            "Legacy transfer init checkpoint dual_encoder setting does not match the target run."
        )
    if source_backbone is not None and source_backbone != args.backbone:
        raise ValueError(
            "Transfer init checkpoint backbone does not match the target run: "
            f"{source_backbone!r} vs {args.backbone!r}"
        )

    return {
        "source_has_config": True,
        "source_image_source": source_data.get("image_source"),
        "source_image_metrics": source_data.get("image_metrics"),
        "source_model_variant": source_variant,
        "source_dual_encoder": source_dual_encoder,
        "source_backbone": source_backbone,
    }


def _apply_checkpoint_initialization(
    model: ProstateImageClassifier | T2ReconDWIFusionClassifier,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if args.init_from_checkpoint is None:
        return {}

    state_dict, source_config = _load_checkpoint_payload(args.init_from_checkpoint)
    source_stats = _validate_init_checkpoint_config(source_config, args)

    model_state = model.state_dict()
    backbone_prefixes = _backbone_prefixes(
        model_variant=args.model_variant,
        dual_encoder=args.dual_encoder,
    )
    init_scope = args.init_scope

    filtered_state_dict: Dict[str, Any] = {}
    skipped_scope = 0
    skipped_missing_target = 0
    skipped_shape = 0
    loaded_backbone = 0

    for key, tensor in state_dict.items():
        is_backbone_key = key.startswith(backbone_prefixes)
        if init_scope == "encoder_only" and not is_backbone_key:
            skipped_scope += 1
            continue

        target_tensor = model_state.get(key)
        if target_tensor is None:
            skipped_missing_target += 1
            continue

        if target_tensor.shape != tensor.shape:
            skipped_shape += 1
            continue

        filtered_state_dict[key] = tensor
        if is_backbone_key:
            loaded_backbone += 1

    if loaded_backbone == 0:
        raise ValueError(
            "Transfer init failed: loaded zero backbone tensors. "
            "Check checkpoint compatibility and init_scope."
        )

    incompatible = model.load_state_dict(filtered_state_dict, strict=False)
    missing_keys = list(incompatible.missing_keys)
    unexpected_keys = list(incompatible.unexpected_keys)

    stats = {
        "path": str(args.init_from_checkpoint),
        "scope": init_scope,
        "loaded_total": len(filtered_state_dict),
        "loaded_backbone": loaded_backbone,
        "skipped_scope": skipped_scope,
        "skipped_missing_target": skipped_missing_target,
        "skipped_shape": skipped_shape,
        "missing_keys": len(missing_keys),
        "unexpected_keys": len(unexpected_keys),
    }
    stats.update(source_stats)
    print(
        "[init] loaded checkpoint="
        f"{stats['path']} scope={stats['scope']} loaded_total={stats['loaded_total']} "
        f"loaded_backbone={stats['loaded_backbone']} skipped_scope={stats['skipped_scope']} "
        f"skipped_missing_target={stats['skipped_missing_target']} skipped_shape={stats['skipped_shape']} "
        f"missing_keys={stats['missing_keys']} unexpected_keys={stats['unexpected_keys']} "
        f"source_image_source={stats.get('source_image_source')} "
        f"source_image_metrics={stats.get('source_image_metrics')} "
        f"source_model_variant={stats.get('source_model_variant')} "
        f"source_dual_encoder={stats.get('source_dual_encoder')} "
        f"source_backbone={stats.get('source_backbone')}"
    )
    return stats


def _validate_fusion_branch_checkpoint_config(
    source_config: dict,
    *,
    args: argparse.Namespace,
    branch: str,
) -> Dict[str, Any]:
    if not source_config:
        return {"source_has_config": False}

    source_mode = source_config.get("mode")
    if source_mode not in {None, "image"}:
        raise ValueError(
            f"Fusion branch init requires an image checkpoint, got mode={source_mode!r}"
        )

    source_model = source_config.get("model", {})
    source_data = source_config.get("data", {})
    source_variant = source_model.get("model_variant")
    source_dual_encoder = bool(source_model.get("dual_encoder", False))
    source_backbone = source_model.get("backbone")
    if source_variant not in {None, MODEL_VARIANT_SINGLE_STREAM}:
        raise ValueError(
            f"{branch} branch init requires a single-stream source checkpoint, got model_variant={source_variant!r}"
        )
    if source_variant is None and source_dual_encoder:
        raise ValueError(
            f"{branch} branch init does not support legacy dual-encoder source checkpoints."
        )
    if source_backbone is not None and source_backbone != args.backbone:
        raise ValueError(
            f"{branch} branch init checkpoint backbone does not match the target run: "
            f"{source_backbone!r} vs {args.backbone!r}"
        )

    expected_source = "dicom" if branch == "t2" else "recon"
    expected_metrics = "axt2" if branch == "t2" else "adc_b1500"
    source_image_source = source_data.get("image_source")
    source_image_metrics = source_data.get("image_metrics")
    source_coil_combine = source_data.get("coil_combine")
    if source_image_source is not None and source_image_source != expected_source:
        raise ValueError(
            f"{branch} branch init requires image_source={expected_source!r}, got {source_image_source!r}"
        )
    if source_image_metrics is not None and source_image_metrics != expected_metrics:
        raise ValueError(
            f"{branch} branch init requires image_metrics={expected_metrics!r}, got {source_image_metrics!r}"
        )
    if branch == "dwi":
        if (
            source_coil_combine is not None
            and source_coil_combine != args.coil_combine
        ):
            raise ValueError(
                f"{branch} branch init requires coil_combine={args.coil_combine!r}, "
                f"got {source_coil_combine!r}"
            )
        target_recipe_mode = getattr(args, "recipe_mode", None)
        source_recipe_mode = source_data.get("recipe_mode")
        if target_recipe_mode is not None or source_recipe_mode is not None:
            if source_recipe_mode != target_recipe_mode:
                raise ValueError(
                    f"{branch} branch init recipe_mode does not match the target run: "
                    f"{source_recipe_mode!r} vs {target_recipe_mode!r}"
                )
            if target_recipe_mode == RECIPE_MODE_SAMPLED_RECIPES:
                source_recipes_per_accession = source_data.get("recipes_per_accession")
                target_recipes_per_accession = _resolved_recipes_per_accession(args)
                if source_recipes_per_accession != target_recipes_per_accession:
                    raise ValueError(
                        f"{branch} branch init recipes_per_accession does not match the target run: "
                        f"{source_recipes_per_accession!r} vs {target_recipes_per_accession!r}"
                    )
                source_quotas = list(source_data.get("recipe_family_quotas", []))
                target_quotas = _resolved_recipe_family_quotas(args)
                if source_quotas != target_quotas:
                    raise ValueError(
                        f"{branch} branch init recipe_family_quotas do not match the target run: "
                        f"{source_quotas!r} vs {target_quotas!r}"
                    )
        else:
            source_averaging_tags = source_data.get("averaging_tags")
            if source_averaging_tags:
                resolved_source_tags = list(source_averaging_tags)
            elif source_data.get("averaging_tag") is not None:
                resolved_source_tags = [str(source_data["averaging_tag"])]
            else:
                resolved_source_tags = []
            target_averaging_tags = _resolved_averaging_tags(args)
            if (
                resolved_source_tags
                and target_averaging_tags
                and resolved_source_tags != target_averaging_tags
            ):
                raise ValueError(
                    f"{branch} branch init averaging_tags do not match the target run: "
                    f"{resolved_source_tags!r} vs {target_averaging_tags!r}"
                )

    return {
        "source_has_config": True,
        "source_image_source": source_image_source,
        "source_image_metrics": source_image_metrics,
        "source_coil_combine": source_coil_combine,
        "source_model_variant": source_variant,
        "source_dual_encoder": source_dual_encoder,
        "source_backbone": source_backbone,
    }


def _apply_fusion_branch_initialization(
    model: T2ReconDWIFusionClassifier,
    args: argparse.Namespace,
) -> Dict[str, Dict[str, Any]]:
    branch_specs = {
        "t2": (args.t2_branch_init_from_checkpoint, "t2_encoder."),
        "dwi": (args.dwi_branch_init_from_checkpoint, "dwi_encoder."),
    }
    source_prefix = "encoder3d."
    model_state = model.state_dict()
    branch_stats: Dict[str, Dict[str, Any]] = {}

    for branch, (path, target_prefix) in branch_specs.items():
        if path is None:
            continue

        state_dict, source_config = _load_checkpoint_payload(path)
        source_stats = _validate_fusion_branch_checkpoint_config(
            source_config,
            args=args,
            branch=branch,
        )

        filtered_state_dict: Dict[str, Any] = {}
        skipped_non_encoder = 0
        skipped_missing_target = 0
        skipped_shape = 0
        loaded_backbone = 0

        for key, tensor in state_dict.items():
            if not key.startswith(source_prefix):
                skipped_non_encoder += 1
                continue

            target_key = f"{target_prefix}{key[len(source_prefix):]}"
            target_tensor = model_state.get(target_key)
            if target_tensor is None:
                skipped_missing_target += 1
                continue
            if target_tensor.shape != tensor.shape:
                skipped_shape += 1
                continue

            filtered_state_dict[target_key] = tensor
            loaded_backbone += 1

        if loaded_backbone == 0:
            raise ValueError(
                f"{branch} branch init failed: loaded zero encoder tensors. Check checkpoint compatibility."
            )

        incompatible = model.load_state_dict(filtered_state_dict, strict=False)
        stats = {
            "path": str(path),
            "loaded_total": len(filtered_state_dict),
            "loaded_backbone": loaded_backbone,
            "skipped_non_encoder": skipped_non_encoder,
            "skipped_missing_target": skipped_missing_target,
            "skipped_shape": skipped_shape,
            "missing_keys": len(list(incompatible.missing_keys)),
            "unexpected_keys": len(list(incompatible.unexpected_keys)),
        }
        stats.update(source_stats)
        branch_stats[branch] = stats
        print(
            f"[init:{branch}] loaded checkpoint={stats['path']} "
            f"loaded_total={stats['loaded_total']} loaded_backbone={stats['loaded_backbone']} "
            f"skipped_non_encoder={stats['skipped_non_encoder']} "
            f"skipped_missing_target={stats['skipped_missing_target']} "
            f"skipped_shape={stats['skipped_shape']} missing_keys={stats['missing_keys']} "
            f"unexpected_keys={stats['unexpected_keys']} "
            f"source_image_source={stats.get('source_image_source')} "
            f"source_image_metrics={stats.get('source_image_metrics')} "
            f"source_coil_combine={stats.get('source_coil_combine')} "
            f"source_model_variant={stats.get('source_model_variant')} "
            f"source_dual_encoder={stats.get('source_dual_encoder')} "
            f"source_backbone={stats.get('source_backbone')}"
        )

    return branch_stats


def _apply_model_initialization(
    model: ProstateImageClassifier | T2ReconDWIFusionClassifier,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if _is_hybrid_model_variant(args.model_variant):
        return {"fusion_branches": _apply_fusion_branch_initialization(model, args)}

    legacy_stats = _apply_checkpoint_initialization(model, args)
    return {"legacy": legacy_stats} if legacy_stats else {}


def build_checkpoint_config(
    args: argparse.Namespace,
    *,
    cohort_mode: str,
    dicom_roots: Sequence[Path] | None,
) -> dict:
    dicom_paths = [str(path) for path in dicom_roots] if dicom_roots else None
    data_config = {
        "image_source": args.image_source,
        "image_metrics": args.image_metrics,
        "coil_combine": args.coil_combine,
        "cohort": cohort_mode,
        "validation_policy": args.validation_policy,
        "dicom_h5_roots": dicom_paths,
        "pathology_csv": str(args.pathology_csv) if args.pathology_csv else None,
        "gleason_positive_grade_group_threshold": args.gleason_positive_grade_group_threshold,
        "split_strategy": args.split_strategy,
        "split_test_frac": args.split_test_frac,
        "split_test_start_date": args.split_test_start_date,
        "split_pool_train_frac": args.split_pool_train_frac,
    }
    if _uses_recipe_mode(args):
        data_config.update(_recipe_config_values(args))
    elif _uses_averaging_tags(args):
        data_config.update(_averaging_tag_config_values(args))

    model_config = {
        "model_variant": args.model_variant,
        "backbone": args.backbone,
        "backbone_pretrained": args.backbone_pretrained,
        "dropout": args.dropout,
        "enable_gleason_head": args.enable_gleason_task,
    }
    if args.model_variant == MODEL_VARIANT_T2_RECON_DWI_FUSION:
        model_config["fusion_architecture"] = FUSION_ARCHITECTURE_SHARED_BOTTLENECK_V1
        model_config["fusion_gate"] = args.fusion_gate
    if args.dual_encoder:
        model_config["dual_encoder"] = True
    if args.freeze_backbone:
        model_config["freeze_backbone"] = True

    training_config = {
        "enable_gleason_task": args.enable_gleason_task,
        "primary_metric": args.primary_metric,
        "pirads_loss_weight": args.pirads_loss_weight,
        "gleason_loss_weight": args.gleason_loss_weight,
    }
    if args.init_from_checkpoint is not None:
        training_config.update(
            {
                "init_from_checkpoint": str(args.init_from_checkpoint),
                "init_scope": args.init_scope,
                "init_freeze_backbone_epochs": args.init_freeze_backbone_epochs,
            }
        )
    if args.t2_branch_init_from_checkpoint is not None:
        training_config["t2_branch_init_from_checkpoint"] = str(
            args.t2_branch_init_from_checkpoint
        )
    if args.dwi_branch_init_from_checkpoint is not None:
        training_config["dwi_branch_init_from_checkpoint"] = str(
            args.dwi_branch_init_from_checkpoint
        )
    if _initialized_fusion_branches(args):
        training_config[
            "freeze_initialized_branch_epochs"
        ] = args.freeze_initialized_branch_epochs

    return {
        "version": 1,
        "mode": "image",
        "model": model_config,
        "data": data_config,
        "training": training_config,
    }


def _build_optimizer(
    model: ProstateImageClassifier | T2ReconDWIFusionClassifier,
    args: argparse.Namespace,
) -> tuple[torch.optim.Optimizer, list[float]]:
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    return optimizer, [group["lr"] for group in optimizer.param_groups]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train paired image-metric classifier")
    parser.add_argument("--h5-root", type=Path, required=True)
    parser.add_argument(
        "--image-source", type=str, choices=IMAGE_SOURCE_CHOICES, default="recon"
    )
    parser.add_argument(
        "--model-variant",
        type=str,
        choices=MODEL_VARIANT_CHOICES,
        default=MODEL_VARIANT_SINGLE_STREAM,
    )
    parser.add_argument(
        "--coil-combine",
        type=str,
        choices=["esc", "rss", "espirit"],
        default=DEFAULT_COIL_COMBINE,
    )
    parser.add_argument("--averaging-tag", type=str, default=DEFAULT_AVERAGING_TAG)
    parser.add_argument(
        "--averaging-tags",
        type=str,
        action="append",
        default=None,
        help="Repeat flag to train on multiple recon averaging tags in one run.",
    )
    parser.add_argument(
        "--recipe-mode",
        type=str,
        choices=RECIPE_MODE_CHOICES,
        default=None,
    )
    parser.add_argument("--recipes-per-accession", type=int, default=None)
    parser.add_argument(
        "--recipe-family-quotas",
        type=str,
        action="append",
        default=None,
        help="Repeat family=count to control sampled_recipes composition.",
    )
    parser.add_argument(
        "--recipe-seed",
        type=int,
        default=DEFAULT_RECIPE_SEED,
    )
    parser.add_argument(
        "--dicom-h5-roots",
        type=Path,
        nargs="+",
        action="append",
        default=None,
        help="One or more directories containing DICOM-derived ADC/b1500 H5 files",
    )
    parser.add_argument(
        "--cohort",
        type=str,
        choices=COHORT_CHOICES,
        default="recon_only",
        help="Cohort selection: recon_only, dicom_only, or aligned (labels + recon + DICOM).",
    )
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument(
        "--pathology-csv",
        type=Path,
        default=None,
        help="Optional pathology CSV containing highest_grade_group per mri_accession.",
    )
    parser.add_argument(
        "--enable-gleason-task",
        action="store_true",
        help="Enable multitask training with auxiliary Gleason/grade-group supervision.",
    )
    parser.add_argument(
        "--gleason-positive-grade-group-threshold",
        type=int,
        default=2,
        help="Define positive Gleason target as grade_group >= threshold (default: 2).",
    )
    parser.add_argument(
        "--image-metrics",
        type=str,
        choices=IMAGE_METRICS_CHOICES,
        default="adc_b1500",
        help="Image metric selection for the requested source (e.g. recon adc_b1500 or DICOM axt2).",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument(
        "--lr-scheduler", type=str, choices=["none", "cosine"], default="none"
    )
    parser.add_argument("--lr-warmup-epochs", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--eval-interval-steps", type=int, default=0)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=6,
        help="Stop if validation primary metric does not improve for this many epochs (0 disables early stopping)",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Number of minibatches to accumulate before an optimizer step",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        choices=["r3d_18", "mc3_18", "r2plus1d_18"],
        default="r3d_18",
    )
    parser.add_argument(
        "--backbone-pretrained",
        action="store_true",
        help="Use Kinetics pretrained weights for the backbone",
    )
    parser.add_argument(
        "--dual-encoder",
        action="store_true",
        help="Use separate encoders for ADC and b1500 instead of channel stacking",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze backbone weights, only train the classification head",
    )
    parser.add_argument(
        "--init-from-checkpoint",
        type=Path,
        default=None,
        help="Optional source checkpoint path for transfer initialization.",
    )
    parser.add_argument(
        "--init-scope",
        type=str,
        choices=["full", "encoder_only"],
        default="full",
        help="Initialization scope: full model weights or encoder-only.",
    )
    parser.add_argument(
        "--init-freeze-backbone-epochs",
        type=int,
        default=0,
        help="Freeze backbone for the first N epochs after init (BN stats keep adapting).",
    )
    parser.add_argument(
        "--t2-branch-init-from-checkpoint",
        type=Path,
        default=None,
        help="Optional T2 single-stream checkpoint used to initialize the T2 branch of the fusion model.",
    )
    parser.add_argument(
        "--dwi-branch-init-from-checkpoint",
        type=Path,
        default=None,
        help="Optional recon DWI single-stream checkpoint used to initialize the DWI branch of the fusion model.",
    )
    parser.add_argument(
        "--freeze-initialized-branch-epochs",
        type=int,
        default=0,
        help="Freeze initialized fusion branches for the first N epochs, then unfreeze.",
    )
    parser.add_argument(
        "--fusion-gate",
        action="store_true",
        help="Apply a simple soft gate over the concatenated T2+DWI feature before the shared fusion bottleneck.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply random augmentations (flips, rotations, intensity jitter, noise) to training data",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate in classification head (default: 0.3)",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing factor for BCE loss (default: 0.0, try 0.1)",
    )
    parser.add_argument(
        "--pirads-loss-weight",
        type=float,
        default=1.0,
        help="Loss weight for PI-RADS/csPCa head.",
    )
    parser.add_argument(
        "--gleason-loss-weight",
        type=float,
        default=1.0,
        help="Loss weight for Gleason auxiliary head when enabled.",
    )
    parser.add_argument(
        "--primary-metric",
        type=str,
        choices=["auroc", "pirads_auroc", "gleason_auroc"],
        default="auroc",
        help="Metric used for checkpoint selection and early stopping.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--date-buffer-days", type=int, default=7)
    parser.add_argument(
        "--split-strategy",
        type=str,
        choices=SPLIT_STRATEGY_CHOICES,
        default="temporal",
        help="Split strategy: temporal (existing) or consecutive_patient_holdout (new).",
    )
    parser.add_argument(
        "--split-test-frac",
        type=float,
        default=0.10,
        help="For consecutive_patient_holdout: most-recent accession fraction used to anchor test start.",
    )
    parser.add_argument(
        "--split-test-start-date",
        type=str,
        default=None,
        help="For consecutive_patient_holdout: explicit test-start date (e.g., 2024-01-01). Overrides split-test-frac.",
    )
    parser.add_argument(
        "--split-pool-train-frac",
        type=float,
        default=0.85,
        help="For consecutive_patient_holdout: fraction of pre-test pool patients assigned to train.",
    )
    parser.add_argument(
        "--validation-policy",
        type=str,
        choices=VALIDATION_POLICY_CHOICES,
        default="aligned_holdout",
        help="Use aligned holdout for validation/testing, or source_internal for source-only pretraining.",
    )
    parser.add_argument("--wandb-project", type=str, default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument(
        "--wandb-group",
        type=str,
        default=None,
        help="Logical experiment group name in W&B",
    )
    parser.add_argument(
        "--wandb-job-type",
        type=str,
        default=None,
        help="Sub-experiment stage label in W&B",
    )
    parser.add_argument(
        "--wandb-tags",
        type=str,
        action="append",
        default=None,
        help="Repeat flag to attach tags in W&B (e.g. --wandb-tags trace_b50_b1000 --wandb-tags seed_22222)",
    )
    parser.add_argument(
        "--wandb-experiment-id",
        type=str,
        default=None,
        help="Stable experiment identifier stored in W&B config for API filtering",
    )
    parser.add_argument("--wandb-mode", type=str, default="online")
    parser.add_argument(
        "--amp-dtype",
        type=str,
        choices=AMP_DTYPE_CHOICES,
        default="none",
        help="Mixed precision mode for CUDA image training.",
    )
    return parser.parse_args(argv)


def run_training(args: argparse.Namespace) -> None:
    setup_logging()
    seed_everything(args.seed)
    _validate_image_args(args)
    averaging_tags = _resolved_averaging_tags(args)
    recipe_family_quotas = _resolved_recipe_family_quotas(args)
    recipes_per_accession = _resolved_recipes_per_accession(args)
    desired_device = torch.device(args.device)
    device = (
        desired_device
        if (desired_device.type == "cpu" or torch.cuda.is_available())
        else torch.device("cpu")
    )
    amp_dtype = (
        torch.bfloat16 if args.amp_dtype == "bf16" and device.type == "cuda" else None
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    primary_metric = args.primary_metric
    if args.enable_gleason_task and primary_metric == "auroc":
        primary_metric = "gleason_auroc"

    cohort_mode = args.cohort
    initialized_fusion_branches = _initialized_fusion_branches(args)
    if initialized_fusion_branches:
        staged_freeze_epochs = args.freeze_initialized_branch_epochs
    else:
        staged_freeze_epochs = (
            args.init_freeze_backbone_epochs
            if args.init_from_checkpoint is not None
            else 0
        )

    dicom_roots = (
        [Path(p) for group in args.dicom_h5_roots for p in group]
        if args.dicom_h5_roots
        else None
    )
    image_keys = list(resolve_image_input_keys(args.image_source, args.image_metrics))
    checkpoint_config = build_checkpoint_config(
        args,
        cohort_mode=cohort_mode,
        dicom_roots=dicom_roots,
    )
    checkpoint_config["training"]["primary_metric"] = primary_metric
    checkpoint_config["training"]["amp_dtype"] = (
        "bf16" if amp_dtype is not None else "none"
    )
    if args.init_from_checkpoint is not None:
        checkpoint_config["training"][
            "init_freeze_backbone_epochs"
        ] = staged_freeze_epochs

    if cohort_mode == "dicom_only" and args.image_source != "dicom":
        raise ValueError("--cohort=dicom_only requires --image-source=dicom")
    if cohort_mode == "recon_only" and args.image_source != "recon":
        raise ValueError("--cohort=recon_only requires --image-source=recon")

    if args.image_source in {"dicom", "hybrid"} or cohort_mode in {
        "dicom_only",
        "recon_only",
        "aligned",
    }:
        if dicom_roots is None:
            raise ValueError(
                "--dicom-h5-roots required when using --image-source=dicom/--image-source=hybrid "
                "or any cohort split policy (aligned/dicom_only/recon_only)."
            )

    # Select source cohort labels
    cohort_stats = None
    alignment_base_labels: List[AccessionLabel]
    if cohort_mode == "dicom_only":
        raw_labels = load_accession_labels_from_csv(
            args.labels_csv,
            pathology_csv=args.pathology_csv,
            gleason_positive_threshold=args.gleason_positive_grade_group_threshold,
        )
        dicom_ids = list_dicom_accessions(dicom_roots)
        labels = [lbl for lbl in raw_labels if lbl.accession_id in dicom_ids]
        if not labels:
            raise ValueError("No labeled DICOM accessions found after filtering.")
        print(
            f"[cohort:dicom_only] labeled={len(raw_labels)} dicom={len(dicom_ids)} "
            f"-> selected={len(labels)}"
        )
        alignment_base_labels = load_accession_labels(
            csv_path=args.labels_csv,
            h5_root=args.h5_root,
            date_buffer_days=args.date_buffer_days,
            pathology_csv=args.pathology_csv,
            gleason_positive_threshold=args.gleason_positive_grade_group_threshold,
        )
    else:
        labels = load_accession_labels(
            csv_path=args.labels_csv,
            h5_root=args.h5_root,
            date_buffer_days=args.date_buffer_days,
            pathology_csv=args.pathology_csv,
            gleason_positive_threshold=args.gleason_positive_grade_group_threshold,
        )
        alignment_base_labels = labels

    if cohort_mode == "aligned":
        labels, cohort_stats = get_aligned_cohort(
            accession_labels=alignment_base_labels,
            recon_h5_root=args.h5_root,
            dicom_h5_roots=dicom_roots,
            require_raw_kspace=False,  # Image baselines don't need raw k-space
        )
        print(
            f"[aligned_cohort] labeled={cohort_stats.labeled_count} "
            f"recon_h5={cohort_stats.recon_h5_count} dicom={cohort_stats.dicom_count} "
            f"-> aligned={cohort_stats.aligned_count}"
        )

        train_ids, val_ids, test_ids = _split_accession_ids(labels, args)
        id_to_label = {lbl.accession_id: lbl for lbl in labels}
        train_labels = [id_to_label[i] for i in train_ids]
        val_labels = [id_to_label[i] for i in val_ids]
        test_labels = [id_to_label[i] for i in test_ids]
        aligned_val_start: pd.Timestamp | None = None
    else:
        aligned_labels, aligned_stats = get_aligned_cohort(
            accession_labels=alignment_base_labels,
            recon_h5_root=args.h5_root,
            dicom_h5_roots=dicom_roots,
            require_raw_kspace=False,
        )
        print(
            f"[aligned_reference] labeled={aligned_stats.labeled_count} "
            f"recon_h5={aligned_stats.recon_h5_count} dicom={aligned_stats.dicom_count} "
            f"-> aligned={aligned_stats.aligned_count}"
        )

        _, aligned_val_ids, aligned_test_ids = _split_accession_ids(
            aligned_labels, args
        )
        aligned_id_map = {lbl.accession_id: lbl for lbl in aligned_labels}
        aligned_val_labels = [aligned_id_map[i] for i in aligned_val_ids]
        aligned_test_labels = [aligned_id_map[i] for i in aligned_test_ids]
        fair_train_labels, aligned_val_start = _build_fair_train_labels(
            source_labels=labels,
            aligned_val_labels=aligned_val_labels,
            aligned_test_labels=aligned_test_labels,
        )
        if args.validation_policy == "source_internal":
            train_labels, val_labels = _split_source_internal_labels(fair_train_labels)
            test_labels = []
            train_ids = [lbl.accession_id for lbl in train_labels]
            val_ids = [lbl.accession_id for lbl in val_labels]
            test_ids = []
        else:
            train_labels = fair_train_labels
            val_labels = aligned_val_labels
            test_labels = aligned_test_labels
            train_ids = [lbl.accession_id for lbl in train_labels]
            val_ids = aligned_val_ids
            test_ids = aligned_test_ids

    print(
        f"[split] strategy={args.split_strategy} validation_policy={args.validation_policy} "
        f"train={len(train_ids)} val={len(val_ids)} test={len(test_ids)} "
        f"source={args.image_source} cohort={cohort_mode}"
    )

    train_summary = summarize_label_distribution(train_labels)
    val_summary = summarize_label_distribution(val_labels)
    test_summary = summarize_label_distribution(test_labels)
    _print_split_stats("train", train_summary)
    _print_split_stats("val", val_summary)
    _print_split_stats("test", test_summary)

    train_augs = build_train_augmentations(image_keys) if args.augment else None
    train_dataset = create_image_dataset(
        accession_labels=train_labels,
        h5_root=args.h5_root,
        image_source=args.image_source,
        image_metrics=args.image_metrics,
        dicom_roots=dicom_roots,
        coil_combine=args.coil_combine,
        averaging_tag=args.averaging_tag,
        averaging_tags=averaging_tags,
        recipe_mode=args.recipe_mode,
        recipes_per_accession=recipes_per_accession,
        recipe_family_quotas=recipe_family_quotas,
        recipe_seed=args.recipe_seed,
        recipe_split="train",
        image_transforms=train_augs,
    )
    val_dataset = create_image_dataset(
        accession_labels=val_labels,
        h5_root=args.h5_root,
        image_source=args.image_source,
        image_metrics=args.image_metrics,
        dicom_roots=dicom_roots,
        coil_combine=args.coil_combine,
        averaging_tag=args.averaging_tag,
        averaging_tags=averaging_tags,
        recipe_mode=args.recipe_mode,
        recipes_per_accession=recipes_per_accession,
        recipe_family_quotas=recipe_family_quotas,
        recipe_seed=args.recipe_seed,
        recipe_split="val",
    )

    if args.recipe_mode == RECIPE_MODE_SAMPLED_RECIPES:
        val_manifest = build_recipe_manifest_rows(
            val_labels,
            recipe_mode=args.recipe_mode,
            recipes_per_accession=recipes_per_accession,
            recipe_family_quotas=recipe_family_quotas,
            recipe_seed=args.recipe_seed,
            recipe_split="val",
        )
        test_manifest = build_recipe_manifest_rows(
            test_labels,
            recipe_mode=args.recipe_mode,
            recipes_per_accession=recipes_per_accession,
            recipe_family_quotas=recipe_family_quotas,
            recipe_seed=args.recipe_seed,
            recipe_split="test",
        )
        manifest_root = args.output_dir / "recipe_manifests"
        write_recipe_manifest(manifest_root / "val.jsonl", val_manifest)
        write_recipe_manifest(manifest_root / "test.jsonl", test_manifest)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_image_model(
        model_variant=args.model_variant,
        input_keys=image_keys,
        dropout=args.dropout,
        backbone=args.backbone,
        backbone_pretrained=args.backbone_pretrained,
        dual_encoder=args.dual_encoder,
        freeze_backbone=args.freeze_backbone,
        enable_gleason_head=args.enable_gleason_task,
        fusion_gate=args.fusion_gate,
    ).to(device)

    init_stats = _apply_model_initialization(model, args)
    if staged_freeze_epochs > 0:
        if initialized_fusion_branches:
            _set_fusion_branches_trainable(
                model,
                branches=initialized_fusion_branches,
                trainable=False,
            )
            print(
                f"[init] froze initialized fusion branches {initialized_fusion_branches} "
                f"for first {staged_freeze_epochs} epochs"
            )
        else:
            _set_backbone_trainable(
                model,
                model_variant=args.model_variant,
                dual_encoder=args.dual_encoder,
                trainable=False,
            )
            print(f"[init] froze backbone for first {staged_freeze_epochs} epochs")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(
        f"[model] Total params: {sum(p.numel() for p in model.parameters()):,}, "
        f"Trainable: {sum(p.numel() for p in trainable_params):,}"
    )
    print(f"[precision] amp_dtype={'bf16' if amp_dtype is not None else 'none'}")

    # Build loss function with optional label smoothing
    from prostate_kspace_classifier.training.losses import build_bce_loss

    loss_fn = build_bce_loss(label_smoothing=args.label_smoothing)
    if args.label_smoothing > 0:
        print(f"[loss] Using label smoothing: {args.label_smoothing}")

    optimizer, base_lrs = _build_optimizer(model, args)
    lr_schedule = create_epoch_scheduler(
        args.lr_scheduler,
        total_epochs=args.epochs,
        warmup_epochs=args.lr_warmup_epochs,
    )

    def _set_epoch_lr(epoch_idx: int) -> float:
        if lr_schedule is None:
            return optimizer.param_groups[0]["lr"]
        multiplier = lr_schedule(epoch_idx)
        for base_lr, param_group in zip(base_lrs, optimizer.param_groups):
            param_group["lr"] = base_lr * multiplier
        return optimizer.param_groups[0]["lr"]

    run_name = args.wandb_run_name or default_run_name(args)
    checkpoint_stem = default_run_name(args)
    wandb_config = {
        "experiment": "image_metrics",
        "model_variant": args.model_variant,
        "image_metrics": args.image_metrics,
        "coil_combine": args.coil_combine,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "backbone": args.backbone,
        "backbone_pretrained": args.backbone_pretrained,
        "freeze_backbone": args.freeze_backbone,
        "dropout": args.dropout,
        "label_smoothing": args.label_smoothing,
        "pathology_csv": str(args.pathology_csv) if args.pathology_csv else None,
        "enable_gleason_task": args.enable_gleason_task,
        "gleason_positive_grade_group_threshold": args.gleason_positive_grade_group_threshold,
        "pirads_loss_weight": args.pirads_loss_weight,
        "gleason_loss_weight": args.gleason_loss_weight,
        "primary_metric": primary_metric,
        "augment": args.augment,
        "seed": args.seed,
        "lr_scheduler": args.lr_scheduler,
        "lr_warmup_epochs": args.lr_warmup_epochs,
        "eval_interval_steps": args.eval_interval_steps,
        "grad_accum_steps": args.grad_accum_steps,
        "early_stopping_patience": args.early_stopping_patience,
        "cohort": cohort_mode,
        "split_strategy": args.split_strategy,
        "split_test_frac": args.split_test_frac,
        "split_test_start_date": args.split_test_start_date,
        "split_pool_train_frac": args.split_pool_train_frac,
        "validation_policy": args.validation_policy,
        "amp_dtype": "bf16" if amp_dtype is not None else "none",
    }
    if _uses_recipe_mode(args):
        wandb_config["recipe_mode"] = args.recipe_mode
        if args.recipe_mode == RECIPE_MODE_SAMPLED_RECIPES:
            wandb_config["recipes_per_accession"] = recipes_per_accession
            wandb_config["recipe_family_quotas"] = recipe_family_quotas
            wandb_config["recipe_seed"] = args.recipe_seed
    if args.dual_encoder:
        wandb_config["dual_encoder"] = True
    if args.model_variant == MODEL_VARIANT_T2_RECON_DWI_FUSION:
        wandb_config["fusion_architecture"] = FUSION_ARCHITECTURE_SHARED_BOTTLENECK_V1
        wandb_config["fusion_gate"] = args.fusion_gate
    if args.init_from_checkpoint:
        wandb_config["init_from_checkpoint"] = str(args.init_from_checkpoint)
        wandb_config["init_scope"] = args.init_scope
        wandb_config["init_freeze_backbone_epochs"] = staged_freeze_epochs
    if args.t2_branch_init_from_checkpoint:
        wandb_config["t2_branch_init_from_checkpoint"] = str(
            args.t2_branch_init_from_checkpoint
        )
    if args.dwi_branch_init_from_checkpoint:
        wandb_config["dwi_branch_init_from_checkpoint"] = str(
            args.dwi_branch_init_from_checkpoint
        )
    if initialized_fusion_branches:
        wandb_config["freeze_initialized_branch_epochs"] = staged_freeze_epochs
    if _uses_recipe_mode(args):
        wandb_config.update(_recipe_config_values(args))
    elif _uses_averaging_tags(args):
        wandb_config.update(_averaging_tag_config_values(args))
    if args.wandb_experiment_id:
        wandb_config["experiment_id"] = args.wandb_experiment_id
    if args.wandb_group:
        wandb_config["wandb_group"] = args.wandb_group
    if args.wandb_job_type:
        wandb_config["wandb_job_type"] = args.wandb_job_type
    if args.wandb_tags:
        wandb_config["wandb_tags"] = list(args.wandb_tags)

    wandb_logger = WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=run_name,
        group=args.wandb_group,
        job_type=args.wandb_job_type,
        mode=args.wandb_mode,
        config=wandb_config,
        tags=args.wandb_tags,
    )
    wandb_logger.log(
        {
            "data/train_count": len(train_ids),
            "data/val_count": len(val_ids),
            "data/test_count": len(test_ids),
            "data/image_source": args.image_source,
            "data/image_metrics": args.image_metrics,
            "data/model_variant": args.model_variant,
            "data/fusion_gate": args.fusion_gate
            if args.model_variant == MODEL_VARIANT_T2_RECON_DWI_FUSION
            else None,
            "data/fusion_architecture": (
                FUSION_ARCHITECTURE_SHARED_BOTTLENECK_V1
                if args.model_variant == MODEL_VARIANT_T2_RECON_DWI_FUSION
                else None
            ),
            "data/cohort": cohort_mode,
            "data/aligned_cohort": cohort_mode == "aligned",
            "data/fair_split_against_aligned": cohort_mode
            in {"dicom_only", "recon_only"},
            "data/aligned_val_start_date": (
                aligned_val_start.date().isoformat()
                if aligned_val_start is not None
                else None
            ),
            "data/validation_policy": args.validation_policy,
            "data/split_strategy": args.split_strategy,
            "data/split_test_frac": args.split_test_frac,
            "data/split_test_start_date": args.split_test_start_date,
            "data/split_pool_train_frac": args.split_pool_train_frac,
            "data/enable_gleason_task": args.enable_gleason_task,
            "data/primary_metric": primary_metric,
            "data/num_averaging_tags": len(averaging_tags)
            if _uses_averaging_tags(args)
            else 0,
            "data/recipe_mode": args.recipe_mode,
            "data/recipes_per_accession": recipes_per_accession or 0,
            "data/amp_dtype": "bf16" if amp_dtype is not None else "none",
        },
        step=0,
    )
    legacy_init_stats = init_stats.get("legacy", {})
    if legacy_init_stats:
        wandb_logger.log(
            {
                "init/loaded_total": float(legacy_init_stats["loaded_total"]),
                "init/loaded_backbone": float(legacy_init_stats["loaded_backbone"]),
                "init/skipped_scope": float(legacy_init_stats["skipped_scope"]),
                "init/skipped_missing_target": float(
                    legacy_init_stats["skipped_missing_target"]
                ),
                "init/skipped_shape": float(legacy_init_stats["skipped_shape"]),
                "init/missing_keys": float(legacy_init_stats["missing_keys"]),
                "init/unexpected_keys": float(legacy_init_stats["unexpected_keys"]),
            },
            step=0,
        )
    for branch_name, branch_stats in init_stats.get("fusion_branches", {}).items():
        wandb_logger.log(
            {
                f"init/{branch_name}_loaded_total": float(branch_stats["loaded_total"]),
                f"init/{branch_name}_loaded_backbone": float(
                    branch_stats["loaded_backbone"]
                ),
                f"init/{branch_name}_skipped_non_encoder": float(
                    branch_stats["skipped_non_encoder"]
                ),
                f"init/{branch_name}_skipped_missing_target": float(
                    branch_stats["skipped_missing_target"]
                ),
                f"init/{branch_name}_skipped_shape": float(
                    branch_stats["skipped_shape"]
                ),
                f"init/{branch_name}_missing_keys": float(branch_stats["missing_keys"]),
                f"init/{branch_name}_unexpected_keys": float(
                    branch_stats["unexpected_keys"]
                ),
            },
            step=0,
        )
    wandb_logger.log(_flatten_split_stats("splits/train", train_summary), step=0)
    wandb_logger.log(_flatten_split_stats("splits/val", val_summary), step=0)
    wandb_logger.log(_flatten_split_stats("splits/test", test_summary), step=0)

    best_primary_metric = -1.0
    last_improvement_epoch = 0
    current_epoch = 0
    best_path = args.output_dir / f"best_{checkpoint_stem}.pt"
    legacy_checkpoint_stem = legacy_single_stream_run_name(args)
    legacy_best_path = (
        args.output_dir / f"best_{legacy_checkpoint_stem}.pt"
        if legacy_checkpoint_stem and legacy_checkpoint_stem != checkpoint_stem
        else None
    )
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)

    def _fmt(metric: float) -> str:
        return f"{metric:.4f}" if math.isfinite(metric) else "nan"

    def _log_val_metrics(step: int, val_metrics: dict) -> None:
        nonlocal best_primary_metric, last_improvement_epoch
        current_primary = float(val_metrics.get(primary_metric, float("nan")))
        if math.isfinite(current_primary) and current_primary > best_primary_metric:
            best_primary_metric = current_primary
            last_improvement_epoch = current_epoch
            payload = {"state_dict": model.state_dict(), "config": checkpoint_config}
            torch.save(payload, best_path)
            if legacy_best_path is not None:
                torch.save(payload, legacy_best_path)
            print(
                f"Saved new best model to {best_path} "
                f"(step={step}, {primary_metric}={_fmt(current_primary)})"
            )
            if legacy_best_path is not None:
                print(
                    f"[compat] also saved legacy checkpoint alias to {legacy_best_path}"
                )

        log_payload: Dict[str, Any] = {
            "val/primary_metric_value": current_primary,
            "val/best_primary_metric": best_primary_metric,
        }
        for key, value in val_metrics.items():
            if isinstance(value, (int, float)):
                log_payload[f"val/{key}"] = float(value)
        wandb_logger.log(log_payload, step=step)

    def _mid_epoch_eval(global_step: int) -> None:
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            loss_fn=loss_fn,
            pirads_loss_weight=args.pirads_loss_weight,
            gleason_loss_weight=args.gleason_loss_weight
            if args.enable_gleason_task
            else 0.0,
            amp_dtype=amp_dtype,
        )
        print(
            f"[val] step {global_step}: "
            f"pirads_auroc={_fmt(float(val_metrics.get('pirads_auroc', float('nan'))))} "
            f"gleason_auroc={_fmt(float(val_metrics.get('gleason_auroc', float('nan'))))} "
            f"best_{primary_metric}={_fmt(best_primary_metric)}"
        )
        _log_val_metrics(global_step, val_metrics)

    for epoch in range(1, args.epochs + 1):
        current_epoch = epoch
        epoch_step = epoch * steps_per_epoch
        if hasattr(train_dataset, "resample_for_epoch"):
            train_dataset.resample_for_epoch(epoch, args.recipe_seed)
        if staged_freeze_epochs > 0 and epoch == (staged_freeze_epochs + 1):
            if initialized_fusion_branches:
                _set_fusion_branches_trainable(
                    model,
                    branches=initialized_fusion_branches,
                    trainable=True,
                )
                trainable_now = sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                )
                print(
                    f"[init] unfroze fusion branches {initialized_fusion_branches} at epoch {epoch} "
                    f"(trainable_params={trainable_now:,})"
                )
            else:
                _set_backbone_trainable(
                    model,
                    model_variant=args.model_variant,
                    dual_encoder=args.dual_encoder,
                    trainable=True,
                )
                trainable_now = sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                )
                print(
                    f"[init] unfroze backbone at epoch {epoch} (trainable_params={trainable_now:,})"
                )
        current_lr = _set_epoch_lr(epoch)
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            loss_fn=loss_fn,
            pirads_loss_weight=args.pirads_loss_weight,
            gleason_loss_weight=args.gleason_loss_weight
            if args.enable_gleason_task
            else 0.0,
            log_interval=args.log_interval,
            progress_bar=True,
            logger=wandb_logger,
            epoch=epoch,
            base_step=(epoch - 1) * steps_per_epoch,
            eval_interval_steps=args.eval_interval_steps,
            eval_callback=_mid_epoch_eval if args.eval_interval_steps else None,
            gradient_accumulation_steps=args.grad_accum_steps,
            amp_dtype=amp_dtype,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            loss_fn=loss_fn,
            pirads_loss_weight=args.pirads_loss_weight,
            gleason_loss_weight=args.gleason_loss_weight
            if args.enable_gleason_task
            else 0.0,
            amp_dtype=amp_dtype,
        )
        _log_val_metrics(epoch_step, val_metrics)
        print(
            f"Epoch {epoch}: "
            f"train_pirads_auroc={_fmt(float(train_metrics.get('pirads_auroc', float('nan'))))} "
            f"train_gleason_auroc={_fmt(float(train_metrics.get('gleason_auroc', float('nan'))))} "
            f"val_pirads_auroc={_fmt(float(val_metrics.get('pirads_auroc', float('nan'))))} "
            f"val_gleason_auroc={_fmt(float(val_metrics.get('gleason_auroc', float('nan'))))} "
            f"best_{primary_metric}={_fmt(best_primary_metric)}"
        )
        train_log_payload: Dict[str, Any] = {"epoch": epoch, "train/lr": current_lr}
        for key, value in train_metrics.items():
            if isinstance(value, (int, float)):
                train_log_payload[f"train/{key}"] = float(value)
        wandb_logger.log(train_log_payload, step=epoch_step)

        if (
            args.early_stopping_patience > 0
            and best_primary_metric >= 0
            and epoch > staged_freeze_epochs
        ):
            epochs_since_best = epoch - last_improvement_epoch
            if epochs_since_best >= args.early_stopping_patience:
                print(
                    f"[early-stop] no {primary_metric} improvement for {epochs_since_best} epochs "
                    f"(patience={args.early_stopping_patience}); stopping at epoch {epoch}"
                )
                break

    wandb_logger.set_summary(
        {
            "best_primary_metric_name": primary_metric,
            "best_primary_metric": best_primary_metric,
        }
    )
    wandb_logger.finish()


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()

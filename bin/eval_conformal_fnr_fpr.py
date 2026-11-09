#!/usr/bin/env python3
"""
Conformal evaluation controlling FNR/FPR with dual thresholds.
Targets: hierarchical ImageNet, knee slice, knee volume.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, to_rgb
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from scipy.optimize import brentq

# Add project root to path for classifier imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classifier.utils.utils import get_config
from classifier.data import kspaceImageNet, ImageNetDataset
from classifier.data.imagenet import HierarchicalImbalancedImageNet
from classifier.data.knee_data import KneeDataset
from classifier.data.knee_vol_data import KneeVolumeDataset
from classifier.data import MRDataModule
from classifier.classification_modules.modules import ARMSModule, BayesDistillModule
from classifier.classification_modules.compat import (
    load_state_dict_with_classifier_fallback,
    infer_pool_type_from_state_dict,
)
from classifier.masks.factory import build_mask_fn
from copy import deepcopy
from classifier.masks.cumulative_masking import (
    CumulativeMaskState,
    apply_mask_to_inputs,
    build_cumulative_batch_mask,
)


COLOR_MAP = {
    "conformal": "#4C72B0",  # blue
    "forced": "#7F7F7F",  # gray
    "entropy": "#1B9E77",  # teal
    "probability": "#D95F02",  # orange
}
FORCED_PRED_THRESHOLD = 0.5  # standard probability threshold on the positive class (vs abstention baselines)


def configure_plot_style():
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 24,
            "axes.labelsize": 18,
            "legend.fontsize": 12,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
        }
    )


def style_axis(ax, show_left: bool):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(show_left)
    ax.set_axisbelow(True)
    ax.grid(True, c="0.9", axis="y")
    ax.grid(True, c="0.9", axis="x")
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.tick_params(axis="both", which="minor", labelsize=15)


def get_method_color(method: str) -> str:
    base = method.split("_", 1)[0]
    return COLOR_MAP.get(base, "#4C72B0")


def build_dataloader(dataset, batch_size: int, num_workers: int, shuffle: bool = False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def split_dataset(dataset, calib_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    n_calib = int(len(indices) * calib_fraction)
    calib_idx = indices[:n_calib]
    test_idx = indices[n_calib:]
    return Subset(dataset, calib_idx), Subset(dataset, test_idx)


def get_datasets(config, calib_fraction: float, seed: int, use_testset: bool):
    if config.dataset == "imagenet":
        if getattr(config, "hierarchical_superclass", None) is None:
            raise ValueError("Non-hierarchical ImageNet should use coverage conformal script.")
        calib = HierarchicalImbalancedImageNet(config, mode="val")
        if use_testset:
            try:
                test = HierarchicalImbalancedImageNet(config, mode="test")
                if len(test) == 0:
                    raise ValueError("Empty test dataset")
                return calib, test
            except Exception:
                return split_dataset(calib, calib_fraction, seed)
        return split_dataset(calib, calib_fraction, seed)

    if config.dataset in ["knee", "knee_vol"]:
        # Mirror MRDataModule's split logic to match training-time validation
        datamodule = MRDataModule(config=config, dev_mode=False)
        calib = datamodule.val_dataset
        test = datamodule.test_dataset
        if use_testset and test is not None and len(test) > 0:
            return calib, test
        return split_dataset(calib, calib_fraction, seed)

    raise ValueError(f"Unsupported dataset for FNR/FPR conformal: {config.dataset}")


def load_model(config, ckpt_path: str, device: torch.device):
    classifier_type = getattr(config, "classifier_type", "arms")
    module_cls = BayesDistillModule if classifier_type in ["bayes_distill", "bayes_distill_arms"] else ARMSModule
    print(classifier_type, module_cls)
    remap_classifier_keys = False
    if getattr(config, "input_type", "kspace") != "image" and getattr(config, "mask_type", "none") == "none":
        # Fully sampled k-space configs sometimes set mask_type=none, which is not supported.
        # Switch to a uniform full-sampling mask to allow model init.
        config.mask_type = "uniform"
        config.k_fraction = getattr(config, "k_fraction", 1.0)
        config.center_fraction = getattr(config, "center_fraction", config.k_fraction / 2)
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        inferred_pool = infer_pool_type_from_state_dict(state_dict, getattr(config, "pool_type", None))
        print(inferred_pool)
        if inferred_pool is not None:
            config.pool_type = inferred_pool
        has_classifier_keys = any(k.startswith("classifier.") for k in state_dict.keys())
        has_model_keys = any(k.startswith("model.") for k in state_dict.keys())
        remap_classifier_keys = has_classifier_keys and not has_model_keys
        print(has_classifier_keys, has_model_keys, state_dict.keys())
    except Exception as e:
        print("Error loading checkpoint path", e)
        pass
    if remap_classifier_keys:
        print("Loading checkpoint with classifier->model key remap.")
    model = module_cls(config=config)
    missing_keys, unexpected_keys = load_state_dict_with_classifier_fallback(
        model, ckpt_path, map_location=device, strict=False, weights_only=False
    )
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Checkpoint load failed: missing/unexpected keys detected. "
            f"Missing keys: {len(missing_keys)}; Unexpected keys: {len(unexpected_keys)}"
        )
    print("Checkpoint loaded with all keys matched.")
    model.to(device)
    model.eval()
    return model


def _move_value_to_device(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move_value_to_device(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_move_value_to_device(v, device) for v in value)
    return value


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    return {key: _move_value_to_device(value, device) for key, value in batch.items()}


def build_score_cache_dir(output_dir: str, seed: int) -> Path:
    return Path(output_dir) / "score_cache" / f"seed_{seed}"


def load_cached_scores(cache_path: Path) -> Optional[Dict]:
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=False)
    return None


def save_cached_scores(cache_path: Path, payload: Dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)


def build_mask_for_sampling_rate(config, sampling_rate: float):
    cfg = deepcopy(config)
    if getattr(cfg, "input_type", "kspace") == "image":
        if cfg.mask_type == "random_rate_patch":
            mask_ratio = 1.0 - sampling_rate
            cfg.min_sampling_rate = mask_ratio
            cfg.max_sampling_rate = mask_ratio
        elif cfg.mask_type == "biased_random_rate_patch":
            cfg.min_sampling_rate = sampling_rate
            cfg.max_sampling_rate = sampling_rate
        elif cfg.mask_type == "patch":
            cfg.patch_mask_ratio = max(0.0, min(1.0, 1.0 - sampling_rate))
    else:
        if cfg.mask_type in ["random_vds", "random_vvds"]:
            cfg.k_fraction = sampling_rate
            cfg.min_k_fraction = sampling_rate
            cfg.max_k_fraction = sampling_rate
        elif cfg.mask_type in ["vds", "vvds", "uniform", "equi", "equi_no_center"]:
            cfg.k_fraction = sampling_rate
            cfg.center_fraction = sampling_rate / 2
        elif cfg.mask_type == "random_rate_uniform":
            cfg.min_sampling_rate = sampling_rate
            cfg.max_sampling_rate = sampling_rate
    return build_mask_fn(cfg, training=False)


def get_probs_labels(
    model,
    dataloader,
    label_names: List[str],
    device: torch.device,
    mask_fn,
    cumulative_state: Optional[CumulativeMaskState] = None,
    input_type: str = "kspace",
    is_volume: bool = False,
):
    all_probs = {name: [] for name in label_names}
    all_labels = {name: [] for name in label_names}
    if cumulative_state is not None:
        cumulative_state.reset_cursor()
    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            if cumulative_state is not None:
                inputs = model._get_input(batch, masked=False)
                if input_type == "kspace":
                    kspace = batch["sc_kspace"]
                    num_slices = kspace.shape[1] if is_volume else None
                    mask = build_cumulative_batch_mask(mask_fn, kspace, cumulative_state, num_slices=num_slices)
                else:
                    mask = build_cumulative_batch_mask(mask_fn, inputs, cumulative_state)
                inputs = apply_mask_to_inputs(inputs, mask)
                logits = model.model(inputs)
            else:
                if isinstance(model, BayesDistillModule):
                    logits = model.forward_undersampled(batch, mask_fn=mask_fn)
                else:
                    inputs = model._get_input(batch, masked=True, mask_fn=mask_fn)
                    logits = model.model(inputs)
            for name in label_names:
                probs = F.softmax(logits[name], dim=1)
                all_probs[name].append(probs.detach().cpu().numpy())
                all_labels[name].append(batch["label"][name].detach().cpu().squeeze(1).numpy())
    for name in label_names:
        all_probs[name] = np.concatenate(all_probs[name], axis=0)
        all_labels[name] = np.concatenate(all_labels[name], axis=0)
    return all_probs, all_labels


def verify_checkpoint(model, dataloader, label_names: List[str], device: torch.device):
    model.eval()
    total_correct = {name: 0 for name in label_names}
    total_count = {name: 0 for name in label_names}
    total_loss = {name: 0.0 for name in label_names}
    all_probs = {name: [] for name in label_names}
    all_labels = {name: [] for name in label_names}
    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            if isinstance(model, BayesDistillModule):
                logits = model.forward_undersampled(batch, mask_fn=model.mask_fn)
            else:
                inputs = model._get_input(batch, masked=True, mask_fn=model.mask_fn)
                logits = model.model(inputs)
            for name in label_names:
                labels = batch["label"][name].detach().squeeze(1)
                probs = F.softmax(logits[name], dim=1)
                preds = torch.argmax(logits[name], dim=1)
                total_correct[name] += int((preds == labels).sum().item())
                total_count[name] += int(labels.numel())
                total_loss[name] += float(
                    F.cross_entropy(logits[name], labels, reduction="sum").item()
                )
                all_probs[name].append(probs.detach().cpu())
                all_labels[name].append(labels.detach().cpu())
    metrics = {}
    aucs = []
    for name in label_names:
        count = max(total_count[name], 1)
        if all_labels[name]:
            probs_cat = torch.cat(all_probs[name], dim=0)
            labels_cat = torch.cat(all_labels[name], dim=0)
            try:
                from classifier.metrics.classification_metrics import evaluate_classifier

                auc = float(
                    evaluate_classifier(
                        preds=probs_cat,
                        labels=labels_cat,
                        num_classes=probs_cat.shape[1],
                    )["auc"]
                )
            except Exception:
                auc = 0.0
        else:
            auc = 0.0
        aucs.append(auc)
        metrics[name] = {
            "accuracy": total_correct[name] / count,
            "loss": total_loss[name] / count,
            "auc": auc,
        }
    val_auc_mean = float(np.mean(aucs)) if aucs else 0.0
    print(f"Checkpoint verification val_auc_mean: {val_auc_mean:.4f}")
    print("Checkpoint verification metrics (per label):")
    for name, stats in metrics.items():
        print(
            f"  {name}: acc={stats['accuracy']:.4f}, "
            f"loss={stats['loss']:.4f}, auc={stats['auc']:.4f}"
        )


def false_rate(predictions: np.ndarray, labels: np.ndarray) -> float:
    if predictions.size == 0 or labels.size == 0:
        return 0.0
    predictions = predictions.astype(int)
    labels = labels.astype(int)
    missed = np.logical_and(predictions == 0, labels == 1).sum()
    positives = (labels == 1).sum()
    return missed / positives if positives > 0 else 0.0


def compute_dual_thresholds(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    alpha_fnr: float,
    alpha_fpr: Optional[float],
    pos_idx: int = 1,
):
    cal_probs = np.asarray(cal_probs)
    cal_labels = np.asarray(cal_labels)
    n_calib = len(cal_labels)
    neg_idx = 1 - pos_idx

    pos_probs = cal_probs[:, pos_idx]
    pos_labels = (cal_labels == pos_idx).astype(int)
    alpha_fnr = float(np.clip(alpha_fnr, 0.0, 1.0))
    target_fnr = ((n_calib + 1) / n_calib) * alpha_fnr - 1 / n_calib

    def fnr_objective(lam: float) -> float:
        predictions = (pos_probs >= lam).astype(int)
        return false_rate(predictions, pos_labels) - target_fnr

    try:
        threshold_pos = float(brentq(fnr_objective, 0.0, 1.0))
    except ValueError as exc:
        pos_count = int(pos_labels.sum())
        if pos_probs.size > 0:
            prob_min = float(np.min(pos_probs))
            prob_max = float(np.max(pos_probs))
        else:
            prob_min = float("nan")
            prob_max = float("nan")
        fnr_at_zero = float(fnr_objective(0.0))
        fnr_at_one = float(fnr_objective(1.0))
        print(
            "brentq failed for FNR threshold. Diagnostics: "
            f"alpha_fnr={alpha_fnr}, target_fnr={target_fnr}, "
            f"pos_count={pos_count}, prob_min={prob_min:.6f}, prob_max={prob_max:.6f}, "
            f"fnr_objective(0)={fnr_at_zero:.6f}, fnr_objective(1)={fnr_at_one:.6f}, "
            f"error={exc}"
        )
        threshold_pos = (
            float(np.quantile(pos_probs[pos_labels == 1], alpha_fnr))
            if (pos_labels == 1).any()
            else float(pos_probs.mean())
        )

    if alpha_fpr is None:
        return {pos_idx: threshold_pos}

    neg_probs = cal_probs[:, neg_idx]
    neg_labels = (cal_labels == neg_idx).astype(int)
    alpha_fpr = float(np.clip(alpha_fpr, 0.0, 1.0))
    target_fpr = ((n_calib + 1) / n_calib) * alpha_fpr - 1 / n_calib

    def fpr_objective(lam: float) -> float:
        predictions = (neg_probs >= lam).astype(int)
        return false_rate(predictions, neg_labels) - target_fpr

    try:
        threshold_neg = float(brentq(fpr_objective, 0.0, 1.0))
    except ValueError as exc:
        neg_count = int(neg_labels.sum())
        if neg_probs.size > 0:
            neg_min = float(np.min(neg_probs))
            neg_max = float(np.max(neg_probs))
        else:
            neg_min = float("nan")
            neg_max = float("nan")
        fpr_at_zero = float(fpr_objective(0.0))
        fpr_at_one = float(fpr_objective(1.0))
        print(
            "brentq failed for FPR threshold. Diagnostics: "
            f"alpha_fpr={alpha_fpr}, target_fpr={target_fpr}, "
            f"neg_count={neg_count}, prob_min={neg_min:.6f}, prob_max={neg_max:.6f}, "
            f"fpr_objective(0)={fpr_at_zero:.6f}, fpr_objective(1)={fpr_at_one:.6f}, "
            f"error={exc}"
        )
        threshold_neg = (
            float(np.quantile(neg_probs[neg_labels == 1], alpha_fpr))
            if (neg_labels == 1).any()
            else float(neg_probs.mean())
        )

    return {neg_idx: threshold_neg, pos_idx: threshold_pos}


def compute_stopped_stats(pred_sets: List[List[int]], labels: np.ndarray) -> Dict[str, float]:
    stopped = [1 if len(s) == 1 else 0 for s in pred_sets]
    stopped_count = int(np.sum(stopped))
    total = len(pred_sets)
    stopped_fraction = stopped_count / total if total else 0.0
    stopped_coverage = 0.0
    for idx, label in enumerate(labels):
        if len(pred_sets[idx]) == 1:
            stopped_coverage += 1.0 if int(label) in pred_sets[idx] else 0.0
    stopped_coverage = stopped_coverage / stopped_count if stopped_count else 0.0
    return {
        "stopped_count": stopped_count,
        "stopped_fraction": stopped_fraction,
        "stopped_coverage": stopped_coverage,
    }


def parse_alpha_list(alpha_arg: str, num_rates: int) -> List[float]:
    alpha_str = str(alpha_arg)
    if "," in alpha_str:
        values = [float(x) for x in alpha_str.split(",") if x.strip()]
        if len(values) != num_rates:
            raise ValueError(
                f"alpha list length {len(values)} must match sampling rates ({num_rates})"
            )
        return values
    return [float(alpha_str)] * num_rates


def parse_optional_alpha_list(alpha_arg: Optional[str], num_rates: int) -> Optional[List[float]]:
    if alpha_arg is None:
        return None
    alpha_str = str(alpha_arg).strip().lower()
    if alpha_str in {"none", "null"}:
        return None
    return parse_alpha_list(alpha_str, num_rates)


def parse_threshold_list(raw_value: Optional[str]) -> List[float]:
    if raw_value is None:
        return []
    raw_str = str(raw_value).strip()
    if not raw_str or raw_str.lower() in {"none", "null"}:
        return []
    values = [float(x) for x in raw_str.split(",") if x.strip()]
    return values


def format_baseline_label(name: str) -> str:
    if name.startswith("probability_"):
        return f"Conf. Threshold {name.split('_', 1)[1]}"
    if name.startswith("entropy_"):
        return f"Entropy Threshold {name.split('_', 1)[1]}"
    if name == "forced":
        return "Forced"
    return name


def _get_threshold_from_name(name: str, prefix: str) -> Optional[float]:
    if not name.startswith(prefix):
        return None
    try:
        return float(name.split("_", 1)[1])
    except (IndexError, ValueError):
        return None


def _shade_color(base_hex: str, strength: float) -> str:
    strength = float(np.clip(strength, 0.0, 1.0))
    base = np.array(to_rgb(base_hex))
    light = base + (1.0 - base) * 0.45
    dark = base * 0.55
    shaded = light * (1.0 - strength) + dark * strength
    return to_hex(shaded, keep_alpha=False)


def normalize_optional_path(path_value: Optional[str]) -> Optional[str]:
    if path_value is None:
        return None
    normalized = str(path_value).strip()
    if not normalized or normalized.lower() in {"none", "null"}:
        return None
    return normalized


def compute_stopped_confusion(preds: np.ndarray, labels: np.ndarray) -> Dict[str, int]:
    tp = fp = tn = fn = 0
    for pred, label in zip(preds, labels):
        if label == 1 and pred == 1:
            tp += 1
        elif label == 1 and pred == 0:
            fn += 1
        elif label == 0 and pred == 1:
            fp += 1
        elif label == 0 and pred == 0:
            tn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def compute_fnr_fpr_from_confusion(tp: int, fp: int, tn: int, fn: int) -> Tuple[float, float]:
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return fnr, fpr


def apply_entropy_threshold(probs: np.ndarray, threshold: float) -> np.ndarray:
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)
    return entropy <= threshold


def apply_probability_threshold(
    probs: np.ndarray, threshold: float, positive_class_idx: int
) -> np.ndarray:
    pos_probs = probs[:, positive_class_idx]
    return pos_probs >= threshold


def compute_baseline_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    stop_mask: np.ndarray,
    total_samples: int,
) -> Dict[str, float]:
    if stop_mask.sum() == 0:
        return {
            "stopped_count": 0,
            "stopped_fraction": 0.0,
            "fnr": 0.0,
            "fpr": 0.0,
        }
    stopped_probs = probs[stop_mask]
    stopped_labels = labels[stop_mask]
    stopped_preds = np.argmax(stopped_probs, axis=1)
    conf = compute_stopped_confusion(stopped_preds, stopped_labels)
    fnr, fpr = compute_fnr_fpr_from_confusion(**conf)
    return {
        "stopped_count": int(stop_mask.sum()),
        "stopped_fraction": float(stop_mask.sum()) / total_samples,
        "fnr": fnr,
        "fpr": fpr,
    }


def evaluate_thresholds(probs: np.ndarray, labels: np.ndarray, thresholds: Dict[int, float]):
    pred_sets = []
    tp = fp = tn = fn = 0
    for i in range(len(labels)):
        pred_set = [k for k, thr in thresholds.items() if probs[i, k] >= thr]
        pred_sets.append(pred_set)
        if len(pred_set) == 1:
            pred = pred_set[0]
            if labels[i] == 1 and pred == 1:
                tp += 1
            elif labels[i] == 1 and pred == 0:
                fn += 1
            elif labels[i] == 0 and pred == 1:
                fp += 1
            elif labels[i] == 0 and pred == 0:
                tn += 1
    coverage = np.mean([labels[i] in pred_sets[i] for i in range(len(labels))])
    avg_set_size = float(np.mean([len(s) for s in pred_sets]))
    stopped = tp + fp + tn + fn
    stopped_fraction = stopped / len(labels) if len(labels) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    balanced_acc = 0.5 * ((tp / (tp + fn)) if (tp + fn) else 0.0 + (tn / (tn + fp)) if (tn + fp) else 0.0)
    stopped_stats = compute_stopped_stats(pred_sets, labels)
    metrics = {
        "coverage": coverage,
        "avg_set_size": avg_set_size,
        "stopped_fraction": stopped_fraction,
        "stopped_count": stopped_stats["stopped_count"],
        "stopped_coverage": stopped_stats["stopped_coverage"],
        "fnr": fnr,
        "fpr": fpr,
        "balanced_accuracy_stopped": balanced_acc,
    }
    return metrics, pred_sets


def plot_sampling_curves(
    output_dir: str,
    label: str,
    per_rate: List[Dict],
    baselines: Dict[str, Dict],
    full_metrics: Optional[Dict[str, float]] = None,
    entropy_threshold: Optional[float] = None,
    probability_threshold: Optional[float] = None,
    alpha_fnr: Optional[float] = None,
):
    os.makedirs(output_dir, exist_ok=True)
    configure_plot_style()
    sampling_rates = [entry["sampling_rate"] for entry in per_rate]
    x_positions = np.arange(len(sampling_rates))
    x_labels = [f"{rate:.2f}" for rate in sampling_rates]
    stopped_fractions = [entry.get("stopped_fraction", 0.0) for entry in per_rate]
    fnr_vals = [entry["fnr"] for entry in per_rate]
    fpr_vals = [entry["fpr"] for entry in per_rate]

    cumulative_fnr = [entry.get("cumulative_fnr", 0.0) for entry in per_rate]
    cumulative_balanced_accuracy = [
        entry.get("cumulative_balanced_accuracy", 0.0) for entry in per_rate
    ]
    label_map = {"conformal": "CASC", "forced": "Forced"}
    prob_thresholds = [
        _get_threshold_from_name(name, "probability_") for name in baselines.keys()
    ]
    prob_thresholds = [v for v in prob_thresholds if v is not None]
    ent_thresholds = [
        _get_threshold_from_name(name, "entropy_") for name in baselines.keys()
    ]
    ent_thresholds = [v for v in ent_thresholds if v is not None]
    prob_min = min(prob_thresholds) if prob_thresholds else 0.0
    prob_max = max(prob_thresholds) if prob_thresholds else 1.0
    ent_min = min(ent_thresholds) if ent_thresholds else 0.0
    ent_max = max(ent_thresholds) if ent_thresholds else 1.0

    fig, ax = plt.subplots(1, 3, figsize=(18, 4), dpi=350, sharey=False)
    ax[0].plot(
        x_positions,
        stopped_fractions,
        marker="o",
        color=get_method_color("conformal"),
        label=label_map["conformal"],
    )
    # No full-sample line for stopped fraction.
    for name, series in baselines.items():
        if name == "forced":
            continue
        if "stopped_fraction" in series:
            line_color = get_method_color(name)
            prob_thr = _get_threshold_from_name(name, "probability_")
            ent_thr = _get_threshold_from_name(name, "entropy_")
            if prob_thr is not None and prob_max > prob_min:
                line_color = _shade_color(
                    get_method_color("probability"),
                    (prob_thr - prob_min) / (prob_max - prob_min),
                )
            elif ent_thr is not None and ent_max > ent_min:
                line_color = _shade_color(
                    get_method_color("entropy"),
                    (ent_thr - ent_min) / (ent_max - ent_min),
                )
            ax[0].plot(
                x_positions,
                series["stopped_fraction"],
                linestyle="--",
                color=line_color,
                label=label_map.get(name, format_baseline_label(name)),
            )
    ax[0].set_xlabel("Sampling rate")
    ax[0].set_ylabel("Fraction Stopped")
    ax[0].set_xticks(x_positions, x_labels, rotation=0)
    ax[0].set_ylim(0, 1.05)

    ax[1].plot(
        x_positions,
        cumulative_fnr,
        marker="o",
        color=get_method_color("conformal"),
        label=label_map["conformal"],
    )
    if alpha_fnr is not None:
        target_color = "#7A6A53"
        ax[1].axhline(alpha_fnr, color=target_color, linestyle=":", linewidth=0.8)
        ax[1].text(
            x_positions[-1],
            alpha_fnr + 0.02,
            f"Target (α={alpha_fnr:.2f})",
            color=target_color,
            fontsize=8,
            ha="right",
            va="bottom",
        )
    if full_metrics and full_metrics.get("fnr") is not None:
        ax[1].axhline(
            full_metrics["fnr"],
            linestyle="--",
            color="#333333",
            label="Complete Samples",
        )
    for name, series in baselines.items():
        if "cumulative_fnr" in series:
            line_color = get_method_color(name)
            prob_thr = _get_threshold_from_name(name, "probability_")
            ent_thr = _get_threshold_from_name(name, "entropy_")
            if prob_thr is not None and prob_max > prob_min:
                line_color = _shade_color(
                    get_method_color("probability"),
                    (prob_thr - prob_min) / (prob_max - prob_min),
                )
            elif ent_thr is not None and ent_max > ent_min:
                line_color = _shade_color(
                    get_method_color("entropy"),
                    (ent_thr - ent_min) / (ent_max - ent_min),
                )
            ax[1].plot(
                x_positions,
                series["cumulative_fnr"],
                linestyle="--",
                color=line_color,
                label=label_map.get(name, format_baseline_label(name)),
            )
    ax[1].set_xlabel("Sampling rate")
    ax[1].set_ylabel("Miscoverage Rate")
    ax[1].set_xticks(x_positions, x_labels, rotation=0)
    ax[1].set_ylim(0, 1.05)

    ax[2].plot(
        x_positions,
        cumulative_balanced_accuracy,
        marker="o",
        color=get_method_color("conformal"),
        label=label_map["conformal"],
    )
    if full_metrics and full_metrics.get("balanced_accuracy_stopped") is not None:
        ax[2].axhline(
            full_metrics["balanced_accuracy_stopped"],
            linestyle="--",
            color="#333333",
            label="Complete Samples",
        )
    for name, series in baselines.items():
        if "cumulative_balanced_accuracy" in series:
            line_color = get_method_color(name)
            prob_thr = _get_threshold_from_name(name, "probability_")
            ent_thr = _get_threshold_from_name(name, "entropy_")
            if prob_thr is not None and prob_max > prob_min:
                line_color = _shade_color(
                    get_method_color("probability"),
                    (prob_thr - prob_min) / (prob_max - prob_min),
                )
            elif ent_thr is not None and ent_max > ent_min:
                line_color = _shade_color(
                    get_method_color("entropy"),
                    (ent_thr - ent_min) / (ent_max - ent_min),
                )
            ax[2].plot(
                x_positions,
                series["cumulative_balanced_accuracy"],
                linestyle="--",
                color=line_color,
                label=label_map.get(name, format_baseline_label(name)),
            )
    ax[2].set_xlabel("Sampling rate")
    ax[2].set_ylabel("Balanced Acc.")
    ax[2].set_xticks(x_positions, x_labels, rotation=0)
    ax[2].set_ylim(0, 1.05)
    handles, labels = [], []
    for axis in ax:
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        for handle, label_text in zip(axis_handles, axis_labels):
            if label_text not in labels:
                handles.append(handle)
                labels.append(label_text)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=max(1, len(labels)),
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
    )
    style_axis(ax[0], show_left=True)
    style_axis(ax[1], show_left=False)
    style_axis(ax[2], show_left=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plot_path = os.path.join(output_dir, f"conformal_risk_control_{label}.png")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot -> {plot_path}")


def plot_aggregate_curves(
    output_dir: str,
    sampling_rates: List[float],
    stopped_fractions: List[float],
    cumulative_fnr: List[float],
    cumulative_balanced_accuracy: List[float],
    baselines: Dict[str, Dict],
    alpha_fnr: Optional[float] = None,
):
    os.makedirs(output_dir, exist_ok=True)
    configure_plot_style()
    x_positions = np.arange(len(sampling_rates))
    x_labels = [f"{rate:.2f}" for rate in sampling_rates]
    label_map = {"conformal": "CASC", "forced": "Forced"}
    prob_thresholds = [
        _get_threshold_from_name(name, "probability_") for name in baselines.keys()
    ]
    prob_thresholds = [v for v in prob_thresholds if v is not None]
    ent_thresholds = [
        _get_threshold_from_name(name, "entropy_") for name in baselines.keys()
    ]
    ent_thresholds = [v for v in ent_thresholds if v is not None]
    prob_min = min(prob_thresholds) if prob_thresholds else 0.0
    prob_max = max(prob_thresholds) if prob_thresholds else 1.0
    ent_min = min(ent_thresholds) if ent_thresholds else 0.0
    ent_max = max(ent_thresholds) if ent_thresholds else 1.0

    fig, ax = plt.subplots(1, 3, figsize=(18, 4), dpi=350, sharey=False)
    ax[0].plot(
        x_positions,
        stopped_fractions,
        marker="o",
        color=get_method_color("conformal"),
        label=label_map["conformal"],
    )
    for name, series in baselines.items():
        if name == "forced":
            continue
        if "stopped_fraction" in series:
            line_color = get_method_color(name)
            prob_thr = _get_threshold_from_name(name, "probability_")
            ent_thr = _get_threshold_from_name(name, "entropy_")
            if prob_thr is not None and prob_max > prob_min:
                line_color = _shade_color(
                    get_method_color("probability"),
                    (prob_thr - prob_min) / (prob_max - prob_min),
                )
            elif ent_thr is not None and ent_max > ent_min:
                line_color = _shade_color(
                    get_method_color("entropy"),
                    (ent_thr - ent_min) / (ent_max - ent_min),
                )
            ax[0].plot(
                x_positions,
                series["stopped_fraction"],
                linestyle="--",
                color=line_color,
                label=label_map.get(name, format_baseline_label(name)),
            )
    ax[0].set_xlabel("Sampling rate")
    ax[0].set_ylabel("Fraction Stopped (All Labels)")
    ax[0].set_xticks(x_positions, x_labels, rotation=0)
    ax[0].set_ylim(0, 1.05)

    ax[1].plot(
        x_positions,
        cumulative_fnr,
        marker="o",
        color=get_method_color("conformal"),
        label=label_map["conformal"],
    )
    if alpha_fnr is not None:
        target_color = "#7A6A53"
        ax[1].axhline(alpha_fnr, color=target_color, linestyle=":", linewidth=0.8)
        ax[1].text(
            x_positions[-1],
            alpha_fnr + 0.02,
            f"Target (α={alpha_fnr:.2f})",
            color=target_color,
            fontsize=8,
            ha="right",
            va="bottom",
        )
    for name, series in baselines.items():
        if "cumulative_fnr" in series:
            line_color = get_method_color(name)
            prob_thr = _get_threshold_from_name(name, "probability_")
            ent_thr = _get_threshold_from_name(name, "entropy_")
            if prob_thr is not None and prob_max > prob_min:
                line_color = _shade_color(
                    get_method_color("probability"),
                    (prob_thr - prob_min) / (prob_max - prob_min),
                )
            elif ent_thr is not None and ent_max > ent_min:
                line_color = _shade_color(
                    get_method_color("entropy"),
                    (ent_thr - ent_min) / (ent_max - ent_min),
                )
            ax[1].plot(
                x_positions,
                series["cumulative_fnr"],
                linestyle="--",
                color=line_color,
                label=label_map.get(name, format_baseline_label(name)),
            )
    ax[1].set_xlabel("Sampling rate")
    ax[1].set_ylabel("Miscoverage Rate")
    ax[1].set_xticks(x_positions, x_labels, rotation=0)
    ax[1].set_ylim(0, 1.05)

    ax[2].plot(
        x_positions,
        cumulative_balanced_accuracy,
        marker="o",
        color=get_method_color("conformal"),
        label=label_map["conformal"],
    )
    for name, series in baselines.items():
        if "cumulative_balanced_accuracy" in series:
            line_color = get_method_color(name)
            prob_thr = _get_threshold_from_name(name, "probability_")
            ent_thr = _get_threshold_from_name(name, "entropy_")
            if prob_thr is not None and prob_max > prob_min:
                line_color = _shade_color(
                    get_method_color("probability"),
                    (prob_thr - prob_min) / (prob_max - prob_min),
                )
            elif ent_thr is not None and ent_max > ent_min:
                line_color = _shade_color(
                    get_method_color("entropy"),
                    (ent_thr - ent_min) / (ent_max - ent_min),
                )
            ax[2].plot(
                x_positions,
                series["cumulative_balanced_accuracy"],
                linestyle="--",
                color=line_color,
                label=label_map.get(name, format_baseline_label(name)),
            )
    ax[2].set_xlabel("Sampling rate")
    ax[2].set_ylabel("Balanced Acc.")
    ax[2].set_xticks(x_positions, x_labels, rotation=0)
    ax[2].set_ylim(0, 1.05)

    handles, labels = [], []
    for axis in ax:
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        for handle, label_text in zip(axis_handles, axis_labels):
            if label_text not in labels:
                handles.append(handle)
                labels.append(label_text)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=max(1, len(labels)),
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
    )
    style_axis(ax[0], show_left=True)
    style_axis(ax[1], show_left=False)
    style_axis(ax[2], show_left=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plot_path = os.path.join(output_dir, "conformal_risk_control_aggregate.png")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved aggregate plot -> {plot_path}")


def compute_full_model_metrics(
    full_config_path: Optional[str],
    full_checkpoint_path: Optional[str],
    batch_size: int,
    num_workers: int,
    label_names: List[str],
    device: torch.device,
    calib_fraction: float,
    seed: int,
    use_testset: bool,
    alpha_fnr: float,
    alpha_fpr: Optional[float],
    positive_class_idx: int,
) -> Optional[Dict[str, Dict[str, float]]]:
    if not full_config_path or not full_checkpoint_path:
        return None
    full_config = get_config(full_config_path)
    full_model = load_model(full_config, full_checkpoint_path, device)
    full_calib_ds, full_test_ds = get_datasets(
        full_config, calib_fraction, seed, use_testset
    )
    full_calib_loader = build_dataloader(
        full_calib_ds, batch_size, num_workers, shuffle=False
    )
    full_test_loader = build_dataloader(
        full_test_ds, batch_size, num_workers, shuffle=False
    )
    full_mask_fn = build_mask_for_sampling_rate(full_config, sampling_rate=1.0)
    full_cal_probs, full_cal_labels = get_probs_labels(
        full_model, full_calib_loader, label_names, device, full_mask_fn
    )
    full_test_probs, full_test_labels = get_probs_labels(
        full_model, full_test_loader, label_names, device, full_mask_fn
    )
    full_metrics_by_label: Dict[str, Dict[str, float]] = {}
    for label in label_names:
        pos_probs = full_test_probs[label][:, positive_class_idx]
        full_preds = (pos_probs >= 0.2).astype(int)
        stopped_conf = compute_stopped_confusion(full_preds, full_test_labels[label])
        stopped_fnr, stopped_fpr = compute_fnr_fpr_from_confusion(**stopped_conf)
        full_balanced_accuracy = 0.5 * ((1.0 - stopped_fnr) + (1.0 - stopped_fpr))
        full_metrics_by_label[label] = {
            "fnr": float(stopped_fnr),
            "balanced_accuracy_stopped": float(full_balanced_accuracy),
        }
    return full_metrics_by_label


def initialize_active_masks(label_names: List[str], labels_by_label: Dict[str, np.ndarray]):
    return {
        label: np.ones(len(labels_by_label[label]), dtype=bool) for label in label_names
    }


def filter_by_active_mask(
    probs: np.ndarray,
    labels: np.ndarray,
    active_mask: Optional[np.ndarray],
):
    if active_mask is None:
        return probs, labels, None
    active_indices = np.where(active_mask)[0]
    return probs[active_indices], labels[active_indices], active_indices


def update_active_mask_from_pred_sets(
    active_mask: np.ndarray, pred_sets: List[List[int]], active_indices: np.ndarray
):
    for idx, pred_set in zip(active_indices, pred_sets):
        if len(pred_set) == 1:
            active_mask[idx] = False


def main():
    parser = argparse.ArgumentParser(description="Conformal evaluation controlling FNR/FPR.")
    parser.add_argument("--config_path", default="configs/partial/imagenet_hierarchical_bayes_random_rate_config.yaml", type=str)
    parser.add_argument("--checkpoint_path", default="/gpfs/data/chopralab/us453/partial_classifiers/results/checkpoints/imagenet/hierarchical_german_shepherd/bayes_random_rate/bayes_imagenet_hierarchical_german_shepherd_bayes_enforcing_random_epoch=36_val_loss=0.226.ckpt", type=str)
    parser.add_argument(
        "--alpha_fnr",
        type=str,
        default="0.03, 0.02, 0.02, 0.01, 0.01, 0.01",
        help="Single alpha or comma-separated list matching sampling rates.",
    )
    parser.add_argument(
        "--alpha_fpr",
        type=str,
        default="0.25",
        help="Single alpha, comma-separated list matching sampling rates, or 'none' to ignore FPR.",
    )
    parser.add_argument("--positive_class_idx", type=int, default=1)
    parser.add_argument("--calib_fraction", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output_dir", type=str, default="./paper_replots/h-imagenet/test_set")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sampling_rates", type=str, default="0.1, 0.2, 0.4, 0.5, 0.75, 1.0")
    parser.add_argument("--use_testset", action="store_true", help="Use test set; otherwise split val into calib/test.")
    parser.add_argument("--entropy_threshold", type=float, default=0.2)
    parser.add_argument("--probability_threshold", type=float, default=0.2)
    parser.add_argument(
        "--entropy_thresholds",
        type=str,
        default="0.1,0.2",
        help="Comma-separated entropy thresholds. Overrides --entropy_threshold when set.",
    )
    parser.add_argument(
        "--probability_thresholds",
        type=str,
        default="0.1, 0.2",
        help="Comma-separated confidence thresholds. Overrides --probability_threshold when set.",
    )
    parser.add_argument(
        "--verify_checkpoint",
        action="store_true",
        help="Run a validation pass to verify checkpoint loading.",
    )
    parser.add_argument("--full_config_path", type=str, default="configs/partial/imagenet_hierarchical_german_shepherd_full_image_config.yaml")
    parser.add_argument("--full_checkpoint_path", type=str, default="/gpfs/data/chopralab/us453/icml_26/results/checkpoints/imagenet/hierarchical_german_shepherd/full/icml26_imagenet_hierarchical/v118j2yf/checkpoints/arms-imagenet-resnet18-epoch=65-best_metric=0.97.ckpt")
    parser.add_argument(
        "--force_recompute_scores",
        action="store_true",
        help="Recompute and overwrite cached scores.",
    )
    parser.add_argument(
        "--filter_stopped",
        action="store_true",
        help="Filter out stopped samples from lower sampling rates on the test set.",
    )
    parser.add_argument(
        "--cumulative_sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ensure higher sampling rates only add to previously sampled masks.",
    )
    args = parser.parse_args()

    config = get_config(args.config_path)
    device = torch.device(args.device)
    label_names = config.label_names.split(",") if isinstance(config.label_names, str) else config.label_names
    input_type = getattr(config, "input_type", "kspace")
    is_volume = "vol" in str(getattr(config, "dataset", "")).lower()

    calib_ds, test_ds = get_datasets(config, args.calib_fraction, args.seed, args.use_testset)
    batch_size = args.batch_size or getattr(config, "val_batch_size", getattr(config, "batch_size", 32))

    calib_loader = build_dataloader(calib_ds, batch_size, args.num_workers, shuffle=False)
    test_loader = build_dataloader(test_ds, batch_size, args.num_workers, shuffle=False)

    model = load_model(config, args.checkpoint_path, device)
    if args.verify_checkpoint:
        verify_checkpoint(model, calib_loader, label_names, device)

    sampling_rates = [float(x) for x in args.sampling_rates.split(",") if x.strip()]
    score_cache_dir = build_score_cache_dir(args.output_dir, args.seed)
    alpha_fnr_values = parse_alpha_list(args.alpha_fnr, len(sampling_rates))
    alpha_fpr_values = parse_optional_alpha_list(args.alpha_fpr, len(sampling_rates))
    full_alpha_idx = int(np.argmax(sampling_rates))
    full_alpha_fnr = alpha_fnr_values[full_alpha_idx]
    full_alpha_fpr = alpha_fpr_values[full_alpha_idx] if alpha_fpr_values is not None else None
    alpha_fnr_arg = str(args.alpha_fnr)
    if "," in alpha_fnr_arg:
        target_alpha_fnr = float(np.sum(alpha_fnr_values))
    else:
        target_alpha_fnr = float(alpha_fnr_values[0]) * len(sampling_rates)

    full_metrics_by_label = compute_full_model_metrics(
        normalize_optional_path(args.full_config_path),
        normalize_optional_path(args.full_checkpoint_path),
        batch_size,
        args.num_workers,
        label_names,
        device,
        args.calib_fraction,
        args.seed,
        args.use_testset,
        full_alpha_fnr,
        full_alpha_fpr,
        args.positive_class_idx,
    )
    entropy_thresholds = (
        parse_threshold_list(args.entropy_thresholds)
        if args.entropy_thresholds is not None
        else []
    )
    if not entropy_thresholds and args.entropy_threshold is not None:
        entropy_thresholds = [float(args.entropy_threshold)]
    probability_thresholds = (
        parse_threshold_list(args.probability_thresholds)
        if args.probability_thresholds is not None
        else []
    )
    if not probability_thresholds and args.probability_threshold is not None:
        probability_thresholds = [float(args.probability_threshold)]
    baseline_names = ["forced"]
    baseline_names.extend([f"entropy_{thr:g}" for thr in entropy_thresholds])
    baseline_names.extend([f"probability_{thr:g}" for thr in probability_thresholds])

    results = {label: {"per_rate": []} for label in label_names}
    aggregate_stopped_fractions: List[float] = []
    aggregate_baseline_stopped: Dict[str, List[float]] = {name: [] for name in baseline_names}
    aggregate_cumulative_fnr: List[float] = []
    aggregate_cumulative_bal_acc: List[float] = []
    aggregate_baseline_cumulative_fnr: Dict[str, List[float]] = {name: [] for name in baseline_names}
    aggregate_baseline_cumulative_bal_acc: Dict[str, List[float]] = {name: [] for name in baseline_names}
    aggregate_active_mask: Optional[np.ndarray] = None
    aggregate_counts: Dict[str, Dict[str, int]] = {
        label: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for label in label_names
    }
    aggregate_baseline_active_masks: Dict[str, Optional[np.ndarray]] = {
        name: None for name in baseline_names
    }
    aggregate_baseline_counts: Dict[str, Dict[str, Dict[str, int]]] = {
        name: {label: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for label in label_names}
        for name in baseline_names
    }
    baseline_results = {label: {name: [] for name in baseline_names} for label in label_names}
    active_masks = None
    baseline_active_masks = None
    baseline_cumulative_masks = None
    baseline_cumulative_counts = None
    cumulative_active_masks = None
    cumulative_counts = {label: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for label in label_names}
    calib_mask_state = CumulativeMaskState(len(calib_ds)) if args.cumulative_sampling else None
    test_mask_state = CumulativeMaskState(len(test_ds)) if args.cumulative_sampling else None

    for rate_idx, sampling_rate in enumerate(sampling_rates):
        mask_fn = build_mask_for_sampling_rate(config, sampling_rate)
        cache_path = score_cache_dir / f"rate_{rate_idx}_sr_{sampling_rate:.6f}.pt"
        cached = None if args.force_recompute_scores else load_cached_scores(cache_path)
        cache_ok = False
        if cached is not None:
            cache_ok = (
                cached.get("sampling_rate") == sampling_rate
                and cached.get("seed") == args.seed
                and cached.get("calib_fraction") == args.calib_fraction
                and cached.get("use_testset") == args.use_testset
                and cached.get("label_names") == label_names
            )
        if cache_ok:
            cal_probs = cached["cal_probs"]
            cal_labels = cached["cal_labels"]
            test_probs = cached["test_probs"]
            test_labels = cached["test_labels"]
        else:
            cal_probs, cal_labels = get_probs_labels(
                model,
                calib_loader,
                label_names,
                device,
                mask_fn,
                cumulative_state=calib_mask_state,
                input_type=input_type,
                is_volume=is_volume,
            )
            test_probs, test_labels = get_probs_labels(
                model,
                test_loader,
                label_names,
                device,
                mask_fn,
                cumulative_state=test_mask_state,
                input_type=input_type,
                is_volume=is_volume,
            )
            save_cached_scores(
                cache_path,
                {
                    "sampling_rate": sampling_rate,
                    "seed": args.seed,
                    "calib_fraction": args.calib_fraction,
                    "use_testset": args.use_testset,
                    "label_names": label_names,
                    "cal_probs": cal_probs,
                    "cal_labels": cal_labels,
                    "test_probs": test_probs,
                    "test_labels": test_labels,
                },
            )
        if args.filter_stopped and active_masks is None:
            active_masks = initialize_active_masks(label_names, test_labels)
            baseline_active_masks = {
                name: initialize_active_masks(label_names, test_labels) for name in baseline_names
            }
        if baseline_cumulative_masks is None:
            baseline_cumulative_masks = {
                name: initialize_active_masks(label_names, test_labels) for name in baseline_names
            }
            baseline_cumulative_counts = {
                name: {
                    label: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for label in label_names
                }
                for name in baseline_names
            }
        if cumulative_active_masks is None:
            cumulative_active_masks = initialize_active_masks(label_names, test_labels)

        per_label_stopped_masks = {}
        full_pred_sets_by_label = {}
        baseline_stopped_masks_by_name = {name: [] for name in baseline_names}
        for label in label_names:
            thresholds = compute_dual_thresholds(
                cal_probs[label],
                cal_labels[label],
                alpha_fnr_values[rate_idx],
                alpha_fpr_values[rate_idx] if alpha_fpr_values is not None else None,
                args.positive_class_idx,
            )
            full_pred_sets = [
                [k for k, thr in thresholds.items() if test_probs[label][idx, k] >= thr]
                for idx in range(len(test_probs[label]))
            ]
            full_pred_sets_by_label[label] = full_pred_sets
            per_label_stopped_masks[label] = np.array(
                [len(ps) == 1 for ps in full_pred_sets], dtype=bool
            )
            for baseline_name in baseline_names:
                if baseline_name == "forced":
                    stop_mask_full = np.ones(len(test_labels[label]), dtype=bool)
                elif baseline_name.startswith("entropy_"):
                    entropy_threshold = float(baseline_name.split("_", 1)[1])
                    stop_mask_full = apply_entropy_threshold(test_probs[label], entropy_threshold)
                else:
                    probability_threshold = float(baseline_name.split("_", 1)[1])
                    stop_mask_full = apply_probability_threshold(
                        test_probs[label], probability_threshold, args.positive_class_idx
                    )
                baseline_stopped_masks_by_name[baseline_name].append(stop_mask_full)
            active_mask = active_masks[label] if args.filter_stopped else None
            total_test_samples = len(test_labels[label])
            cumulative_mask = cumulative_active_masks[label]
            cumulative_indices = np.where(cumulative_mask)[0]
            cumulative_pred_sets = []
            for idx in cumulative_indices:
                pred_set = [
                    k for k, thr in thresholds.items() if test_probs[label][idx, k] >= thr
                ]
                cumulative_pred_sets.append((idx, pred_set))
            newly_stopped = [(idx, ps) for idx, ps in cumulative_pred_sets if len(ps) == 1]
            if newly_stopped:
                for idx, ps in newly_stopped:
                    pred = ps[0]
                    label_val = int(test_labels[label][idx])
                    if label_val == 1 and pred == 1:
                        cumulative_counts[label]["tp"] += 1
                    elif label_val == 1 and pred == 0:
                        cumulative_counts[label]["fn"] += 1
                    elif label_val == 0 and pred == 1:
                        cumulative_counts[label]["fp"] += 1
                    else:
                        cumulative_counts[label]["tn"] += 1
                cumulative_mask[[idx for idx, _ in newly_stopped]] = False
            tp = cumulative_counts[label]["tp"]
            fn = cumulative_counts[label]["fn"]
            fp = cumulative_counts[label]["fp"]
            tn = cumulative_counts[label]["tn"]
            cumulative_fnr = fn / (fn + tp) if (fn + tp) else 0.0
            cumulative_fpr = fp / (fp + tn) if (fp + tn) else 0.0
            cumulative_balanced_accuracy = 0.5 * (
                (tp / (tp + fn) if (tp + fn) else 0.0)
                + (tn / (tn + fp) if (tn + fp) else 0.0)
            )
            filtered_probs, filtered_labels, active_indices = filter_by_active_mask(
                test_probs[label], test_labels[label], active_mask
            )
            if len(filtered_labels) == 0:
                metrics = {
                    "coverage": 0.0,
                    "avg_set_size": 0.0,
                    "stopped_fraction": 0.0,
                    "stopped_count": 0,
                    "stopped_coverage": 0.0,
                    "fnr": 0.0,
                    "fpr": 0.0,
                    "balanced_accuracy_stopped": 0.0,
                }
                pred_sets = []
            else:
                metrics, pred_sets = evaluate_thresholds(filtered_probs, filtered_labels, thresholds)
                if args.filter_stopped and total_test_samples > 0:
                    metrics["stopped_fraction"] = metrics["stopped_count"] / total_test_samples
            baseline_total = total_test_samples if args.filter_stopped else len(test_labels[label])

            def append_baseline_result(name, stop_mask):
                if baseline_total == 0:
                    baseline_results[label][name].append(
                        {
                            "stopped_count": 0,
                            "stopped_fraction": 0.0,
                            "fnr": 0.0,
                            "fpr": 0.0,
                            "cumulative_fnr": 0.0,
                            "cumulative_fpr": 0.0,
                        }
                    )
                    return
                if name == "forced":
                    forced_preds = (
                        filtered_probs[:, args.positive_class_idx] >= FORCED_PRED_THRESHOLD
                    ).astype(int)
                    conf = compute_stopped_confusion(forced_preds, filtered_labels)
                    fnr, fpr = compute_fnr_fpr_from_confusion(**conf)
                    baseline_metrics = {
                        "stopped_count": baseline_total,
                        "stopped_fraction": 1.0 if baseline_total > 0 else 0.0,
                        "fnr": fnr,
                        "fpr": fpr,
                    }
                elif name.startswith("probability_"):
                    stopped_probs = filtered_probs[stop_mask]
                    stopped_labels = filtered_labels[stop_mask]
                    if stopped_labels.size == 0:
                        baseline_metrics = {
                            "stopped_count": 0,
                            "stopped_fraction": 0.0,
                            "fnr": 0.0,
                            "fpr": 0.0,
                        }
                    else:
                        preds = (
                            stopped_probs[:, args.positive_class_idx] >= FORCED_PRED_THRESHOLD
                        ).astype(int)
                        conf = compute_stopped_confusion(preds, stopped_labels)
                        fnr, fpr = compute_fnr_fpr_from_confusion(**conf)
                        baseline_metrics = {
                            "stopped_count": int(stop_mask.sum()),
                            "stopped_fraction": int(stop_mask.sum()) / baseline_total
                            if baseline_total > 0
                            else 0.0,
                            "fnr": fnr,
                            "fpr": fpr,
                        }
                else:
                    baseline_metrics = compute_baseline_metrics(
                        filtered_probs, filtered_labels, stop_mask, len(filtered_labels)
                    )
                if name == "forced":
                    baseline_metrics["stopped_count"] = baseline_total
                    baseline_metrics["stopped_fraction"] = 1.0 if baseline_total > 0 else 0.0
                elif args.filter_stopped and baseline_active_mask is not None:
                    baseline_active_mask[active_indices_baseline[stop_mask]] = False
                    cumulative_stopped = total_test_samples - int(baseline_active_mask.sum())
                    baseline_metrics["stopped_count"] = cumulative_stopped
                    baseline_metrics["stopped_fraction"] = (
                        cumulative_stopped / total_test_samples if total_test_samples > 0 else 0.0
                    )
                else:
                    baseline_metrics["stopped_count"] = int(stop_mask.sum())
                    baseline_metrics["stopped_fraction"] = (
                        int(stop_mask.sum()) / baseline_total if baseline_total > 0 else 0.0
                    )
                if baseline_cumulative_masks is None or baseline_cumulative_counts is None:
                    baseline_metrics["cumulative_fnr"] = 0.0
                    baseline_metrics["cumulative_fpr"] = 0.0
                    baseline_metrics["cumulative_balanced_accuracy"] = 0.0
                else:
                    base_indices = (
                        np.arange(len(test_labels[label]))
                        if active_indices_baseline is None
                        else active_indices_baseline
                    )
                    cumulative_mask = baseline_cumulative_masks[name][label]
                    newly_stopped_mask = stop_mask & cumulative_mask[base_indices]
                    new_indices = base_indices[newly_stopped_mask]
                    if len(new_indices) > 0:
                        if name == "forced" or name.startswith("probability_"):
                            new_preds = (
                                test_probs[label][new_indices, args.positive_class_idx]
                                >= FORCED_PRED_THRESHOLD
                            ).astype(int)
                        else:
                            new_preds = np.argmax(test_probs[label][new_indices], axis=1)
                        new_labels = test_labels[label][new_indices]
                        counts = baseline_cumulative_counts[name][label]
                        for pred_val, label_val in zip(new_preds, new_labels):
                            if label_val == 1 and pred_val == 1:
                                counts["tp"] += 1
                            elif label_val == 1 and pred_val == 0:
                                counts["fn"] += 1
                            elif label_val == 0 and pred_val == 1:
                                counts["fp"] += 1
                            else:
                                counts["tn"] += 1
                        cumulative_mask[new_indices] = False
                    counts = baseline_cumulative_counts[name][label]
                    tp = counts["tp"]
                    fn = counts["fn"]
                    fp = counts["fp"]
                    tn = counts["tn"]
                    baseline_metrics["cumulative_fnr"] = fn / (fn + tp) if (fn + tp) else 0.0
                    baseline_metrics["cumulative_fpr"] = fp / (fp + tn) if (fp + tn) else 0.0
                    baseline_metrics["cumulative_balanced_accuracy"] = 0.5 * (
                        (tp / (tp + fn) if (tp + fn) else 0.0)
                        + (tn / (tn + fp) if (tn + fp) else 0.0)
                    )
                baseline_results[label][name].append(baseline_metrics)

            for baseline_name in baseline_names:
                if baseline_name == "forced":
                    baseline_active_mask = None
                    active_indices_baseline = None
                    filtered_probs = test_probs[label]
                    filtered_labels = test_labels[label]
                    stop_mask = np.ones(len(filtered_labels), dtype=bool)
                elif baseline_name.startswith("entropy_"):
                    baseline_active_mask = (
                        baseline_active_masks[baseline_name][label]
                        if args.filter_stopped
                        else None
                    )
                    if args.filter_stopped and baseline_active_mask is not None:
                        active_indices_baseline = np.where(baseline_active_mask)[0]
                        filtered_probs = test_probs[label][active_indices_baseline]
                        filtered_labels = test_labels[label][active_indices_baseline]
                    else:
                        active_indices_baseline = None
                        filtered_probs = test_probs[label]
                        filtered_labels = test_labels[label]
                    entropy_threshold = float(baseline_name.split("_", 1)[1])
                    stop_mask = apply_entropy_threshold(filtered_probs, entropy_threshold)
                else:
                    baseline_active_mask = (
                        baseline_active_masks[baseline_name][label]
                        if args.filter_stopped
                        else None
                    )
                    if args.filter_stopped and baseline_active_mask is not None:
                        active_indices_baseline = np.where(baseline_active_mask)[0]
                        filtered_probs = test_probs[label][active_indices_baseline]
                        filtered_labels = test_labels[label][active_indices_baseline]
                    else:
                        active_indices_baseline = None
                        filtered_probs = test_probs[label]
                        filtered_labels = test_labels[label]
                    probability_threshold = float(baseline_name.split("_", 1)[1])
                    stop_mask = apply_probability_threshold(
                        filtered_probs, probability_threshold, args.positive_class_idx
                    )

                append_baseline_result(baseline_name, stop_mask)

            if args.filter_stopped and active_indices is not None:
                update_active_mask_from_pred_sets(active_mask, pred_sets, active_indices)
                cumulative_stopped = total_test_samples - int(active_mask.sum())
                metrics["stopped_count"] = cumulative_stopped
                metrics["stopped_fraction"] = (
                    cumulative_stopped / total_test_samples if total_test_samples > 0 else 0.0
                )

            results[label]["per_rate"].append(
                {
                    "sampling_rate": sampling_rate,
                    "thresholds": thresholds,
                    "cumulative_fnr": cumulative_fnr,
                    "cumulative_fpr": cumulative_fpr,
                    "cumulative_balanced_accuracy": cumulative_balanced_accuracy,
                    **metrics,
                }
            )

        if len(label_names) > 1:
            all_masks = list(per_label_stopped_masks.values())
            total_samples = len(test_labels[label_names[0]])
            if aggregate_active_mask is None:
                aggregate_active_mask = np.ones(total_samples, dtype=bool)
            if all_masks:
                aggregate_mask = np.logical_and.reduce(all_masks)
                newly_stopped = aggregate_active_mask & aggregate_mask
                if np.any(newly_stopped):
                    for label in label_names:
                        preds = np.array(
                            [full_pred_sets_by_label[label][idx][0] for idx in np.where(newly_stopped)[0]]
                        )
                        labels_arr = test_labels[label][newly_stopped]
                        counts = aggregate_counts[label]
                        counts["tp"] += int(((labels_arr == 1) & (preds == 1)).sum())
                        counts["fn"] += int(((labels_arr == 1) & (preds == 0)).sum())
                        counts["fp"] += int(((labels_arr == 0) & (preds == 1)).sum())
                        counts["tn"] += int(((labels_arr == 0) & (preds == 0)).sum())
                    aggregate_active_mask[newly_stopped] = False
                aggregate_stopped_fractions.append(
                    (total_samples - int(aggregate_active_mask.sum())) / total_samples
                    if total_samples > 0
                    else 0.0
                )

            avg_cum_fnr = float(
                np.mean(
                    [
                        compute_fnr_fpr_from_confusion(**aggregate_counts[label])[0]
                        for label in label_names
                    ]
                )
            )
            avg_cum_bal = float(
                np.mean(
                    [
                        0.5
                        * (
                            (aggregate_counts[label]["tp"] / (aggregate_counts[label]["tp"] + aggregate_counts[label]["fn"]))
                            if (aggregate_counts[label]["tp"] + aggregate_counts[label]["fn"])
                            else 0.0
                        )
                        + 0.5
                        * (
                            (aggregate_counts[label]["tn"] / (aggregate_counts[label]["tn"] + aggregate_counts[label]["fp"]))
                            if (aggregate_counts[label]["tn"] + aggregate_counts[label]["fp"])
                            else 0.0
                        )
                        for label in label_names
                    ]
                )
            )
            aggregate_cumulative_fnr.append(avg_cum_fnr)
            aggregate_cumulative_bal_acc.append(avg_cum_bal)

            for baseline_name in baseline_names:
                base_masks = baseline_stopped_masks_by_name[baseline_name]
                if aggregate_baseline_active_masks[baseline_name] is None:
                    aggregate_baseline_active_masks[baseline_name] = np.ones(total_samples, dtype=bool)
                base_active = aggregate_baseline_active_masks[baseline_name]
                if base_masks:
                    base_agg = np.logical_and.reduce(base_masks)
                    newly_stopped_base = base_active & base_agg
                    if np.any(newly_stopped_base):
                        for label in label_names:
                            preds = np.argmax(test_probs[label][newly_stopped_base], axis=1)
                            labels_arr = test_labels[label][newly_stopped_base]
                            counts = aggregate_baseline_counts[baseline_name][label]
                            counts["tp"] += int(((labels_arr == 1) & (preds == 1)).sum())
                            counts["fn"] += int(((labels_arr == 1) & (preds == 0)).sum())
                            counts["fp"] += int(((labels_arr == 0) & (preds == 1)).sum())
                            counts["tn"] += int(((labels_arr == 0) & (preds == 0)).sum())
                        base_active[newly_stopped_base] = False
                    aggregate_baseline_stopped[baseline_name].append(
                        (total_samples - int(base_active.sum())) / total_samples
                        if total_samples > 0
                        else 0.0
                    )
                avg_base_fnr = float(
                    np.mean(
                        [
                            compute_fnr_fpr_from_confusion(**aggregate_baseline_counts[baseline_name][label])[0]
                            for label in label_names
                        ]
                    )
                )
                avg_base_bal = float(
                    np.mean(
                        [
                            0.5
                            * (
                                (aggregate_baseline_counts[baseline_name][label]["tp"]
                                 / (aggregate_baseline_counts[baseline_name][label]["tp"]
                                    + aggregate_baseline_counts[baseline_name][label]["fn"]))
                                if (
                                    aggregate_baseline_counts[baseline_name][label]["tp"]
                                    + aggregate_baseline_counts[baseline_name][label]["fn"]
                                )
                                else 0.0
                            )
                            + 0.5
                            * (
                                (aggregate_baseline_counts[baseline_name][label]["tn"]
                                 / (aggregate_baseline_counts[baseline_name][label]["tn"]
                                    + aggregate_baseline_counts[baseline_name][label]["fp"]))
                                if (
                                    aggregate_baseline_counts[baseline_name][label]["tn"]
                                    + aggregate_baseline_counts[baseline_name][label]["fp"]
                                )
                                else 0.0
                            )
                            for label in label_names
                        ]
                    )
                )
                aggregate_baseline_cumulative_fnr[baseline_name].append(avg_base_fnr)
                aggregate_baseline_cumulative_bal_acc[baseline_name].append(avg_base_bal)

    for label in label_names:
        if results[label]["per_rate"]:
            baselines = {}
            for baseline_name in baseline_names:
                baselines[baseline_name] = {
                    "stopped_fraction": [
                        m["stopped_fraction"] for m in baseline_results[label][baseline_name]
                    ],
                    "fnr": [m["fnr"] for m in baseline_results[label][baseline_name]],
                    "fpr": [m["fpr"] for m in baseline_results[label][baseline_name]],
                    "cumulative_fnr": [
                        m["cumulative_fnr"] for m in baseline_results[label][baseline_name]
                    ],
                    "cumulative_fpr": [
                        m["cumulative_fpr"] for m in baseline_results[label][baseline_name]
                    ],
                    "cumulative_balanced_accuracy": [
                        m["cumulative_balanced_accuracy"]
                        for m in baseline_results[label][baseline_name]
                    ],
                }
            full_metrics = full_metrics_by_label.get(label) if full_metrics_by_label else None
            plot_sampling_curves(
                args.output_dir,
                label,
                results[label]["per_rate"],
                baselines,
                full_metrics=full_metrics,
                entropy_threshold=args.entropy_threshold,
                probability_threshold=args.probability_threshold,
                alpha_fnr=target_alpha_fnr,
            )

    if len(label_names) > 1 and aggregate_stopped_fractions:
        aggregate_baselines = {
            name: {
                "stopped_fraction": aggregate_baseline_stopped[name],
                "cumulative_fnr": aggregate_baseline_cumulative_fnr[name],
                "cumulative_balanced_accuracy": aggregate_baseline_cumulative_bal_acc[name],
            }
            for name in baseline_names
        }
        plot_aggregate_curves(
            args.output_dir,
            sampling_rates,
            aggregate_stopped_fractions,
            aggregate_cumulative_fnr,
            aggregate_cumulative_bal_acc,
            aggregate_baselines,
            alpha_fnr=target_alpha_fnr,
        )

    result_path = Path(args.output_dir) / "conformal_fnr_fpr_results.yaml"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        for label, info in results.items():
            f.write(f"{label}:\n")
            for entry in info["per_rate"]:
                f.write(f"  - sampling_rate: {entry['sampling_rate']}\n")
                f.write(f"    thresholds: {entry['thresholds']}\n")
                f.write(f"    coverage: {entry['coverage']}\n")
                f.write(f"    avg_set_size: {entry['avg_set_size']}\n")
                f.write(f"    stopped_count: {entry.get('stopped_count', 0)}\n")
                f.write(f"    stopped_fraction: {entry['stopped_fraction']}\n")
                f.write(f"    stopped_coverage: {entry.get('stopped_coverage', 0.0)}\n")
                f.write(f"    fnr: {entry['fnr']}\n")
                f.write(f"    fpr: {entry['fpr']}\n")
                f.write(f"    balanced_accuracy_stopped: {entry['balanced_accuracy_stopped']}\n")
    print(f"Saved results -> {result_path}")


if __name__ == "__main__":
    main()

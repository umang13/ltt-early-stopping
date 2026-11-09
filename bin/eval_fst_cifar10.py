#!/usr/bin/env python3
"""
CIFAR-10: LTT + FST lambda selection for band-based early stopping (miscoverage risk).

Uses time-normalized ambiguity bands from a small estimation split, then calibrates
lambda on held-out calibration trajectories with Hoeffding p-values and fixed-sequence
testing at FWER level delta.
"""

from __future__ import annotations

import argparse
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classifier.utils.utils import get_config
from classifier.data import kspaceCIFAR10
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

from fst.fixed_sequence import fst_select_from_pvalues
from fst.ltt import (
    compute_grid_pvalues,
    grid_risks_and_ns,
    grid_utilities,
    ltt_select_lambda,
)


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


def split_estimation_set(dataset, est_size: int, seed: int):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    est_count = min(est_size, len(indices))
    est_idx = indices[:est_count]
    remaining_idx = indices[est_count:]
    return Subset(dataset, est_idx), Subset(dataset, remaining_idx)


def get_val_test_datasets(config, use_testset: bool):
    if config.dataset != "cifar10":
        raise ValueError(f"Unsupported dataset: {config.dataset}")
    val_ds = kspaceCIFAR10(config, mode="val")
    test_ds = None
    if use_testset:
        try:
            test_ds = kspaceCIFAR10(config, mode="test")
            if len(test_ds) == 0:
                raise ValueError("Empty test dataset")
        except Exception:
            test_ds = None
    return val_ds, test_ds


def load_model(config, ckpt_path: str, device: torch.device):
    classifier_type = getattr(config, "classifier_type", "arms")
    module_cls = BayesDistillModule if classifier_type in ["bayes_distill", "bayes_distill_arms"] else ARMSModule
    state_dict = None
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        inferred_pool = infer_pool_type_from_state_dict(state_dict, getattr(config, "pool_type", None))
        if inferred_pool is not None:
            config.pool_type = inferred_pool
    except Exception:
        pass
    try:
        model = module_cls.load_from_checkpoint(
            ckpt_path, config=config, map_location=device, strict=False
        )
    except Exception as exc:
        print(f"load_from_checkpoint failed, falling back to manual load: {exc}")
        model = module_cls(config=config)
        load_state_dict_with_classifier_fallback(
            model, ckpt_path, map_location=device, strict=False, weights_only=False
        )
    # Audit how many parameters were actually restored vs. left at random init.
    if state_dict is not None:
        try:
            model_keys = set(model.state_dict().keys())
            ckpt_keys = set(state_dict.keys())
            missing = sorted(model_keys - ckpt_keys)
            unexpected = sorted(ckpt_keys - model_keys)
            print(
                f"load_model: restored {len(model_keys & ckpt_keys)} keys; "
                f"{len(missing)} missing (still random init), "
                f"{len(unexpected)} unexpected"
            )
            if missing:
                preview = ", ".join(missing[:8])
                more = "" if len(missing) <= 8 else f" (+{len(missing)-8} more)"
                print(f"  MISSING (random init left in place): {preview}{more}")
            if unexpected:
                preview = ", ".join(unexpected[:8])
                more = "" if len(unexpected) <= 8 else f" (+{len(unexpected)-8} more)"
                print(f"  UNEXPECTED (in ckpt, not in model): {preview}{more}")
        except Exception as exc:
            print(f"load_model: key audit failed: {exc}")
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


def collect_prob_tensor(
    model,
    config,
    dataset,
    sampling_rates: List[float],
    label_names: List[str],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    cumulative_sampling: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns probs (n, T, K), labels (n,) for first label name."""
    loader = build_dataloader(dataset, batch_size, num_workers, shuffle=False)
    input_type = getattr(config, "input_type", "kspace")
    is_volume = "vol" in str(getattr(config, "dataset", "")).lower()
    mask_state = CumulativeMaskState(len(dataset)) if cumulative_sampling else None
    probs_by_rate = []
    labels_arr = None
    for rate in sampling_rates:
        mask_fn = build_mask_for_sampling_rate(config, rate)
        probs, labels = get_probs_labels(
            model,
            loader,
            label_names,
            device,
            mask_fn,
            cumulative_state=mask_state,
            input_type=input_type,
            is_volume=is_volume,
        )
        key = label_names[0]
        probs_by_rate.append(probs[key])
        if labels_arr is None:
            labels_arr = labels[key]
    probs_arr = np.stack(probs_by_rate, axis=1)
    return probs_arr, labels_arr


def build_score_cache_dir(output_dir: str, seed: int) -> Path:
    return Path(output_dir) / "score_cache" / f"seed_{seed}"


def load_cached_scores(cache_path: Path) -> Optional[Dict]:
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=False)
    return None


def save_cached_scores(cache_path: Path, payload: Dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)


def collect_prob_tensor_cached(
    model,
    config,
    dataset,
    split_name: str,
    sampling_rates: List[float],
    label_names: List[str],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    cumulative_sampling: bool,
    cache_dir: Path,
    cache_key: Dict,
    force_recompute: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cache layer around collect_prob_tensor. One cache file per split, keyed by
    cache_key (seed, calib_fraction, est_size, use_testset, cumulative_sampling,
    sampling_rates, checkpoint_path, label_names, dataset_size).

    Two runs with identical cache_keys reuse the saved probs/labels and never
    re-run the model, which is what makes selection deterministic across runs
    even when the model itself isn't perfectly seeded.
    """
    cache_path = cache_dir / f"{split_name}.pt"
    cached = None if force_recompute else load_cached_scores(cache_path)
    if cached is not None:
        keys_match = all(cached.get(k) == cache_key[k] for k in cache_key)
        has_payload = "probs" in cached and "labels" in cached
        if keys_match and has_payload:
            print(f"score cache HIT  split={split_name:>5}  ({cache_path})")
            return cached["probs"], cached["labels"]
        print(
            f"score cache MISS split={split_name:>5}  (key mismatch -> recomputing)"
        )
    else:
        print(f"score cache MISS split={split_name:>5}  ({cache_path})")
    probs, labels = collect_prob_tensor(
        model, config, dataset, sampling_rates, label_names,
        device, batch_size, num_workers, cumulative_sampling,
    )
    payload = {**cache_key, "probs": probs, "labels": labels}
    save_cached_scores(cache_path, payload)
    return probs, labels


def build_true_class_baseline(
    est_probs: np.ndarray,
    est_labels: np.ndarray,
    delta_est: float,
) -> np.ndarray:
    num_steps = est_probs.shape[1]
    out = np.zeros(num_steps, dtype=float)
    for t in range(num_steps):
        true_scores = est_probs[np.arange(len(est_labels)), t, est_labels]
        out[t] = float(np.quantile(true_scores, 1.0 - delta_est))
    return out


def build_max_score_baseline(
    est_probs: np.ndarray,
    delta_est: float,
) -> np.ndarray:
    max_scores = np.max(est_probs, axis=2)  # (n, T)
    return np.quantile(max_scores, 1.0 - delta_est, axis=0).astype(float)


def stop_by_max_score(
    probs: np.ndarray,
    thresholds_t: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    n, t_steps, _ = probs.shape
    stop_indices = np.full(n, t_steps - 1, dtype=int)
    stopped = np.zeros(n, dtype=bool)
    max_scores = np.max(probs, axis=2)
    for t in range(t_steps):
        stop_mask = max_scores[:, t] >= thresholds_t[t]
        newly = np.logical_and(~stopped, stop_mask)
        stop_indices[newly] = t
        stopped[newly] = True
    return stop_indices, stopped


def prediction_set_by_threshold(
    probs_t: np.ndarray,
    threshold_t: np.ndarray,
) -> np.ndarray:
    return probs_t >= threshold_t[:, None]


def stop_by_set_size(
    probs: np.ndarray,
    thresholds_t: np.ndarray,
    set_size_max: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n, t_steps, _ = probs.shape
    stop_indices = np.full(n, t_steps - 1, dtype=int)
    stopped = np.zeros(n, dtype=bool)
    for t in range(t_steps):
        sets_t = prediction_set_by_threshold(probs[:, t, :], np.full(n, thresholds_t[t]))
        set_sizes = sets_t.sum(axis=1)
        stop_mask = set_sizes <= set_size_max
        newly = np.logical_and(~stopped, stop_mask)
        stop_indices[newly] = t
        stopped[newly] = True
    return stop_indices, stopped


def miscoverage_losses_from_threshold_sets(
    probs: np.ndarray,
    labels: np.ndarray,
    stop_indices: np.ndarray,
    stopped: np.ndarray,
    thresholds_t: np.ndarray,
) -> np.ndarray:
    n = labels.shape[0]
    losses = np.zeros(n, dtype=float)
    if not np.any(stopped):
        return losses
    idx = np.where(stopped)[0]
    t = stop_indices[idx]
    sets = prediction_set_by_threshold(probs[idx, t, :], thresholds_t[t])
    covered = sets[np.arange(len(idx)), labels[idx]]
    losses[idx] = 1.0 - covered.astype(float)
    return losses


def argmax_miscoverage_losses_from_stops(
    probs: np.ndarray,
    labels: np.ndarray,
    stop_indices: np.ndarray,
    stopped: np.ndarray,
) -> np.ndarray:
    """
    Per-sample loss = 1{argmax(probs at stop time) != y} for stopped samples,
    0 for samples that did not stop. Aligns calibration risk with the
    test-time argmax accuracy used in the comparison plots.
    """
    n = labels.shape[0]
    losses = np.zeros(n, dtype=float)
    if not np.any(stopped):
        return losses
    idx = np.where(stopped)[0]
    t = stop_indices[idx]
    preds = np.argmax(probs[idx, t, :], axis=1)
    losses[idx] = (preds != labels[idx]).astype(float)
    return losses


def apply_stopping_rule(
    probs: np.ndarray,
    stopping_rule: str,
    thresholds_t: np.ndarray,
    set_size_max: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if stopping_rule == "max_score":
        return stop_by_max_score(probs, thresholds_t)
    return stop_by_set_size(probs, thresholds_t, set_size_max)


def per_sample_losses(
    probs: np.ndarray,
    labels: np.ndarray,
    stop_indices: np.ndarray,
    stopped: np.ndarray,
    thresholds_t: np.ndarray,
    loss_kind: str,
) -> np.ndarray:
    if loss_kind == "argmax_miscoverage":
        return argmax_miscoverage_losses_from_stops(probs, labels, stop_indices, stopped)
    if loss_kind == "set_miscoverage":
        return miscoverage_losses_from_threshold_sets(
            probs, labels, stop_indices, stopped, thresholds_t
        )
    raise ValueError(f"Unknown loss_kind: {loss_kind}")


def loss_matrix_for_rule(
    probs: np.ndarray,
    labels: np.ndarray,
    lambdas: np.ndarray,
    stopping_rule: str,
    threshold_baseline_t: np.ndarray,
    set_size_max: int,
    loss_kind: str = "set_miscoverage",
) -> np.ndarray:
    n = labels.shape[0]
    m = lambdas.shape[0]
    out = np.zeros((n, m), dtype=float)
    for j, lam in enumerate(lambdas):
        thresholds_t = np.clip(float(lam) * threshold_baseline_t, 0.0, 1.0)
        stop_indices, stopped = apply_stopping_rule(
            probs, stopping_rule, thresholds_t, set_size_max
        )
        out[:, j] = per_sample_losses(
            probs, labels, stop_indices, stopped, thresholds_t, loss_kind
        )
    return out


def per_lambda_calibration_arrays(
    probs: np.ndarray,
    labels: np.ndarray,
    lambdas: np.ndarray,
    stopping_rule: str,
    threshold_baseline_t: np.ndarray,
    set_size_max: int,
    loss_kind: str = "set_miscoverage",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (losses_mat, stopped_mat, stop_idx_mat) each (n, m).
    losses_mat[i, j] is the per-sample miscoverage at the lambda_j stop time
    (0 for samples that did not stop). Use loss_kind="argmax_miscoverage"
    when you want the calibration risk to align with the test-time argmax
    accuracy (recommended for cross-method comparison).
    """
    n = labels.shape[0]
    m = lambdas.shape[0]
    losses_mat = np.zeros((n, m), dtype=float)
    stopped_mat = np.zeros((n, m), dtype=bool)
    stop_idx_mat = np.zeros((n, m), dtype=int)
    for j, lam in enumerate(lambdas):
        thresholds_t = np.clip(float(lam) * threshold_baseline_t, 0.0, 1.0)
        stop_indices, stopped = apply_stopping_rule(
            probs, stopping_rule, thresholds_t, set_size_max
        )
        losses_mat[:, j] = per_sample_losses(
            probs, labels, stop_indices, stopped, thresholds_t, loss_kind
        )
        stopped_mat[:, j] = stopped
        stop_idx_mat[:, j] = stop_indices
    return losses_mat, stopped_mat, stop_idx_mat


def summarize_test(
    probs: np.ndarray,
    labels: np.ndarray,
    lam: float,
    stopping_rule: str,
    threshold_baseline_t: np.ndarray,
    set_size_max: int,
) -> Dict[str, float]:
    thresholds_t = np.clip(float(lam) * threshold_baseline_t, 0.0, 1.0)
    stop_indices, stopped = apply_stopping_rule(
        probs, stopping_rule, thresholds_t, set_size_max
    )
    losses = miscoverage_losses_from_threshold_sets(
        probs, labels, stop_indices, stopped, thresholds_t
    )
    return {
        "mean_risk": float(np.mean(losses)),
        "stopped_fraction": float(np.mean(stopped)),
        "mean_stop_index": float(np.mean(stop_indices)),
    }


def conformal_threshold_from_probs(
    probs_t: np.ndarray,
    labels: np.ndarray,
    alpha: float,
) -> float:
    true_scores = probs_t[np.arange(len(labels)), labels]
    sorted_scores = np.sort(true_scores)[::-1]
    n = len(sorted_scores)
    idx = int(np.ceil((1 - alpha) * (n + 1))) - 1
    idx = min(max(idx, 0), n - 1)
    return float(sorted_scores[idx])


def build_casc_alpha_schedule(
    num_steps: int,
    alpha_total: float,
    schedule: str,
) -> np.ndarray:
    if num_steps <= 0:
        return np.array([], dtype=float)
    if schedule == "uniform":
        weights = np.ones(num_steps, dtype=float)
    elif schedule == "linear_increasing":
        weights = np.arange(1, num_steps + 1, dtype=float)
    elif schedule == "linear_decreasing":
        weights = np.arange(num_steps, 0, -1, dtype=float)
    elif schedule == "geometric":
        # More risk mass earlier; still sums exactly to alpha_total after normalization.
        weights = np.array([0.5 ** t for t in range(num_steps)], dtype=float)
    else:
        raise ValueError(f"Unknown casc alpha schedule: {schedule}")
    alpha_k = alpha_total * (weights / np.sum(weights))
    return alpha_k.astype(float)


def casc_curve(
    calib_probs: np.ndarray,
    calib_labels: np.ndarray,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    alpha_total: float,
    alpha_schedule: str,
) -> Dict[str, List[float]]:
    n_test, t_steps, _ = test_probs.shape
    alpha_k = build_casc_alpha_schedule(t_steps, alpha_total, alpha_schedule)
    active = np.ones(n_test, dtype=bool)
    cumulative_correct = 0
    cumulative_total = 0
    stopped_counts = []
    cumulative_acc = []
    for t in range(t_steps):
        thr = conformal_threshold_from_probs(calib_probs[:, t, :], calib_labels, float(alpha_k[t]))
        sets_t = test_probs[:, t, :] >= thr
        set_sizes = sets_t.sum(axis=1)
        newly_stopped = np.where(active & (set_sizes == 1))[0]
        if newly_stopped.size > 0:
            preds = np.argmax(sets_t[newly_stopped, :], axis=1)
            cumulative_correct += int(np.sum(preds == test_labels[newly_stopped]))
            cumulative_total += int(newly_stopped.size)
            active[newly_stopped] = False
        stopped_counts.append(int(n_test - np.sum(active)))
        acc = (cumulative_correct / cumulative_total) if cumulative_total > 0 else 0.0
        cumulative_acc.append(float(acc))
    return {
        "stopped_count": stopped_counts,
        "accuracy": cumulative_acc,
        "alpha_k": alpha_k.tolist(),
        "alpha_k_sum": float(np.sum(alpha_k)),
    }


def fst_curve(
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    lambda_hat: float,
    stopping_rule: str,
    baseline_t: np.ndarray,
    set_size_max: int,
) -> Dict[str, List[float]]:
    n_test, t_steps, _ = test_probs.shape
    thresholds_t = np.clip(float(lambda_hat) * baseline_t, 0.0, 1.0)
    stop_indices, stopped = apply_stopping_rule(
        test_probs, stopping_rule, thresholds_t, set_size_max
    )
    stopped_counts = []
    cumulative_acc = []
    for t in range(t_steps):
        stopped_by_t = stopped & (stop_indices <= t)
        count = int(np.sum(stopped_by_t))
        stopped_counts.append(count)
        if count == 0:
            cumulative_acc.append(0.0)
            continue
        idx = np.where(stopped_by_t)[0]
        t_stop = stop_indices[idx]
        preds = np.argmax(test_probs[idx, t_stop, :], axis=1)
        cumulative_acc.append(float(np.mean(preds == test_labels[idx])))
    return {"stopped_count": stopped_counts, "accuracy": cumulative_acc}


def forced_curve(
    test_probs: np.ndarray,
    test_labels: np.ndarray,
) -> Dict[str, List[float]]:
    n_test, t_steps, _ = test_probs.shape
    stopped_counts = [n_test] * t_steps
    accuracies = []
    for t in range(t_steps):
        preds = np.argmax(test_probs[:, t, :], axis=1)
        accuracies.append(float(np.mean(preds == test_labels)))
    return {"stopped_count": stopped_counts, "accuracy": accuracies}


def _multiclass_curves_from_first_stop(
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    stop_indices: np.ndarray,
) -> Dict[str, List[float]]:
    """
    Per-rate cumulative stopped fraction and accuracy of stopped samples, matching
    ``fst_curve``: each sample contributes at its first-stop timestep with argmax prediction.
    """
    n_test, t_steps, _ = test_probs.shape
    stopped = stop_indices >= 0
    stopped_counts: List[int] = []
    cumulative_acc: List[float] = []
    for t in range(t_steps):
        stopped_by_t = stopped & (stop_indices <= t)
        count = int(np.sum(stopped_by_t))
        stopped_counts.append(count)
        if count == 0:
            cumulative_acc.append(0.0)
            continue
        idx = np.where(stopped_by_t)[0]
        t_stop = stop_indices[idx]
        preds = np.argmax(test_probs[idx, t_stop, :], axis=1)
        cumulative_acc.append(float(np.mean(preds == test_labels[idx])))
    return {"stopped_count": stopped_counts, "accuracy": cumulative_acc}


def _first_stop_indices_max_class_probability(
    test_probs: np.ndarray,
    confidence_threshold: float,
) -> np.ndarray:
    """First timestep where max softmax probability is >= ``confidence_threshold`` (else -1)."""
    n_test, t_steps, _nc = test_probs.shape
    first = np.full(n_test, -1, dtype=int)
    thr = float(confidence_threshold)
    for t in range(t_steps):
        maxp = np.max(test_probs[:, t, :], axis=1)
        ok = maxp >= thr
        m = (first < 0) & ok
        first[m] = t
    return first


def _first_stop_indices_row_entropy(
    test_probs: np.ndarray,
    entropy_threshold: float,
) -> np.ndarray:
    """First timestep where Shannon entropy of the class distribution is <= ``entropy_threshold``."""
    n_test, t_steps, _nc = test_probs.shape
    first = np.full(n_test, -1, dtype=int)
    thr = float(entropy_threshold)
    for t in range(t_steps):
        p = np.clip(test_probs[:, t, :], 1e-12, 1.0)
        ent = -np.sum(p * np.log(p), axis=1)
        ok = ent <= thr
        m = (first < 0) & ok
        first[m] = t
    return first


def fixed_threshold_multiclass_schedule_curves(
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    *,
    confidence_threshold: Optional[float] = None,
    entropy_threshold: Optional[float] = None,
) -> Tuple[Optional[Dict[str, List[float]]], Optional[Dict[str, List[float]]]]:
    """
    Optional baselines aligned with ``eval_conformal_coverage``:
    stop at the first sampling rate (timestep) where max probability >= ``confidence_threshold``
    or row entropy <= ``entropy_threshold``.
    """
    tl = np.asarray(test_labels, dtype=int)
    conf_curve: Optional[Dict[str, List[float]]] = None
    ent_curve: Optional[Dict[str, List[float]]] = None
    if confidence_threshold is not None:
        si = _first_stop_indices_max_class_probability(test_probs, float(confidence_threshold))
        conf_curve = _multiclass_curves_from_first_stop(test_probs, tl, si)
    if entropy_threshold is not None:
        si = _first_stop_indices_row_entropy(test_probs, float(entropy_threshold))
        ent_curve = _multiclass_curves_from_first_stop(test_probs, tl, si)
    return conf_curve, ent_curve


def plot_stopped_vs_rate(
    output_dir: Path,
    sampling_rates: List[float],
    fst_stopped_counts: List[int],
    casc_stopped_counts: List[int],
    stopping_rule: str,
    method_label: str = "FST",
    method_tag: str = "fst",
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
    ax.plot(sampling_rates, fst_stopped_counts, marker="o", color="#4C72B0", label=method_label)
    ax.plot(sampling_rates, casc_stopped_counts, marker="o", color="#55A868", label="CASC")
    ax.set_xlabel("Sampling rate")
    ax.set_ylabel("Number of stopped samples")
    ax.set_title("Stopped samples vs sampling rate")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        output_dir / f"{method_tag}_stopped_vs_sampling_rate_{stopping_rule}.png",
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_stopped_accuracy_vs_rate(
    output_dir: Path,
    sampling_rates: List[float],
    fst_stats: Dict[str, List[float]],
    forced_stats: Dict[str, List[float]],
    casc_stats: Dict[str, List[float]],
    stopping_rule: str,
    method_label: str = "FST",
    method_tag: str = "fst",
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
    ax.plot(
        sampling_rates,
        fst_stats["accuracy"],
        marker="o",
        color="#4C72B0",
        label=method_label,
    )
    ax.plot(
        sampling_rates,
        forced_stats["accuracy"],
        marker="o",
        color="#7F7F7F",
        label="Forced",
    )
    ax.plot(
        sampling_rates,
        casc_stats["accuracy"],
        marker="o",
        color="#55A868",
        label="CASC",
    )
    ax.set_xlabel("Sampling rate")
    ax.set_ylabel("Accuracy of stopped samples")
    ax.set_title("Stopped-sample accuracy vs sampling rate")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        output_dir / f"{method_tag}_stopped_accuracy_vs_sampling_rate_comparison_{stopping_rule}.png",
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="FST + LTT stopping on CIFAR-10.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/partial/cifar10_bayes_biased_random_rate_config.yaml",
    )
    parser.add_argument("--checkpoint_path", type=str, default="/gpfs/data/chopralab/us453/icml_26/results/checkpoints/cifar10/bayes_biased_random_rate/cifar10_baseline_epoch=57_val_accuracy=0.7336.ckpt")
    parser.add_argument("--alpha", type=float, default=0.1, help="Target risk level for H0: R > alpha.")
    parser.add_argument("--delta", type=float, default=0.05, help="FWER level for FST.")
    parser.add_argument("--calib_fraction", type=float, default=0.5)
    parser.add_argument("--est_size", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./fst_results/cifar10")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--sampling_rates",
        type=str,
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
    )
    parser.add_argument("--use_testset", action="store_true")
    parser.add_argument("--num_lambdas", type=int, default=201)
    parser.add_argument(
        "--stopping_rule",
        type=str,
        default="max_score",
        choices=["max_score", "set_size"],
        help="Decision function D_lambda(x_t).",
    )
    parser.add_argument(
        "--set_size_max",
        type=int,
        default=1,
        help="Used only for stopping_rule=set_size.",
    )
    parser.add_argument(
        "--delta_est",
        type=float,
        default=0.1,
        help="Quantile level on estimation set for per-time baseline thresholds.",
    )
    parser.add_argument(
        "--casc_alpha_schedule",
        type=str,
        default="uniform",
        choices=["uniform", "linear_increasing", "linear_decreasing", "geometric"],
        help="Allocate CASC risk across rates with sum_k alpha_k <= alpha.",
    )
    parser.add_argument(
        "--cumulative_sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--selection_method",
        type=str,
        default="fst",
        choices=["fst", "ltt"],
        help="FST (fixed sequence) or LTT (Holm step-down + utility selection).",
    )
    parser.add_argument(
        "--risk_mode",
        type=str,
        default="unconditional_with_abstain",
        choices=["unconditional_with_abstain", "conditional_on_stopped"],
        help=(
            "How to define R(lambda): mean over all calibration samples with "
            "abstainers contributing --abstain_penalty, or mean over stopped only."
        ),
    )
    parser.add_argument(
        "--abstain_penalty",
        type=float,
        default=0.0,
        help="Loss assigned to non-stopped (abstained) calibration samples in unconditional mode.",
    )
    parser.add_argument(
        "--utility",
        type=str,
        default="stopped_count",
        choices=["stopped_count", "neg_mean_stop_index", "neg_calib_risk"],
        help="Utility function used by LTT to pick a lambda from the safe set.",
    )
    parser.add_argument(
        "--default_lambda_idx",
        type=int,
        default=None,
        help="If LTT safe set is empty, fall back to lambdas[default_lambda_idx]. If unset, returns None.",
    )
    parser.add_argument(
        "--loss_kind",
        type=str,
        default="set_miscoverage",
        choices=["set_miscoverage", "argmax_miscoverage"],
        help=(
            "Per-sample calibration loss. set_miscoverage = 1{y not in {k: prob_k>=threshold}} "
            "(legacy). argmax_miscoverage = 1{argmax(probs at stop) != y}, which aligns the "
            "calibration risk with the test-time argmax accuracy and avoids the trivial "
            "set-coverage degeneracy at small lambda."
        ),
    )
    parser.add_argument(
        "--force_recompute_scores",
        action="store_true",
        help="Recompute probabilities/labels and overwrite the score cache.",
    )
    args = parser.parse_args()

    # Seed all RNGs before any model is constructed so that random sub-modules
    # whose weights are *not* restored by load_from_checkpoint(strict=False)
    # get a reproducible initialization.
    import random as _py_random
    _py_random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    config = get_config(args.config_path)
    device = torch.device(args.device)
    label_names = config.label_names.split(",") if isinstance(config.label_names, str) else config.label_names

    val_ds, test_ds = get_val_test_datasets(config, args.use_testset)
    est_ds, remaining = split_estimation_set(val_ds, args.est_size, args.seed)
    if args.use_testset and test_ds is not None:
        calib_ds = remaining
    else:
        calib_ds, test_ds = split_dataset(remaining, args.calib_fraction, args.seed)

    batch_size = args.batch_size or getattr(config, "val_batch_size", getattr(config, "batch_size", 32))
    model = load_model(config, args.checkpoint_path, device)
    sampling_rates = [float(x) for x in args.sampling_rates.split(",") if x.strip()]

    cache_dir = build_score_cache_dir(args.output_dir, args.seed)
    base_cache_key: Dict = {
        "seed": int(args.seed),
        "calib_fraction": float(args.calib_fraction),
        "est_size": int(args.est_size),
        "use_testset": bool(args.use_testset),
        "cumulative_sampling": bool(args.cumulative_sampling),
        "sampling_rates": tuple(sampling_rates),
        "checkpoint_path": str(args.checkpoint_path),
        "label_names": tuple(label_names),
    }

    est_probs, est_labels = collect_prob_tensor_cached(
        model, config, est_ds, "est", sampling_rates, label_names, device,
        batch_size, args.num_workers, args.cumulative_sampling,
        cache_dir,
        {**base_cache_key, "split": "est", "dataset_size": int(len(est_ds))},
        args.force_recompute_scores,
    )
    calib_probs, calib_labels = collect_prob_tensor_cached(
        model, config, calib_ds, "calib", sampling_rates, label_names, device,
        batch_size, args.num_workers, args.cumulative_sampling,
        cache_dir,
        {**base_cache_key, "split": "calib", "dataset_size": int(len(calib_ds))},
        args.force_recompute_scores,
    )
    test_probs, test_labels = collect_prob_tensor_cached(
        model, config, test_ds, "test", sampling_rates, label_names, device,
        batch_size, args.num_workers, args.cumulative_sampling,
        cache_dir,
        {**base_cache_key, "split": "test", "dataset_size": int(len(test_ds))},
        args.force_recompute_scores,
    )

    if args.stopping_rule == "max_score":
        baseline_t = build_max_score_baseline(est_probs, args.delta_est)
        lambdas = np.linspace(1.0, 0.0, args.num_lambdas)  # conservative -> aggressive
    else:
        baseline_t = build_true_class_baseline(est_probs, est_labels.astype(int), args.delta_est)
        lambdas = np.linspace(0.0, 1.0, args.num_lambdas)  # conservative -> aggressive

    losses_mat, stopped_mat, stop_idx_mat = per_lambda_calibration_arrays(
        calib_probs,
        calib_labels.astype(int),
        lambdas,
        args.stopping_rule,
        baseline_t,
        args.set_size_max,
        loss_kind=args.loss_kind,
    )
    n_cal = calib_labels.shape[0]
    risks, ns = grid_risks_and_ns(
        losses_mat, stopped_mat, args.risk_mode, args.abstain_penalty
    )
    p_values = compute_grid_pvalues(risks, ns, args.alpha, B=1.0)
    utilities = grid_utilities(stopped_mat, stop_idx_mat, risks, args.utility)

    if args.selection_method == "fst":
        lambda_hat, meta = fst_select_from_pvalues(lambdas, p_values, args.delta)
        meta_key = "fst_meta"
    else:
        lambda_hat, meta = ltt_select_lambda(
            lambdas,
            risks,
            p_values,
            args.delta,
            utilities,
            default_lambda_idx=args.default_lambda_idx,
        )
        meta_key = "ltt_meta"

    print(f"selection_method={args.selection_method} risk_mode={args.risk_mode}")
    print("trace (conservative -> aggressive):")
    for j in range(lambdas.shape[0]):
        print(
            f"  j={j:4d} lambda={float(lambdas[j]):.4f} "
            f"r_hat={float(risks[j]):.5f} n={int(ns[j])} "
            f"p={float(p_values[j]):.6g} u={float(utilities[j]):.4f}"
        )
    print(f"{args.selection_method} reason={meta['reason']}")
    if lambda_hat is None:
        print("No lambda selected (calibration set did not pass at FWER delta).")
    else:
        print(f"lambda_hat={lambda_hat:.6f}")

    out: Dict = {
        "alpha": args.alpha,
        "delta": args.delta,
        "n_cal": n_cal,
        "n_est": int(est_labels.shape[0]),
        "n_test": int(test_labels.shape[0]),
        "stopping_rule": args.stopping_rule,
        "set_size_max": args.set_size_max,
        "delta_est": args.delta_est,
        "casc_alpha_schedule": args.casc_alpha_schedule,
        "selection_method": args.selection_method,
        "risk_mode": args.risk_mode,
        "abstain_penalty": float(args.abstain_penalty),
        "utility": args.utility,
        "default_lambda_idx": args.default_lambda_idx,
        "loss_kind": args.loss_kind,
        "lambda_hat": None if lambda_hat is None else float(lambda_hat),
        meta_key: meta,
        "test_summary": None,
    }
    if lambda_hat is not None:
        out["test_summary"] = summarize_test(
            test_probs,
            test_labels.astype(int),
            lambda_hat,
            args.stopping_rule,
            baseline_t,
            args.set_size_max,
        )
        print("Test summary:", out["test_summary"])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    curves = {}
    if lambda_hat is not None:
        fst_stats = fst_curve(
            test_probs,
            test_labels.astype(int),
            float(lambda_hat),
            args.stopping_rule,
            baseline_t,
            args.set_size_max,
        )
        forced_stats = forced_curve(test_probs, test_labels.astype(int))
        casc_stats = casc_curve(
            calib_probs,
            calib_labels.astype(int),
            test_probs,
            test_labels.astype(int),
            args.alpha,
            args.casc_alpha_schedule,
        )
        method_label = args.selection_method.upper()
        method_tag = args.selection_method
        plot_stopped_vs_rate(
            out_dir,
            sampling_rates,
            fst_stats["stopped_count"],
            casc_stats["stopped_count"],
            args.stopping_rule,
            method_label=method_label,
            method_tag=method_tag,
        )
        plot_stopped_accuracy_vs_rate(
            out_dir,
            sampling_rates,
            fst_stats,
            forced_stats,
            casc_stats,
            args.stopping_rule,
            method_label=method_label,
            method_tag=method_tag,
        )
        curves = {args.selection_method: fst_stats, "forced": forced_stats, "casc": casc_stats}

    out_path = out_dir / f"{args.selection_method}_cifar10_summary_{args.stopping_rule}.json"

    def _json_default(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(type(obj))

    with open(out_path, "w") as handle:
        out["curves"] = curves
        json.dump(out, handle, indent=2, default=_json_default)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

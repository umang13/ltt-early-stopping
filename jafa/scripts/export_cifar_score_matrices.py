#!/usr/bin/env python3
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
RL_DIR = REPO_ROOT / "rl"
sys.path.insert(0, str(RL_DIR))

import main as rl_main  # noqa: E402


DEFAULT_RUN_DIR = (
    "rl/results/train_rl_cifar_grid_fixed_split_top8/"
    "rl_cifar_frozen_fixedsplit_top8_2_random_seed_123_"
    "r_cost_m0p025_initial_visible_count_4"
)
DEFAULT_OUTPUT_DIR = "rl/results/score_matrices/cifar_best"


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_json(path):
    with open(path) as f:
        return json.load(f)


def parse_splits(value):
    splits = [item.strip() for item in value.split(",") if item.strip()]
    allowed = {"val", "test"}
    unknown = sorted(set(splits) - allowed)
    if unknown:
        raise ValueError(f"unknown split(s): {unknown}")
    if not splits:
        raise ValueError("expected at least one split")
    return splits


def load_run_args(run_dir, cli_args):
    config = load_json(run_dir / "run_config.json")
    args = argparse.Namespace(**config)
    if cli_args.n_envs is not None:
        args.n_envs = cli_args.n_envs
    if cli_args.num_workers is not None:
        args.num_workers = cli_args.num_workers
    if cli_args.disable_cuda:
        args.disable_cuda = True
    args.device = torch.device(
        "cuda" if not args.disable_cuda and torch.cuda.is_available() else "cpu"
    )
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if args.device.type == "cuda":
        torch.cuda.manual_seed(args.random_seed)
    return args, config


def _empty_float(batch_size, n_steps, fill=np.nan):
    return np.full((batch_size, n_steps), fill, dtype=np.float32)


def _initial_patch_order(initial_mask, n_features):
    order = np.full((initial_mask.shape[0], n_features), -1, dtype=np.int16)
    for i in range(initial_mask.shape[0]):
        visible = np.flatnonzero(initial_mask[i] == 1)
        order[i, : visible.size] = visible.astype(np.int16)
    return order


def _batch_scores(step_runner, imgs, initial_mask, labels, visible_counts, args):
    n = imgs.shape[0]
    n_features = initial_mask.shape[1]
    n_steps = len(visible_counts)
    max_visible_count = int(visible_counts[-1])

    probs_out = np.empty((n, n_steps, 10), dtype=np.float32)
    predictions = np.empty((n, n_steps), dtype=np.int64)
    score_max_prob = np.empty((n, n_steps), dtype=np.float32)
    q_stop_out = np.empty((n, n_steps), dtype=np.float32)
    q_best_reveal = _empty_float(n, n_steps)
    q_mean_reveal = _empty_float(n, n_steps)
    score_q_margin_max = _empty_float(n, n_steps)
    score_q_margin_mean = _empty_float(n, n_steps)
    has_reveal_action = np.zeros((n, n_steps), dtype=bool)

    mask = initial_mask.copy()
    initial_count = int(mask.sum(axis=1)[0])
    patch_order = _initial_patch_order(initial_mask, n_features)
    natural_stop_t = np.full(n, -1, dtype=np.int16)
    natural_stop_visible_count = np.full(n, -1, dtype=np.int16)

    imgs_t = torch.from_numpy(imgs).to(args.device)

    with torch.no_grad():
        for col, visible_count in enumerate(visible_counts):
            if int(mask.sum(axis=1)[0]) != int(visible_count):
                raise RuntimeError(
                    f"mask has {int(mask.sum(axis=1)[0])} visible patches, "
                    f"expected {visible_count}"
                )

            mask_t = torch.from_numpy(mask).to(args.device)
            logits, q_values = step_runner.model(imgs_t, mask_t)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            q_np = q_values.detach().cpu().numpy()

            probs_out[:, col] = probs
            predictions[:, col] = probs.argmax(axis=1)
            score_max_prob[:, col] = probs.max(axis=1)
            q_stop = q_np[:, n_features]
            q_stop_out[:, col] = q_stop

            reveal_q = q_np[:, :n_features]
            unseen = mask == 0
            unseen_count = unseen.sum(axis=1)
            has_reveal = unseen_count > 0
            has_reveal_action[:, col] = has_reveal

            if has_reveal.any():
                masked_reveal_q = np.where(unseen, reveal_q, -np.inf)
                best_reveal = masked_reveal_q.max(axis=1)
                mean_reveal = np.divide(
                    (reveal_q * unseen).sum(axis=1),
                    unseen_count,
                    out=np.full(n, np.nan, dtype=np.float32),
                    where=unseen_count > 0,
                )

                q_best_reveal[has_reveal, col] = best_reveal[has_reveal]
                q_mean_reveal[has_reveal, col] = mean_reveal[has_reveal]
                score_q_margin_max[has_reveal, col] = (
                    q_stop[has_reveal] - best_reveal[has_reveal]
                )
                score_q_margin_mean[has_reveal, col] = (
                    q_stop[has_reveal] - mean_reveal[has_reveal]
                )

                should_stop = q_stop > best_reveal
                newly_stopped = (natural_stop_t < 0) & should_stop
                natural_stop_t[newly_stopped] = col
                natural_stop_visible_count[newly_stopped] = visible_count

            if visible_count >= max_visible_count:
                break

            if not has_reveal.all():
                raise RuntimeError("cannot continue trajectory when no reveal action exists")
            actions = np.where(unseen, reveal_q, -np.inf).argmax(axis=1)
            mask[np.arange(n), actions] = 1
            patch_order[:, visible_count] = actions.astype(np.int16)

    never_stopped = natural_stop_t < 0
    if max_visible_count >= n_features:
        natural_stop_t[never_stopped] = n_steps - 1
        natural_stop_visible_count[never_stopped] = visible_counts[-1]
    is_correct = predictions == labels[:, None]

    return {
        "labels": labels.astype(np.int64),
        "predictions": predictions,
        "is_correct": is_correct,
        "probs": probs_out,
        "score_max_prob": score_max_prob,
        "score_q_margin_max": score_q_margin_max,
        "score_q_margin_mean": score_q_margin_mean,
        "q_stop": q_stop_out,
        "q_best_reveal": q_best_reveal,
        "q_mean_reveal": q_mean_reveal,
        "has_reveal_action": has_reveal_action,
        "initial_masks": initial_mask.astype(np.uint8),
        "patch_order": patch_order,
        "natural_stop_t": natural_stop_t,
        "natural_stop_visible_count": natural_stop_visible_count,
    }


def _append_batch(accum, batch):
    for key, value in batch.items():
        accum.setdefault(key, []).append(value)


def _concat_batches(accum):
    return {key: np.concatenate(values, axis=0) for key, values in accum.items()}


def export_split(step_runner, data, visible_counts, args, max_samples=None):
    accum = {}
    data.reset()
    total = 0
    target_n = data.n_data if max_samples is None else min(max_samples, data.n_data)
    while total < target_n:
        batch_size = min(args.n_envs, target_n - total)
        imgs, mask, labels, _ = data.next_batch(batch_size)
        batch = _batch_scores(step_runner, imgs, mask, labels, visible_counts, args)
        _append_batch(accum, batch)
        total += batch_size
        if total % (args.n_envs * 10) == 0 or total == target_n:
            print(f"  exported {total}/{target_n}")

    out = _concat_batches(accum)
    out["visible_counts"] = np.asarray(visible_counts, dtype=np.int16)
    return out


def save_npz(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    print(f"Wrote {path}")


def validate_arrays(arrays):
    probs = arrays["probs"]
    if not np.allclose(probs.sum(axis=-1), 1.0, atol=1e-5):
        raise AssertionError("class probabilities do not sum to 1")
    if not np.allclose(arrays["score_max_prob"], probs.max(axis=-1), atol=1e-6):
        raise AssertionError("score_max_prob does not match probs.max")

    has_reveal = arrays["has_reveal_action"]
    margin = arrays["q_stop"] - arrays["q_best_reveal"]
    if not np.allclose(
        arrays["score_q_margin_max"][has_reveal],
        margin[has_reveal],
        atol=1e-5,
    ):
        raise AssertionError("score_q_margin_max does not match q_stop - q_best_reveal")

    expected_correct = arrays["predictions"] == arrays["labels"][:, None]
    if not np.array_equal(arrays["is_correct"], expected_correct):
        raise AssertionError("is_correct does not match predictions == labels")

    for order in arrays["patch_order"]:
        revealed = order[order >= 0]
        if len(np.unique(revealed)) != len(revealed):
            raise AssertionError("patch_order contains duplicate patches")


def write_metadata(path, cli_args, run_config, visible_counts, outputs):
    metadata = {
        "run_dir": str(resolve_path(cli_args.run_dir)),
        "checkpoint": cli_args.checkpoint,
        "checkpoint_path": str(resolve_path(cli_args.run_dir) / f"{cli_args.checkpoint}.model"),
        "splits": parse_splits(cli_args.splits),
        "output_files": outputs,
        "max_samples": cli_args.max_samples,
        "visible_counts": list(map(int, visible_counts)),
        "trajectory": "continue_after_stop_with_rl_patch_order",
        "natural_stop_t": "column index into visible_counts where greedy DQN first chooses STOP; -1 if not observed in a truncated export",
        "natural_stop_visible_count": "visible patch count where greedy DQN first chooses STOP; -1 if not observed in a truncated export",
        "q_margin_at_full_image": "NaN because no reveal action remains at 64 visible patches",
        "score_definitions": {
            "score_max_prob": "max_y softmax(classifier_logits)[y]",
            "score_q_margin_max": "Q(STOP) - max_j Q(reveal patch j), over unseen patches",
            "score_q_margin_mean": "Q(STOP) - mean_j Q(reveal patch j), over unseen patches",
        },
        "run_config": run_config,
    }
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--checkpoint", default="trained_best_acc")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--max_visible_count", type=int, default=64)
    parser.add_argument("--n_envs", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--disable_cuda", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    run_dir = resolve_path(args.run_dir)
    output_dir = resolve_path(args.output_dir)
    checkpoint_path = run_dir / f"{args.checkpoint}.model"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    run_args, run_config = load_run_args(run_dir, args)
    _, step_runner, _, valenv, _, testenv = rl_main.build_cifar_components(
        run_args, print_model=False
    )
    if run_args.freeze_after_pretrain:
        step_runner.freeze_encoder_classifier()
    state_dict = torch.load(checkpoint_path, map_location=run_args.device)
    step_runner.model.load_state_dict(state_dict)
    step_runner.model.eval()

    initial_visible_count = int(run_args.initial_visible_count)
    if args.max_visible_count < initial_visible_count or args.max_visible_count > 64:
        raise ValueError(
            f"max_visible_count must be in [{initial_visible_count}, 64], "
            f"got {args.max_visible_count}"
        )
    visible_counts = np.arange(
        initial_visible_count, args.max_visible_count + 1, dtype=np.int16
    )

    split_to_data = {"val": valenv.data, "test": testenv.data}
    outputs = {}
    for split in parse_splits(args.splits):
        print(f"Exporting {split}")
        arrays = export_split(
            step_runner,
            split_to_data[split],
            visible_counts,
            run_args,
            max_samples=args.max_samples,
        )
        validate_arrays(arrays)
        out_path = output_dir / f"{split}_scores.npz"
        save_npz(out_path, arrays)
        outputs[split] = str(out_path)

    write_metadata(output_dir / "metadata.json", args, run_config, visible_counts, outputs)


if __name__ == "__main__":
    main()

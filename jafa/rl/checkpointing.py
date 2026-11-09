import json
import os

import numpy as np
import wandb


def parse_checkpoint_budgets(value, n_features):
    budgets = []
    for raw in str(value).split(","):
        raw = raw.strip()
        if not raw:
            continue
        budget = int(raw)
        if budget < 0 or budget > n_features:
            raise ValueError(f"checkpoint budget {budget} outside [0, {n_features}]")
        budgets.append(budget)
    if not budgets:
        raise ValueError("checkpoint_budgets must contain at least one value")
    return budgets


def init_checkpoint_state(args, budgets):
    checkpoints = {
        "best_acc": {
            "checkpoint": "trained_best_acc",
            "path": os.path.join(args.save_path, "trained_best_acc.model"),
            "accuracy": -np.inf,
            "mean_patches": None,
            "median_patches": None,
            "min_patches": None,
            "max_patches": None,
            "returns_mean": None,
            "update": None,
            "budget": None,
        }
    }
    for budget in budgets:
        checkpoints[f"budget{budget}"] = {
            "checkpoint": f"trained_best_budget{budget}",
            "path": os.path.join(args.save_path, f"trained_best_budget{budget}.model"),
            "accuracy": -np.inf,
            "mean_patches": None,
            "median_patches": None,
            "min_patches": None,
            "max_patches": None,
            "returns_mean": None,
            "update": None,
            "budget": budget,
        }
    write_checkpoint_summary(args, checkpoints)
    return checkpoints


def checkpoint_rows(checkpoints):
    rows = []
    for name, state in checkpoints.items():
        if state["update"] is None:
            continue
        rows.append(
            {
                "name": name,
                "checkpoint": state["checkpoint"],
                "accuracy": state["accuracy"],
                "mean_patches": state["mean_patches"],
                "median_patches": state["median_patches"],
                "min_patches": state["min_patches"],
                "max_patches": state["max_patches"],
                "returns_mean": state["returns_mean"],
                "update": state["update"],
                "budget": state["budget"],
                "path": os.path.basename(state["path"]),
            }
        )
    return rows


def write_checkpoint_summary(args, checkpoints):
    payload = {
        "run_dir": args.save_path,
        "random_seed": args.random_seed,
        "r_cost": args.r_cost,
        "initial_visible_count": args.initial_visible_count,
        "freeze_after_pretrain": args.freeze_after_pretrain,
        "checkpoints": checkpoint_rows(checkpoints),
    }
    output_path = os.path.join(args.save_path, "checkpoint_summary.json")
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def checkpoint_wandb_payload(checkpoints):
    payload = {}
    for name, state in checkpoints.items():
        if state["update"] is None:
            continue
        prefix = "checkpoint/best_acc" if name == "best_acc" else f"checkpoint/{name}"
        payload[f"{prefix}_accuracy"] = state["accuracy"]
        payload[f"{prefix}_mean_patches"] = state["mean_patches"]
        payload[f"{prefix}_update"] = state["update"]
    return payload


def _update_state(state, accuracy, summary, update):
    state.update(
        accuracy=accuracy,
        mean_patches=summary["mean_patches"],
        median_patches=summary["median_patches"],
        min_patches=summary["min_patches"],
        max_patches=summary["max_patches"],
        returns_mean=summary["returns_mean"],
        update=update,
    )


def maybe_save_checkpoints(step_runner, args, checkpoints, update, summary, min_update):
    if update <= min_update:
        return

    changed = False
    accuracy = summary["accuracy"]
    mean_patches = summary["mean_patches"]
    best_acc = checkpoints["best_acc"]
    if accuracy >= best_acc["accuracy"]:
        _update_state(best_acc, accuracy, summary, update)
        step_runner.save(best_acc["path"])
        changed = True

    for state in checkpoints.values():
        budget = state["budget"]
        if budget is None or mean_patches > budget:
            continue
        if accuracy >= state["accuracy"]:
            _update_state(state, accuracy, summary, update)
            step_runner.save(state["path"])
            changed = True

    if changed:
        write_checkpoint_summary(args, checkpoints)

    payload = checkpoint_wandb_payload(checkpoints)
    if payload:
        payload["rl/update"] = update
        wandb.log(payload)

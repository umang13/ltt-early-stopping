import argparse
import itertools
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


BOOLEAN_FLAGS = {
    "batchnorm",
    "complete",
    "disable_cuda",
    "done_action_train",
    "dropout",
    "freeze_after_pretrain",
}


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def grid_runs(config):
    common = dict(config.get("common", {}))
    explicit_runs = config.get("runs")
    grid = config.get("grid", {})
    keys = list(grid)
    base_runs = explicit_runs if explicit_runs else [{}]
    if not keys:
        return [dict(common, **run) for run in base_runs]
    runs = []
    for base_run in base_runs:
        for values in itertools.product(*(grid[key] for key in keys)):
            run = dict(common)
            run.update(base_run)
            run.update(dict(zip(keys, values)))
            runs.append(run)
    return runs


def run_label(config, index, args):
    parts = [config.get("name", "rl_cifar"), str(index)]
    for key in ("random_seed", "r_cost", "initial_visible_count"):
        if key in args:
            value = str(args[key]).replace("-", "m").replace(".", "p")
            parts.append(f"{key}_{value}")
    return "_".join(parts)


def add_default_run_name(config, index, args):
    if args.get("wandb_run_name"):
        return
    args["wandb_run_name"] = run_label(config, index, args)


def cli_value(value):
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def cli_args(args):
    out = []
    for key, value in args.items():
        if value is None:
            continue
        flag = f"--{key}"
        if key in BOOLEAN_FLAGS:
            if value:
                out.append(flag)
            continue
        out.extend([flag, cli_value(value)])
    return out


def resolve_index(explicit_index):
    if explicit_index is not None:
        return explicit_index
    return int(os.environ["SLURM_ARRAY_TASK_ID"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rl/rl_cifar_grid.yaml")
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--count", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    config = load_config(config_path)
    runs = grid_runs(config)
    if args.count:
        print(len(runs))
        return

    index = resolve_index(args.index)
    run_args = runs[index]
    add_default_run_name(config, index, run_args)
    command = [
        sys.executable,
        "-u",
        config.get("entrypoint", "rl/main.py"),
        *cli_args(run_args),
    ]
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    subprocess.run(command, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()

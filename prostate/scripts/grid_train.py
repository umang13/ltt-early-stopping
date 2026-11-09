#!/usr/bin/env python
"""Simple grid runner that calls training scripts directly."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import yaml

from scripts.train_image_metrics import parse_args as parse_image_args
from scripts.train_image_metrics import run_training as run_image_training
from scripts.train_kspace_post_grappa import parse_args as parse_kspace_args
from scripts.train_kspace_post_grappa import run_training as run_kspace_training


TRAINERS = {
    "image": {"parse": parse_image_args, "run": run_image_training},
    "kspace": {"parse": parse_kspace_args, "run": run_kspace_training},
}


def _generate_grid(grid_params: Dict[str, Iterable[Any]]) -> List[Dict[str, Any]]:
    if not grid_params:
        return [{}]
    keys = sorted(grid_params.keys())
    values = [grid_params[key] for key in keys]
    combinations = itertools.product(*values)
    return [dict(zip(keys, combo)) for combo in combinations]


def _dict_to_cli(args_dict: Dict[str, Any]) -> List[str]:
    cli: List[str] = []
    for key in sorted(args_dict.keys()):
        value = args_dict[key]
        if value is None:
            continue
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                cli.append(flag)
            continue
        if isinstance(value, (list, tuple)):
            for item in value:
                cli.extend([flag, str(item)])
            continue
        cli.extend([flag, str(value)])
    return cli


def _load_config(config_path: Path) -> Dict[str, Any]:
    with Path(config_path).open("r") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config must be a mapping")
    return config


def _materialize_runs(
    *,
    mode: str,
    base_args: Dict[str, Any],
    grid_params: Dict[str, Iterable[Any]],
    label: str | None,
) -> List[Dict[str, Any]]:
    if mode not in TRAINERS:
        raise ValueError(f"Unknown mode '{mode}'. Expected one of {list(TRAINERS.keys())}")
    combos = _generate_grid(grid_params)
    runs: List[Dict[str, Any]] = []
    for overrides in combos:
        merged = {**base_args, **overrides}
        merged.pop("mode", None)  # ignore accidental inline mode entries
        runs.append({"mode": mode, "args": merged, "label": label})
    return runs


def _experiment_runs(exp: Dict[str, Any], exp_idx: int) -> List[Dict[str, Any]]:
    if not isinstance(exp, dict):
        raise ValueError("Each experiment entry must be a mapping")
    exp_mode = exp.get("mode")
    if exp_mode is None:
        raise ValueError("Each experiment entry must set 'mode'")

    base_args = exp.get("base_args", {})
    grid_params = exp.get("grid_params", {})
    label = exp.get("name") or exp.get("label") or f"experiment_{exp_idx}"
    variants = exp.get("variants")
    if variants is None:
        return _materialize_runs(
            mode=exp_mode,
            base_args=base_args,
            grid_params=grid_params,
            label=label,
        )

    if grid_params:
        raise ValueError("Experiment entries using 'variants' must not also set top-level 'grid_params'")
    if not isinstance(variants, list):
        raise ValueError("variants must be a list when provided")

    runs: List[Dict[str, Any]] = []
    for variant_idx, variant in enumerate(variants):
        if not isinstance(variant, dict):
            raise ValueError("Each variant entry must be a mapping")
        variant_base_args = {**base_args, **variant.get("base_args", {})}
        variant_grid_params = variant.get("grid_params", {})
        variant_label = variant.get("label")
        run_label = label if variant_label is None else f"{label}:{variant_label}"
        runs.extend(
            _materialize_runs(
                mode=exp_mode,
                base_args=variant_base_args,
                grid_params=variant_grid_params,
                label=run_label,
            )
        )
    return runs


def run_grid(config_path: Path, index: int | None, dry_run: bool) -> None:
    config = _load_config(config_path)

    experiments = config.get("experiments")
    runs: List[Dict[str, Any]] = []

    if experiments:
        if not isinstance(experiments, list):
            raise ValueError("experiments must be a list when provided")
        for exp_idx, exp in enumerate(experiments):
            runs.extend(_experiment_runs(exp, exp_idx))
    else:
        exp_mode = config.get("mode")
        if exp_mode is None:
            raise ValueError("Config without 'experiments' must supply a top-level 'mode'")
        base_args = config.get("base_args", {})
        grid_params = config.get("grid_params", {})
        runs.extend(
            _materialize_runs(
                mode=exp_mode,
                base_args=base_args,
                grid_params=grid_params,
                label=config.get("name"),
            )
        )

    if not runs:
        raise ValueError("Grid configuration produced no runs")

    if index is not None:
        if index < 0 or index >= len(runs):
            raise ValueError(f"Index {index} is out of range for grid of size {len(runs)}")
        runs = [runs[index]]

    total = len(runs)
    for run_idx, run in enumerate(runs):
        run_mode = run["mode"]
        if run_mode not in TRAINERS:
            raise ValueError(f"Unknown mode '{run_mode}' in run #{run_idx}")
        trainer = TRAINERS[run_mode]
        cli_args = _dict_to_cli(run["args"])
        label = run.get("label")
        label_suffix = f" [{label}]" if label else ""
        print(
            f"[grid] Run {run_idx + 1}/{total}{label_suffix} -> "
            f"{' '.join(['python', trainer['parse'].__module__, *cli_args])}"
        )
        if dry_run:
            continue
        parsed_args = trainer["parse"](cli_args)
        trainer["run"](parsed_args)


def parse_cli(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid launcher for training scripts")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--index", type=int, default=None, help="Optional grid index (for SLURM arrays)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--count", action="store_true", help="Print total grid size and exit")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_cli()
    if args.count:
        config = _load_config(args.config)
        experiments = config.get("experiments", [config])
        total = 0
        for exp_idx, exp in enumerate(experiments if isinstance(experiments, list) else [experiments]):
            total += len(_experiment_runs(exp, exp_idx))
        print(f"Total grid combinations: {total}")
        print(f"SLURM array range: --array=0-{total - 1}")
        return
    run_grid(args.config, args.index, args.dry_run)


if __name__ == "__main__":
    main()

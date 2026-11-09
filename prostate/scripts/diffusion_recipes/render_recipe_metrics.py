"""Render ADC/b1500 sidecar metrics for arbitrary ESPIRiT recipes."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from prostate_kspace_classifier.data.recipes.parse import parse_recipe_id
from prostate_kspace_classifier.data.recipes.render import render_recipe_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--combine", type=str, default="espirit")
    parser.add_argument("--recipe", type=str, action="append", required=True)
    parser.add_argument("--include-trace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recipes = [
        parse_recipe_id(recipe_id, family="manual", averaging_tag=recipe_id)
        for recipe_id in args.recipe
    ]
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.input_path, "r") as input_handle, h5py.File(
        args.output_path, "w"
    ) as output_handle:
        output_handle.create_dataset("metadata/source_path", data=str(args.input_path))
        output_handle.create_dataset("metadata/combine", data=args.combine)
        output_handle.create_dataset(
            "metadata/recipes",
            data=np.asarray([spec.recipe_id for spec in recipes], dtype="S128"),
        )
        for spec in recipes:
            rendered = render_recipe_metrics(
                input_handle,
                args.combine,
                spec,
                include_trace=args.include_trace,
            )
            for metric_name, value in rendered.items():
                output_handle.create_dataset(
                    f"recipes/{spec.recipe_id}/{metric_name}",
                    data=value,
                )


if __name__ == "__main__":
    main()

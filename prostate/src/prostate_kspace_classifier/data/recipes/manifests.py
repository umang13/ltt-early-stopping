"""Recipe manifest helpers for deterministic validation and test splits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from prostate_kspace_classifier.data.metadata_index import AccessionLabel

from .sampling import resolve_recipes_for_accession


def build_recipe_manifest_rows(
    accession_labels: Sequence[AccessionLabel],
    *,
    recipe_mode: str | None,
    recipes_per_accession: int | None,
    recipe_family_quotas: list[str] | tuple[str, ...] | None,
    recipe_seed: int,
    recipe_split: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label in accession_labels:
        specs = resolve_recipes_for_accession(
            label.accession_id,
            recipe_mode=recipe_mode,
            recipes_per_accession=recipes_per_accession,
            recipe_family_quotas=recipe_family_quotas,
            recipe_seed=recipe_seed,
            recipe_split=recipe_split,
        )
        for spec in specs:
            rows.append(
                {
                    "accession_id": label.accession_id,
                    "recipe_id": spec.recipe_id,
                    "recipe_family": spec.family,
                    "averaging_tag": spec.averaging_tag,
                    "counts_b50": list(spec.counts_b50),
                    "counts_b1000": list(spec.counts_b1000),
                    "direction_mask": list(spec.direction_mask),
                }
            )
    return rows


def write_recipe_manifest(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")

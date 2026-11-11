"""Recipe sampling for fixed and stochastic ESPIRiT regimes."""

from __future__ import annotations

import hashlib
import random
from collections import Counter

from .parse import (
    build_active_direction_recipe,
    build_legacy_tag_recipes,
    build_xyz_grid48_recipes,
)
from .types import (
    AXES,
    AXIS_TO_INDEX,
    B1000_MAX_COUNT,
    B50_MAX_COUNT,
    DEFAULT_RECIPE_FAMILY_QUOTAS,
    DEFAULT_RECIPES_PER_ACCESSION,
    FAMILY_DUAL_DIR,
    FAMILY_LEGACY_ANCHOR,
    FAMILY_SINGLE_DIR,
    FAMILY_XYZ_ASYMMETRIC,
    FAMILY_XYZ_EQUAL,
    RECIPE_MODE_LEGACY_TAGS,
    RECIPE_MODE_SAMPLED_RECIPES,
    RECIPE_MODE_XYZ_GRID48,
    RecipeSpec,
)


PAIR_DIRECTIONS = (("x", "y"), ("x", "z"), ("y", "z"))


def parse_recipe_family_quotas(
    quota_strings: list[str] | tuple[str, ...] | None,
) -> list[tuple[str, int]]:
    source = quota_strings if quota_strings else DEFAULT_RECIPE_FAMILY_QUOTAS
    quotas: list[tuple[str, int]] = []
    for item in source:
        family, count = item.split("=", 1)
        quotas.append((family, int(count)))
    return quotas


def resolve_recipe_group_labels(
    recipe_mode: str | None,
    recipe_family_quotas: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    if recipe_mode == RECIPE_MODE_LEGACY_TAGS:
        return [spec.averaging_tag for spec in build_legacy_tag_recipes()]
    if recipe_mode == RECIPE_MODE_XYZ_GRID48:
        return [spec.recipe_id for spec in build_xyz_grid48_recipes()]
    if recipe_mode == RECIPE_MODE_SAMPLED_RECIPES:
        return [family for family, _ in parse_recipe_family_quotas(recipe_family_quotas)]
    return []


def _stable_seed(*parts: object) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _random_count_tuple(
    rng: random.Random,
    active_directions: tuple[str, ...],
    *,
    tied: bool = False,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    counts_b50 = [0, 0, 0]
    counts_b1000 = [0, 0, 0]
    if tied:
        b50_count = rng.randint(1, B50_MAX_COUNT)
        b1000_count = rng.randint(1, B1000_MAX_COUNT)
        for axis in active_directions:
            index = AXIS_TO_INDEX[axis]
            counts_b50[index] = b50_count
            counts_b1000[index] = b1000_count
        return tuple(counts_b50), tuple(counts_b1000)

    for axis in active_directions:
        index = AXIS_TO_INDEX[axis]
        counts_b50[index] = rng.randint(1, B50_MAX_COUNT)
        counts_b1000[index] = rng.randint(1, B1000_MAX_COUNT)
    return tuple(counts_b50), tuple(counts_b1000)


def _append_unique(
    recipes: list[RecipeSpec],
    used_recipe_ids: set[str],
    spec: RecipeSpec,
) -> bool:
    if spec.recipe_id in used_recipe_ids:
        return False
    used_recipe_ids.add(spec.recipe_id)
    recipes.append(spec)
    return True


def _sample_single_dir(
    rng: random.Random,
    *,
    count: int,
    used_recipe_ids: set[str],
) -> list[RecipeSpec]:
    recipes: list[RecipeSpec] = []
    axis_sequence = ["x", "y", "z"]
    if count > 3:
        axis_sequence.extend(rng.choice(AXES) for _ in range(count - 3))
    for axis in axis_sequence[:count]:
        while True:
            counts_b50, counts_b1000 = _random_count_tuple(rng, (axis,))
            spec = build_active_direction_recipe(
                (axis,),
                counts_b50,
                counts_b1000,
                family=FAMILY_SINGLE_DIR,
                averaging_tag=FAMILY_SINGLE_DIR,
            )
            if _append_unique(recipes, used_recipe_ids, spec):
                break
    return recipes


def _sample_dual_dir(
    rng: random.Random,
    *,
    count: int,
    used_recipe_ids: set[str],
) -> list[RecipeSpec]:
    recipes: list[RecipeSpec] = []
    pair_sequence = [("x", "y"), ("x", "y"), ("x", "z"), ("x", "z"), ("y", "z"), ("y", "z")]
    if count > 6:
        pair_sequence.extend(rng.choice(PAIR_DIRECTIONS) for _ in range(count - 6))
    for active_directions in pair_sequence[:count]:
        while True:
            counts_b50, counts_b1000 = _random_count_tuple(rng, active_directions)
            spec = build_active_direction_recipe(
                active_directions,
                counts_b50,
                counts_b1000,
                family=FAMILY_DUAL_DIR,
                averaging_tag=FAMILY_DUAL_DIR,
            )
            if _append_unique(recipes, used_recipe_ids, spec):
                break
    return recipes


def _sample_legacy_anchor(
    rng: random.Random,
    *,
    count: int,
    used_recipe_ids: set[str],
) -> list[RecipeSpec]:
    recipes: list[RecipeSpec] = []
    pool = build_legacy_tag_recipes()
    rng.shuffle(pool)
    for spec in pool:
        sampled = RecipeSpec(
            recipe_id=spec.recipe_id,
            family=FAMILY_LEGACY_ANCHOR,
            averaging_tag=FAMILY_LEGACY_ANCHOR,
            active_directions=spec.active_directions,
            counts_b50=spec.counts_b50,
            counts_b1000=spec.counts_b1000,
        )
        if _append_unique(recipes, used_recipe_ids, sampled) and len(recipes) == count:
            break
    return recipes


def _sample_xyz_equal(
    rng: random.Random,
    *,
    count: int,
    used_recipe_ids: set[str],
) -> list[RecipeSpec]:
    recipes: list[RecipeSpec] = []
    pool = build_xyz_grid48_recipes()
    rng.shuffle(pool)
    for spec in pool:
        sampled = RecipeSpec(
            recipe_id=spec.recipe_id,
            family=FAMILY_XYZ_EQUAL,
            averaging_tag=FAMILY_XYZ_EQUAL,
            active_directions=spec.active_directions,
            counts_b50=spec.counts_b50,
            counts_b1000=spec.counts_b1000,
        )
        if _append_unique(recipes, used_recipe_ids, sampled) and len(recipes) == count:
            break
    return recipes


def _sample_xyz_asymmetric(
    rng: random.Random,
    *,
    count: int,
    used_recipe_ids: set[str],
) -> list[RecipeSpec]:
    recipes: list[RecipeSpec] = []
    active_directions = ("x", "y", "z")
    while len(recipes) < count:
        counts_b50, counts_b1000 = _random_count_tuple(rng, active_directions)
        if len({counts_b50[0], counts_b50[1], counts_b50[2]}) == 1 and len(
            {counts_b1000[0], counts_b1000[1], counts_b1000[2]}
        ) == 1:
            continue
        spec = build_active_direction_recipe(
            active_directions,
            counts_b50,
            counts_b1000,
            family=FAMILY_XYZ_ASYMMETRIC,
            averaging_tag=FAMILY_XYZ_ASYMMETRIC,
        )
        _append_unique(recipes, used_recipe_ids, spec)
    return recipes


def resolve_recipes_for_accession(
    accession_id: str,
    *,
    recipe_mode: str | None,
    recipes_per_accession: int | None,
    recipe_family_quotas: list[str] | tuple[str, ...] | None,
    recipe_seed: int,
    recipe_split: str,
    recipe_epoch: int = 0,
) -> list[RecipeSpec]:
    if recipe_mode == RECIPE_MODE_LEGACY_TAGS:
        return build_legacy_tag_recipes()
    if recipe_mode == RECIPE_MODE_XYZ_GRID48:
        return build_xyz_grid48_recipes()
    if recipe_mode != RECIPE_MODE_SAMPLED_RECIPES:
        return []

    quotas = parse_recipe_family_quotas(recipe_family_quotas)
    target_count = recipes_per_accession or DEFAULT_RECIPES_PER_ACCESSION
    if sum(count for _, count in quotas) != target_count:
        raise ValueError("recipes_per_accession must equal the sum of recipe_family_quotas")

    rng = random.Random(
        _stable_seed(recipe_seed, recipe_split, recipe_epoch, accession_id)
    )
    recipes: list[RecipeSpec] = []
    used_recipe_ids: set[str] = set()
    for family, count in quotas:
        if family == FAMILY_SINGLE_DIR:
            recipes.extend(
                _sample_single_dir(rng, count=count, used_recipe_ids=used_recipe_ids)
            )
        elif family == FAMILY_DUAL_DIR:
            recipes.extend(
                _sample_dual_dir(rng, count=count, used_recipe_ids=used_recipe_ids)
            )
        elif family == FAMILY_LEGACY_ANCHOR:
            recipes.extend(
                _sample_legacy_anchor(
                    rng, count=count, used_recipe_ids=used_recipe_ids
                )
            )
        elif family == FAMILY_XYZ_EQUAL:
            recipes.extend(
                _sample_xyz_equal(rng, count=count, used_recipe_ids=used_recipe_ids)
            )
        elif family == FAMILY_XYZ_ASYMMETRIC:
            recipes.extend(
                _sample_xyz_asymmetric(
                    rng, count=count, used_recipe_ids=used_recipe_ids
                )
            )
        else:
            raise ValueError(f"Unsupported recipe family '{family}'")
    if len(recipes) != target_count:
        raise ValueError(
            f"Resolved {len(recipes)} recipes but expected {target_count}"
        )
    return recipes


def recipe_family_counter(
    recipe_mode: str | None,
    recipe_family_quotas: list[str] | tuple[str, ...] | None,
) -> Counter[str]:
    if recipe_mode == RECIPE_MODE_SAMPLED_RECIPES:
        return Counter(dict(parse_recipe_family_quotas(recipe_family_quotas)))
    if recipe_mode == RECIPE_MODE_LEGACY_TAGS:
        return Counter({FAMILY_LEGACY_ANCHOR: len(build_legacy_tag_recipes())})
    if recipe_mode == RECIPE_MODE_XYZ_GRID48:
        return Counter({FAMILY_XYZ_EQUAL: len(build_xyz_grid48_recipes())})
    return Counter()

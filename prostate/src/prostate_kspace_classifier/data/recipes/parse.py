"""Recipe parsing and deterministic fixed recipe sets."""

from __future__ import annotations

from .types import (
    AXES,
    AXIS_TO_INDEX,
    FAMILY_LEGACY_ANCHOR,
    FAMILY_XYZ_EQUAL,
    LEGACY_TAGS,
    RecipeSpec,
)


def format_recipe_id(
    active_directions: tuple[str, ...],
    counts_b50: tuple[int, int, int],
    counts_b1000: tuple[int, int, int],
) -> str:
    direction_token = "".join(active_directions)
    b50_tokens = [
        f"50{axis}{counts_b50[AXIS_TO_INDEX[axis]]}" for axis in active_directions
    ]
    b1000_tokens = [
        f"1000{axis}{counts_b1000[AXIS_TO_INDEX[axis]]}" for axis in active_directions
    ]
    return f"{direction_token}__{'_'.join(b50_tokens)}__{'_'.join(b1000_tokens)}"


def _parse_shell_counts(
    shell_spec: str,
    expected_prefix: str,
    active_directions: tuple[str, ...],
) -> tuple[int, int, int]:
    counts = [0, 0, 0]
    tokens = [token for token in shell_spec.split("_") if token]
    seen_axes: set[str] = set()
    for token in tokens:
        axis = token[len(expected_prefix)]
        count = int(token[len(expected_prefix) + 1 :])
        seen_axes.add(axis)
        counts[AXIS_TO_INDEX[axis]] = count
    if tuple(sorted(seen_axes)) != active_directions:
        raise ValueError(
            f"Shell spec axes {sorted(seen_axes)} do not match active directions {active_directions}"
        )
    return tuple(counts)


def parse_recipe_id(
    recipe_id: str,
    *,
    family: str,
    averaging_tag: str | None = None,
) -> RecipeSpec:
    direction_token, b50_spec, b1000_spec = recipe_id.split("__")
    active_directions = tuple(direction_token)
    if active_directions != tuple(sorted(set(active_directions))):
        raise ValueError(
            f"Recipe directions must be sorted and unique: {recipe_id!r}"
        )
    counts_b50 = _parse_shell_counts(b50_spec, "50", active_directions)
    counts_b1000 = _parse_shell_counts(b1000_spec, "1000", active_directions)
    if any(counts_b50[AXIS_TO_INDEX[axis]] <= 0 for axis in active_directions):
        raise ValueError(f"Active b50 counts must be positive: {recipe_id!r}")
    if any(counts_b1000[AXIS_TO_INDEX[axis]] <= 0 for axis in active_directions):
        raise ValueError(f"Active b1000 counts must be positive: {recipe_id!r}")
    return RecipeSpec(
        recipe_id=recipe_id,
        family=family,
        averaging_tag=averaging_tag if averaging_tag is not None else recipe_id,
        active_directions=active_directions,
        counts_b50=counts_b50,
        counts_b1000=counts_b1000,
    )


def legacy_tag_to_recipe(tag: str, *, averaging_tag: str | None = None) -> RecipeSpec:
    _, n50, _, n1000 = tag.split("_")
    counts_b50 = (int(n50), int(n50), int(n50))
    counts_b1000 = (int(n1000), int(n1000), int(n1000))
    return parse_recipe_id(
        format_recipe_id(("x", "y", "z"), counts_b50, counts_b1000),
        family=FAMILY_LEGACY_ANCHOR,
        averaging_tag=averaging_tag if averaging_tag is not None else tag,
    )


def build_legacy_tag_recipes() -> list[RecipeSpec]:
    return [legacy_tag_to_recipe(tag) for tag in LEGACY_TAGS]


def build_xyz_grid48_recipes() -> list[RecipeSpec]:
    recipes: list[RecipeSpec] = []
    for n50 in range(1, 5):
        for n1000 in range(1, 13):
            counts_b50 = (n50, n50, n50)
            counts_b1000 = (n1000, n1000, n1000)
            recipe_id = format_recipe_id(("x", "y", "z"), counts_b50, counts_b1000)
            recipes.append(
                parse_recipe_id(
                    recipe_id,
                    family=FAMILY_XYZ_EQUAL,
                    averaging_tag=recipe_id,
                )
            )
    return recipes


def build_active_direction_recipe(
    active_directions: tuple[str, ...],
    counts_b50: tuple[int, int, int],
    counts_b1000: tuple[int, int, int],
    *,
    family: str,
    averaging_tag: str,
) -> RecipeSpec:
    return parse_recipe_id(
        format_recipe_id(active_directions, counts_b50, counts_b1000),
        family=family,
        averaging_tag=averaging_tag,
    )

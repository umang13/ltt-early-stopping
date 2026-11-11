"""Types and constants for ESPIRiT recipe rendering."""

from __future__ import annotations

from dataclasses import dataclass


AXES = ("x", "y", "z")
AXIS_TO_INDEX = {axis: index for index, axis in enumerate(AXES)}
B50_MAX_COUNT = 4
B1000_MAX_COUNT = 12

RECIPE_MODE_LEGACY_TAGS = "legacy_tags"
RECIPE_MODE_XYZ_GRID48 = "xyz_grid48"
RECIPE_MODE_SAMPLED_RECIPES = "sampled_recipes"
RECIPE_MODE_CHOICES = (
    RECIPE_MODE_LEGACY_TAGS,
    RECIPE_MODE_XYZ_GRID48,
    RECIPE_MODE_SAMPLED_RECIPES,
)

DEFAULT_RECIPES_PER_ACCESSION = 24
DEFAULT_RECIPE_SEED = 1337
DEFAULT_RECIPE_FAMILY_QUOTAS = (
    "single_dir=4",
    "dual_dir=8",
    "legacy_anchor=4",
    "xyz_equal=4",
    "xyz_asymmetric=4",
)

FAMILY_SINGLE_DIR = "single_dir"
FAMILY_DUAL_DIR = "dual_dir"
FAMILY_LEGACY_ANCHOR = "legacy_anchor"
FAMILY_XYZ_EQUAL = "xyz_equal"
FAMILY_XYZ_ASYMMETRIC = "xyz_asymmetric"

LEGACY_TAGS = (
    "b50_1_b1000_1",
    "b50_2_b1000_1",
    "b50_2_b1000_2",
    "b50_3_b1000_2",
    "b50_3_b1000_3",
    "b50_4_b1000_3",
    "b50_4_b1000_4",
    "b50_4_b1000_5",
    "b50_4_b1000_6",
    "b50_4_b1000_7",
    "b50_4_b1000_8",
    "b50_4_b1000_9",
    "b50_4_b1000_10",
    "b50_4_b1000_11",
    "b50_4_b1000_12",
)


@dataclass(frozen=True)
class RecipeSpec:
    """One rendered ADC/b1500 recipe."""

    recipe_id: str
    family: str
    averaging_tag: str
    active_directions: tuple[str, ...]
    counts_b50: tuple[int, int, int]
    counts_b1000: tuple[int, int, int]

    @property
    def direction_mask(self) -> tuple[int, int, int]:
        return tuple(
            1 if axis in self.active_directions else 0 for axis in AXES
        )

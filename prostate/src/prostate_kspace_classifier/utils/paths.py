"""Utilities to resolve project-relative paths."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def data_path(*parts: str) -> Path:
    """Return data directory path given optional sub-parts."""

    return PROJECT_ROOT / "data" / Path(*parts)


def output_path(*parts: str) -> Path:
    """Return outputs directory path and ensure it exists."""

    path = PROJECT_ROOT / "outputs" / Path(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

"""Simple logging configuration helper."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(log_file: Path | None = None) -> None:
    """Configure root logger with console and optional file handlers."""

    handlers = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
    )

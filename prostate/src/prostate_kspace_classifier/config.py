"""Dataclass-backed configuration objects used by training scripts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class DataConfig:
    """Paths and dataset settings shared by image and k-space modes."""

    metadata_csv: Path
    h5_root: Path
    mode: str = "image"  # "image" or "kspace"
    num_workers: int = 8
    batch_size: int = 2
    seed: int = 1337


@dataclass
class ImageModelConfig:
    """Hyper-parameters for the ADC + b1500 classifier."""

    backbone: str = "resnet3d18"
    pretrained: bool = False
    in_channels: int = 2
    dropout: float = 0.1


@dataclass
class KspaceModelConfig:
    """Hyper-parameters for the k-space classifier stack."""

    fftconv_kernel_size: int = 5
    fftconv_channels: int = 32
    num_slices: int = 192
    complex_mode: str = "separate"
    dropout: float = 0.1


@dataclass
class OptimConfig:
    """Optimizer and scheduler knobs."""

    lr: float = 1e-4
    weight_decay: float = 1e-5
    betas: tuple[float, float] = (0.9, 0.999)
    max_epochs: int = 50
    warmup_epochs: int = 3


@dataclass
class ExperimentConfig:
    """Full configuration tree consumed by training scripts."""

    data: DataConfig
    optimizer: OptimConfig = field(default_factory=OptimConfig)
    image_model: Optional[ImageModelConfig] = None
    kspace_model: Optional[KspaceModelConfig] = None
    extras: Dict[str, Any] = field(default_factory=dict)


def load_yaml_config(path: Path) -> ExperimentConfig:
    """Load a YAML file into an :class:`ExperimentConfig`."""

    import yaml  # Local import to keep optional dependency cheap.

    raw: Dict[str, Any] = yaml.safe_load(Path(path).read_text())
    data_cfg = DataConfig(**raw["data"])
    optim_cfg = OptimConfig(**raw.get("optimizer", {}))
    image_cfg = (
        ImageModelConfig(**raw["image_model"]) if raw.get("image_model") else None
    )
    kspace_cfg = (
        KspaceModelConfig(**raw["kspace_model"]) if raw.get("kspace_model") else None
    )
    extras = raw.get("extras", {})
    return ExperimentConfig(
        data=data_cfg,
        optimizer=optim_cfg,
        image_model=image_cfg,
        kspace_model=kspace_cfg,
        extras=extras,
    )

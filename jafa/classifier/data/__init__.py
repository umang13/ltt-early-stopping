from .cifar10 import kspaceCIFAR10

__all__ = [
    "kspaceCIFAR10",
    "ImageNetDataset",
    "kspaceImageNet",
    "HierarchicalImbalancedImageNet",
    "HierarchicalImbalancedKspaceImageNet",
    "KneeDataset",
    "KneeVolumeDataset",
    "MRDataModule",
    "MRDataset",
    "MRVolumeDataset",
]


def __getattr__(name):
    if name in {
        "ImageNetDataset",
        "kspaceImageNet",
        "HierarchicalImbalancedImageNet",
        "HierarchicalImbalancedKspaceImageNet",
    }:
        from .imagenet import (
            HierarchicalImbalancedImageNet,
            HierarchicalImbalancedKspaceImageNet,
            ImageNetDataset,
            kspaceImageNet,
        )

        return {
            "ImageNetDataset": ImageNetDataset,
            "kspaceImageNet": kspaceImageNet,
            "HierarchicalImbalancedImageNet": HierarchicalImbalancedImageNet,
            "HierarchicalImbalancedKspaceImageNet": HierarchicalImbalancedKspaceImageNet,
        }[name]
    if name == "KneeDataset":
        from .knee_data import KneeDataset

        return KneeDataset
    if name == "KneeVolumeDataset":
        from .knee_vol_data import KneeVolumeDataset

        return KneeVolumeDataset
    if name == "MRDataModule":
        from .mr_datamodule import MRDataModule

        return MRDataModule
    if name == "MRDataset":
        from .mr_dataset import MRDataset

        return MRDataset
    if name == "MRVolumeDataset":
        from .mr_vol_dataset import MRVolumeDataset

        return MRVolumeDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

from classifier.masks.uniform import UniformRandomMask
from classifier.masks.vds import VariableDensityMask
from classifier.masks.equispaced import EquiSpacedMask
from classifier.masks.random_vds import RandomVariableDensityMask
from classifier.masks.random_volumetric_vds import RandomVolumetricVariableDensityMask
from classifier.masks.volumetric_vds import VolumetricVariableDensityMask
from classifier.masks.random_rate_mask import (
    RandomRateUniformMask,
)
import torch

def build_mask_fn(config, training=True):
    """
    Build mask function based on input_type and mask_type.
    """
    device = getattr(config, "device", None)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_type = getattr(config, "input_type", "kspace")
    mask_type = getattr(config, "mask_type", "none")

    if input_type == "image":
        from classifier.masks.patch_mask import PatchMask
        from classifier.masks.random_rate_mask import (
            RandomRatePatchMask,
            BiasedRandomRatePatchMask,
        )

        if mask_type == "none":
            class IdentityMask:
                def __call__(self, x, batch_size=None):
                    return x

            return IdentityMask()

        if mask_type == "patch":
            patch_size = getattr(config, "patch_size", 4)
            mask_ratio = getattr(config, "patch_mask_ratio", 0.5)
            return PatchMask(
                patch_size=patch_size,
                mask_ratio=mask_ratio,
                mask_value=0.0,
            )
        if mask_type == "random_rate_patch":
            patch_size = getattr(config, "patch_size", 4)
            min_rate = getattr(config, "min_sampling_rate", 0.1)
            max_rate = getattr(config, "max_sampling_rate", 0.9)
            return RandomRatePatchMask(
                patch_size=patch_size,
                min_rate=min_rate,
                max_rate=max_rate,
                mask_value=0.0,
            )
        if mask_type == "biased_random_rate_patch":
            patch_size = getattr(config, "patch_size", 4)
            min_rate = getattr(config, "min_sampling_rate", 0.1)
            max_rate = getattr(config, "max_sampling_rate", 0.9)
            bias_threshold = getattr(config, "bias_threshold", 0.15)
            low_weight = getattr(config, "bias_low_rate_weight", 0.7)
            return BiasedRandomRatePatchMask(
                patch_size=patch_size,
                min_sampling_rate=min_rate,
                max_sampling_rate=max_rate,
                bias_threshold=bias_threshold,
                low_rate_weight=low_weight,
                mask_value=0.0,
            )

        raise NotImplementedError(f"Image mask type {mask_type} not implemented")



    mask_configs = dict(
        target_shape=tuple(config.kspace_shape),
        k_fraction=config.k_fraction,
        sampled_indices=tuple(config.sampled_indices),
        center_fraction=getattr(config, "center_fraction", config.k_fraction / 2),
        in_channels=getattr(config, "in_channels", 1),
        batch_size=config.batch_size if training else config.val_batch_size,
        device=device,
        is_complex=getattr(config, "is_complex", True),
    )

    if mask_type == "uniform":
        return UniformRandomMask(**mask_configs)
    if mask_type == "vds":
        return VariableDensityMask(**mask_configs)
    if mask_type == "equi":
        return EquiSpacedMask(**mask_configs)
    if mask_type == "equi_no_center":
        return EquiSpacedMask(**mask_configs)
    if mask_type == "vvds":
        return VolumetricVariableDensityMask(**mask_configs)
    if mask_type == "random_vds":
        min_k_fraction = getattr(config, "min_k_fraction", getattr(config, "min_sampling_rate", None))
        max_k_fraction = getattr(config, "max_k_fraction", getattr(config, "max_sampling_rate", None))
        return RandomVariableDensityMask(
            **mask_configs,
            min_k_fraction=min_k_fraction,
            max_k_fraction=max_k_fraction,
        )
    if mask_type == "random_vvds":
        min_k_fraction = getattr(config, "min_k_fraction", getattr(config, "min_sampling_rate", None))
        max_k_fraction = getattr(config, "max_k_fraction", getattr(config, "max_sampling_rate", None))
        return RandomVolumetricVariableDensityMask(
            **mask_configs,
            min_k_fraction=min_k_fraction,
            max_k_fraction=max_k_fraction,
        )
    if mask_type == "random_rate_uniform":
        min_rate = getattr(config, "min_sampling_rate", 0.1)
        max_rate = getattr(config, "max_sampling_rate", 1.0)
        return RandomRateUniformMask(
            target_shape=mask_configs["target_shape"],
            min_rate=min_rate,
            max_rate=max_rate,
            center_fraction=mask_configs["center_fraction"],
            batch_size=mask_configs["batch_size"],
            sampled_indices=mask_configs["sampled_indices"],
            in_channels=mask_configs["in_channels"],
            is_complex=mask_configs["is_complex"],
            device=mask_configs["device"],
        )
    raise NotImplementedError(f"K-space mask type {mask_type} not implemented")

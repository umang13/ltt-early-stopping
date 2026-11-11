"""Render ADC/b1500 from directional prefix means."""

from __future__ import annotations

from typing import Dict

import h5py
import numpy as np

from .types import AXES, AXIS_TO_INDEX, RecipeSpec


ADC_SCALE = -1e6
B_VALUES = [50, 1000]
CROP_SIZE = (100, 100)


def trace(img_dict: Dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    trace_b50 = np.cbrt(img_dict["b50x"] * img_dict["b50y"] * img_dict["b50z"])
    trace_b1000 = np.cbrt(
        img_dict["b1000x"] * img_dict["b1000y"] * img_dict["b1000z"]
    )
    return trace_b50, trace_b1000


def adc(
    raw_images: np.ndarray,
    adc_scale: float,
    b_values: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    if np.mean(raw_images) < 1e-3:
        raw_images = 1e5 * raw_images

    log_image = np.log(raw_images + 1.0)
    sum_log_image = np.mean(log_image, axis=2)

    X = np.column_stack((b_values, np.ones(2)))
    Y = sum_log_image.reshape(-1, len(b_values)).T

    res = np.linalg.lstsq(X, Y, rcond=None)[0]
    tmp = res[0, :].reshape(sum_log_image.shape[:2])
    b0_img = np.exp(res[1, :].reshape(sum_log_image.shape[:2]))
    b0_img[np.isnan(b0_img)] = 0

    adc_map = tmp * adc_scale
    adc_map[(adc_map < 0) | (np.isnan(adc_map))] = 0
    return adc_map, b0_img


def b1500(
    adc_map: np.ndarray,
    b0_img: np.ndarray,
    adc_scale: float,
    b_values: list[int],
) -> np.ndarray:
    noise_level = 12
    noise_threshold_max_adc = 300
    calculated_b_value = 1500
    noise_threshold_min_b0 = noise_level

    minimal_pixel_fraction = 0.01
    b0_intensity = b0_img[
        (adc_map < noise_threshold_max_adc) & (b0_img > noise_threshold_min_b0)
    ]
    if len(b0_intensity) > ((minimal_pixel_fraction * adc_map.size) + 1):
        noise_level = np.percentile(b0_intensity, 50) * 3

    noise_estimation_adc_offset = 1000
    adc_offset = np.where(
        (noise_level > 0) & (b0_img < noise_level),
        noise_estimation_adc_offset
        * np.sqrt(np.maximum(1 - ((b0_img / noise_level) ** 2), 0)),
        0,
    )

    neg_calc_b_value = calculated_b_value / adc_scale
    neg_max_b_value = b_values[-1] / adc_scale
    tmp_exponent = (
        (neg_calc_b_value - neg_max_b_value) * np.maximum(adc_map, adc_offset)
        + neg_max_b_value * adc_map
    )
    return b0_img * np.exp(tmp_exponent)


def _center_crop_im(volume: np.ndarray, crop_size: tuple[int, int]) -> np.ndarray:
    crop_h, crop_w = crop_size
    _, height, width = volume.shape
    start_h = (height - crop_h) // 2
    start_w = (width - crop_w) // 2
    return volume[:, start_h : start_h + crop_h, start_w : start_w + crop_w]


def _postprocess_volume(volume: np.ndarray) -> np.ndarray:
    processed = np.flip(volume, axis=0).copy()
    processed = _center_crop_im(processed, CROP_SIZE)
    return processed.astype(np.float32)


def _read_directional_prefixes(
    handle: h5py.File,
    combine: str,
    spec: RecipeSpec,
) -> Dict[str, np.ndarray]:
    images: Dict[str, np.ndarray] = {}
    for axis in spec.active_directions:
        axis_index = AXIS_TO_INDEX[axis]
        b50_count = spec.counts_b50[axis_index]
        b1000_count = spec.counts_b1000[axis_index]
        images[f"b50{axis}"] = handle[f"bank/{combine}/b50/{axis}"][b50_count - 1]
        images[f"b1000{axis}"] = handle[f"bank/{combine}/b1000/{axis}"][b1000_count - 1]
    return images


def _compute_adc_b1500_from_images(
    images: Dict[str, np.ndarray],
    spec: RecipeSpec,
) -> tuple[np.ndarray, np.ndarray]:
    active_directions = spec.active_directions
    recon_shape = images[f"b50{active_directions[0]}"].shape
    adc_inputs = np.zeros(recon_shape + (len(active_directions), 2), dtype=np.float32)
    for direction_index, axis in enumerate(active_directions):
        adc_inputs[:, :, :, direction_index, 0] = images[f"b50{axis}"]
        adc_inputs[:, :, :, direction_index, 1] = images[f"b1000{axis}"]
    adc_slices, b0_slices = zip(
        *[
            adc(adc_inputs[slice_index, ...], ADC_SCALE, B_VALUES)
            for slice_index in range(adc_inputs.shape[0])
        ]
    )
    adc_map = np.asarray(adc_slices, dtype=np.float32)
    b0_img = np.asarray(b0_slices, dtype=np.float32)
    return adc_map, b1500(adc_map, b0_img, ADC_SCALE, B_VALUES).astype(np.float32)


def render_recipe_metrics(
    handle: h5py.File,
    combine: str,
    spec: RecipeSpec,
    *,
    include_trace: bool = False,
) -> dict[str, np.ndarray]:
    images = _read_directional_prefixes(handle, combine, spec)
    adc_map, b1500_map = _compute_adc_b1500_from_images(images, spec)
    rendered = {
        "adc_map": _postprocess_volume(adc_map),
        "b1500": _postprocess_volume(b1500_map),
    }
    if include_trace and spec.active_directions == AXES:
        trace_b50, trace_b1000 = trace(images)
        rendered["trace_b50"] = _postprocess_volume(trace_b50)
        rendered["trace_b1000"] = _postprocess_volume(trace_b1000)
    return rendered

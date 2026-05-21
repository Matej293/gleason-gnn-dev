from __future__ import annotations

import numpy as np
from skimage.segmentation import felzenszwalb

from .superpixels import generate_slic_superpixels


def _felzenszwalb_rgb(image_rgb: np.ndarray, scale: float, sigma: float, min_size: int) -> np.ndarray:
    """Compatibility shim across scikit-image versions."""
    try:
        return felzenszwalb(
            image_rgb,
            scale=float(scale),
            sigma=float(sigma),
            min_size=int(min_size),
            channel_axis=-1,
        )
    except TypeError:
        # Older scikit-image used ``multichannel`` instead of ``channel_axis``.
        return felzenszwalb(
            image_rgb,
            scale=float(scale),
            sigma=float(sigma),
            min_size=int(min_size),
            multichannel=True,
        )


def generate_felzenszwalb_superpixels(
    image_rgb: np.ndarray,
    tissue_mask: np.ndarray,
    scale: float = 100.0,
    sigma: float = 0.8,
    min_size: int = 20,
    start_label: int = 0,
) -> np.ndarray:
    """
    Generate Felzenszwalb superpixels constrained to tissue regions.

    Non-tissue pixels are assigned -1 so downstream code can ignore them.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected image_rgb shape [H, W, 3].")
    if tissue_mask.shape != image_rgb.shape[:2]:
        raise ValueError("tissue_mask must match image spatial shape.")

    segments = _felzenszwalb_rgb(
        image_rgb=image_rgb,
        scale=float(scale),
        sigma=float(sigma),
        min_size=int(min_size),
    ).astype(np.int32, copy=False)

    mask = tissue_mask.astype(bool)
    out = np.full(segments.shape, -1, dtype=np.int32)
    valid = mask & (segments >= 0)
    if not np.any(valid):
        return out

    # Relabel valid regions to compact IDs so graph node IDs stay dense.
    _, inv = np.unique(segments[valid].astype(np.int64, copy=False), return_inverse=True)
    out[valid] = inv.astype(np.int32, copy=False) + int(start_label)
    return out


def generate_superpixels(
    image_rgb: np.ndarray,
    tissue_mask: np.ndarray,
    method: str = "slic",
    num_segments: int = 300,
    compactness: float = 10.0,
    slic_sigma: float = 1.0,
    felzenszwalb_scale: float = 100.0,
    felzenszwalb_sigma: float = 0.8,
    felzenszwalb_min_size: int = 20,
    start_label: int = 0,
) -> np.ndarray:
    mode = str(method).strip().lower()
    if mode == "slic":
        return generate_slic_superpixels(
            image_rgb=image_rgb,
            tissue_mask=tissue_mask,
            num_segments=int(num_segments),
            compactness=float(compactness),
            sigma=float(slic_sigma),
            start_label=int(start_label),
        )
    if mode == "felzenszwalb":
        return generate_felzenszwalb_superpixels(
            image_rgb=image_rgb,
            tissue_mask=tissue_mask,
            scale=float(felzenszwalb_scale),
            sigma=float(felzenszwalb_sigma),
            min_size=int(felzenszwalb_min_size),
            start_label=int(start_label),
        )
    raise ValueError(f"Unsupported superpixel method: {method}")

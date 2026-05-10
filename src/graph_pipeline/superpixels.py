from __future__ import annotations

import numpy as np
from skimage.segmentation import slic


def generate_slic_superpixels(
    image_rgb: np.ndarray,
    tissue_mask: np.ndarray,
    num_segments: int = 300,
    compactness: float = 10.0,
    sigma: float = 1.0,
    start_label: int = 0,
) -> np.ndarray:
    """
    Generate SLIC superpixels constrained to tissue regions.

    Non-tissue pixels are assigned -1 so downstream code can ignore them.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected image_rgb shape [H, W, 3].")
    if tissue_mask.shape != image_rgb.shape[:2]:
        raise ValueError("tissue_mask must match image spatial shape.")

    mask = tissue_mask.astype(bool)
    segments = slic(
        image_rgb,
        n_segments=int(num_segments),
        compactness=float(compactness),
        sigma=float(sigma),
        start_label=int(start_label),
        mask=mask,
        channel_axis=-1,
    )
    segments = segments.astype(np.int32, copy=False)
    segments[~mask] = -1
    return segments


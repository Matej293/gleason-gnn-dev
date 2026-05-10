from __future__ import annotations

import numpy as np


def _safe_entropy(prob_mean: np.ndarray, eps: float = 1e-8) -> float:
    p = np.clip(prob_mean.astype(np.float64, copy=False), eps, 1.0)
    return float(-np.sum(p * np.log(p)))


def compute_node_features(
    image_rgb: np.ndarray,
    superpixels: np.ndarray,
    seg_probs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-node features.

    Returns:
        node_ids: [N]
        features: [N, F]
    Features:
        mean_rgb(3), std_rgb(3), area(1), centroid_xy(2), mean_seg_probs(4), entropy(1)
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected image_rgb [H, W, 3].")
    if superpixels.shape != image_rgb.shape[:2]:
        raise ValueError("superpixels must match image spatial shape.")
    if seg_probs.ndim != 3 or seg_probs.shape[0] != 4:
        raise ValueError("seg_probs must be [4, H, W].")
    if tuple(seg_probs.shape[1:]) != image_rgb.shape[:2]:
        raise ValueError("seg_probs spatial shape must match image.")

    node_ids = np.unique(superpixels)
    node_ids = node_ids[node_ids >= 0]

    image_f = image_rgb.astype(np.float32)
    features: list[np.ndarray] = []
    for node_id in node_ids.tolist():
        m = superpixels == node_id
        ys, xs = np.where(m)
        if ys.size == 0:
            continue

        pix = image_f[m]
        mean_rgb = pix.mean(axis=0)
        std_rgb = pix.std(axis=0)
        area = float(m.sum())
        centroid_x = float(xs.mean())
        centroid_y = float(ys.mean())
        probs = seg_probs[:, m]
        mean_probs = probs.mean(axis=1)
        entropy = _safe_entropy(mean_probs)

        feat = np.concatenate(
            [
                mean_rgb,
                std_rgb,
                np.asarray([area, centroid_x, centroid_y], dtype=np.float32),
                mean_probs.astype(np.float32),
                np.asarray([entropy], dtype=np.float32),
            ],
            axis=0,
        ).astype(np.float32)
        features.append(feat)

    if not features:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 14), dtype=np.float32)
    return node_ids.astype(np.int64), np.vstack(features).astype(np.float32)


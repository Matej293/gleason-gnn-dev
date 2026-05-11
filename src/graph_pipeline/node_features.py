from __future__ import annotations

import numpy as np


def _safe_entropy(prob_mean: np.ndarray, eps: float = 1e-8) -> float:
    p = np.clip(prob_mean.astype(np.float64, copy=False), eps, 1.0)
    return float(-np.sum(p * np.log(p)))


def _boundary_touch_ratio(mask: np.ndarray) -> float:
    h, w = mask.shape
    boundary = np.zeros_like(mask, dtype=bool)
    boundary[0, :] = True
    boundary[h - 1, :] = True
    boundary[:, 0] = True
    boundary[:, w - 1] = True
    total = float(mask.sum())
    if total <= 0.0:
        return 0.0
    return float((mask & boundary).sum()) / total


def _adjacent_nodes(superpixels: np.ndarray) -> dict[int, set[int]]:
    node_ids = np.unique(superpixels)
    node_ids = node_ids[node_ids >= 0]
    adj: dict[int, set[int]] = {int(n): set() for n in node_ids.tolist()}
    h, w = superpixels.shape

    def _add(a: int, b: int) -> None:
        if a < 0 or b < 0 or a == b:
            return
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    for y in range(h):
        for x in range(w):
            src = int(superpixels[y, x])
            if x + 1 < w:
                _add(src, int(superpixels[y, x + 1]))
            if y + 1 < h:
                _add(src, int(superpixels[y + 1, x]))
    return adj


def compute_node_features(image_rgb: np.ndarray, superpixels: np.ndarray, seg_probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    prob_mean_by_node: dict[int, np.ndarray] = {}
    masks_by_node: dict[int, np.ndarray] = {}
    for node_id in node_ids.tolist():
        m = superpixels == node_id
        if m.any():
            prob_mean_by_node[int(node_id)] = seg_probs[:, m].mean(axis=1).astype(np.float32)
            masks_by_node[int(node_id)] = m

    adjacency = _adjacent_nodes(superpixels)
    features: list[np.ndarray] = []
    for node_id in node_ids.tolist():
        m = masks_by_node.get(int(node_id))
        if m is None:
            continue
        ys, xs = np.where(m)
        pix = image_f[m]
        mean_rgb = pix.mean(axis=0)
        std_rgb = pix.std(axis=0)
        area = float(m.sum())
        centroid_x = float(xs.mean())
        centroid_y = float(ys.mean())

        probs = seg_probs[:, m]
        mean_probs = probs.mean(axis=1)
        std_probs = probs.std(axis=1)
        top2 = np.sort(mean_probs)[-2:]
        margin = float(top2[-1] - top2[-2])
        entropy = _safe_entropy(mean_probs)

        neighbors = np.asarray(sorted(adjacency.get(int(node_id), set())), dtype=np.int64)
        if neighbors.size > 0:
            nmean = np.stack([prob_mean_by_node.get(int(n), mean_probs) for n in neighbors.tolist()], axis=0).mean(axis=0)
            contrast = float(np.mean(np.abs(mean_probs - nmean)))
        else:
            contrast = 0.0

        perimeter_proxy = float(np.count_nonzero(np.logical_xor(m, np.pad(m, 1, mode="edge")[1:-1, :-2])))
        compactness = float((4.0 * np.pi * area) / max(perimeter_proxy * perimeter_proxy, 1.0))
        boundary_touch = _boundary_touch_ratio(m)

        feat = np.concatenate([
            mean_rgb,
            std_rgb,
            np.asarray([area, centroid_x, centroid_y], dtype=np.float32),
            mean_probs.astype(np.float32),
            np.asarray([entropy], dtype=np.float32),
            std_probs.astype(np.float32),
            np.asarray([margin, contrast, compactness, boundary_touch], dtype=np.float32),
        ], axis=0).astype(np.float32)
        features.append(feat)

    if not features:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 22), dtype=np.float32)
    return node_ids.astype(np.int64), np.vstack(features).astype(np.float32)

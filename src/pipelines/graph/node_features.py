from __future__ import annotations

import numpy as np


def _safe_entropy_rows(prob_means: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    p = np.clip(prob_means.astype(np.float64, copy=False), eps, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def _group_mean_std(inv: np.ndarray, values: np.ndarray, n_groups: int, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vals = values.astype(np.float64, copy=False)
    sums = np.bincount(inv, weights=vals, minlength=n_groups)
    sq_sums = np.bincount(inv, weights=(vals * vals), minlength=n_groups)
    means = sums / counts
    vars_ = np.maximum((sq_sums / counts) - (means * means), 0.0)
    stds = np.sqrt(vars_)
    return means, stds


def _touch_undirected_pairs(superpixels: np.ndarray) -> np.ndarray:
    left = superpixels[:, :-1].reshape(-1)
    right = superpixels[:, 1:].reshape(-1)
    up = superpixels[:-1, :].reshape(-1)
    down = superpixels[1:, :].reshape(-1)

    a = np.concatenate([left, up], axis=0).astype(np.int64, copy=False)
    b = np.concatenate([right, down], axis=0).astype(np.int64, copy=False)
    valid = (a >= 0) & (b >= 0) & (a != b)
    if not np.any(valid):
        return np.zeros((0, 2), dtype=np.int64)

    av = a[valid]
    bv = b[valid]
    lo = np.minimum(av, bv)
    hi = np.maximum(av, bv)
    pairs = np.stack([lo, hi], axis=1)
    return np.unique(pairs, axis=0)


def _adjacency_from_pairs(node_ids: np.ndarray, pairs: np.ndarray) -> list[np.ndarray]:
    n = int(node_ids.shape[0])
    if n == 0:
        return []
    if pairs.size == 0:
        return [np.zeros((0,), dtype=np.int64) for _ in range(n)]

    src = np.searchsorted(node_ids, pairs[:, 0])
    dst = np.searchsorted(node_ids, pairs[:, 1])
    src_d = np.concatenate([src, dst], axis=0)
    dst_d = np.concatenate([dst, src], axis=0)

    order = np.argsort(src_d, kind="mergesort")
    src_s = src_d[order]
    dst_s = dst_d[order]

    out: list[np.ndarray] = []
    for idx in range(n):
        lo = int(np.searchsorted(src_s, idx, side="left"))
        hi = int(np.searchsorted(src_s, idx, side="right"))
        if hi <= lo:
            out.append(np.zeros((0,), dtype=np.int64))
        else:
            out.append(np.unique(dst_s[lo:hi]).astype(np.int64, copy=False))
    return out


def _horizontal_transition_counts(superpixels: np.ndarray, node_ids: np.ndarray) -> np.ndarray:
    """Match the legacy horizontal XOR perimeter proxy, but compute for all nodes at once."""
    left = superpixels[:, :-1].reshape(-1)
    right = superpixels[:, 1:].reshape(-1)
    diff = left != right
    if not np.any(diff):
        return np.zeros((node_ids.shape[0],), dtype=np.float64)

    a = left[diff]
    b = right[diff]
    out = np.zeros((node_ids.shape[0],), dtype=np.float64)

    a_valid = a >= 0
    if np.any(a_valid):
        a_idx = np.searchsorted(node_ids, a[a_valid])
        out += np.bincount(a_idx, minlength=node_ids.shape[0]).astype(np.float64)

    b_valid = b >= 0
    if np.any(b_valid):
        b_idx = np.searchsorted(node_ids, b[b_valid])
        out += np.bincount(b_idx, minlength=node_ids.shape[0]).astype(np.float64)

    return out


def compute_node_features(image_rgb: np.ndarray, superpixels: np.ndarray, seg_probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected image_rgb [H, W, 3].")
    if superpixels.shape != image_rgb.shape[:2]:
        raise ValueError("superpixels must match image spatial shape.")
    if seg_probs.ndim != 3 or seg_probs.shape[0] != 4:
        raise ValueError("seg_probs must be [4, H, W].")
    if tuple(seg_probs.shape[1:]) != image_rgb.shape[:2]:
        raise ValueError("seg_probs spatial shape must match image.")

    valid = superpixels >= 0
    if not np.any(valid):
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 22), dtype=np.float32)

    h, w = superpixels.shape
    labels = superpixels[valid].astype(np.int64, copy=False)
    node_ids, inv = np.unique(labels, return_inverse=True)
    n_nodes = int(node_ids.shape[0])

    ys, xs = np.nonzero(valid)
    counts = np.bincount(inv, minlength=n_nodes).astype(np.float64)

    image_f = image_rgb.astype(np.float32, copy=False)
    mean_rgb_cols: list[np.ndarray] = []
    std_rgb_cols: list[np.ndarray] = []
    for c in range(3):
        mean_c, std_c = _group_mean_std(inv, image_f[..., c][valid], n_nodes, counts)
        mean_rgb_cols.append(mean_c)
        std_rgb_cols.append(std_c)
    mean_rgb = np.stack(mean_rgb_cols, axis=1)
    std_rgb = np.stack(std_rgb_cols, axis=1)

    mean_probs_cols: list[np.ndarray] = []
    std_probs_cols: list[np.ndarray] = []
    seg_probs_f = seg_probs.astype(np.float32, copy=False)
    for c in range(4):
        mean_c, std_c = _group_mean_std(inv, seg_probs_f[c][valid], n_nodes, counts)
        mean_probs_cols.append(mean_c)
        std_probs_cols.append(std_c)
    mean_probs = np.stack(mean_probs_cols, axis=1)
    std_probs = np.stack(std_probs_cols, axis=1)

    area = counts
    centroid_x = np.bincount(inv, weights=xs.astype(np.float64), minlength=n_nodes) / counts
    centroid_y = np.bincount(inv, weights=ys.astype(np.float64), minlength=n_nodes) / counts

    sorted_probs = np.sort(mean_probs, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    entropy = _safe_entropy_rows(mean_probs)

    pairs = _touch_undirected_pairs(superpixels)
    neighbors_by_idx = _adjacency_from_pairs(node_ids, pairs)
    contrast = np.zeros((n_nodes,), dtype=np.float64)
    for idx in range(n_nodes):
        neighbors = neighbors_by_idx[idx]
        if neighbors.size == 0:
            continue
        neighbor_mean = mean_probs[neighbors].mean(axis=0)
        contrast[idx] = float(np.mean(np.abs(mean_probs[idx] - neighbor_mean)))

    perim = _horizontal_transition_counts(superpixels, node_ids)
    compactness = (4.0 * np.pi * area) / np.maximum(perim * perim, 1.0)

    on_boundary = (ys == 0) | (ys == (h - 1)) | (xs == 0) | (xs == (w - 1))
    boundary_counts = np.bincount(inv[on_boundary], minlength=n_nodes).astype(np.float64)
    boundary_touch = boundary_counts / counts

    feats = np.concatenate(
        [
            mean_rgb,
            std_rgb,
            np.stack([area, centroid_x, centroid_y], axis=1),
            mean_probs,
            entropy[:, None],
            std_probs,
            np.stack([margin, contrast, compactness, boundary_touch], axis=1),
        ],
        axis=1,
    ).astype(np.float32, copy=False)

    return node_ids.astype(np.int64, copy=False), feats

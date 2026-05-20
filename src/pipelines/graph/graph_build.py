from __future__ import annotations

import numpy as np


def _unique_undirected_pairs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return unique undirected pairs [u,v] with u<v and labels >=0."""
    if a.size == 0 or b.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    a_i = a.astype(np.int64, copy=False).reshape(-1)
    b_i = b.astype(np.int64, copy=False).reshape(-1)
    valid = (a_i >= 0) & (b_i >= 0) & (a_i != b_i)
    if not np.any(valid):
        return np.zeros((0, 2), dtype=np.int64)
    av = a_i[valid]
    bv = b_i[valid]
    lo = np.minimum(av, bv)
    hi = np.maximum(av, bv)
    pairs = np.stack([lo, hi], axis=1)
    return np.unique(pairs, axis=0)


def _to_bidirectional_edge_index(undirected_pairs: np.ndarray) -> np.ndarray:
    if undirected_pairs.size == 0:
        return np.zeros((2, 0), dtype=np.int64)
    rev = undirected_pairs[:, ::-1]
    doubled = np.concatenate([undirected_pairs, rev], axis=0)
    return doubled.T.astype(np.int64, copy=False)


def _node_ids_and_centroids(superpixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = superpixels >= 0
    if not np.any(valid):
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 2), dtype=np.float64)

    ys, xs = np.nonzero(valid)
    labels = superpixels[valid].astype(np.int64, copy=False)
    node_ids, inv = np.unique(labels, return_inverse=True)
    counts = np.bincount(inv, minlength=node_ids.shape[0]).astype(np.float64)

    sum_x = np.bincount(inv, weights=xs.astype(np.float64), minlength=node_ids.shape[0])
    sum_y = np.bincount(inv, weights=ys.astype(np.float64), minlength=node_ids.shape[0])
    centroids = np.stack([sum_x / counts, sum_y / counts], axis=1)
    return node_ids.astype(np.int64, copy=False), centroids.astype(np.float64, copy=False)


def build_touch_adjacency_edges(superpixels: np.ndarray) -> np.ndarray:
    """
    Build undirected edges between touching superpixels.

    Returns edge_index with shape [2, E] in COO format.
    """
    if superpixels.ndim != 2:
        raise ValueError("superpixels must be [H, W].")

    horiz_pairs = _unique_undirected_pairs(superpixels[:, :-1], superpixels[:, 1:])
    vert_pairs = _unique_undirected_pairs(superpixels[:-1, :], superpixels[1:, :])
    if horiz_pairs.size == 0 and vert_pairs.size == 0:
        return np.zeros((2, 0), dtype=np.int64)

    merged = np.concatenate([horiz_pairs, vert_pairs], axis=0)
    undirected = np.unique(merged, axis=0)
    return _to_bidirectional_edge_index(undirected)


def build_knn_centroid_edges(
    superpixels: np.ndarray,
    k: int = 2,
    max_distance: float | None = None,
) -> np.ndarray:
    if superpixels.ndim != 2:
        raise ValueError("superpixels must be [H, W].")

    k = max(int(k), 0)
    if k <= 0:
        return np.zeros((2, 0), dtype=np.int64)

    node_ids, centroids = _node_ids_and_centroids(superpixels)
    n = int(node_ids.shape[0])
    if n <= 1:
        return np.zeros((2, 0), dtype=np.int64)

    # Pairwise centroid distance matrix.
    deltas = centroids[:, None, :] - centroids[None, :, :]
    dists = np.sqrt(np.sum(deltas * deltas, axis=2, dtype=np.float64), dtype=np.float64)
    np.fill_diagonal(dists, np.inf)

    if max_distance is not None:
        max_d = float(max_distance)
        if max_d < 0.0:
            raise ValueError("max_distance must be >= 0 or None.")
    else:
        max_d = None

    edges: set[tuple[int, int]] = set()
    for src_idx in range(n):
        row = dists[src_idx]
        if max_d is None:
            candidates = np.arange(n, dtype=np.int64)
            candidates = candidates[candidates != src_idx]
        else:
            candidates = np.where(row <= max_d)[0].astype(np.int64, copy=False)
        if candidates.size == 0:
            continue

        # Deterministic tie-break by centroid distance, then node id.
        order = np.lexsort((node_ids[candidates], row[candidates]))
        selected = candidates[order[:k]]
        src_id = int(node_ids[src_idx])
        for dst_idx in selected.tolist():
            dst_id = int(node_ids[dst_idx])
            a, b = (src_id, dst_id) if src_id < dst_id else (dst_id, src_id)
            if a != b:
                edges.add((a, b))

    if not edges:
        return np.zeros((2, 0), dtype=np.int64)

    undirected = np.asarray(sorted(edges), dtype=np.int64)
    return _to_bidirectional_edge_index(undirected)


def build_edges(
    superpixels: np.ndarray,
    policy: str = "touch",
    knn_k: int = 2,
    knn_max_distance: float | None = None,
) -> np.ndarray:
    mode = str(policy).strip().lower()
    touch = build_touch_adjacency_edges(superpixels)
    if mode == "touch":
        return touch

    knn = build_knn_centroid_edges(superpixels, k=knn_k, max_distance=knn_max_distance)
    if mode == "knn":
        return knn

    if mode == "touch_plus_knn":
        if touch.shape[1] == 0:
            return knn
        if knn.shape[1] == 0:
            return touch
        merged = np.concatenate([touch.T, knn.T], axis=0)
        uniq = np.unique(merged, axis=0)
        return uniq.astype(np.int64).T

    raise ValueError(f"Unsupported edge policy: {policy}")

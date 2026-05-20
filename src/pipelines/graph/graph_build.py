from __future__ import annotations

import numpy as np


def build_touch_adjacency_edges(superpixels: np.ndarray) -> np.ndarray:
    """
    Build undirected edges between touching superpixels.

    Returns edge_index with shape [2, E] in COO format.
    """
    if superpixels.ndim != 2:
        raise ValueError("superpixels must be [H, W].")

    edges: set[tuple[int, int]] = set()
    h, w = superpixels.shape

    def _add(a: int, b: int) -> None:
        if a < 0 or b < 0 or a == b:
            return
        x, y = (a, b) if a < b else (b, a)
        edges.add((x, y))

    for y in range(h):
        for x in range(w):
            src = int(superpixels[y, x])
            if x + 1 < w:
                _add(src, int(superpixels[y, x + 1]))
            if y + 1 < h:
                _add(src, int(superpixels[y + 1, x]))

    if not edges:
        return np.zeros((2, 0), dtype=np.int64)

    undirected = sorted(edges)
    doubled: list[tuple[int, int]] = []
    for a, b in undirected:
        doubled.append((a, b))
        doubled.append((b, a))
    edge_index = np.asarray(doubled, dtype=np.int64).T
    return edge_index


def build_knn_centroid_edges(
    superpixels: np.ndarray,
    k: int = 2,
    max_distance: float | None = None,
) -> np.ndarray:
    if superpixels.ndim != 2:
        raise ValueError("superpixels must be [H, W].")
    node_ids = np.unique(superpixels)
    node_ids = node_ids[node_ids >= 0]
    if node_ids.size <= 1:
        return np.zeros((2, 0), dtype=np.int64)

    centroids: dict[int, tuple[float, float]] = {}
    for node_id in node_ids.tolist():
        ys, xs = np.where(superpixels == node_id)
        if ys.size == 0:
            continue
        centroids[int(node_id)] = (float(xs.mean()), float(ys.mean()))
    ids = sorted(centroids.keys())
    if len(ids) <= 1:
        return np.zeros((2, 0), dtype=np.int64)

    edges: set[tuple[int, int]] = set()
    for src in ids:
        sx, sy = centroids[src]
        dists = []
        for dst in ids:
            if dst == src:
                continue
            dx, dy = centroids[dst]
            d = float(((sx - dx) ** 2 + (sy - dy) ** 2) ** 0.5)
            if max_distance is None or d <= float(max_distance):
                dists.append((d, dst))
        dists.sort(key=lambda x: (x[0], x[1]))
        for _, dst in dists[: max(int(k), 0)]:
            a, b = (src, dst) if src < dst else (dst, src)
            if a != b:
                edges.add((a, b))

    if not edges:
        return np.zeros((2, 0), dtype=np.int64)
    doubled = []
    for a, b in sorted(edges):
        doubled.append((a, b))
        doubled.append((b, a))
    return np.asarray(doubled, dtype=np.int64).T


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

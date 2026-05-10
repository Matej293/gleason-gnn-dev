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


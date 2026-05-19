from __future__ import annotations

import numpy as np


def extract_superpixel_boundaries(superpixels: np.ndarray) -> np.ndarray:
    if superpixels.ndim != 2:
        raise ValueError("superpixels must be [H, W].")
    h, w = superpixels.shape
    boundaries = np.zeros((h, w), dtype=bool)
    diff_right = superpixels[:, :-1] != superpixels[:, 1:]
    boundaries[:, :-1] |= diff_right
    boundaries[:, 1:] |= diff_right
    diff_down = superpixels[:-1, :] != superpixels[1:, :]
    boundaries[:-1, :] |= diff_down
    boundaries[1:, :] |= diff_down
    return boundaries


def extract_node_centroids(
    node_ids: np.ndarray,
    superpixels: np.ndarray,
) -> dict[int, tuple[float, float]]:
    centroids: dict[int, tuple[float, float]] = {}
    for nid in node_ids.tolist():
        yy, xx = np.nonzero(superpixels == int(nid))
        if yy.size == 0:
            continue
        centroids[int(nid)] = (float(np.mean(xx)), float(np.mean(yy)))
    return centroids


def unique_undirected_edges(edge_index: np.ndarray) -> list[tuple[int, int]]:
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must be [2,E].")
    edges: set[tuple[int, int]] = set()
    for a, b in edge_index.T.tolist():
        ai, bi = int(a), int(b)
        if ai == bi:
            continue
        edges.add((ai, bi) if ai < bi else (bi, ai))
    return sorted(edges)

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch

from src.pipelines.graph.graph_build import build_edges, build_touch_adjacency_edges
from src.pipelines.graph.node_features import compute_node_features
from src.pipelines.graph.node_labels import assign_majority_node_labels
from src.pipelines.graph.superpixel_methods import generate_superpixels


def _load_build_graphs_module():
    mod_path = Path(__file__).resolve().parents[1] / "src" / "cli" / "build_superpixel_graphs.py"
    spec = importlib.util.spec_from_file_location("build_superpixel_graphs", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed loading build_superpixel_graphs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_touch_adjacency_edges_bidirectional() -> None:
    sp = np.array([[0, 0, 1], [0, 2, 2]], dtype=np.int32)
    edge_index = build_touch_adjacency_edges(sp)
    edges = set((int(a), int(b)) for a, b in edge_index.T.tolist())
    assert (0, 1) in edges and (1, 0) in edges
    assert (0, 2) in edges and (2, 0) in edges
    assert (1, 2) in edges and (2, 1) in edges


def test_assign_majority_node_labels_with_ambiguity_mask() -> None:
    sp = np.array([[0, 0, 1], [0, 1, 1]], dtype=np.int32)
    gt = np.array([[1, 1, 2], [1, 2, 1]], dtype=np.int64)
    labels, train_mask = assign_majority_node_labels(
        superpixels=sp,
        hard_mask=gt,
        min_majority_fraction=0.7,
    )
    assert labels.tolist() == [1, 2]
    assert train_mask.tolist() == [True, False]


def test_compute_node_features_shape() -> None:
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    sp = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 3, 3],
            [2, 2, 3, 3],
        ],
        dtype=np.int32,
    )
    probs = np.zeros((4, 4, 4), dtype=np.float32)
    probs[0, :, :] = 1.0
    node_ids, feats = compute_node_features(img, sp, probs)
    assert node_ids.tolist() == [0, 1, 2, 3]
    assert feats.shape == (4, 22)


def test_compute_node_features_neighbor_contrast_uses_adjacent_nodes() -> None:
    img = np.zeros((3, 3, 3), dtype=np.uint8)
    sp = np.array(
        [
            [0, 0, 1],
            [0, 2, 1],
            [2, 2, 1],
        ],
        dtype=np.int32,
    )
    probs = np.zeros((4, 3, 3), dtype=np.float32)
    probs[:, :, :] = np.array([0.1, 0.8, 0.1, 0.0], dtype=np.float32)[:, None, None]
    probs[:, sp == 0] = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)[:, None]
    probs[:, sp == 1] = np.array([0.0, 0.9, 0.1, 0.0], dtype=np.float32)[:, None]
    probs[:, sp == 2] = np.array([0.0, 0.1, 0.9, 0.0], dtype=np.float32)[:, None]
    node_ids, feats = compute_node_features(img, sp, probs)
    id_to_row = {int(n): i for i, n in enumerate(node_ids.tolist())}
    contrast_idx = 19
    # Node 0 neighbors are {1,2}; contrast should be finite positive and bounded.
    assert feats[id_to_row[0], contrast_idx] > 0.0
    assert np.isfinite(feats[id_to_row[0], contrast_idx])


def test_build_edges_touch_plus_knn_superset_of_touch() -> None:
    sp = np.array([[0, 0, 1], [0, 2, 2]], dtype=np.int32)
    touch = build_edges(sp, policy="touch")
    hybrid = build_edges(sp, policy="touch_plus_knn", knn_k=1)
    touch_set = set((int(a), int(b)) for a, b in touch.T.tolist())
    hybrid_set = set((int(a), int(b)) for a, b in hybrid.T.tolist())
    assert touch_set.issubset(hybrid_set)


def test_superpixel_preset_resolution() -> None:
    mod = _load_build_graphs_module()
    args = argparse.Namespace(superpixel_preset="med", num_segments=111, compactness=2.0)
    num_segments, compactness = mod._resolve_superpixel_params(args)
    assert num_segments == 300
    assert compactness == 10.0


def test_extract_logits_supports_tensor_dict_and_sequence() -> None:
    mod = _load_build_graphs_module()
    t = torch.randn(2, 4, 8, 8)
    assert mod._extract_logits(t) is t
    assert mod._extract_logits({"out": t}) is t
    assert mod._extract_logits([t]) is t


def test_build_touch_adjacency_edges_ignores_background_label_minus_one() -> None:
    sp = np.array(
        [
            [-1, -1, 5],
            [-1, 9, 5],
        ],
        dtype=np.int32,
    )
    edge_index = build_touch_adjacency_edges(sp)
    edges = set((int(a), int(b)) for a, b in edge_index.T.tolist())
    assert edges == {(5, 9), (9, 5)}


def test_compute_node_features_supports_non_contiguous_superpixel_ids() -> None:
    img = np.zeros((2, 3, 3), dtype=np.uint8)
    sp = np.array([[2, 2, 7], [2, 7, 7]], dtype=np.int32)
    probs = np.zeros((4, 2, 3), dtype=np.float32)
    probs[1, :, :] = 1.0
    node_ids, feats = compute_node_features(img, sp, probs)
    assert node_ids.tolist() == [2, 7]
    assert feats.shape == (2, 22)


def test_generate_superpixels_felzenszwalb_respects_tissue_mask() -> None:
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[..., 0] = np.tile(np.arange(16, dtype=np.uint8), (16, 1))
    tissue = np.zeros((16, 16), dtype=np.uint8)
    tissue[3:13, 3:13] = 1

    sp = generate_superpixels(
        image_rgb=img,
        tissue_mask=tissue,
        method="felzenszwalb",
        felzenszwalb_scale=80.0,
        felzenszwalb_sigma=0.8,
        felzenszwalb_min_size=8,
    )

    assert sp.shape == tissue.shape
    assert np.all(sp[tissue == 0] == -1)
    assert np.any(sp[tissue == 1] >= 0)


def test_generate_superpixels_raises_for_unsupported_method() -> None:
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    tissue = np.ones((4, 4), dtype=np.uint8)

    try:
        _ = generate_superpixels(image_rgb=img, tissue_mask=tissue, method="unknown")
    except ValueError as exc:
        assert "Unsupported superpixel method" in str(exc)
        return

    raise AssertionError("Expected ValueError for unsupported superpixel method")

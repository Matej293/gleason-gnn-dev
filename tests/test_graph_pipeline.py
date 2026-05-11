from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch

from src.graph_pipeline.graph_build import build_touch_adjacency_edges
from src.graph_pipeline.node_features import compute_node_features
from src.graph_pipeline.node_labels import assign_majority_node_labels


def _load_build_graphs_module():
    mod_path = Path(__file__).resolve().parents[1] / "scripts" / "build_superpixel_graphs.py"
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

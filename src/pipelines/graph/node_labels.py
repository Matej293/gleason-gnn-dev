from __future__ import annotations

import numpy as np


def assign_majority_node_labels(
    superpixels: np.ndarray,
    hard_mask: np.ndarray,
    ignore_mask: np.ndarray | None = None,
    min_majority_fraction: float = 0.6,
    num_classes: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assign labels per superpixel using majority class.

    Returns:
        labels: [N] int64 (0..num_classes-1)
        train_mask: [N] bool (False for ambiguous/empty nodes)
    """
    if hard_mask.shape != superpixels.shape:
        raise ValueError("hard_mask must match superpixels shape.")
    if ignore_mask is not None and ignore_mask.shape != superpixels.shape:
        raise ValueError("ignore_mask must match superpixels shape.")

    node_ids = np.unique(superpixels)
    node_ids = node_ids[node_ids >= 0]
    labels = np.zeros((node_ids.size,), dtype=np.int64)
    train_mask = np.zeros((node_ids.size,), dtype=bool)

    for i, node_id in enumerate(node_ids.tolist()):
        pix = superpixels == node_id
        if ignore_mask is not None:
            pix = pix & (ignore_mask == 0)
        values = hard_mask[pix]
        if values.size == 0:
            continue
        counts = np.bincount(values.astype(np.int64), minlength=num_classes)[:num_classes]
        majority = int(np.argmax(counts))
        frac = float(counts[majority]) / float(max(values.size, 1))
        labels[i] = majority
        train_mask[i] = frac >= float(min_majority_fraction)

    return labels, train_mask


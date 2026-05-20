from __future__ import annotations

import numpy as np

RAW_VALID_LABELS = {0, 1, 3, 4, 5, 6}
RAW_TO_CLASS = {0: 0, 1: 0, 6: 0, 3: 1, 4: 2, 5: 3}
NUM_CLASSES = 4


def validate_raw_labels(raw_mask: np.ndarray) -> tuple[bool, list[int]]:
    unique = set(np.unique(raw_mask).tolist())
    invalid = sorted(unique.difference(RAW_VALID_LABELS))
    return len(invalid) == 0, invalid


def remap_raw_mask(raw_mask: np.ndarray) -> np.ndarray:
    ok, invalid = validate_raw_labels(raw_mask)
    if not ok:
        raise ValueError(f"Invalid labels found: {invalid}")

    out = np.zeros_like(raw_mask, dtype=np.uint8)
    for raw, cls in RAW_TO_CLASS.items():
        out[raw_mask == raw] = cls
    return out

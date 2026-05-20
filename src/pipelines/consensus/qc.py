from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import label as ndi_label


@dataclass
class QCConfig:
    tiny_cancer_frac: float = 0.001
    extreme_cancer_ratio: float = 3.0
    low_loo_dice: float = 0.35
    high_fragment_count: int = 300
    suspicious_island_px: int = 16
    min_large_cancer_px: int = 200


def class_stats(mask: np.ndarray, num_classes: int) -> dict[str, Any]:
    total = int(mask.size)
    counts = {str(c): int((mask == c).sum()) for c in range(num_classes)}
    fractions = {k: (v / total if total else 0.0) for k, v in counts.items()}
    cancer_frac = float(1.0 - fractions.get("0", 0.0))
    return {
        "total_pixels": total,
        "counts": counts,
        "fractions": fractions,
        "cancer_fraction": cancer_frac,
    }


def fragmentation_stats(mask: np.ndarray) -> dict[str, Any]:
    stats = {}
    for cls in (1, 2, 3):
        comp_map, n = ndi_label(mask == cls)
        if n == 0:
            stats[str(cls)] = {"components": 0, "tiny_components": 0, "largest_component_px": 0}
            continue
        counts = np.bincount(comp_map.ravel())[1:]
        tiny = int((counts <= 16).sum())
        stats[str(cls)] = {
            "components": int(n),
            "tiny_components": tiny,
            "largest_component_px": int(counts.max()),
        }
    return stats


def qc_flags_for_rater(
    stats: dict[str, Any],
    frag: dict[str, Any],
    median_cancer_frac: float,
    loo_dice: float | None,
    config: QCConfig,
) -> list[str]:
    flags = []
    cancer_frac = stats["cancer_fraction"]

    if cancer_frac < config.tiny_cancer_frac and median_cancer_frac > 0.05:
        flags.append("all_background_or_benign")

    if median_cancer_frac > 0:
        ratio = max(cancer_frac / median_cancer_frac, median_cancer_frac / max(cancer_frac, 1e-8))
        if ratio > config.extreme_cancer_ratio:
            flags.append("extreme_cancer_fraction")

    g5_frac = stats["fractions"]["3"]
    if g5_frac > 0.0 and g5_frac > 2.0 * max(1e-8, median_cancer_frac):
        flags.append("extreme_grade5_fraction")

    for cls in ("1", "2", "3"):
        c = frag[cls]["components"]
        if c > config.high_fragment_count:
            flags.append("suspicious_fragmentation")
            break

    if loo_dice is not None and loo_dice < config.low_loo_dice:
        flags.append("low_leave_one_out_dice")

    # Empty grade regions when others likely have tumor burden.
    if stats["counts"]["1"] == 0 and stats["counts"]["2"] == 0 and stats["counts"]["3"] == 0 and median_cancer_frac > 0.1:
        flags.append("suspicious_empty_grade_regions")

    return sorted(set(flags))


def decide_rater_status(flags: list[str], has_invalid_labels: bool, n_raters: int) -> tuple[str, float]:
    if has_invalid_labels:
        return "exclude", 0.0

    hard_exclude_flags = {"all_background_or_benign"}
    if hard_exclude_flags.intersection(flags) and n_raters >= 4:
        return "exclude", 0.0

    soft_flags = {"low_leave_one_out_dice", "suspicious_fragmentation", "extreme_cancer_fraction", "extreme_grade5_fraction", "suspicious_empty_grade_regions"}
    bad = len(soft_flags.intersection(flags))
    if bad >= 2:
        return "down_weight", 0.5
    if bad == 1:
        return "down_weight", 0.7
    return "keep", 1.0

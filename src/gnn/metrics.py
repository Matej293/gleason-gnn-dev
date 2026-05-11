from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CLASS_NAMES = ("benign", "g3", "g4", "g5")
NUM_CLASSES = 4


def json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if np.isfinite(v) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def per_class_f1_from_cm(cm: np.ndarray) -> np.ndarray:
    out = np.full((cm.shape[0],), np.nan, dtype=np.float64)
    for i in range(cm.shape[0]):
        tp = float(cm[i, i])
        fp = float(cm[:, i].sum() - tp)
        fn = float(cm[i, :].sum() - tp)
        denom = 2.0 * tp + fp + fn
        if denom > 0:
            out[i] = (2.0 * tp) / denom
    return out


def balanced_accuracy_from_cm(cm: np.ndarray) -> float:
    recalls = []
    for i in range(cm.shape[0]):
        denom = float(cm[i, :].sum())
        if denom > 0:
            recalls.append(float(cm[i, i]) / denom)
    if not recalls:
        return float("nan")
    return float(np.mean(recalls))


def metric_dict_from_cm(cm: np.ndarray) -> dict:
    per_f1 = per_class_f1_from_cm(cm)
    return {
        "macro_f1": float(np.nanmean(per_f1)) if np.any(~np.isnan(per_f1)) else float("nan"),
        "balanced_accuracy": balanced_accuracy_from_cm(cm),
        "per_class_f1": {CLASS_NAMES[i]: float(per_f1[i]) for i in range(NUM_CLASSES)},
    }


@dataclass
class CaseEval:
    image_id: str
    y_true: np.ndarray
    y_pred: np.ndarray


def aggregate_case_metrics(cases: list[CaseEval]) -> tuple[dict, dict, np.ndarray, dict[str, dict]]:
    case_rows: dict[str, dict] = {}
    cms = []
    for c in cases:
        cm = confusion_matrix(c.y_true, c.y_pred)
        cms.append(cm)
        case_rows[c.image_id] = metric_dict_from_cm(cm)

    global_cm = np.sum(cms, axis=0) if cms else np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    micro = metric_dict_from_cm(global_cm)

    macro_f1_vals = [v["macro_f1"] for v in case_rows.values() if not np.isnan(v["macro_f1"])]
    bal_vals = [v["balanced_accuracy"] for v in case_rows.values() if not np.isnan(v["balanced_accuracy"])]
    per_class_means = {}
    for cls in CLASS_NAMES:
        vals = [v["per_class_f1"][cls] for v in case_rows.values() if not np.isnan(v["per_class_f1"][cls])]
        per_class_means[cls] = float(np.mean(vals)) if vals else float("nan")

    per_case_mean = {
        "macro_f1": float(np.mean(macro_f1_vals)) if macro_f1_vals else float("nan"),
        "balanced_accuracy": float(np.mean(bal_vals)) if bal_vals else float("nan"),
        "per_class_f1": per_class_means,
    }

    return micro, per_case_mean, global_cm, case_rows

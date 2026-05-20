from __future__ import annotations

import math

import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_fill_holes
from skimage.measure import label

from src.eval.eval_utils import (
    compute_multiclass_metrics,
    compute_multiclass_metrics_from_pred,
    postprocess_predictions,
)


def test_multiclass_metrics_keys_and_ranges():
    logits = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
    logits[:, 1, 2:6, 2:6] = 5.0

    hard = torch.zeros((1, 8, 8), dtype=torch.long)
    hard[:, 2:6, 2:6] = 1

    ignore = torch.zeros((1, 8, 8), dtype=torch.uint8)

    m = compute_multiclass_metrics(logits, hard, ignore, include_background_in_dice=False)
    for k in (
        "macro_dice",
        "macro_f1",
        "micro_f1",
        "cohen_kappa",
        "challenge_score",
        "weighted_macro_dice",
        "grade5_dice",
        "miou",
        "grade5_iou",
        "dice_benign",
        "dice_g3",
        "dice_g4",
        "dice_g5",
        "iou_benign",
        "iou_g3",
        "iou_g4",
        "iou_g5",
        "iou_tumor_vs_benign",
        "sensitivity",
        "precision",
        "ignored_pixel_fraction",
        "tumor_pixels_ignored_fraction",
        "hd95_mean",
        "hd95_g3",
        "hd95_g4",
        "hd95_g5",
        "asd_mean",
        "asd_g3",
        "asd_g4",
        "asd_g5",
    ):
        assert k in m
    assert 0.0 <= m["macro_dice"] <= 1.0
    assert 0.0 <= m["miou"] <= 1.0
    assert 0.0 <= m["sensitivity"] <= 1.0
    assert 0.0 <= m["precision"] <= 1.0
    assert 0.0 <= m["iou_tumor_vs_benign"] <= 1.0
    assert 0.0 <= m["ignored_pixel_fraction"] <= 1.0
    assert -1.0 <= m["cohen_kappa"] <= 1.0


def test_postprocess_predictions_forces_benign_outside_valid_and_removes_tiny_components():
    pred = torch.zeros((1, 12, 12), dtype=torch.long)
    pred[:, 0:2, 0:2] = 1
    pred[:, 4:10, 4:10] = 1
    ignore = torch.zeros((1, 12, 12), dtype=torch.uint8)
    ignore[:, :3, :] = 1
    tissue = torch.ones((1, 12, 12), dtype=torch.uint8)
    tissue[:, :, :2] = 0

    out = postprocess_predictions(
        pred=pred,
        ignore_mask=ignore,
        tissue_mask=tissue,
        min_component_size_by_class={1: 8},
    )
    assert int(out[:, :3, :].sum().item()) == 0
    assert int(out[:, :, :2].sum().item()) == 0
    assert int((out == 1).sum().item()) == 36


def test_metrics_from_pred_matches_logits_path():
    logits = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
    logits[:, 2, 1:7, 1:7] = 7.0
    hard = torch.zeros((1, 8, 8), dtype=torch.long)
    hard[:, 1:7, 1:7] = 2
    ignore = torch.zeros((1, 8, 8), dtype=torch.uint8)

    m_logits = compute_multiclass_metrics(logits, hard, ignore, include_background_in_dice=False)
    pred = logits.argmax(dim=1)
    m_pred = compute_multiclass_metrics_from_pred(pred, hard, ignore, include_background_in_dice=False)
    assert abs(float(m_logits["macro_dice"]) - float(m_pred["macro_dice"])) < 1e-8
    assert abs(float(m_logits["weighted_macro_dice"]) - float(m_pred["weighted_macro_dice"])) < 1e-8
    assert abs(float(m_logits["miou"]) - float(m_pred["miou"])) < 1e-8


def test_absent_class_metrics_return_nan_and_are_excluded_from_macro():
    pred = torch.zeros((1, 8, 8), dtype=torch.long)
    hard = torch.zeros((1, 8, 8), dtype=torch.long)
    ignore = torch.zeros((1, 8, 8), dtype=torch.uint8)
    m = compute_multiclass_metrics_from_pred(
        pred=pred,
        hard_mask=hard,
        ignore_mask=ignore,
        include_background_in_dice=False,
    )
    assert math.isnan(float(m["dice_g3"]))
    assert math.isnan(float(m["dice_g4"]))
    assert math.isnan(float(m["dice_g5"]))
    assert math.isnan(float(m["macro_dice"]))
    assert math.isnan(float(m["hd95_g3"]))
    assert math.isnan(float(m["hd95_g4"]))
    assert math.isnan(float(m["hd95_g5"]))
    assert math.isnan(float(m["hd95_mean"]))
    assert math.isnan(float(m["asd_g3"]))
    assert math.isnan(float(m["asd_g4"]))
    assert math.isnan(float(m["asd_g5"]))
    assert math.isnan(float(m["asd_mean"]))


def test_postprocess_improves_metrics_when_raw_predicts_outside_tissue():
    pred_raw = torch.zeros((1, 8, 8), dtype=torch.long)
    pred_raw[:, :3, :3] = 1  # false positive region outside tissue
    pred_raw[:, 5:7, 5:7] = 1  # true positive region
    hard = torch.zeros((1, 8, 8), dtype=torch.long)
    hard[:, 5:7, 5:7] = 1
    ignore = torch.zeros((1, 8, 8), dtype=torch.uint8)
    tissue = torch.ones((1, 8, 8), dtype=torch.uint8)
    tissue[:, :3, :3] = 0

    m_raw = compute_multiclass_metrics_from_pred(
        pred=pred_raw,
        hard_mask=hard,
        ignore_mask=ignore,
        include_background_in_dice=False,
    )
    pred_post = postprocess_predictions(
        pred=pred_raw,
        ignore_mask=ignore,
        tissue_mask=tissue,
        min_component_size_by_class={},
    )
    m_post = compute_multiclass_metrics_from_pred(
        pred=pred_post,
        hard_mask=hard,
        ignore_mask=ignore,
        include_background_in_dice=False,
    )

    assert float(m_post["dice_benign"]) > float(m_raw["dice_benign"])
    assert float(m_post["iou_tumor_vs_benign"]) > float(m_raw["iou_tumor_vs_benign"])


def test_postprocess_fills_internal_holes_created_by_component_pruning():
    pred = torch.ones((1, 16, 16), dtype=torch.long)
    pred[:, 6:8, 6:8] = 2
    ignore = torch.zeros((1, 16, 16), dtype=torch.uint8)

    out = postprocess_predictions(
        pred=pred,
        ignore_mask=ignore,
        tissue_mask=None,
        min_component_size_by_class={2: 16},
    )

    assert int((out == 0).sum().item()) == 0
    assert int((out == 2).sum().item()) == 0
    assert int((out == 1).sum().item()) == int(out.numel())


def test_weighted_macro_matches_macro_on_balanced_support():
    pred = torch.zeros((1, 4, 4), dtype=torch.long)
    hard = torch.zeros((1, 4, 4), dtype=torch.long)
    ignore = torch.zeros((1, 4, 4), dtype=torch.uint8)

    hard[:, 0:2, 0:2] = 1
    hard[:, 0:2, 2:4] = 2
    pred[:, 0:2, 0:2] = 1
    pred[:, 0:2, 2:4] = 2

    m = compute_multiclass_metrics_from_pred(
        pred=pred,
        hard_mask=hard,
        ignore_mask=ignore,
        include_background_in_dice=False,
    )
    assert abs(float(m["weighted_macro_dice"]) - float(m["macro_dice"])) < 1e-8


def test_weighted_macro_reflects_imbalanced_support():
    pred = torch.zeros((1, 6, 6), dtype=torch.long)
    hard = torch.zeros((1, 6, 6), dtype=torch.long)
    ignore = torch.zeros((1, 6, 6), dtype=torch.uint8)

    hard[:, 0:4, :] = 1  # 24 pixels
    hard[:, 4:6, 0:2] = 2  # 4 pixels
    pred[:, 0:3, :] = 1  # misses one row of class-1 region
    pred[:, 4:6, :] = 2  # overpredicts class-2 region

    m = compute_multiclass_metrics_from_pred(
        pred=pred,
        hard_mask=hard,
        ignore_mask=ignore,
        include_background_in_dice=False,
    )
    macro = float(m["macro_dice"])
    weighted = float(m["weighted_macro_dice"])
    assert 0.0 <= macro <= 1.0
    assert 0.0 <= weighted <= 1.0
    assert weighted != macro


def test_challenge_score_matches_formula_identity():
    pred = torch.zeros((1, 8, 8), dtype=torch.long)
    hard = torch.zeros((1, 8, 8), dtype=torch.long)
    ignore = torch.zeros((1, 8, 8), dtype=torch.uint8)

    hard[:, 1:5, 1:5] = 1
    pred[:, 2:6, 2:6] = 1

    m = compute_multiclass_metrics_from_pred(
        pred=pred,
        hard_mask=hard,
        ignore_mask=ignore,
        include_background_in_dice=False,
    )
    expected = float(m["cohen_kappa"] + ((m["macro_f1"] + m["micro_f1"]) / 2.0))
    assert abs(float(m["challenge_score"]) - expected) < 1e-8


def test_boundary_metrics_improve_with_better_overlap():
    hard = torch.zeros((1, 16, 16), dtype=torch.long)
    hard[:, 4:12, 4:12] = 1
    ignore = torch.zeros((1, 16, 16), dtype=torch.uint8)

    pred_perfect = hard.clone()
    pred_shifted = torch.zeros_like(hard)
    pred_shifted[:, 6:14, 6:14] = 1

    m_perfect = compute_multiclass_metrics_from_pred(
        pred=pred_perfect,
        hard_mask=hard,
        ignore_mask=ignore,
        include_background_in_dice=False,
    )
    m_shifted = compute_multiclass_metrics_from_pred(
        pred=pred_shifted,
        hard_mask=hard,
        ignore_mask=ignore,
        include_background_in_dice=False,
    )

    assert float(m_perfect["hd95_g3"]) <= float(m_shifted["hd95_g3"])
    assert float(m_perfect["asd_g3"]) <= float(m_shifted["asd_g3"])
    assert float(m_perfect["hd95_mean"]) <= float(m_shifted["hd95_mean"])
    assert float(m_perfect["asd_mean"]) <= float(m_shifted["asd_mean"])


def _legacy_compute_multiclass_metrics_from_pred(
    pred: torch.Tensor,
    hard_mask: torch.Tensor,
    ignore_mask: torch.Tensor,
    include_background_in_dice: bool,
) -> dict[str, float]:
    valid = ignore_mask == 0
    valid_n = int(valid.sum().item())

    num_classes = 4
    dice_values: list[float] = []
    iou_values: list[float] = []
    tp_per_class: list[float] = []
    fp_per_class: list[float] = []
    fn_per_class: list[float] = []
    for c in range(num_classes):
        p = (pred == c) & valid
        t = (hard_mask == c) & valid
        tp_c = float((p & t).sum().item())
        fp_c = float((p & (~t)).sum().item())
        fn_c = float(((~p) & t).sum().item())
        tp_per_class.append(tp_c)
        fp_per_class.append(fp_c)
        fn_per_class.append(fn_c)
        denom = p.sum().item() + t.sum().item()
        union = (p | t).sum().item()
        if denom == 0:
            dice_values.append(float("nan"))
        else:
            inter = (p & t).sum().item()
            dice_values.append((2.0 * inter + 1e-5) / (denom + 1e-5))
        if union == 0:
            iou_values.append(float("nan"))
        else:
            inter = (p & t).sum().item()
            iou_values.append((inter + 1e-5) / (union + 1e-5))

    start = 0 if include_background_in_dice else 1
    used = [d for d in dice_values[start:] if not math.isnan(d)]
    used_iou = [x for x in iou_values[start:] if not math.isnan(x)]
    macro_dice = float(sum(used) / len(used)) if used else float("nan")
    weighted_num = 0.0
    weighted_den = 0.0
    for c in range(start, num_classes):
        support = int(((hard_mask == c) & valid).sum().item())
        dice_c = dice_values[c]
        if support > 0 and not math.isnan(dice_c):
            weighted_num += float(support) * float(dice_c)
            weighted_den += float(support)
    weighted_macro_dice = (
        float(weighted_num / weighted_den) if weighted_den > 0.0 else float("nan")
    )

    macro_f1 = macro_dice
    tp_sum = float(sum(tp_per_class))
    fp_sum = float(sum(fp_per_class))
    fn_sum = float(sum(fn_per_class))
    micro_f1 = (
        (2.0 * tp_sum) / ((2.0 * tp_sum) + fp_sum + fn_sum + 1e-8)
        if valid_n > 0
        else float("nan")
    )

    if valid_n > 0:
        conf = torch.zeros((num_classes, num_classes), dtype=torch.float64, device=pred.device)
        pred_v = pred[valid].long().clamp(0, num_classes - 1).view(-1)
        hard_v = hard_mask[valid].long().clamp(0, num_classes - 1).view(-1)
        flat_idx = (hard_v * num_classes) + pred_v
        conf.view(-1).index_add_(
            0,
            flat_idx,
            torch.ones_like(flat_idx, dtype=torch.float64, device=pred.device),
        )
        po = float(conf.diag().sum().item() / float(valid_n))
        row = conf.sum(dim=1)
        col = conf.sum(dim=0)
        pe = float((row * col).sum().item() / float(valid_n * valid_n))
        denom = 1.0 - pe
        cohen_kappa = float((po - pe) / denom) if abs(denom) > 1e-12 else float("nan")
    else:
        cohen_kappa = float("nan")

    if any(math.isnan(v) for v in (cohen_kappa, macro_f1, micro_f1)):
        challenge_score = float("nan")
    else:
        challenge_score = float(cohen_kappa + ((macro_f1 + micro_f1) / 2.0))

    miou = float(sum(used_iou) / len(used_iou)) if used_iou else float("nan")
    grade5_dice = dice_values[3] if len(dice_values) > 3 else float("nan")
    grade5_iou = iou_values[3] if len(iou_values) > 3 else float("nan")

    p_pos = (pred > 0) & valid
    t_pos = (hard_mask > 0) & valid
    tp = float((p_pos & t_pos).sum().item())
    fn = float((~p_pos & t_pos).sum().item())
    fp = float((p_pos & ~t_pos).sum().item())

    sens = (tp + 1e-6) / (tp + fn + 1e-6) if (tp + fn) > 0 else float("nan")
    prec = (tp + 1e-6) / (tp + fp + 1e-6) if (tp + fp) > 0 else float("nan")
    iou_tumor = (tp + 1e-6) / (tp + fp + fn + 1e-6) if (tp + fp + fn) > 0 else float("nan")

    ignored_fraction = float((~valid).float().mean().item())
    tumor_pixels = hard_mask > 0
    tumor_ignored_den = float(tumor_pixels.sum().item())
    tumor_ignored_num = float((tumor_pixels & (~valid)).sum().item())
    tumor_ignored_fraction = (
        (tumor_ignored_num / tumor_ignored_den) if tumor_ignored_den > 0 else float("nan")
    )

    return {
        "macro_dice": macro_dice,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "cohen_kappa": cohen_kappa,
        "challenge_score": challenge_score,
        "weighted_macro_dice": weighted_macro_dice,
        "grade5_dice": grade5_dice,
        "miou": miou,
        "grade5_iou": grade5_iou,
        "dice_benign": dice_values[0] if len(dice_values) > 0 else float("nan"),
        "dice_g3": dice_values[1] if len(dice_values) > 1 else float("nan"),
        "dice_g4": dice_values[2] if len(dice_values) > 2 else float("nan"),
        "dice_g5": dice_values[3] if len(dice_values) > 3 else float("nan"),
        "iou_benign": iou_values[0] if len(iou_values) > 0 else float("nan"),
        "iou_g3": iou_values[1] if len(iou_values) > 1 else float("nan"),
        "iou_g4": iou_values[2] if len(iou_values) > 2 else float("nan"),
        "iou_g5": iou_values[3] if len(iou_values) > 3 else float("nan"),
        "iou_tumor_vs_benign": iou_tumor,
        "sensitivity": sens,
        "precision": prec,
        "ignored_pixel_fraction": ignored_fraction,
        "tumor_pixels_ignored_fraction": tumor_ignored_fraction,
    }


def _legacy_postprocess_predictions(
    pred: torch.Tensor,
    ignore_mask: torch.Tensor,
    tissue_mask: torch.Tensor | None,
    min_component_size_by_class: dict[int, int] | None,
) -> torch.Tensor:
    out = pred.clone().long()
    valid = ignore_mask == 0
    if tissue_mask is not None:
        valid = valid & (tissue_mask > 0)
    out[~valid] = 0

    if not min_component_size_by_class:
        return out

    def _fill_internal_tumor_holes(sample: np.ndarray, sample_valid: np.ndarray) -> np.ndarray:
        tumor = (sample > 0) & sample_valid
        holes = np.logical_and(binary_fill_holes(tumor), ~tumor) & sample_valid
        if not holes.any():
            return sample
        hole_comps = label(holes, connectivity=2)
        hole_ids = np.unique(hole_comps)
        for hid in hole_ids:
            hid_int = int(hid)
            if hid_int == 0:
                continue
            hole = hole_comps == hid_int
            ring = np.logical_and(binary_dilation(hole), ~hole)
            neighbor_classes = sample[np.logical_and(ring, sample > 0)]
            if neighbor_classes.size == 0:
                continue
            counts = np.bincount(neighbor_classes.astype(np.int64), minlength=4)
            if counts.shape[0] <= 1:
                continue
            fill_cls = int(np.argmax(counts[1:]) + 1)
            sample[hole] = fill_cls
        return sample

    out_np = out.detach().cpu().numpy()
    valid_np = valid.detach().cpu().numpy()
    for b in range(out_np.shape[0]):
        sample = out_np[b]
        sample_valid = valid_np[b]
        for cls, min_size in min_component_size_by_class.items():
            cls_id = int(cls)
            min_sz = int(min_size)
            if cls_id <= 0 or min_sz <= 1:
                continue
            mask = (sample == cls_id) & sample_valid
            if not mask.any():
                continue
            comps = label(mask, connectivity=2)
            ids, counts = np.unique(comps, return_counts=True)
            keep_ids = {int(i) for i, c in zip(ids, counts) if int(i) != 0 and int(c) >= min_sz}
            cleaned = np.isin(comps, list(keep_ids))
            sample[np.logical_and(mask, ~cleaned)] = 0
        sample = _fill_internal_tumor_holes(sample, sample_valid)
        out_np[b] = sample

    return torch.from_numpy(out_np).to(device=out.device, dtype=out.dtype)


def _assert_metric_dicts_close(actual: dict[str, float], expected: dict[str, float], atol: float = 1e-8) -> None:
    assert set(actual.keys()) == set(expected.keys())
    for key in actual:
        av = float(actual[key])
        ev = float(expected[key])
        if math.isnan(ev):
            assert math.isnan(av), key
        else:
            assert abs(av - ev) <= atol, key


def test_metrics_fast_path_matches_legacy_reference_for_raw_and_post() -> None:
    gen = torch.Generator().manual_seed(7)

    hard = torch.randint(0, 4, (2, 25, 31), generator=gen, dtype=torch.long)
    pred_raw = torch.randint(0, 4, (2, 25, 31), generator=gen, dtype=torch.long)
    ignore = (torch.rand((2, 25, 31), generator=gen) < 0.18).to(torch.uint8)
    tissue = (torch.rand((2, 25, 31), generator=gen) > 0.08).to(torch.uint8)
    pred_post = postprocess_predictions(
        pred=pred_raw,
        ignore_mask=ignore,
        tissue_mask=tissue,
        min_component_size_by_class={1: 5, 2: 9, 3: 11},
    )

    valid = ignore == 0
    valid_n = int(valid.sum().item())
    if valid_n > 0:
        hard_valid_support = torch.bincount(
            hard[valid].long().clamp(0, 3).reshape(-1),
            minlength=4,
        ).to(dtype=torch.float64)
    else:
        hard_valid_support = torch.zeros((4,), dtype=torch.float64)
    ignored_fraction = float((~valid).float().mean().item())
    tumor_pixels = hard > 0
    tumor_ignored_den = float(tumor_pixels.sum().item())
    tumor_ignored_num = float((tumor_pixels & (~valid)).sum().item())
    tumor_ignored_fraction = (
        tumor_ignored_num / tumor_ignored_den if tumor_ignored_den > 0 else float("nan")
    )

    for pred in (pred_raw, pred_post):
        actual = compute_multiclass_metrics_from_pred(
            pred=pred,
            hard_mask=hard,
            ignore_mask=ignore,
            include_background_in_dice=False,
            include_boundary_metrics=False,
        )
        expected = _legacy_compute_multiclass_metrics_from_pred(
            pred=pred,
            hard_mask=hard,
            ignore_mask=ignore,
            include_background_in_dice=False,
        )
        _assert_metric_dicts_close(actual, expected)

        actual_precomputed = compute_multiclass_metrics_from_pred(
            pred=pred,
            hard_mask=hard,
            ignore_mask=ignore,
            include_background_in_dice=False,
            include_boundary_metrics=False,
            valid_mask=valid,
            valid_n=valid_n,
            hard_valid_support=hard_valid_support,
            ignored_pixel_fraction=ignored_fraction,
            tumor_pixels_ignored_fraction=tumor_ignored_fraction,
        )
        _assert_metric_dicts_close(actual_precomputed, expected)


def test_postprocess_matches_legacy_reference_on_representative_batch() -> None:
    pred = torch.zeros((4, 24, 24), dtype=torch.long)
    ignore = torch.zeros((4, 24, 24), dtype=torch.uint8)
    tissue = torch.ones((4, 24, 24), dtype=torch.uint8)

    pred[0, 4:12, 4:12] = 1

    pred[1, 1:3, 1:3] = 2
    pred[1, 10:17, 10:17] = 2

    pred[2, 5:19, 5:19] = 1
    pred[2, 10:12, 10:12] = 0
    pred[2, 14, 14] = 3

    pred[3, 2:5, 2:5] = 3
    pred[3, 12:14, 12:14] = 1
    ignore[3, :4, :] = 1
    tissue[3, :, :2] = 0

    min_comp = {1: 8, 2: 12, 3: 7}

    actual = postprocess_predictions(
        pred=pred,
        ignore_mask=ignore,
        tissue_mask=tissue,
        min_component_size_by_class=min_comp,
    )
    expected = _legacy_postprocess_predictions(
        pred=pred,
        ignore_mask=ignore,
        tissue_mask=tissue,
        min_component_size_by_class=min_comp,
    )

    assert torch.equal(actual, expected)



def test_postprocess_matches_legacy_reference_for_named_edge_fixtures() -> None:
    min_comp = {1: 6, 2: 8, 3: 5}

    # Fixture 1: no tumor present.
    pred_no_tumor = torch.zeros((1, 20, 20), dtype=torch.long)
    ignore_no_tumor = torch.zeros((1, 20, 20), dtype=torch.uint8)
    tissue_no_tumor = torch.ones((1, 20, 20), dtype=torch.uint8)

    # Fixture 2: tiny component removal.
    pred_tiny = torch.zeros((1, 20, 20), dtype=torch.long)
    pred_tiny[:, 2:4, 2:4] = 2
    pred_tiny[:, 8:16, 8:16] = 2
    ignore_tiny = torch.zeros((1, 20, 20), dtype=torch.uint8)
    tissue_tiny = torch.ones((1, 20, 20), dtype=torch.uint8)

    # Fixture 3: internal hole filling after pruning.
    pred_hole = torch.ones((1, 20, 20), dtype=torch.long)
    pred_hole[:, 8:11, 8:11] = 3
    ignore_hole = torch.zeros((1, 20, 20), dtype=torch.uint8)
    tissue_hole = torch.ones((1, 20, 20), dtype=torch.uint8)

    # Fixture 4: ignore/tissue masking interaction.
    pred_masked = torch.zeros((1, 20, 20), dtype=torch.long)
    pred_masked[:, 1:6, 1:6] = 1
    pred_masked[:, 10:15, 10:15] = 3
    ignore_masked = torch.zeros((1, 20, 20), dtype=torch.uint8)
    ignore_masked[:, :3, :] = 1
    tissue_masked = torch.ones((1, 20, 20), dtype=torch.uint8)
    tissue_masked[:, :, :4] = 0

    fixtures = [
        (pred_no_tumor, ignore_no_tumor, tissue_no_tumor),
        (pred_tiny, ignore_tiny, tissue_tiny),
        (pred_hole, ignore_hole, tissue_hole),
        (pred_masked, ignore_masked, tissue_masked),
    ]

    for pred, ignore, tissue in fixtures:
        actual = postprocess_predictions(
            pred=pred,
            ignore_mask=ignore,
            tissue_mask=tissue,
            min_component_size_by_class=min_comp,
        )
        expected = _legacy_postprocess_predictions(
            pred=pred,
            ignore_mask=ignore,
            tissue_mask=tissue,
            min_component_size_by_class=min_comp,
        )
        assert torch.equal(actual, expected)

from __future__ import annotations

import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.eval_utils import collate_consensus_batch, compute_multiclass_metrics_from_pred
from src.metric_config import LEGACY_METRIC_TRACK_KEYS
from src.train_deconver import _compute_training_loss, _infer_logits, _resize_targets_for_logits, validate


class _FixedLogitsModel(torch.nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("fixed_logits", logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] != self.fixed_logits.shape[0]:
            raise RuntimeError("Batch size mismatch for fixed logits model.")
        return self.fixed_logits


def _legacy_validate_reference(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor,
    image_weight_map: dict[str, float],
    metric_track_keys: tuple[str, ...],
) -> dict[str, float]:
    metric_keys = tuple(metric_track_keys)
    num_metrics = len(metric_keys)
    sums_raw = np.zeros(num_metrics, dtype=np.float64)
    counts_raw = np.zeros(num_metrics, dtype=np.int64)

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    n_batches = 0

    model.eval()
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device)
            soft_probs = batch["soft_probs"].to(device)
            hard_mask = batch["hard_mask"].to(device)
            ignore_mask = batch["ignore_mask"].to(device)
            image_ids = [str(x) for x in batch["image_id"]]
            sample_w = torch.tensor(
                [float(image_weight_map.get(i, 1.0)) for i in image_ids],
                device=device,
                dtype=torch.float32,
            )

            logits = _infer_logits(
                model=model,
                images=images,
                inference_mode="resized_full",
                resized_sliding_window_patch_size=(64, 64),
                resized_sliding_window_overlap=0.25,
            ).clamp(-15.0, 15.0)
            loss, _ = _compute_training_loss(
                logits=logits,
                scales=[logits],
                aux=None,
                hard_mask=hard_mask,
                soft_probs=soft_probs,
                ignore_mask=ignore_mask,
                sample_weights=sample_w,
                class_weights=class_weights,
                use_confidence_mask=False,
                confidence_threshold=0.0,
                soft_loss_type="ce",
                loss_variant="soft_dice",
                lambda_soft=1.0,
                lambda_dice=1.0,
                include_background_in_dice=False,
                exclude_absent_classes_in_dice_loss=False,
                model_name="deconver",
                pspnet_loss_mode="consensus",
                pspnet_aux_weight=0.4,
                pspnet_soft_weight=0.3,
                pspnet_soft_term="ce",
            )

            hard_rs, _, ignore_rs = _resize_targets_for_logits(
                logits=logits,
                hard_mask=hard_mask,
                soft_probs=soft_probs,
                ignore_mask=ignore_mask,
            )
            valid = ignore_rs == 0
            valid_n = int(valid.sum().item())
            if valid_n > 0:
                hard_valid_support = torch.bincount(
                    hard_rs[valid].long().clamp(0, 3).reshape(-1),
                    minlength=4,
                ).to(dtype=torch.float64)
            else:
                hard_valid_support = torch.zeros((4,), dtype=torch.float64, device=hard_rs.device)

            ignored_fraction = float((~valid).float().mean().item())
            tumor_pixels = hard_rs > 0
            tumor_ignored_den = float(tumor_pixels.sum().item())
            tumor_ignored_num = float((tumor_pixels & (~valid)).sum().item())
            tumor_ignored_fraction = (
                tumor_ignored_num / tumor_ignored_den if tumor_ignored_den > 0 else float("nan")
            )

            pred_raw = logits.argmax(dim=1)

            m_raw = compute_multiclass_metrics_from_pred(
                pred=pred_raw,
                hard_mask=hard_rs,
                ignore_mask=ignore_rs,
                include_background_in_dice=False,
                include_boundary_metrics=False,
                boundary_metric_cfg={},
                valid_mask=valid,
                valid_n=valid_n,
                hard_valid_support=hard_valid_support,
                ignored_pixel_fraction=ignored_fraction,
                tumor_pixels_ignored_fraction=tumor_ignored_fraction,
            )
            raw_vals = np.array([float(m_raw.get(k, float("nan"))) for k in metric_keys], dtype=np.float64)
            raw_ok = ~np.isnan(raw_vals)
            sums_raw[raw_ok] += raw_vals[raw_ok]
            counts_raw[raw_ok] += 1

            loss_sum += loss.detach().to(dtype=torch.float64)
            n_batches += 1

    out = {"val_loss": float((loss_sum / max(1, n_batches)).item())}
    for idx, key in enumerate(metric_keys):
        out[f"val_raw/{key}"] = (
            float(sums_raw[idx] / counts_raw[idx]) if counts_raw[idx] > 0 else float("nan")
        )
    return out


def _assert_close_with_nan(actual: float, expected: float, *, atol: float = 1e-8) -> None:
    if math.isnan(expected):
        assert math.isnan(actual)
    else:
        assert abs(actual - expected) <= atol


def test_validate_matches_legacy_reference_for_all_tracked_keys() -> None:
    gen = torch.Generator().manual_seed(123)
    b, h, w = 2, 18, 22

    images = torch.rand((b, 3, h, w), generator=gen, dtype=torch.float32)
    hard = torch.randint(0, 4, (b, h, w), generator=gen, dtype=torch.long)
    ignore = (torch.rand((b, h, w), generator=gen) < 0.15).to(torch.uint8)

    soft = torch.nn.functional.one_hot(hard.clamp(0, 3), num_classes=4).permute(0, 3, 1, 2).float()

    # Fixed logits model so both paths see identical predictions/loss numerics.
    fixed_logits = torch.randn((b, 4, h, w), generator=gen, dtype=torch.float32)
    model = _FixedLogitsModel(fixed_logits)

    samples = []
    for i in range(b):
        samples.append(
            {
                "image": images[i],
                "soft_probs": soft[i],
                "hard_mask": hard[i],
                "ignore_mask": ignore[i],
                "image_id": f"case_{i}",
            }
        )

    loader = DataLoader(samples, batch_size=b, shuffle=False, collate_fn=collate_consensus_batch)
    device = torch.device("cpu")
    class_weights = torch.ones((4,), dtype=torch.float32)
    image_weight_map = {f"case_{i}": 1.0 + (0.1 * i) for i in range(b)}

    expected = _legacy_validate_reference(
        model=model,
        loader=loader,
        device=device,
        class_weights=class_weights,
        image_weight_map=image_weight_map,
        metric_track_keys=LEGACY_METRIC_TRACK_KEYS,
    )

    actual = validate(
        model=model,
        loader=loader,
        device=device,
        class_weights=class_weights,
        image_weight_map=image_weight_map,
        use_confidence_mask=False,
        confidence_threshold=0.0,
        soft_loss_type="ce",
        loss_variant="soft_dice",
        lambda_soft=1.0,
        lambda_dice=1.0,
        include_background_in_dice=False,
        min_component_size_by_class={1: 4, 2: 6, 3: 8},
        inference_mode="resized_full",
        resized_sliding_window_patch_size=(64, 64),
        resized_sliding_window_overlap=0.25,
        use_amp=False,
        amp_dtype=torch.float16,
        model_name="deconver",
        pspnet_loss_mode="consensus",
        pspnet_soft_weight=0.3,
        pspnet_soft_term="ce",
        pspnet_aux_weight=0.4,
        metric_track_keys=LEGACY_METRIC_TRACK_KEYS,
        include_boundary_metrics=False,
        boundary_metric_cfg={},
        enable_channels_last=False,
    )

    assert set(actual.keys()) == set(expected.keys())
    for key in sorted(expected.keys()):
        _assert_close_with_nan(float(actual[key]), float(expected[key]))

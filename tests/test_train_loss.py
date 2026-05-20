from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

from src.trainers.segmentation import (
    _compute_training_loss,
    _hard_dice_per_class,
    _hard_dice_valid_class_mask,
    _make_valid_mask,
    _nanmean_tensor,
    _resize_targets_for_logits,
    _soft_loss_map,
)


def _baseline_single_scale_loss(
    logits: torch.Tensor,
    hard_mask: torch.Tensor,
    soft_probs: torch.Tensor,
    ignore_mask: torch.Tensor,
    sample_weights: torch.Tensor,
    class_weights: torch.Tensor,
    use_confidence_mask: bool,
    confidence_threshold: float,
    soft_loss_type: str,
    loss_variant: str,
    lambda_soft: float,
    lambda_dice: float,
    include_background_in_dice: bool,
    exclude_absent_classes_in_dice_loss: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Non-finite logits passed to loss.")
    hard_rs, soft_rs, ignore_rs = _resize_targets_for_logits(
        logits=logits,
        hard_mask=hard_mask,
        soft_probs=soft_probs,
        ignore_mask=ignore_mask,
    )
    if not torch.isfinite(soft_rs).all():
        raise FloatingPointError("Non-finite soft targets after resize.")
    if not torch.isfinite(hard_rs.float()).all():
        raise FloatingPointError("Non-finite hard targets after resize.")
    if not torch.isfinite(ignore_rs.float()).all():
        raise FloatingPointError("Non-finite ignore mask after resize.")

    valid_mask = _make_valid_mask(
        ignore_mask=ignore_rs,
        soft_probs=soft_rs,
        use_confidence_mask=use_confidence_mask,
        confidence_threshold=confidence_threshold,
    )

    soft_map = _soft_loss_map(logits, soft_rs, loss_type=soft_loss_type)
    expected_cls_weight = (soft_rs * class_weights.view(1, -1, 1, 1)).sum(dim=1)
    soft_map = soft_map * expected_cls_weight

    pixel_weight = sample_weights.view(-1, 1, 1)
    valid_float = valid_mask.float()
    soft_num = (soft_map * valid_float * pixel_weight).sum()
    soft_den = (valid_float * pixel_weight).sum().clamp_min(1e-8)
    soft_loss = soft_num / soft_den

    probs = F.softmax(logits.float(), dim=1)
    if loss_variant == "focal_dice":
        target = F.one_hot(
            hard_rs.long().clamp(0, logits.shape[1] - 1), num_classes=logits.shape[1]
        ).permute(0, 3, 1, 2).float()
        ce = F.cross_entropy(logits.float(), hard_rs.long(), reduction="none")
        pt = (probs * target).sum(dim=1).clamp(1e-6, 1.0)
        focal_gamma = 2.0
        focal_map = ((1.0 - pt) ** focal_gamma) * ce
        hard_cls_weight = class_weights[hard_rs.long()].float()
        focal_num = (focal_map * hard_cls_weight * valid_float * pixel_weight).sum()
        focal_den = (hard_cls_weight * valid_float * pixel_weight).sum().clamp_min(1e-8)
        soft_loss = focal_num / focal_den

    dice_c = _hard_dice_per_class(
        probs=probs,
        hard_mask=hard_rs,
        valid_mask=valid_mask,
        num_classes=logits.shape[1],
    )
    dice_valid_mask = _hard_dice_valid_class_mask(
        hard_mask=hard_rs,
        valid_mask=valid_mask,
        num_classes=logits.shape[1],
    )

    if include_background_in_dice:
        dice_used = dice_c
        dice_valid_used = dice_valid_mask
    else:
        dice_used = dice_c[1:]
        dice_valid_used = dice_valid_mask[1:]

    if loss_variant == "tversky_dice":
        target = F.one_hot(
            hard_rs.long().clamp(0, logits.shape[1] - 1), num_classes=logits.shape[1]
        ).permute(0, 3, 1, 2).float()
        valid = valid_mask.unsqueeze(1).float()
        p = probs * valid
        t = target * valid
        fp = (p * (1.0 - t)).sum(dim=(0, 2, 3))
        fn = ((1.0 - p) * t).sum(dim=(0, 2, 3))
        tp = (p * t).sum(dim=(0, 2, 3))
        alpha = 0.3
        beta = 0.7
        tversky = (tp + 1e-5) / (tp + (alpha * fp) + (beta * fn) + 1e-5)
        tversky_used = tversky if include_background_in_dice else tversky[1:]
        if exclude_absent_classes_in_dice_loss:
            hard_dice_loss = 1.0 - _nanmean_tensor(tversky_used, dice_valid_used)
        else:
            hard_dice_loss = 1.0 - tversky_used.mean()
    else:
        if exclude_absent_classes_in_dice_loss:
            hard_dice_loss = 1.0 - _nanmean_tensor(dice_used, dice_valid_used)
        else:
            hard_dice_loss = 1.0 - dice_used.mean()

    total = (lambda_soft * soft_loss) + (lambda_dice * hard_dice_loss)
    stats = {
        "soft_loss": float(soft_loss.detach().cpu().item()),
        "hard_dice_loss": float(hard_dice_loss.detach().cpu().item()),
        "valid_fraction": float(valid_mask.float().mean().detach().cpu().item()),
    }
    return total, stats


def _baseline_consensus_loss(
    outputs: list[torch.Tensor] | torch.Tensor,
    hard_mask: torch.Tensor,
    soft_probs: torch.Tensor,
    ignore_mask: torch.Tensor,
    sample_weights: torch.Tensor,
    class_weights: torch.Tensor,
    use_confidence_mask: bool,
    confidence_threshold: float,
    soft_loss_type: str,
    loss_variant: str,
    lambda_soft: float,
    lambda_dice: float,
    include_background_in_dice: bool,
    exclude_absent_classes_in_dice_loss: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    if isinstance(outputs, torch.Tensor):
        return _baseline_single_scale_loss(
            logits=outputs,
            hard_mask=hard_mask,
            soft_probs=soft_probs,
            ignore_mask=ignore_mask,
            sample_weights=sample_weights,
            class_weights=class_weights,
            use_confidence_mask=use_confidence_mask,
            confidence_threshold=confidence_threshold,
            soft_loss_type=soft_loss_type,
            loss_variant=loss_variant,
            lambda_soft=lambda_soft,
            lambda_dice=lambda_dice,
            include_background_in_dice=include_background_in_dice,
            exclude_absent_classes_in_dice_loss=exclude_absent_classes_in_dice_loss,
        )

    raw = [1.0 / (2**i) for i in range(len(outputs))]
    total_w = sum(raw)
    weights = [w / total_w for w in raw]

    total_loss = torch.zeros((), device=outputs[0].device, dtype=torch.float32)
    soft_acc = 0.0
    dice_acc = 0.0
    valid_acc = 0.0
    for out, w in zip(outputs, weights):
        l, stats = _baseline_single_scale_loss(
            logits=out,
            hard_mask=hard_mask,
            soft_probs=soft_probs,
            ignore_mask=ignore_mask,
            sample_weights=sample_weights,
            class_weights=class_weights,
            use_confidence_mask=use_confidence_mask,
            confidence_threshold=confidence_threshold,
            soft_loss_type=soft_loss_type,
            loss_variant=loss_variant,
            lambda_soft=lambda_soft,
            lambda_dice=lambda_dice,
            include_background_in_dice=include_background_in_dice,
            exclude_absent_classes_in_dice_loss=exclude_absent_classes_in_dice_loss,
        )
        total_loss = total_loss + (w * l)
        soft_acc += w * stats["soft_loss"]
        dice_acc += w * stats["hard_dice_loss"]
        valid_acc += w * stats["valid_fraction"]

    return total_loss, {
        "soft_loss": soft_acc,
        "hard_dice_loss": dice_acc,
        "valid_fraction": valid_acc,
    }


def _baseline_gleason_ce_loss(
    logits: torch.Tensor,
    hard_mask: torch.Tensor,
    soft_probs: torch.Tensor,
    ignore_mask: torch.Tensor,
    use_confidence_mask: bool,
    confidence_threshold: float,
    include_background_in_dice: bool,
    exclude_absent_classes_in_dice_loss: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    hard_rs, soft_rs, ignore_rs = _resize_targets_for_logits(
        logits=logits,
        hard_mask=hard_mask,
        soft_probs=soft_probs,
        ignore_mask=ignore_mask,
    )
    valid_mask = _make_valid_mask(
        ignore_mask=ignore_rs,
        soft_probs=soft_rs,
        use_confidence_mask=use_confidence_mask,
        confidence_threshold=confidence_threshold,
    )
    target = hard_rs.clone()
    target[ignore_rs != 0] = 255
    loss = F.cross_entropy(logits, target, ignore_index=255)

    probs = F.softmax(logits.float(), dim=1)
    dice_c = _hard_dice_per_class(
        probs=probs,
        hard_mask=hard_rs,
        valid_mask=valid_mask,
        num_classes=logits.shape[1],
    )
    dice_valid_mask = _hard_dice_valid_class_mask(
        hard_mask=hard_rs,
        valid_mask=valid_mask,
        num_classes=logits.shape[1],
    )
    if include_background_in_dice:
        dice_used = dice_c
        dice_valid_used = dice_valid_mask
    else:
        dice_used = dice_c[1:]
        dice_valid_used = dice_valid_mask[1:]
    if exclude_absent_classes_in_dice_loss:
        hard_dice_loss = 1.0 - _nanmean_tensor(dice_used, dice_valid_used)
    else:
        hard_dice_loss = 1.0 - dice_used.mean()

    valid_fraction = float((target != 255).float().mean().detach().cpu().item())
    stats = {
        "soft_loss": float(loss.detach().cpu().item()),
        "hard_dice_loss": float(hard_dice_loss.detach().cpu().item()),
        "valid_fraction": valid_fraction,
    }
    return loss, stats


def _baseline_soft_target_term_loss(
    logits: torch.Tensor,
    soft_probs: torch.Tensor,
    ignore_mask: torch.Tensor,
    sample_weights: torch.Tensor,
    class_weights: torch.Tensor,
    use_confidence_mask: bool,
    confidence_threshold: float,
    soft_loss_type: str,
) -> torch.Tensor:
    _, soft_rs, ignore_rs = _resize_targets_for_logits(
        logits=logits,
        hard_mask=torch.zeros_like(ignore_mask),
        soft_probs=soft_probs,
        ignore_mask=ignore_mask,
    )
    valid_mask = _make_valid_mask(
        ignore_mask=ignore_rs,
        soft_probs=soft_rs,
        use_confidence_mask=use_confidence_mask,
        confidence_threshold=confidence_threshold,
    )
    soft_map = _soft_loss_map(logits, soft_rs, loss_type=soft_loss_type)
    expected_cls_weight = (soft_rs * class_weights.view(1, -1, 1, 1)).sum(dim=1)
    soft_map = soft_map * expected_cls_weight
    pixel_weight = sample_weights.view(-1, 1, 1)
    valid_float = valid_mask.float()
    soft_num = (soft_map * valid_float * pixel_weight).sum()
    soft_den = (valid_float * pixel_weight).sum().clamp_min(1e-8)
    return soft_num / soft_den


def _baseline_compute_training_loss(**kwargs: torch.Tensor | list[torch.Tensor] | str | float | bool | None):
    logits = kwargs["logits"]
    scales = kwargs["scales"]
    aux = kwargs["aux"]
    hard_mask = kwargs["hard_mask"]
    soft_probs = kwargs["soft_probs"]
    ignore_mask = kwargs["ignore_mask"]
    sample_weights = kwargs["sample_weights"]
    class_weights = kwargs["class_weights"]
    use_confidence_mask = kwargs["use_confidence_mask"]
    confidence_threshold = kwargs["confidence_threshold"]
    soft_loss_type = kwargs["soft_loss_type"]
    loss_variant = kwargs["loss_variant"]
    lambda_soft = kwargs["lambda_soft"]
    lambda_dice = kwargs["lambda_dice"]
    include_background_in_dice = kwargs["include_background_in_dice"]
    exclude_absent_classes_in_dice_loss = kwargs["exclude_absent_classes_in_dice_loss"]
    model_name = kwargs["model_name"]
    pspnet_loss_mode = kwargs["pspnet_loss_mode"]
    pspnet_aux_weight = kwargs["pspnet_aux_weight"]
    pspnet_soft_weight = kwargs["pspnet_soft_weight"]
    pspnet_soft_term = kwargs["pspnet_soft_term"]

    if model_name == "pspnet" and pspnet_loss_mode == "gleason_ce":
        loss, stats = _baseline_gleason_ce_loss(
            logits=logits,
            hard_mask=hard_mask,
            soft_probs=soft_probs,
            ignore_mask=ignore_mask,
            use_confidence_mask=use_confidence_mask,
            confidence_threshold=confidence_threshold,
            include_background_in_dice=include_background_in_dice,
            exclude_absent_classes_in_dice_loss=exclude_absent_classes_in_dice_loss,
        )
        if aux is not None:
            aux_loss, _ = _baseline_gleason_ce_loss(
                logits=aux,
                hard_mask=hard_mask,
                soft_probs=soft_probs,
                ignore_mask=ignore_mask,
                use_confidence_mask=use_confidence_mask,
                confidence_threshold=confidence_threshold,
                include_background_in_dice=include_background_in_dice,
                exclude_absent_classes_in_dice_loss=exclude_absent_classes_in_dice_loss,
            )
            loss = loss + (float(pspnet_aux_weight) * aux_loss)
        return loss, stats

    if model_name == "pspnet" and pspnet_loss_mode == "gleason_ce_soft":
        loss, stats = _baseline_gleason_ce_loss(
            logits=logits,
            hard_mask=hard_mask,
            soft_probs=soft_probs,
            ignore_mask=ignore_mask,
            use_confidence_mask=use_confidence_mask,
            confidence_threshold=confidence_threshold,
            include_background_in_dice=include_background_in_dice,
            exclude_absent_classes_in_dice_loss=exclude_absent_classes_in_dice_loss,
        )
        if aux is not None:
            aux_loss, _ = _baseline_gleason_ce_loss(
                logits=aux,
                hard_mask=hard_mask,
                soft_probs=soft_probs,
                ignore_mask=ignore_mask,
                use_confidence_mask=use_confidence_mask,
                confidence_threshold=confidence_threshold,
                include_background_in_dice=include_background_in_dice,
                exclude_absent_classes_in_dice_loss=exclude_absent_classes_in_dice_loss,
            )
            loss = loss + (float(pspnet_aux_weight) * aux_loss)
        soft_term = _baseline_soft_target_term_loss(
            logits=logits,
            soft_probs=soft_probs,
            ignore_mask=ignore_mask,
            sample_weights=sample_weights,
            class_weights=class_weights,
            use_confidence_mask=use_confidence_mask,
            confidence_threshold=confidence_threshold,
            soft_loss_type=pspnet_soft_term,
        )
        loss = loss + (float(pspnet_soft_weight) * soft_term)
        stats["soft_loss"] = float(soft_term.detach().cpu().item())
        return loss, stats

    loss, stats = _baseline_consensus_loss(
        outputs=scales if len(scales) > 1 else logits,
        hard_mask=hard_mask,
        soft_probs=soft_probs,
        ignore_mask=ignore_mask,
        sample_weights=sample_weights,
        class_weights=class_weights,
        use_confidence_mask=use_confidence_mask,
        confidence_threshold=confidence_threshold,
        soft_loss_type=soft_loss_type,
        loss_variant=loss_variant,
        lambda_soft=lambda_soft,
        lambda_dice=lambda_dice,
        include_background_in_dice=include_background_in_dice,
        exclude_absent_classes_in_dice_loss=exclude_absent_classes_in_dice_loss,
    )
    if model_name == "pspnet" and aux is not None:
        aux = aux.clamp(-15.0, 15.0)
        aux_loss, _ = _baseline_consensus_loss(
            outputs=aux,
            hard_mask=hard_mask,
            soft_probs=soft_probs,
            ignore_mask=ignore_mask,
            sample_weights=sample_weights,
            class_weights=class_weights,
            use_confidence_mask=use_confidence_mask,
            confidence_threshold=confidence_threshold,
            soft_loss_type=soft_loss_type,
            loss_variant=loss_variant,
            lambda_soft=lambda_soft,
            lambda_dice=lambda_dice,
            include_background_in_dice=include_background_in_dice,
            exclude_absent_classes_in_dice_loss=exclude_absent_classes_in_dice_loss,
        )
        loss = loss + (float(pspnet_aux_weight) * aux_loss)
    return loss, stats


def _make_loss_inputs(loss_variant: str, model_name: str, pspnet_loss_mode: str) -> dict[str, object]:
    torch.manual_seed(7)
    b, c, h, w = 2, 4, 10, 12
    logits = torch.randn(b, c, h, w, dtype=torch.float32)
    scales = [logits, (logits * 0.5) + 0.25]
    aux = (logits * -0.25) + 0.15

    hard_mask = torch.randint(0, c, (b, h + 2, w + 1), dtype=torch.long)
    raw_soft = torch.rand(b, c, h + 2, w + 1, dtype=torch.float32)
    soft_probs = raw_soft / raw_soft.sum(dim=1, keepdim=True).clamp_min(1e-8)
    ignore_mask = torch.zeros(b, h + 2, w + 1, dtype=torch.uint8)
    ignore_mask[:, 0, :] = 1
    ignore_mask[:, :, 0] = 1

    sample_weights = torch.tensor([1.0, 1.4], dtype=torch.float32)
    class_weights = torch.tensor([1.0, 1.1, 0.9, 1.3], dtype=torch.float32)

    return {
        "logits": logits,
        "scales": scales,
        "aux": aux if model_name == "pspnet" else None,
        "hard_mask": hard_mask,
        "soft_probs": soft_probs,
        "ignore_mask": ignore_mask,
        "sample_weights": sample_weights,
        "class_weights": class_weights,
        "use_confidence_mask": True,
        "confidence_threshold": 0.55,
        "soft_loss_type": "ce",
        "loss_variant": loss_variant,
        "lambda_soft": 0.8,
        "lambda_dice": 1.2,
        "include_background_in_dice": False,
        "exclude_absent_classes_in_dice_loss": True,
        "model_name": model_name,
        "pspnet_loss_mode": pspnet_loss_mode,
        "pspnet_aux_weight": 0.5,
        "pspnet_soft_weight": 0.2,
        "pspnet_soft_term": "kl",
    }


@pytest.mark.parametrize("loss_variant", ["soft_dice", "focal_dice", "tversky_dice"])
def test_compute_training_loss_consensus_parity(loss_variant: str) -> None:
    kwargs = _make_loss_inputs(loss_variant=loss_variant, model_name="deconver", pspnet_loss_mode="consensus")
    expected_loss, expected_stats = _baseline_compute_training_loss(**kwargs)
    got_loss, got_stats = _compute_training_loss(**kwargs)

    torch.testing.assert_close(got_loss, expected_loss, rtol=1e-6, atol=1e-7)
    assert got_stats.keys() == expected_stats.keys()
    for key in expected_stats:
        assert got_stats[key] == pytest.approx(expected_stats[key], rel=1e-6, abs=1e-7)


@pytest.mark.parametrize("mode", ["consensus", "gleason_ce", "gleason_ce_soft"])
def test_compute_training_loss_pspnet_modes_parity(mode: str) -> None:
    kwargs = _make_loss_inputs(loss_variant="soft_dice", model_name="pspnet", pspnet_loss_mode=mode)
    if mode == "consensus":
        kwargs["soft_loss_type"] = "kl"
        kwargs["loss_variant"] = "focal_dice"
    expected_loss, expected_stats = _baseline_compute_training_loss(**kwargs)
    got_loss, got_stats = _compute_training_loss(**kwargs)

    torch.testing.assert_close(got_loss, expected_loss, rtol=1e-6, atol=1e-7)
    assert got_stats.keys() == expected_stats.keys()
    for key in expected_stats:
        assert got_stats[key] == pytest.approx(expected_stats[key], rel=1e-6, abs=1e-7)


@pytest.mark.parametrize("loss_variant", ["soft_dice", "focal_dice", "tversky_dice"])
def test_compute_training_loss_gradients_finite_single_step(loss_variant: str) -> None:
    kwargs = _make_loss_inputs(loss_variant=loss_variant, model_name="deconver", pspnet_loss_mode="consensus")
    logits = kwargs["logits"].clone().detach().requires_grad_(True)
    kwargs = copy.deepcopy(kwargs)
    kwargs["logits"] = logits
    kwargs["scales"] = [logits]
    kwargs["aux"] = None

    loss, _ = _compute_training_loss(**kwargs)
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_compute_training_loss_nonfinite_logits_raises() -> None:
    kwargs = _make_loss_inputs(loss_variant="soft_dice", model_name="deconver", pspnet_loss_mode="consensus")
    bad_logits = kwargs["logits"].clone()
    bad_logits[0, 0, 0, 0] = float("nan")
    kwargs["logits"] = bad_logits
    kwargs["scales"] = [bad_logits]

    with pytest.raises(FloatingPointError, match="Non-finite logits passed to loss"):
        _compute_training_loss(**kwargs)

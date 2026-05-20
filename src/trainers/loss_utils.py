from __future__ import annotations

import torch
import torch.nn.functional as F


def resolve_epoch_lambda_weights(
    cfg: dict,
    epoch: int,
    base_lambda_soft: float,
    base_lambda_dice: float,
) -> tuple[float, float]:
    if not bool(cfg.get("loss_schedule_enabled", False)):
        return base_lambda_soft, base_lambda_dice
    switch_epoch = max(1, int(cfg.get("loss_schedule_transition_epoch", 15)))
    warm_soft = float(cfg.get("lambda_soft_warmup", base_lambda_soft))
    warm_dice = float(cfg.get("lambda_dice_warmup", base_lambda_dice))
    final_soft = float(cfg.get("lambda_soft_final", base_lambda_soft))
    final_dice = float(cfg.get("lambda_dice_final", base_lambda_dice))
    if epoch <= switch_epoch:
        return warm_soft, warm_dice
    return final_soft, final_dice


def resize_targets_for_logits(
    logits: torch.Tensor,
    hard_mask: torch.Tensor,
    soft_probs: torch.Tensor,
    ignore_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    spatial = logits.shape[2:]
    if hard_mask.shape[1:] == spatial:
        return hard_mask, soft_probs, ignore_mask

    hard_rs = (
        F.interpolate(
            hard_mask.unsqueeze(1).float(),
            size=spatial,
            mode="nearest",
        )
        .squeeze(1)
        .long()
    )
    ignore_rs = (
        F.interpolate(
            ignore_mask.unsqueeze(1).float(),
            size=spatial,
            mode="nearest",
        )
        .squeeze(1)
        .to(ignore_mask.dtype)
    )

    soft_rs = F.interpolate(
        soft_probs.float(),
        size=spatial,
        mode="bilinear",
        align_corners=False,
    )
    soft_rs = torch.nan_to_num(soft_rs, nan=0.0, posinf=1.0, neginf=0.0)
    soft_rs = torch.clamp(soft_rs, min=0.0)
    soft_sum = torch.clamp(soft_rs.sum(dim=1, keepdim=True), min=1e-8)
    soft_rs = soft_rs / soft_sum
    return hard_rs, soft_rs, ignore_rs


def make_valid_mask(
    ignore_mask: torch.Tensor,
    soft_probs: torch.Tensor,
    use_confidence_mask: bool,
    confidence_threshold: float,
) -> torch.Tensor:
    valid = ignore_mask == 0
    if use_confidence_mask:
        conf = soft_probs.max(dim=1).values
        valid = valid & (conf >= confidence_threshold)
    return valid


def soft_loss_map(
    logits: torch.Tensor,
    soft_probs: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    log_p = F.log_softmax(logits.float(), dim=1)
    if loss_type == "kl":
        return F.kl_div(log_p, soft_probs.float(), reduction="none").sum(dim=1)
    return -(soft_probs.float() * log_p).sum(dim=1)


def hard_dice_per_class(
    probs: torch.Tensor,
    hard_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-5,
) -> torch.Tensor:
    target = F.one_hot(
        hard_mask.long().clamp(0, num_classes - 1), num_classes=num_classes
    )
    target = target.permute(0, 3, 1, 2).float()
    valid = valid_mask.unsqueeze(1).float()

    p = probs * valid
    t = target * valid

    intersection = (p * t).sum(dim=(0, 2, 3))
    denom = p.sum(dim=(0, 2, 3)) + t.sum(dim=(0, 2, 3))
    dice = (2.0 * intersection + smooth) / (denom + smooth)
    return dice


def hard_dice_valid_class_mask(
    hard_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    target = F.one_hot(
        hard_mask.long().clamp(0, num_classes - 1), num_classes=num_classes
    ).permute(0, 3, 1, 2).float()
    valid = valid_mask.unsqueeze(1).float()
    per_class_target = (target * valid).sum(dim=(0, 2, 3))
    return per_class_target > 0.0


def nanmean_tensor(values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    vals = values[valid_mask]
    if vals.numel() == 0:
        return values.new_tensor(0.0)
    return vals.mean()


def build_scale_loss_context(
    *,
    logits: torch.Tensor,
    hard_mask: torch.Tensor,
    soft_probs: torch.Tensor,
    ignore_mask: torch.Tensor,
    sample_weights: torch.Tensor,
    class_weights: torch.Tensor,
    use_confidence_mask: bool,
    confidence_threshold: float,
    require_finite_logits: bool = False,
    check_resized_finite: bool = False,
    include_probs: bool = True,
    include_hard_terms: bool = True,
) -> dict[str, torch.Tensor]:
    if require_finite_logits and not torch.isfinite(logits).all():
        raise FloatingPointError("Non-finite logits passed to loss.")

    hard_rs, soft_rs, ignore_rs = resize_targets_for_logits(
        logits=logits,
        hard_mask=hard_mask,
        soft_probs=soft_probs,
        ignore_mask=ignore_mask,
    )
    if check_resized_finite:
        if not torch.isfinite(soft_rs).all():
            raise FloatingPointError("Non-finite soft targets after resize.")
        if not torch.isfinite(hard_rs.float()).all():
            raise FloatingPointError("Non-finite hard targets after resize.")
        if not torch.isfinite(ignore_rs.float()).all():
            raise FloatingPointError("Non-finite ignore mask after resize.")

    valid_mask = make_valid_mask(
        ignore_mask=ignore_rs,
        soft_probs=soft_rs,
        use_confidence_mask=use_confidence_mask,
        confidence_threshold=confidence_threshold,
    )
    pixel_weight = sample_weights.view(-1, 1, 1)
    class_weights_4d = class_weights.view(1, -1, 1, 1)
    expected_cls_weight = (soft_rs * class_weights_4d).sum(dim=1)

    ctx: dict[str, torch.Tensor] = {
        "hard_rs": hard_rs,
        "soft_rs": soft_rs,
        "ignore_rs": ignore_rs,
        "valid_mask": valid_mask,
        "valid_float": valid_mask.float(),
        "pixel_weight": pixel_weight,
        "class_weights_4d": class_weights_4d,
        "expected_cls_weight": expected_cls_weight,
    }

    probs: torch.Tensor | None = None
    if include_probs or include_hard_terms:
        probs = F.softmax(logits.float(), dim=1)
        ctx["probs"] = probs

    if include_hard_terms:
        if probs is None:
            probs = F.softmax(logits.float(), dim=1)
            ctx["probs"] = probs
        num_classes = logits.shape[1]
        target_one_hot = F.one_hot(
            hard_rs.long().clamp(0, num_classes - 1), num_classes=num_classes
        ).permute(0, 3, 1, 2).float()
        valid = valid_mask.unsqueeze(1).float()
        probs_valid = probs * valid
        target_valid = target_one_hot * valid
        per_class_target = target_valid.sum(dim=(0, 2, 3))
        dice_intersection = (probs_valid * target_valid).sum(dim=(0, 2, 3))
        dice_denom = probs_valid.sum(dim=(0, 2, 3)) + per_class_target
        dice_per_class = (2.0 * dice_intersection + 1e-5) / (dice_denom + 1e-5)

        ctx["target_one_hot"] = target_one_hot
        ctx["probs_valid"] = probs_valid
        ctx["target_valid"] = target_valid
        ctx["dice_per_class"] = dice_per_class
        ctx["dice_valid_mask"] = per_class_target > 0.0
        ctx["hard_cls_weight"] = class_weights[hard_rs.long()].float()

    return ctx


def single_scale_loss_from_context(
    *,
    logits: torch.Tensor,
    ctx: dict[str, torch.Tensor],
    soft_loss_type: str,
    loss_variant: str,
    lambda_soft: float,
    lambda_dice: float,
    include_background_in_dice: bool,
    exclude_absent_classes_in_dice_loss: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    soft_map = soft_loss_map(logits, ctx["soft_rs"], loss_type=soft_loss_type)
    soft_map = soft_map * ctx["expected_cls_weight"]

    soft_num = (soft_map * ctx["valid_float"] * ctx["pixel_weight"]).sum()
    soft_den = (ctx["valid_float"] * ctx["pixel_weight"]).sum().clamp_min(1e-8)
    soft_loss = soft_num / soft_den

    if loss_variant == "focal_dice":
        ce = F.cross_entropy(logits.float(), ctx["hard_rs"].long(), reduction="none")
        pt = (ctx["probs"] * ctx["target_one_hot"]).sum(dim=1).clamp(1e-6, 1.0)
        focal_gamma = 2.0
        focal_map = ((1.0 - pt) ** focal_gamma) * ce
        focal_num = (
            focal_map * ctx["hard_cls_weight"] * ctx["valid_float"] * ctx["pixel_weight"]
        ).sum()
        focal_den = (
            ctx["hard_cls_weight"] * ctx["valid_float"] * ctx["pixel_weight"]
        ).sum().clamp_min(1e-8)
        soft_loss = focal_num / focal_den

    dice_c = ctx["dice_per_class"]
    dice_valid_mask = ctx["dice_valid_mask"]
    if include_background_in_dice:
        dice_used = dice_c
        dice_valid_used = dice_valid_mask
    else:
        dice_used = dice_c[1:]
        dice_valid_used = dice_valid_mask[1:]

    if loss_variant == "tversky_dice":
        fp = (ctx["probs_valid"] * (1.0 - ctx["target_valid"])).sum(dim=(0, 2, 3))
        fn = ((1.0 - ctx["probs_valid"]) * ctx["target_valid"]).sum(dim=(0, 2, 3))
        tp = (ctx["probs_valid"] * ctx["target_valid"]).sum(dim=(0, 2, 3))
        alpha = 0.3
        beta = 0.7
        tversky = (tp + 1e-5) / (tp + (alpha * fp) + (beta * fn) + 1e-5)
        tversky_used = tversky if include_background_in_dice else tversky[1:]
        if exclude_absent_classes_in_dice_loss:
            hard_dice_loss = 1.0 - nanmean_tensor(tversky_used, dice_valid_used)
        else:
            hard_dice_loss = 1.0 - tversky_used.mean()
    else:
        if exclude_absent_classes_in_dice_loss:
            hard_dice_loss = 1.0 - nanmean_tensor(dice_used, dice_valid_used)
        else:
            hard_dice_loss = 1.0 - dice_used.mean()

    total = (lambda_soft * soft_loss) + (lambda_dice * hard_dice_loss)

    with torch.no_grad():
        stats = {
            "soft_loss": float(soft_loss.detach().cpu().item()),
            "hard_dice_loss": float(hard_dice_loss.detach().cpu().item()),
            "valid_fraction": float(ctx["valid_float"].mean().detach().cpu().item()),
        }
    return total, stats


def single_scale_loss(
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
    ctx = build_scale_loss_context(
        logits=logits,
        hard_mask=hard_mask,
        soft_probs=soft_probs,
        ignore_mask=ignore_mask,
        sample_weights=sample_weights,
        class_weights=class_weights,
        use_confidence_mask=use_confidence_mask,
        confidence_threshold=confidence_threshold,
        require_finite_logits=True,
        check_resized_finite=True,
        include_probs=True,
        include_hard_terms=True,
    )
    return single_scale_loss_from_context(
        logits=logits,
        ctx=ctx,
        soft_loss_type=soft_loss_type,
        loss_variant=loss_variant,
        lambda_soft=lambda_soft,
        lambda_dice=lambda_dice,
        include_background_in_dice=include_background_in_dice,
        exclude_absent_classes_in_dice_loss=exclude_absent_classes_in_dice_loss,
    )


def consensus_loss(
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
        return single_scale_loss(
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
        l, stats = single_scale_loss(
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


def gleason_ce_loss_from_context(
    *,
    logits: torch.Tensor,
    ctx: dict[str, torch.Tensor],
    include_background_in_dice: bool,
    exclude_absent_classes_in_dice_loss: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    target = ctx["hard_rs"].clone()
    target[ctx["ignore_rs"] != 0] = 255
    loss = F.cross_entropy(logits, target, ignore_index=255)

    dice_c = ctx["dice_per_class"]
    dice_valid_mask = ctx["dice_valid_mask"]
    if include_background_in_dice:
        dice_used = dice_c
        dice_valid_used = dice_valid_mask
    else:
        dice_used = dice_c[1:]
        dice_valid_used = dice_valid_mask[1:]
    if exclude_absent_classes_in_dice_loss:
        hard_dice_loss = 1.0 - nanmean_tensor(dice_used, dice_valid_used)
    else:
        hard_dice_loss = 1.0 - dice_used.mean()

    valid_fraction = float((target != 255).float().mean().detach().cpu().item())
    stats = {
        "soft_loss": float(loss.detach().cpu().item()),
        "hard_dice_loss": float(hard_dice_loss.detach().cpu().item()),
        "valid_fraction": valid_fraction,
    }
    return loss, stats


def gleason_ce_loss(
    logits: torch.Tensor,
    hard_mask: torch.Tensor,
    soft_probs: torch.Tensor,
    ignore_mask: torch.Tensor,
    use_confidence_mask: bool,
    confidence_threshold: float,
    include_background_in_dice: bool,
    exclude_absent_classes_in_dice_loss: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    ctx = build_scale_loss_context(
        logits=logits,
        hard_mask=hard_mask,
        soft_probs=soft_probs,
        ignore_mask=ignore_mask,
        sample_weights=torch.ones((logits.shape[0],), device=logits.device, dtype=torch.float32),
        class_weights=torch.ones((logits.shape[1],), device=logits.device, dtype=torch.float32),
        use_confidence_mask=use_confidence_mask,
        confidence_threshold=confidence_threshold,
        include_probs=True,
        include_hard_terms=True,
    )
    return gleason_ce_loss_from_context(
        logits=logits,
        ctx=ctx,
        include_background_in_dice=include_background_in_dice,
        exclude_absent_classes_in_dice_loss=exclude_absent_classes_in_dice_loss,
    )


def soft_target_term_loss_from_context(
    *,
    logits: torch.Tensor,
    ctx: dict[str, torch.Tensor],
    soft_loss_type: str,
) -> torch.Tensor:
    soft_map = soft_loss_map(logits, ctx["soft_rs"], loss_type=soft_loss_type)
    soft_map = soft_map * ctx["expected_cls_weight"]
    soft_num = (soft_map * ctx["valid_float"] * ctx["pixel_weight"]).sum()
    soft_den = (ctx["valid_float"] * ctx["pixel_weight"]).sum().clamp_min(1e-8)
    return soft_num / soft_den


def soft_target_term_loss(
    logits: torch.Tensor,
    soft_probs: torch.Tensor,
    ignore_mask: torch.Tensor,
    sample_weights: torch.Tensor,
    class_weights: torch.Tensor,
    use_confidence_mask: bool,
    confidence_threshold: float,
    soft_loss_type: str,
) -> torch.Tensor:
    ctx = build_scale_loss_context(
        logits=logits,
        hard_mask=ignore_mask.new_zeros(ignore_mask.shape),
        soft_probs=soft_probs,
        ignore_mask=ignore_mask,
        sample_weights=sample_weights,
        class_weights=class_weights,
        use_confidence_mask=use_confidence_mask,
        confidence_threshold=confidence_threshold,
        include_probs=False,
        include_hard_terms=False,
    )
    return soft_target_term_loss_from_context(
        logits=logits,
        ctx=ctx,
        soft_loss_type=soft_loss_type,
    )


def compute_training_loss(
    *,
    logits: torch.Tensor,
    scales: list[torch.Tensor],
    aux: torch.Tensor | None,
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
    model_name: str,
    pspnet_loss_mode: str,
    pspnet_aux_weight: float,
    pspnet_soft_weight: float,
    pspnet_soft_term: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    def _build_ctx_for_scale(
        scale_logits: torch.Tensor,
        *,
        include_hard_terms: bool = True,
        require_finite: bool = False,
        check_resized_finite: bool = False,
    ) -> dict[str, torch.Tensor]:
        return build_scale_loss_context(
            logits=scale_logits,
            hard_mask=hard_mask,
            soft_probs=soft_probs,
            ignore_mask=ignore_mask,
            sample_weights=sample_weights,
            class_weights=class_weights,
            use_confidence_mask=use_confidence_mask,
            confidence_threshold=confidence_threshold,
            require_finite_logits=require_finite,
            check_resized_finite=check_resized_finite,
            include_probs=include_hard_terms,
            include_hard_terms=include_hard_terms,
        )

    if model_name == "pspnet" and pspnet_loss_mode == "gleason_ce":
        ctx = _build_ctx_for_scale(logits, include_hard_terms=True)
        loss, stats = gleason_ce_loss_from_context(
            logits=logits,
            ctx=ctx,
            include_background_in_dice=include_background_in_dice,
            exclude_absent_classes_in_dice_loss=exclude_absent_classes_in_dice_loss,
        )
        if aux is not None:
            aux_ctx = _build_ctx_for_scale(aux, include_hard_terms=True)
            aux_loss, _ = gleason_ce_loss_from_context(
                logits=aux,
                ctx=aux_ctx,
                include_background_in_dice=include_background_in_dice,
                exclude_absent_classes_in_dice_loss=exclude_absent_classes_in_dice_loss,
            )
            loss = loss + (float(pspnet_aux_weight) * aux_loss)
        return loss, stats

    if model_name == "pspnet" and pspnet_loss_mode == "gleason_ce_soft":
        ctx = _build_ctx_for_scale(logits, include_hard_terms=True)
        loss, stats = gleason_ce_loss_from_context(
            logits=logits,
            ctx=ctx,
            include_background_in_dice=include_background_in_dice,
            exclude_absent_classes_in_dice_loss=exclude_absent_classes_in_dice_loss,
        )
        if aux is not None:
            aux_ctx = _build_ctx_for_scale(aux, include_hard_terms=True)
            aux_loss, _ = gleason_ce_loss_from_context(
                logits=aux,
                ctx=aux_ctx,
                include_background_in_dice=include_background_in_dice,
                exclude_absent_classes_in_dice_loss=exclude_absent_classes_in_dice_loss,
            )
            loss = loss + (float(pspnet_aux_weight) * aux_loss)

        soft_ctx = _build_ctx_for_scale(logits, include_hard_terms=False)
        soft_term = soft_target_term_loss_from_context(
            logits=logits,
            ctx=soft_ctx,
            soft_loss_type=pspnet_soft_term,
        )
        loss = loss + (float(pspnet_soft_weight) * soft_term)
        stats["soft_loss"] = float(soft_term.detach().cpu().item())
        return loss, stats

    if len(scales) > 1:
        raw = [1.0 / (2**i) for i in range(len(scales))]
        total_w = sum(raw)
        weights = [w / total_w for w in raw]

        total_loss = torch.zeros((), device=scales[0].device, dtype=torch.float32)
        soft_acc = 0.0
        dice_acc = 0.0
        valid_acc = 0.0
        for scale_logits, w in zip(scales, weights):
            ctx = _build_ctx_for_scale(
                scale_logits,
                include_hard_terms=True,
                require_finite=True,
                check_resized_finite=True,
            )
            scale_loss, scale_stats = single_scale_loss_from_context(
                logits=scale_logits,
                ctx=ctx,
                soft_loss_type=soft_loss_type,
                loss_variant=loss_variant,
                lambda_soft=lambda_soft,
                lambda_dice=lambda_dice,
                include_background_in_dice=include_background_in_dice,
                exclude_absent_classes_in_dice_loss=exclude_absent_classes_in_dice_loss,
            )
            total_loss = total_loss + (w * scale_loss)
            soft_acc += w * scale_stats["soft_loss"]
            dice_acc += w * scale_stats["hard_dice_loss"]
            valid_acc += w * scale_stats["valid_fraction"]
        loss = total_loss
        stats = {
            "soft_loss": soft_acc,
            "hard_dice_loss": dice_acc,
            "valid_fraction": valid_acc,
        }
    else:
        main_ctx = _build_ctx_for_scale(
            logits,
            include_hard_terms=True,
            require_finite=True,
            check_resized_finite=True,
        )
        loss, stats = single_scale_loss_from_context(
            logits=logits,
            ctx=main_ctx,
            soft_loss_type=soft_loss_type,
            loss_variant=loss_variant,
            lambda_soft=lambda_soft,
            lambda_dice=lambda_dice,
            include_background_in_dice=include_background_in_dice,
            exclude_absent_classes_in_dice_loss=exclude_absent_classes_in_dice_loss,
        )

    if model_name == "pspnet" and aux is not None:
        aux = aux.clamp(-15.0, 15.0)
        aux_ctx = _build_ctx_for_scale(
            aux,
            include_hard_terms=True,
            require_finite=True,
            check_resized_finite=True,
        )
        aux_loss, _ = single_scale_loss_from_context(
            logits=aux,
            ctx=aux_ctx,
            soft_loss_type=soft_loss_type,
            loss_variant=loss_variant,
            lambda_soft=lambda_soft,
            lambda_dice=lambda_dice,
            include_background_in_dice=include_background_in_dice,
            exclude_absent_classes_in_dice_loss=exclude_absent_classes_in_dice_loss,
        )
        loss = loss + (float(pspnet_aux_weight) * aux_loss)
    return loss, stats

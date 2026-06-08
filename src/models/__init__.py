from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn
from src.models.pspnet import PSPNet

_DECONVER_ROOT = Path(__file__).parent / "deconver"
if str(_DECONVER_ROOT) not in sys.path:
    sys.path.insert(0, str(_DECONVER_ROOT))

try:
    from deconver.deconver import Deconver as _Deconver

    _DECONVER_AVAILABLE = True
except ImportError:
    _Deconver = None  # type: ignore[assignment,misc]
    _DECONVER_AVAILABLE = False


def build_model(cfg: dict) -> nn.Module:
    name = str(cfg.get("model", "deconver")).lower()
    spatial_dims = int(cfg.get("spatial_dims", 2))
    if spatial_dims != 2:
        raise ValueError(f"Only spatial_dims=2 is supported, got {spatial_dims}")

    in_channels = int(cfg.get("input_channels", 3))
    if in_channels == 0:
        raise ValueError("input_channels must be > 0")

    out_channels = int(cfg.get("out_channels", 4))
    if name == "pspnet":
        encoder_weights_cfg = cfg.get("pspnet_encoder_weights", None)
        if encoder_weights_cfg is None:
            encoder_weights: str | None = "imagenet" if bool(cfg.get("pspnet_pretrained_backbone", True)) else None
        else:
            encoder_weights_str = str(encoder_weights_cfg).strip().lower()
            encoder_weights = None if encoder_weights_str == "none" else encoder_weights_str

        return PSPNet(
            in_channels=in_channels,
            out_channels=out_channels,
            use_aux=bool(cfg.get("pspnet_use_aux", True)),
            pretrained_backbone=bool(cfg.get("pspnet_pretrained_backbone", True)),
            encoder_name=str(cfg.get("pspnet_encoder_name", "resnet101")).strip(),
            encoder_weights=encoder_weights,
        )

    if name != "deconver":
        raise ValueError(
            f"Unsupported model {name!r}. Expected 'deconver' or 'pspnet'."
        )
    if not _DECONVER_AVAILABLE:
        raise ValueError(
            "model='deconver' requested but the Deconver package could not be "
            "imported from src/models/deconver/. "
            "Check that the submodule is present and its dependencies are installed."
        )

    encoder_depth_cfg = list(cfg.get("deconver_encoder_depth", [1, 1, 1, 1]))
    decoder_depth_cfg = list(
        cfg.get(
            "deconver_decoder_depth",
            [1] * max(1, len(encoder_depth_cfg) - 1),
        )
    )

    deep_supervision = bool(cfg.get("deep_supervision", False))
    num_deep_supr: bool | int = False
    if deep_supervision:
        num_deep_supr = max(1, len(encoder_depth_cfg) - 1)

    return _Deconver(  # type: ignore[misc]
        in_channels=in_channels,
        out_channels=out_channels,
        spatial_dims=2,
        encoder_depth=tuple(encoder_depth_cfg),
        encoder_width=tuple(cfg.get("deconver_encoder_width", [64, 128, 256, 512])),
        strides=tuple(cfg.get("deconver_strides", [1, 2, 2, 2])),
        decoder_depth=tuple(decoder_depth_cfg),
        norm=nn.InstanceNorm2d,
        kernel_size=tuple(cfg.get("deconver_kernel_size", [3, 3])),
        groups=cfg.get("deconver_groups", -1),
        ratio=cfg.get("deconver_ndc_ratio", 4),
        fp32_islands=cfg.get("deconver_fp32_islands", False),
        fp32_scope=cfg.get("deconver_fp32_scope", "update_only"),
        num_deep_supr=num_deep_supr,
    )


__all__ = ["build_model"]

from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn

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
    if name != "deconver":
        raise ValueError(f"Only model='deconver' is supported, got {name!r}")
    if not _DECONVER_AVAILABLE:
        raise ValueError(
            "model='deconver' requested but the Deconver package could not be "
            "imported from src/models/deconver/. "
            "Check that the submodule is present and its dependencies are installed."
        )

    spatial_dims = int(cfg.get("spatial_dims", 2))
    if spatial_dims != 2:
        raise ValueError(f"Only spatial_dims=2 is supported, got {spatial_dims}")

    in_channels = int(cfg.get("input_channels", 3))
    if in_channels == 0:
        raise ValueError("input_channels must be > 0")

    deep_supervision = bool(cfg.get("deep_supervision", False))
    num_deep_supr: bool | int = False
    if deep_supervision:
        encoder_depth = cfg.get("deconver_encoder_depth", [1, 1, 1, 1])
        num_deep_supr = max(1, len(encoder_depth) - 1)

    return _Deconver(  # type: ignore[misc]
        in_channels=in_channels,
        out_channels=int(cfg.get("out_channels", 4)),
        spatial_dims=2,
        encoder_depth=tuple(cfg.get("deconver_encoder_depth", [1, 1, 1, 1])),
        encoder_width=tuple(cfg.get("deconver_encoder_width", [64, 128, 256, 512])),
        strides=tuple(cfg.get("deconver_strides", [1, 2, 2, 2])),
        norm=nn.InstanceNorm2d,
        kernel_size=tuple(cfg.get("deconver_kernel_size", [3, 3])),
        groups=cfg.get("deconver_groups", -1),
        ratio=cfg.get("deconver_ndc_ratio", 4),
        fp32_islands=cfg.get("deconver_fp32_islands", False),
        fp32_scope=cfg.get("deconver_fp32_scope", "update_only"),
        num_deep_supr=num_deep_supr,
    )


__all__ = ["build_model"]

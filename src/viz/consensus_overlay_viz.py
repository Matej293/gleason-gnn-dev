from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.viz.visualization import CLASS_COLORS, CLASS_LABELS, colorize_mask


IGNORE_COLOR = np.asarray((0, 255, 255), dtype=np.float32)


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected tensor/array, got shape {arr.shape}")
    return arr


def _to_rgb_uint8(image: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        arr = image.detach().cpu().float().numpy()
    else:
        arr = np.asarray(image, dtype=np.float32)

    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected image with 1 or 3 channels, got shape {arr.shape}")

    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    else:
        arr = np.zeros_like(arr)
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def render_gt_overlay(
    image: torch.Tensor | np.ndarray,
    hard_mask: torch.Tensor | np.ndarray,
    ignore_mask: torch.Tensor | np.ndarray | None = None,
    alpha: float = 0.45,
) -> Image.Image:
    if not (0.0 <= float(alpha) <= 1.0):
        raise ValueError(f"alpha must be in [0,1], got {alpha}")

    rgb = _to_rgb_uint8(image).astype(np.float32)
    gt = _to_numpy(hard_mask).astype(np.int64)
    gt_rgb = colorize_mask(gt).astype(np.float32)

    blended = (1.0 - float(alpha)) * rgb + float(alpha) * gt_rgb

    if ignore_mask is not None:
        ign = _to_numpy(ignore_mask) > 0
        if ign.any():
            blended[ign] = 0.65 * blended[ign] + 0.35 * IGNORE_COLOR

    out = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def render_gt_panel(
    image: torch.Tensor | np.ndarray,
    hard_mask: torch.Tensor | np.ndarray,
    ignore_mask: torch.Tensor | np.ndarray | None,
    image_id: str,
    alpha: float = 0.45,
) -> Image.Image:
    rgb = _to_rgb_uint8(image)
    gt = _to_numpy(hard_mask).astype(np.int64)
    ign = _to_numpy(ignore_mask).astype(np.uint8) if ignore_mask is not None else None

    overlay = np.asarray(render_gt_overlay(image=image, hard_mask=hard_mask, ignore_mask=ignore_mask, alpha=alpha))
    gt_rgb = colorize_mask(gt)
    ignore_rgb = np.zeros_like(gt_rgb, dtype=np.uint8)
    if ign is not None:
        ignore_rgb[ign > 0] = np.asarray((0, 255, 255), dtype=np.uint8)

    panels = [
        ("Input RGB", rgb),
        ("GT Overlay", overlay),
        ("GT Mask", gt_rgb),
        ("Ignore", ignore_rgb),
    ]

    tile_h, tile_w = rgb.shape[0], rgb.shape[1]
    gap = 8
    title_h = 18
    legend_h = 56
    canvas_h = title_h + tile_h + legend_h + (3 * gap)
    canvas_w = (tile_w * len(panels)) + (gap * (len(panels) + 1))
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(24, 24, 24))
    draw = ImageDraw.Draw(canvas)

    x = gap
    y = gap + title_h
    for title, arr in panels:
        tile = Image.fromarray(arr, mode="RGB")
        canvas.paste(tile, (x, y))
        draw.text((x, gap), title, fill=(240, 240, 240))
        x += tile_w + gap

    draw.text((gap, y + tile_h + gap), image_id, fill=(230, 230, 230))

    lx = gap
    ly = y + tile_h + gap + 18
    for cls_idx in sorted(CLASS_LABELS):
        color = CLASS_COLORS[cls_idx]
        draw.rectangle((lx, ly, lx + 12, ly + 12), fill=color)
        draw.text((lx + 16, ly), CLASS_LABELS[cls_idx], fill=(240, 240, 240))
        lx += 70
    draw.rectangle((lx, ly, lx + 12, ly + 12), fill=(0, 255, 255))
    draw.text((lx + 16, ly), "Ignore", fill=(240, 240, 240))
    return canvas


def save_gt_overlay_png(
    output_path: Path,
    image: torch.Tensor | np.ndarray,
    hard_mask: torch.Tensor | np.ndarray,
    ignore_mask: torch.Tensor | np.ndarray | None = None,
    alpha: float = 0.45,
    image_format: str = "PNG",
    compress_level: int = 1,
    optimize: bool = False,
    quality: int = 85,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = render_gt_overlay(
        image=image,
        hard_mask=hard_mask,
        ignore_mask=ignore_mask,
        alpha=alpha,
    )
    fmt = str(image_format).upper()
    if fmt == "PNG":
        img.save(output_path, format="PNG", compress_level=int(compress_level), optimize=bool(optimize))
    elif fmt in ("JPG", "JPEG"):
        img.save(output_path, format="JPEG", quality=int(quality), optimize=bool(optimize))
    elif fmt == "WEBP":
        img.save(output_path, format="WEBP", quality=int(quality), method=4)
    else:
        raise ValueError(f"Unsupported image_format for overlay: {image_format}")
    return output_path


def save_gt_panel_png(
    output_path: Path,
    image: torch.Tensor | np.ndarray,
    hard_mask: torch.Tensor | np.ndarray,
    ignore_mask: torch.Tensor | np.ndarray | None,
    image_id: str,
    alpha: float = 0.45,
    image_format: str = "PNG",
    compress_level: int = 1,
    optimize: bool = False,
    quality: int = 85,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel = render_gt_panel(
        image=image,
        hard_mask=hard_mask,
        ignore_mask=ignore_mask,
        image_id=image_id,
        alpha=alpha,
    )
    fmt = str(image_format).upper()
    if fmt == "PNG":
        panel.save(output_path, format="PNG", compress_level=int(compress_level), optimize=bool(optimize))
    elif fmt in ("JPG", "JPEG"):
        panel.save(output_path, format="JPEG", quality=int(quality), optimize=bool(optimize))
    elif fmt == "WEBP":
        panel.save(output_path, format="WEBP", quality=int(quality), method=4)
    else:
        raise ValueError(f"Unsupported image_format for panel: {image_format}")
    return output_path


__all__ = [
    "render_gt_overlay",
    "render_gt_panel",
    "save_gt_overlay_png",
    "save_gt_panel_png",
]

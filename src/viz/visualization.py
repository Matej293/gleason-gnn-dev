from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (90, 90, 90),  # benign
    1: (46, 204, 113),  # G3
    2: (241, 196, 15),  # G4
    3: (231, 76, 60),  # G5
}

CLASS_LABELS: dict[int, str] = {
    0: "Benign",
    1: "G3",
    2: "G4",
    3: "G5",
}


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


def _to_rgb_image(image: torch.Tensor | np.ndarray) -> Image.Image:
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
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected image with 1 or 3 channels, got shape {arr.shape}")

    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    else:
        arr = np.zeros_like(arr)
    arr_u8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr_u8, mode="RGB")


def colorize_mask(mask: torch.Tensor | np.ndarray) -> np.ndarray:
    m = _to_numpy(mask).astype(np.int64)
    out = np.zeros((m.shape[0], m.shape[1], 3), dtype=np.uint8)
    for cls_idx, color in CLASS_COLORS.items():
        out[m == int(cls_idx)] = np.asarray(color, dtype=np.uint8)
    return out


def _overlay_ignore(base_rgb: np.ndarray, ignore_mask: np.ndarray | None) -> np.ndarray:
    if ignore_mask is None:
        return base_rgb
    out = base_rgb.copy()
    ign = (ignore_mask > 0).astype(np.uint8)
    if ign.any():
        out[ign > 0] = (
            0.65 * out[ign > 0].astype(np.float32) + 0.35 * np.asarray([0, 255, 255], dtype=np.float32)
        ).astype(np.uint8)
    return out


def _error_map(gt_mask: np.ndarray, pred_mask: np.ndarray, ignore_mask: np.ndarray | None) -> np.ndarray:
    h, w = gt_mask.shape
    err = np.zeros((h, w, 3), dtype=np.uint8)
    gt_pos = gt_mask > 0
    pred_pos = pred_mask > 0
    fn = gt_pos & (~pred_pos)
    fp = pred_pos & (~gt_pos)
    tp = gt_pos & pred_pos
    err[tp] = np.asarray((90, 90, 90), dtype=np.uint8)
    err[fp] = np.asarray((0, 120, 255), dtype=np.uint8)
    err[fn] = np.asarray((255, 80, 80), dtype=np.uint8)
    if ignore_mask is not None:
        ign = ignore_mask > 0
        err[ign] = (0, 255, 255)
    return err


def render_case_panel(
    image: torch.Tensor | np.ndarray,
    gt_mask: torch.Tensor | np.ndarray,
    pred_mask: torch.Tensor | np.ndarray,
    ignore_mask: torch.Tensor | np.ndarray | None,
    image_id: str,
    metrics: dict[str, Any] | None = None,
) -> Image.Image:
    rgb = np.asarray(_to_rgb_image(image))
    gt = _to_numpy(gt_mask).astype(np.int64)
    pred = _to_numpy(pred_mask).astype(np.int64)
    ign = _to_numpy(ignore_mask).astype(np.uint8) if ignore_mask is not None else None

    gt_rgb = _overlay_ignore(colorize_mask(gt), ign)
    pred_rgb = _overlay_ignore(colorize_mask(pred), ign)
    err_rgb = _error_map(gt, pred, ign)

    panels = [
        ("Input RGB", rgb),
        ("Ground Truth", gt_rgb),
        ("Prediction", pred_rgb),
        ("Error Map", err_rgb),
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

    subtitle = image_id
    if metrics:
        macro = metrics.get("macro_dice")
        g5 = metrics.get("grade5_dice")
        if macro is not None or g5 is not None:
            subtitle += f" | macro_dice={macro} | grade5_dice={g5}"
    draw.text((gap, y + tile_h + gap), subtitle, fill=(230, 230, 230))

    lx = gap
    ly = y + tile_h + gap + 18
    for cls_idx in sorted(CLASS_LABELS):
        color = CLASS_COLORS[cls_idx]
        draw.rectangle((lx, ly, lx + 12, ly + 12), fill=color)
        draw.text((lx + 16, ly), CLASS_LABELS[cls_idx], fill=(240, 240, 240))
        lx += 70
    draw.rectangle((lx, ly, lx + 12, ly + 12), fill=(0, 120, 255))
    draw.text((lx + 16, ly), "FP", fill=(240, 240, 240))
    lx += 50
    draw.rectangle((lx, ly, lx + 12, ly + 12), fill=(255, 80, 80))
    draw.text((lx + 16, ly), "FN", fill=(240, 240, 240))
    lx += 50
    draw.rectangle((lx, ly, lx + 12, ly + 12), fill=(0, 255, 255))
    draw.text((lx + 16, ly), "Ignore", fill=(240, 240, 240))
    return canvas


def save_case_panel(
    output_path: Path,
    image: torch.Tensor | np.ndarray,
    gt_mask: torch.Tensor | np.ndarray,
    pred_mask: torch.Tensor | np.ndarray,
    ignore_mask: torch.Tensor | np.ndarray | None,
    image_id: str,
    metrics: dict[str, Any] | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel = render_case_panel(
        image=image,
        gt_mask=gt_mask,
        pred_mask=pred_mask,
        ignore_mask=ignore_mask,
        image_id=image_id,
        metrics=metrics,
    )
    panel.save(output_path, format="PNG")
    return output_path


from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from src.consensus_overlay_viz import (
    render_gt_overlay,
    render_gt_panel,
    save_gt_overlay_png,
    save_gt_panel_png,
)


def test_render_gt_overlay_is_deterministic_and_rgb() -> None:
    image = torch.linspace(0, 1, steps=3 * 16 * 16, dtype=torch.float32).reshape(3, 16, 16)
    hard = torch.zeros((16, 16), dtype=torch.long)
    hard[4:12, 4:12] = 3

    out1 = np.asarray(render_gt_overlay(image=image, hard_mask=hard, ignore_mask=None, alpha=0.5))
    out2 = np.asarray(render_gt_overlay(image=image, hard_mask=hard, ignore_mask=None, alpha=0.5))

    assert out1.shape == (16, 16, 3)
    assert out1.dtype == np.uint8
    assert np.array_equal(out1, out2)


def test_render_gt_overlay_applies_ignore_tint() -> None:
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    hard = np.ones((12, 12), dtype=np.uint8)
    ignore = np.zeros((12, 12), dtype=np.uint8)
    ignore[:, :6] = 1

    out_no_ignore = np.asarray(render_gt_overlay(image=image, hard_mask=hard, ignore_mask=None, alpha=0.45))
    out_ignore = np.asarray(render_gt_overlay(image=image, hard_mask=hard, ignore_mask=ignore, alpha=0.45))

    assert np.any(out_no_ignore[:, :6] != out_ignore[:, :6])
    assert np.array_equal(out_no_ignore[:, 7:], out_ignore[:, 7:])


def test_save_gt_outputs_write_png(tmp_path) -> None:
    image = torch.rand((3, 20, 20), dtype=torch.float32)
    hard = torch.randint(0, 4, (20, 20), dtype=torch.long)
    ignore = torch.zeros((20, 20), dtype=torch.uint8)

    overlay_path = tmp_path / "overlay.png"
    panel_path = tmp_path / "panel.png"

    save_gt_overlay_png(overlay_path, image=image, hard_mask=hard, ignore_mask=ignore, alpha=0.4)
    save_gt_panel_png(panel_path, image=image, hard_mask=hard, ignore_mask=ignore, image_id="case001", alpha=0.4)

    assert overlay_path.exists()
    assert panel_path.exists()

    with Image.open(overlay_path) as im_overlay:
        assert im_overlay.format == "PNG"
        assert im_overlay.size == (20, 20)

    with Image.open(panel_path) as im_panel:
        assert im_panel.format == "PNG"
        assert im_panel.size[0] > 20
        assert im_panel.size[1] > 20


def test_render_gt_panel_produces_nonempty_canvas() -> None:
    image = torch.zeros((3, 10, 10), dtype=torch.float32)
    hard = torch.zeros((10, 10), dtype=torch.long)
    ignore = torch.ones((10, 10), dtype=torch.uint8)

    panel = render_gt_panel(
        image=image,
        hard_mask=hard,
        ignore_mask=ignore,
        image_id="all_ignore",
        alpha=0.45,
    )
    assert panel.size[0] > 0
    assert panel.size[1] > 0

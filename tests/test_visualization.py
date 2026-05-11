from __future__ import annotations

import numpy as np
import torch

from src.visualization import CLASS_COLORS, colorize_mask, render_case_panel


def test_colorize_mask_class_mapping_is_deterministic() -> None:
    mask = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    out1 = colorize_mask(mask)
    out2 = colorize_mask(mask)
    assert np.array_equal(out1, out2)
    for cls_idx, color in CLASS_COLORS.items():
        y, x = divmod(cls_idx, 2)
        assert tuple(out1[y, x].tolist()) == color


def test_render_case_panel_handles_empty_masks() -> None:
    image = torch.zeros((3, 16, 16), dtype=torch.float32)
    gt = torch.zeros((16, 16), dtype=torch.long)
    pred = torch.zeros((16, 16), dtype=torch.long)
    panel = render_case_panel(
        image=image,
        gt_mask=gt,
        pred_mask=pred,
        ignore_mask=None,
        image_id="empty_case",
        metrics={"macro_dice": "1.0000", "grade5_dice": "n/a"},
    )
    assert panel.size[0] > 0
    assert panel.size[1] > 0


def test_render_case_panel_handles_all_ignore() -> None:
    image = torch.ones((3, 20, 20), dtype=torch.float32)
    gt = torch.zeros((20, 20), dtype=torch.long)
    pred = torch.ones((20, 20), dtype=torch.long)
    ignore = torch.ones((20, 20), dtype=torch.uint8)
    panel = render_case_panel(
        image=image,
        gt_mask=gt,
        pred_mask=pred,
        ignore_mask=ignore,
        image_id="all_ignore",
        metrics=None,
    )
    assert panel.size[0] >= 120
    assert panel.size[1] > 70

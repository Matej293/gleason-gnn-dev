from __future__ import annotations

import pytest

from src.models import build_model


def test_build_model_rejects_non_deconver():
    with pytest.raises(ValueError):
        build_model({"model": "unet3d", "spatial_dims": 2, "input_channels": 3, "out_channels": 4})


def test_build_model_rejects_non_2d():
    with pytest.raises(ValueError):
        build_model({"model": "deconver", "spatial_dims": 3, "input_channels": 3, "out_channels": 4})


def test_build_unet_lite_forward_shape():
    torch = pytest.importorskip("torch")
    model = build_model(
        {
            "model": "unet_lite",
            "spatial_dims": 2,
            "input_channels": 3,
            "out_channels": 4,
            "unet_lite_base_channels": 16,
        }
    )
    x = torch.randn(2, 3, 128, 128)
    y = model(x)
    assert tuple(y.shape) == (2, 4, 128, 128)

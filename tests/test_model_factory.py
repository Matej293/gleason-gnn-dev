from __future__ import annotations

import pytest

from src.models import build_model


def test_build_model_rejects_non_deconver():
    with pytest.raises(ValueError):
        build_model({"model": "unet3d", "spatial_dims": 2, "input_channels": 3, "out_channels": 4})


def test_build_model_rejects_non_spatial():
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


def test_build_pspnet_gleason_forward_shape():
    torch = pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    model = build_model(
        {
            "model": "pspnet_gleason",
            "spatial_dims": 2,
            "input_channels": 3,
            "out_channels": 4,
            "pspnet_use_aux": True,
            "pspnet_pretrained_backbone": False,
        }
    )
    x = torch.randn(1, 3, 128, 128)
    model.eval()
    with torch.no_grad():
        y = model(x)
    assert isinstance(y, dict)
    assert tuple(y["out"].shape) == (1, 4, 128, 128)
    assert tuple(y["aux"].shape) == (1, 4, 128, 128)


def test_build_pspnet_gleason_forward_shape_no_aux():
    torch = pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    model = build_model(
        {
            "model": "pspnet_gleason",
            "spatial_dims": 2,
            "input_channels": 3,
            "out_channels": 4,
            "pspnet_use_aux": False,
            "pspnet_pretrained_backbone": False,
        }
    )
    x = torch.randn(1, 3, 128, 128)
    model.eval()
    with torch.no_grad():
        y = model(x)
    assert isinstance(y, torch.Tensor)
    assert tuple(y.shape) == (1, 4, 128, 128)

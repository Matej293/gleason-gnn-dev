from __future__ import annotations

import pytest

from src.train_deconver import _resolve_training_best_checkpoint_source


def test_training_best_checkpoint_source_accepts_raw() -> None:
    assert _resolve_training_best_checkpoint_source("raw") == "raw"


def test_training_best_checkpoint_source_rejects_post() -> None:
    with pytest.raises(ValueError, match="raw-only"):
        _resolve_training_best_checkpoint_source("post")

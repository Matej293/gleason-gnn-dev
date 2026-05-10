from __future__ import annotations

import pytest

from src.train_deconver_2d import _build_split_rows, _val_presence_shortfalls


def _row(i: int, *, has_cancer: bool, has_g3: bool, has_g4: bool, has_grade5: bool) -> dict:
    return {
        "dataset_index": i,
        "image_id": f"case_{i:03d}",
        "has_cancer": has_cancer,
        "has_g3": has_g3,
        "has_g4": has_g4,
        "has_grade5": has_grade5,
        "qc_fail": False,
        "qc_suspicious": False,
    }


def test_val_presence_shortfalls_reports_missing_classes():
    val_rows = [
        _row(0, has_cancer=True, has_g3=True, has_g4=False, has_grade5=False),
        _row(1, has_cancer=True, has_g3=False, has_g4=True, has_grade5=False),
    ]
    shortfalls = _val_presence_shortfalls(
        val_rows=val_rows,
        required={"n_g3_pos_images": 1, "n_g4_pos_images": 1, "n_g5_pos_images": 2},
    )
    assert shortfalls == {"n_g5_pos_images": (0, 2)}


def test_build_split_rows_enforces_val_presence_minimums():
    rows = []
    for i in range(20):
        rows.append(
            _row(
                i,
                has_cancer=True,
                has_g3=(i % 2 == 0),
                has_g4=(i % 3 == 0),
                has_grade5=False,
            )
        )

    with pytest.raises(RuntimeError, match="Failed to build split meeting validation class-presence minimums"):
        _build_split_rows(
            rows=rows,
            split_mode="iter_80_20",
            seed=42,
            required_val_presence={
                "n_g3_pos_images": 1,
                "n_g4_pos_images": 1,
                "n_g5_pos_images": 1,
            },
            max_attempts=3,
        )


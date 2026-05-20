from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def load_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path), dtype=np.int32)


def save_png_mask(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def save_npz_compressed(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def discover_image_ids(maps_root: Path, raters: list[str]) -> dict[str, dict[str, Path]]:
    by_image: dict[str, dict[str, Path]] = {}
    for rid in raters:
        map_dir = maps_root / f"Maps{rid[1:]}_T"
        if not map_dir.exists():
            continue
        for p in map_dir.glob("*_classimg_nonconvex.png"):
            image_id = p.name.replace("_classimg_nonconvex.png", "")
            by_image.setdefault(image_id, {})[rid] = p
    return by_image


def find_source_image(image_id: str, train_dir: Path, test_dir: Path) -> Path | None:
    for base in (train_dir, test_dir):
        candidate_jpg = base / f"{image_id}.jpg"
        if candidate_jpg.exists():
            return candidate_jpg
        candidate_png = base / f"{image_id}.png"
        if candidate_png.exists():
            return candidate_png
    return None


def run_metadata(config: Any, gpu_info: dict[str, Any], commit_hash: str | None) -> dict[str, Any]:
    payload = asdict(config)
    payload["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    payload["git_commit"] = commit_hash
    payload["gpu"] = gpu_info
    return payload


def try_git_commit_hash(cwd: Path) -> str | None:
    git_head = cwd / ".git"
    if not git_head.exists():
        return None
    head_file = git_head / "HEAD"
    if not head_file.exists():
        return None
    ref = head_file.read_text(encoding="utf-8").strip()
    if ref.startswith("ref:"):
        ref_path = git_head / ref.split(" ", 1)[1]
        if ref_path.exists():
            return ref_path.read_text(encoding="utf-8").strip()
    return ref

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, disk, remove_small_holes, remove_small_objects
from torch.utils.data import Dataset


def build_tissue_mask_from_image(
    image_rgb: np.ndarray,
    close_radius: int = 3,
    min_object_size: int = 4096,
    min_hole_size: int = 4096,
) -> np.ndarray:
    """Return binary tissue mask where 1=tissue, 0=background."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected RGB image with shape [H, W, 3].")

    gray = image_rgb.astype(np.float32).mean(axis=2)
    th = threshold_otsu(gray)
    # Histology background is typically bright, tissue darker.
    tissue = gray < th
    tissue = binary_closing(tissue, footprint=disk(close_radius))
    tissue = remove_small_objects(tissue, min_size=min_object_size)
    tissue = remove_small_holes(tissue, area_threshold=min_hole_size)
    return tissue.astype(np.uint8)


def clean_ignore_mask(
    ignore_mask: np.ndarray,
    tissue_mask: np.ndarray,
    enforce_background_ignore: bool = True,
) -> np.ndarray:
    """Clean ignore mask and optionally force non-tissue/background to ignore."""
    if ignore_mask.shape != tissue_mask.shape:
        raise ValueError("ignore_mask and tissue_mask must have same shape.")
    out = (ignore_mask > 0).astype(np.uint8)
    if enforce_background_ignore:
        out[tissue_mask == 0] = 1
    return out


class GleasonConsensusDataset(Dataset):
    """
    Simple dataset for consensus outputs + source histology image.

    Expects:
      data_root/
        Train_imgs|Test_imgs/<image_id>.jpg
        consensus/<image_id>/
          consensus_hard_mask.png
          consensus_probs_compact.npz  (key: 'probs', [4,H,W], float16/float32)
          ignore_mask.png
    """

    def __init__(
        self,
        data_root: str | Path,
        consensus_root: str | Path,
        image_subdirs: tuple[str, ...] = ("Train_imgs", "Test_imgs"),
        transform: Callable | None = None,
        renormalize_probs: bool = True,
        enforce_background_ignore: bool = True,
        otsu_close_radius: int = 3,
        otsu_min_object_size: int = 4096,
        otsu_min_hole_size: int = 4096,
    ) -> None:
        self.data_root = Path(data_root)
        self.consensus_root = Path(consensus_root)
        self.image_subdirs = image_subdirs
        self.transform = transform
        self.renormalize_probs = renormalize_probs
        self.enforce_background_ignore = enforce_background_ignore
        self.otsu_close_radius = otsu_close_radius
        self.otsu_min_object_size = otsu_min_object_size
        self.otsu_min_hole_size = otsu_min_hole_size

        self.items = self._discover_items()
        if not self.items:
            raise RuntimeError("No valid consensus samples found.")

    def _find_image_path(self, image_id: str) -> Path | None:
        for sub in self.image_subdirs:
            for ext in (".jpg", ".png", ".jpeg"):
                p = self.data_root / sub / f"{image_id}{ext}"
                if p.exists():
                    return p
        return None

    def _discover_items(self) -> list[dict]:
        items: list[dict] = []
        if not self.consensus_root.exists():
            return items

        for d in sorted(self.consensus_root.iterdir()):
            if not d.is_dir():
                continue
            image_id = d.name
            image_path = self._find_image_path(image_id)
            if image_path is None:
                continue

            hard = d / "consensus_hard_mask.png"
            soft = d / "consensus_probs_compact.npz"
            ign = d / "ignore_mask.png"
            if not (hard.exists() and soft.exists() and ign.exists()):
                continue

            items.append(
                {
                    "image_id": image_id,
                    "image_path": image_path,
                    "hard_path": hard,
                    "soft_path": soft,
                    "ignore_path": ign,
                }
            )
        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]

        image = np.array(Image.open(item["image_path"]).convert("RGB"), dtype=np.uint8)
        hard = np.array(Image.open(item["hard_path"]), dtype=np.uint8)
        ignore = np.array(Image.open(item["ignore_path"]), dtype=np.uint8)
        probs = np.load(item["soft_path"])["probs"].astype(np.float32)  # [4,H,W]

        if probs.ndim != 3 or probs.shape[0] != 4:
            raise ValueError(f"Invalid probs shape for {item['image_id']}: {probs.shape}")
        if hard.shape != image.shape[:2] or ignore.shape != image.shape[:2]:
            raise ValueError(f"Shape mismatch for {item['image_id']}")

        if self.renormalize_probs:
            probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
            probs /= np.clip(probs.sum(axis=0, keepdims=True), 1e-8, None)

        tissue = build_tissue_mask_from_image(
            image,
            close_radius=self.otsu_close_radius,
            min_object_size=self.otsu_min_object_size,
            min_hole_size=self.otsu_min_hole_size,
        )
        ignore_clean = clean_ignore_mask(
            ignore,
            tissue,
            enforce_background_ignore=self.enforce_background_ignore,
        )

        sample = {
            "image_id": item["image_id"],
            "image": torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0,  # [3,H,W]
            "soft_probs": torch.from_numpy(probs).float(),  # [4,H,W]
            "hard_mask": torch.from_numpy(hard.astype(np.int64)),  # [H,W]
            "ignore_mask": torch.from_numpy(ignore_clean.astype(np.uint8)),  # [H,W], 1=ignore
            "tissue_mask": torch.from_numpy(tissue.astype(np.uint8)),  # [H,W], 1=tissue
        }

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

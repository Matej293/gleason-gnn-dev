from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from monai.data.utils import dense_patch_slices
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.morphology import closing, disk, remove_small_holes, remove_small_objects
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
    tissue = gray < th
    tissue = closing(tissue, footprint=disk(close_radius))
    # skimage>=0.26 deprecates min_size/area_threshold in favor of max_size
    # with <= semantics; use (threshold - 1) to keep prior strict-< behavior.
    obj_max_size = max(0, int(min_object_size) - 1)
    hole_max_size = max(0, int(min_hole_size) - 1)
    tissue = remove_small_objects(tissue, max_size=obj_max_size)
    tissue = remove_small_holes(tissue, max_size=hole_max_size)
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
    Dataset for consensus outputs + source histology image.

    Expects:
      data_root/
        Train_imgs|Test_imgs/<image_id>.jpg
      consensus_root/<image_id>/
        consensus_hard_mask.png
        consensus_probs_compact.npz  (key: 'probs', [4,H,W], float16/float32)
        ignore_mask.png
        qc_report.json (optional)
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
        probs_eps: float = 1e-8,
        load_qc_report: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.consensus_root = Path(consensus_root)
        # Keep compatibility with older docs/configs that used "Trains_imgs".
        normalized_subdirs: list[str] = []
        for sub in image_subdirs:
            sub_norm = "Train_imgs" if str(sub) == "Trains_imgs" else str(sub)
            if sub_norm not in normalized_subdirs:
                normalized_subdirs.append(sub_norm)
        self.image_subdirs = tuple(normalized_subdirs)
        self.transform = transform
        self.renormalize_probs = renormalize_probs
        self.enforce_background_ignore = enforce_background_ignore
        self.otsu_close_radius = int(otsu_close_radius)
        self.otsu_min_object_size = int(otsu_min_object_size)
        self.otsu_min_hole_size = int(otsu_min_hole_size)
        self.probs_eps = float(probs_eps)
        self.load_qc_report = bool(load_qc_report)

        self.items = self._discover_items()
        if not self.items:
            raise RuntimeError("No valid consensus samples found.")

    def _find_image_path(self, image_id: str) -> tuple[Path, str] | None:
        for sub in self.image_subdirs:
            for ext in (".jpg", ".png", ".jpeg"):
                p = self.data_root / sub / f"{image_id}{ext}"
                if p.exists():
                    return p, sub
        return None

    def _discover_items(self) -> list[dict]:
        items: list[dict] = []
        if not self.consensus_root.exists():
            return items

        for d in sorted(self.consensus_root.iterdir()):
            if not d.is_dir():
                continue
            image_id = d.name
            image_path_meta = self._find_image_path(image_id)
            if image_path_meta is None:
                continue
            image_path, image_subdir = image_path_meta

            hard = d / "consensus_hard_mask.png"
            soft = d / "consensus_probs_compact.npz"
            ign = d / "ignore_mask.png"
            qc = d / "qc_report.json"
            if not (hard.exists() and soft.exists() and ign.exists()):
                continue

            items.append(
                {
                    "image_id": image_id,
                    "image_path": image_path,
                    "image_subdir": image_subdir,
                    "hard_path": hard,
                    "soft_path": soft,
                    "ignore_path": ign,
                    "qc_path": qc,
                }
            )
        return items

    def __len__(self) -> int:
        return len(self.items)

    def _load_probs(self, path: Path, image_id: str) -> np.ndarray:
        probs = np.load(path)["probs"].astype(np.float32)
        if probs.ndim != 3 or probs.shape[0] != 4:
            raise ValueError(f"Invalid probs shape for {image_id}: {probs.shape}")

        probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
        probs = np.clip(probs, 0.0, None)

        if self.renormalize_probs:
            probs_sum = probs.sum(axis=0, keepdims=True)
            nonzero = probs_sum >= self.probs_eps
            probs = np.divide(
                probs,
                np.clip(probs_sum, self.probs_eps, None),
                out=np.zeros_like(probs, dtype=np.float32),
                where=nonzero,
            )
            # Fallback pixels with degenerate all-zero distributions to benign.
            zero_mask = ~nonzero[0]
            if zero_mask.any():
                probs[0, zero_mask] = 1.0

        return probs

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]

        image = np.array(Image.open(item["image_path"]).convert("RGB"), dtype=np.uint8)
        hard = np.array(Image.open(item["hard_path"]), dtype=np.uint8)
        ignore = np.array(Image.open(item["ignore_path"]), dtype=np.uint8)
        probs = self._load_probs(item["soft_path"], image_id=str(item["image_id"]))

        if hard.shape != image.shape[:2] or ignore.shape != image.shape[:2]:
            raise ValueError(f"Shape mismatch for {item['image_id']}")

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
            "image": torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0,
            "soft_probs": torch.from_numpy(probs).float(),
            "hard_mask": torch.from_numpy(hard.astype(np.int64)),
            "ignore_mask": torch.from_numpy(ignore_clean.astype(np.uint8)),
            "tissue_mask": torch.from_numpy(tissue.astype(np.uint8)),
        }

        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class SlidingWindowPatchDataset(Dataset):
    """Patch-level dataset using deterministic MONAI dense sliding windows."""

    def __init__(
        self,
        base_dataset: GleasonConsensusDataset,
        source_indices: list[int] | tuple[int, ...],
        patch_size: tuple[int, int],
        overlap: float = 0.5,
        patch_tissue_filter_enabled: bool = True,
        patch_min_tissue_fraction: float = 0.0,
        transform: Callable | None = None,
    ) -> None:
        if not isinstance(base_dataset, GleasonConsensusDataset):
            raise TypeError("base_dataset must be a GleasonConsensusDataset instance.")
        if not isinstance(source_indices, (list, tuple)) or not source_indices:
            raise ValueError("source_indices must be a non-empty list/tuple of dataset indices.")

        patch_h = int(patch_size[0])
        patch_w = int(patch_size[1])
        if patch_h <= 0 or patch_w <= 0:
            raise ValueError(f"patch_size entries must be > 0, got [{patch_h}, {patch_w}].")

        overlap_f = float(overlap)
        if overlap_f < 0.0 or overlap_f >= 1.0:
            raise ValueError(f"overlap must be in [0.0, 1.0), got {overlap_f}.")

        patch_min_tissue_fraction_f = float(patch_min_tissue_fraction)
        if patch_min_tissue_fraction_f < 0.0 or patch_min_tissue_fraction_f > 1.0:
            raise ValueError(
                "patch_min_tissue_fraction must be in [0.0, 1.0], "
                f"got {patch_min_tissue_fraction_f}."
            )

        self.base_dataset = base_dataset
        self.source_indices = tuple(int(i) for i in source_indices)
        self.patch_size = (patch_h, patch_w)
        self.overlap = overlap_f
        self.patch_tissue_filter_enabled = bool(patch_tissue_filter_enabled)
        self.patch_min_tissue_fraction = patch_min_tissue_fraction_f
        self.transform = transform
        self.scan_interval = (
            max(1, int(round(self.patch_size[0] * (1.0 - self.overlap)))),
            max(1, int(round(self.patch_size[1] * (1.0 - self.overlap)))),
        )

        self.total_candidate_patches = 0
        self.kept_patches = 0
        self.skipped_patches = 0
        self.keep_ratio = 0.0

        self.patch_items = self._build_patch_items()
        self.kept_patches = len(self.patch_items)
        self.skipped_patches = self.total_candidate_patches - self.kept_patches
        self.keep_ratio = (
            float(self.kept_patches) / float(self.total_candidate_patches)
            if self.total_candidate_patches > 0
            else 0.0
        )

        if not self.patch_items:
            raise RuntimeError("No sliding-window patches were generated.")

    def _build_patch_items(self) -> list[dict[str, object]]:
        patch_items: list[dict[str, object]] = []

        for source_index in self.source_indices:
            if source_index < 0 or source_index >= len(self.base_dataset.items):
                raise IndexError(f"source index out of range: {source_index}")

            item = self.base_dataset.items[source_index]
            height, width, tissue_mask = self._load_image_geometry_and_tissue(
                source_index=source_index,
                item=item,
            )

            slices = dense_patch_slices(
                image_size=(int(height), int(width)),
                patch_size=self.patch_size,
                scan_interval=self.scan_interval,
                return_slice=True,
            )

            image_id = str(item["image_id"])
            for y_slice, x_slice in slices:
                self.total_candidate_patches += 1
                if tissue_mask is not None:
                    patch_tissue = tissue_mask[y_slice, x_slice]
                    tissue_fraction = float(patch_tissue.float().mean().item())
                    if self.patch_min_tissue_fraction <= 0.0:
                        if tissue_fraction <= 0.0:
                            continue
                    elif tissue_fraction < self.patch_min_tissue_fraction:
                        continue

                patch_items.append(
                    {
                        "source_index": int(source_index),
                        "image_id": image_id,
                        "y_slice": y_slice,
                        "x_slice": x_slice,
                    }
                )

        return patch_items

    def _load_image_geometry_and_tissue(
        self,
        *,
        source_index: int,
        item: dict[str, object],
    ) -> tuple[int, int, torch.Tensor | None]:
        image_path = Path(str(item["image_path"]))

        if self.patch_tissue_filter_enabled and self.base_dataset.transform is None:
            with Image.open(image_path) as img:
                image_rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
            height, width = int(image_rgb.shape[0]), int(image_rgb.shape[1])
            tissue_mask_np = build_tissue_mask_from_image(
                image_rgb,
                close_radius=self.base_dataset.otsu_close_radius,
                min_object_size=self.base_dataset.otsu_min_object_size,
                min_hole_size=self.base_dataset.otsu_min_hole_size,
            )
            tissue_mask = torch.from_numpy(tissue_mask_np.astype(np.uint8, copy=False))
            return height, width, tissue_mask

        with Image.open(image_path) as img:
            width, height = img.size

        if not self.patch_tissue_filter_enabled:
            return int(height), int(width), None

        sample = self.base_dataset[source_index]
        tissue_mask = sample.get("tissue_mask", None)
        if tissue_mask is None:
            raise ValueError(
                "SlidingWindowPatchDataset requires 'tissue_mask' from base_dataset samples "
                "when patch_tissue_filter_enabled=True."
            )
        if not isinstance(tissue_mask, torch.Tensor):
            tissue_mask = torch.as_tensor(tissue_mask)
        return int(height), int(width), tissue_mask

    def __len__(self) -> int:
        return len(self.patch_items)

    def __getitem__(self, idx: int) -> dict:
        patch = self.patch_items[idx]
        source_index = int(patch["source_index"])
        sample = self.base_dataset[source_index]

        y_slice = patch["y_slice"]
        x_slice = patch["x_slice"]

        out: dict[str, object] = {"image_id": str(sample["image_id"])}
        out["image"] = sample["image"][:, y_slice, x_slice].contiguous()
        out["hard_mask"] = sample["hard_mask"][y_slice, x_slice].contiguous()
        out["ignore_mask"] = sample["ignore_mask"][y_slice, x_slice].contiguous()

        if "soft_probs" in sample:
            out["soft_probs"] = sample["soft_probs"][:, y_slice, x_slice].contiguous()
        if "tissue_mask" in sample:
            out["tissue_mask"] = sample["tissue_mask"][y_slice, x_slice].contiguous()

        if self.transform is not None:
            out = self.transform(out)
        return out


__all__ = [
    "GleasonConsensusDataset",
    "SlidingWindowPatchDataset",
    "build_tissue_mask_from_image",
    "clean_ignore_mask",
]

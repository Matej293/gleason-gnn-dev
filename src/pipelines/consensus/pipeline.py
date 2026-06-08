from __future__ import annotations

import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation
from tqdm import tqdm

from . import metrics
from .fusion import run_multiclass_weighted_vote
from .io import (
    discover_image_ids,
    find_source_image,
    load_mask,
    run_metadata,
    save_json,
    save_npz_compressed,
    save_png_mask,
    try_git_commit_hash,
)
from .labels import NUM_CLASSES, remap_raw_mask, validate_raw_labels
from .postprocess import (
    PostConfig,
    apply_grade5_safeguard,
    boundary_length_4conn,
    boundary_disagreement_penalty,
    choose_hard_mask,
    count_small_components_by_class,
    confidence_uncertainty_maps,
    gpu_info,
    make_ignore_mask_with_threshold,
    normalize_probs,
    refine_hard_mask_classes,
)
from .qc import QCConfig, class_stats, decide_rater_status, fragmentation_stats, qc_flags_for_rater


@dataclass
class ConsensusConfig:
    dataset_root: str = "data"
    output_root: str = "data/consensus"
    raters: list[str] = field(default_factory=lambda: ["p1", "p2", "p3", "p4", "p5", "p6"])
    num_classes: int = NUM_CLASSES
    enable_gpu: bool = True
    strict_ignore: bool = False
    workers: int = 1
    ignore_threshold_loose: float = 0.30
    ignore_threshold_strict: float = 0.50
    target_ignore_tissue_frac: float = 0.05
    target_ignore_total_frac: float = 0.12
    ignore_threshold_min: float = 0.05
    ignore_threshold_max: float = 0.35
    auto_calibrate_ignore_threshold: bool = True
    disable_boundary_penalty: bool = False
    single_rater_ignore_policy: str = "confidence_mask"

    qc: QCConfig = field(default_factory=QCConfig)
    post: PostConfig = field(default_factory=PostConfig)


class ConsensusMaskBuilder:
    def __init__(self, config: ConsensusConfig):
        self.config = config
        self.dataset_root = Path(config.dataset_root)
        self.output_root = Path(config.output_root)
        self.maps_root = self.dataset_root
        self.train_dir = self.dataset_root / "Train_imgs"
        if not self.train_dir.exists():
            self.train_dir = self.dataset_root / "Trains_imgs"
        self.test_dir = self.dataset_root / "Test_imgs"

    def _process_raters(self, by_rater_paths: dict[str, Path]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        sem_maps = {}
        report: dict[str, Any] = {"available_pathologist_ids": sorted(by_rater_paths)}
        invalids = {}

        for rid, path in sorted(by_rater_paths.items()):
            raw = load_mask(path)
            ok, invalid_labels = validate_raw_labels(raw)
            if not ok:
                invalids[rid] = invalid_labels
                continue
            sem_maps[rid] = remap_raw_mask(raw)

        report["invalid_labels"] = invalids
        report["raw_annotation_paths"] = {k: str(v) for k, v in by_rater_paths.items()}
        return sem_maps, report

    def _qc_and_status(
        self, sem_maps: dict[str, np.ndarray], invalids: dict[str, Any], available_raters: list[str]
    ) -> tuple[dict[str, Any], dict[str, str], dict[str, float]]:
        valid_raters = sorted(sem_maps)
        all_raters = sorted(set(available_raters))
        pairwise = metrics.pairwise_agreement(sem_maps, self.config.num_classes)
        loo = metrics.leave_one_out_agreement(sem_maps, self.config.num_classes)

        stats = {r: class_stats(sem_maps[r], self.config.num_classes) for r in valid_raters}
        frags = {r: fragmentation_stats(sem_maps[r]) for r in valid_raters}
        med_cancer = float(np.median([stats[r]["cancer_fraction"] for r in valid_raters])) if valid_raters else 0.0

        statuses: dict[str, str] = {}
        weights: dict[str, float] = {}
        flags: dict[str, list[str]] = {}

        for r in all_raters:
            if r in invalids:
                statuses[r] = "exclude"
                weights[r] = 0.0
                flags[r] = ["invalid_labels"]
                continue

            if r not in sem_maps:
                statuses[r] = "exclude"
                weights[r] = 0.0
                flags[r] = ["missing_or_unreadable_map"]
                continue

            loo_d = loo[r]["dice_multiclass"] if r in loo else None
            f = qc_flags_for_rater(stats[r], frags[r], med_cancer, loo_d, self.config.qc)
            st, w = decide_rater_status(f, False, len(valid_raters))
            statuses[r] = st
            weights[r] = w
            flags[r] = f

        qc_block = {
            "class_pixel_counts_per_pathologist": {r: stats[r]["counts"] for r in valid_raters},
            "class_fractions_per_pathologist": {r: stats[r]["fractions"] for r in valid_raters},
            "fragmentation_per_pathologist": frags,
            "pairwise_dice_scores_between_pathologists": pairwise,
            "leave_one_out_agreement_per_pathologist": loo,
            "flags_for_suspicious_maps": flags,
            "status_per_pathologist": statuses,
            "weights_per_pathologist": weights,
            # Backward-compatible aliases.
            "pairwise_dice": pairwise,
            "leave_one_out": loo,
            "flags_per_pathologist": flags,
        }
        return qc_block, statuses, weights

    @staticmethod
    def _boundary_band_from_masks(masks: list[np.ndarray], dilate_px: int) -> np.ndarray:
        if not masks:
            return np.zeros((0, 0), dtype=bool)
        boundary_union = np.zeros_like(masks[0], dtype=bool)
        for m in masks:
            b = np.zeros_like(m, dtype=bool)
            b[:, 1:] |= m[:, 1:] != m[:, :-1]
            b[1:, :] |= m[1:, :] != m[:-1, :]
            if dilate_px > 0:
                b = binary_dilation(b, iterations=dilate_px)
            boundary_union |= b
        return boundary_union

    @staticmethod
    def _tissue_mask_from_maps(masks: list[np.ndarray], hard: np.ndarray) -> np.ndarray:
        tissue = np.zeros_like(hard, dtype=bool)
        for m in masks:
            tissue |= m > 0
        if not np.any(tissue):
            tissue = hard > 0
        if not np.any(tissue):
            tissue = np.ones_like(hard, dtype=bool)
        return tissue

    def _calibrate_ignore_threshold(
        self,
        conf: np.ndarray,
        base_threshold: float,
        tissue_mask: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        th_min = float(self.config.ignore_threshold_min)
        th_max = float(self.config.ignore_threshold_max)
        threshold = float(np.clip(base_threshold, th_min, th_max))
        ignore = make_ignore_mask_with_threshold(conf, threshold=threshold)
        if not self.config.auto_calibrate_ignore_threshold:
            return ignore, float(threshold)

        target_tissue = float(self.config.target_ignore_tissue_frac)
        target_total = float(self.config.target_ignore_total_frac)
        tissue_denom = float(max(1, int(tissue_mask.sum())))
        total_denom = float(ignore.size)

        while threshold > th_min:
            ignored_tissue = float(np.logical_and(ignore > 0, tissue_mask).sum()) / tissue_denom
            ignored_total = float((ignore > 0).sum()) / total_denom
            if ignored_tissue <= target_tissue and ignored_total <= target_total:
                break
            next_threshold = max(th_min, threshold - 0.01)
            if abs(next_threshold - threshold) < 1e-8:
                break
            threshold = next_threshold
            ignore = make_ignore_mask_with_threshold(conf, threshold=threshold)
        return ignore, float(threshold)

    def _run_consensus(
        self, sem_maps: dict[str, np.ndarray], statuses: dict[str, str], weights: dict[str, float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        kept = [r for r, s in statuses.items() if s != "exclude"]
        if not kept:
            raise RuntimeError("No valid raters after QC")

        masks = [sem_maps[r] for r in kept]
        used_weights = [float(weights.get(r, 1.0)) for r in kept]
        reliable_flags = [weights[r] >= 0.7 for r in kept]

        if len(masks) == 1:
            hard = masks[0].astype(np.uint8)
            probs = np.full((self.config.num_classes, *hard.shape), 0.05 / (self.config.num_classes - 1), dtype=np.float32)
            for c in range(self.config.num_classes):
                probs[c][hard == c] = 0.95
            conf = probs.max(axis=0)
            policy = str(self.config.single_rater_ignore_policy).strip().lower()
            if policy == "all_ignore":
                ignore = np.ones_like(hard, dtype=np.uint8)
            else:
                threshold = self.config.ignore_threshold_strict if self.config.strict_ignore else self.config.ignore_threshold_loose
                threshold = float(np.clip(threshold, self.config.ignore_threshold_min, self.config.ignore_threshold_max))
                ignore = make_ignore_mask_with_threshold(conf, threshold=threshold)
                if np.all(ignore > 0):
                    min_threshold = float(self.config.ignore_threshold_min)
                    ignore = make_ignore_mask_with_threshold(conf, threshold=min_threshold)
                    # Guarantee trainable supervision under single-rater fallback, unless explicitly all-ignore policy is requested.
                    keep = conf >= float(np.max(conf))
                    ignore = (~keep).astype(np.uint8)
            return hard, probs, ignore, {
                "effective_fusion_mode": "single_rater_fallback",
                "single_rater_ignore_policy": policy,
                "n_kept_raters": 1,
                "used_weights_per_pathologist": {kept[0]: float(used_weights[0])},
            }

        probs_raw = run_multiclass_weighted_vote(masks, used_weights, self.config.num_classes)
        probs = normalize_probs(probs_raw, epsilon=self.config.post.epsilon, use_gpu=self.config.enable_gpu)
        probs = apply_grade5_safeguard(probs, masks, reliable_flags, self.config.post.grade5_floor)
        if not np.isfinite(probs).all():
            raise RuntimeError("Non-finite probabilities detected after normalization/safeguard")

        hard = choose_hard_mask(probs)
        conf, unc = confidence_uncertainty_maps(probs, use_gpu=self.config.enable_gpu)
        apply_boundary_penalty = self.config.post.apply_boundary_penalty and (not self.config.disable_boundary_penalty)
        if apply_boundary_penalty:
            conf = boundary_disagreement_penalty(
                hard,
                masks,
                conf,
                self.config.post.boundary_dilate_px,
                agreement_clip_min=0.75,
            )

        threshold = self.config.ignore_threshold_strict if self.config.strict_ignore else self.config.ignore_threshold_loose
        tissue_mask = self._tissue_mask_from_maps(masks, hard)
        ignore, threshold = self._calibrate_ignore_threshold(conf, threshold, tissue_mask)
        hard_before_refine = hard
        hard = refine_hard_mask_classes(
            hard,
            num_classes=self.config.num_classes,
            edge_smooth_open_px=self.config.post.edge_smooth_open_px,
            edge_smooth_close_px=self.config.post.edge_smooth_close_px,
            remove_small_islands_px=self.config.post.remove_small_islands_px,
            fill_small_holes_px=self.config.post.fill_small_holes_px,
        )
        boundary_band = self._boundary_band_from_masks(masks, self.config.post.boundary_dilate_px)
        ignored_total_fraction = float((ignore > 0).mean())
        tissue_denom = float(max(1, int(tissue_mask.sum())))
        ignored_tissue_fraction = float(np.logical_and(ignore > 0, tissue_mask).sum()) / tissue_denom
        boundary_denom = float(max(1, int(boundary_band.sum())))
        ignored_boundary_fraction = float(np.logical_and(ignore > 0, boundary_band).sum()) / boundary_denom
        return hard, probs, ignore, {
            "effective_fusion_mode": "weighted",
            "n_kept_raters": len(kept),
            "used_weights_per_pathologist": {r: float(weights.get(r, 0.0)) for r in kept},
            "ignore_threshold_used": float(threshold),
            "ignored_total_fraction": ignored_total_fraction,
            "ignored_tissue_fraction": ignored_tissue_fraction,
            "ignored_boundary_fraction": ignored_boundary_fraction,
            "boundary_length_before_refine": boundary_length_4conn(hard_before_refine),
            "boundary_length_after_refine": boundary_length_4conn(hard),
            "small_component_count_before_refine": count_small_components_by_class(
                hard_before_refine,
                self.config.num_classes,
                self.config.post.remove_small_islands_px,
            ),
            "small_component_count_after_refine": count_small_components_by_class(
                hard,
                self.config.num_classes,
                self.config.post.remove_small_islands_px,
            ),
            "excessive_tissue_ignore": bool(ignored_tissue_fraction > 0.20),
        }

    def process_image(self, image_id: str, by_rater_paths: dict[str, Path]) -> dict[str, Any]:
        out_dir = self.output_root / image_id
        out_dir.mkdir(parents=True, exist_ok=True)

        sem_maps, report = self._process_raters(by_rater_paths)
        invalids = report["invalid_labels"]

        try:
            qc, statuses, weights = self._qc_and_status(
                sem_maps, invalids, report.get("available_pathologist_ids", sorted(by_rater_paths))
            )
            hard, probs, ignore, fusion_summary = self._run_consensus(sem_maps, statuses, weights)

            save_png_mask(out_dir / "consensus_hard_mask.png", hard.astype(np.uint8))
            save_npz_compressed(
                out_dir / "consensus_probs_compact.npz",
                probs=probs.astype(np.float16),
            )
            save_png_mask(out_dir / "ignore_mask.png", (ignore > 0).astype(np.uint8))

            source_image = find_source_image(image_id, self.train_dir, self.test_dir)
            report.update(
                {
                    "image_id": image_id,
                    "source_image": str(source_image) if source_image else None,
                    "available_pathologist_ids": sorted(by_rater_paths),
                    "map_decisions": {
                        "keep": sorted([k for k, v in statuses.items() if v == "keep"]),
                        "down_weight": sorted([k for k, v in statuses.items() if v == "down_weight"]),
                        "exclude": sorted([k for k, v in statuses.items() if v == "exclude"]),
                    },
                    "excluded_pathologists": sorted([k for k, v in statuses.items() if v == "exclude"]),
                    "down_weighted_pathologists": sorted([k for k, v in statuses.items() if v == "down_weight"]),
                    "any_map_excluded_or_downweighted": any(v != "keep" for v in statuses.values()),
                    "final_thresholds_used": {
                        "low_loo_dice": self.config.qc.low_loo_dice,
                        "tiny_cancer_frac": self.config.qc.tiny_cancer_frac,
                        "extreme_cancer_ratio": self.config.qc.extreme_cancer_ratio,
                        "high_fragment_count": self.config.qc.high_fragment_count,
                        "ignore_mode": "strict" if self.config.strict_ignore else "loose",
                        "ignore_confidence_threshold": (
                            self.config.ignore_threshold_strict
                            if self.config.strict_ignore
                            else self.config.ignore_threshold_loose
                        ),
                        "grade5_floor": self.config.post.grade5_floor,
                    },
                    "consensus_fusion": fusion_summary,
                    "ignored_total_fraction": fusion_summary.get("ignored_total_fraction"),
                    "ignored_tissue_fraction": fusion_summary.get("ignored_tissue_fraction"),
                    "ignored_boundary_fraction": fusion_summary.get("ignored_boundary_fraction"),
                    "boundary_length_before_refine": fusion_summary.get("boundary_length_before_refine"),
                    "boundary_length_after_refine": fusion_summary.get("boundary_length_after_refine"),
                    "small_component_count_before_refine": fusion_summary.get("small_component_count_before_refine"),
                    "small_component_count_after_refine": fusion_summary.get("small_component_count_after_refine"),
                    "excessive_tissue_ignore": fusion_summary.get("excessive_tissue_ignore", False),
                    "storage_mode": "compact_float16_npz",
                }
            )
            report.update(qc)
            report["status"] = "ok"
        except Exception as exc:
            report["status"] = "failed"
            report["error"] = str(exc)
            report["traceback"] = traceback.format_exc()

        save_json(out_dir / "qc_report.json", report)
        return report

    def process_all(self) -> dict[str, Any]:
        by_image = discover_image_ids(self.maps_root, self.config.raters)
        self.output_root.mkdir(parents=True, exist_ok=True)

        reports: dict[str, Any] = {}
        items = sorted(by_image.items())
        success = 0
        failed = 0

        if self.config.workers <= 1:
            pbar = tqdm(items, total=len(items), desc="Consensus", unit="img")
            for image_id, paths in pbar:
                rep = self.process_image(image_id, paths)
                reports[image_id] = rep
                if rep.get("status") == "ok":
                    success += 1
                else:
                    failed += 1
                pbar.set_postfix(image=image_id, ok=success, fail=failed)
        else:
            # Parallelize across images in CPU mode.
            with ProcessPoolExecutor(max_workers=self.config.workers) as ex:
                futures = {
                    ex.submit(_process_image_worker, self.config, image_id, {k: str(v) for k, v in paths.items()}): image_id
                    for image_id, paths in items
                }
                pbar = tqdm(total=len(items), desc=f"Consensus ({self.config.workers} workers)", unit="img")
                for fut in as_completed(futures):
                    image_id = futures[fut]
                    rep = fut.result()
                    reports[image_id] = rep
                    if rep.get("status") == "ok":
                        success += 1
                    else:
                        failed += 1
                    pbar.update(1)
                    pbar.set_postfix(last=image_id, ok=success, fail=failed)
                pbar.close()

        meta = run_metadata(self.config, gpu_info(), try_git_commit_hash(Path.cwd()))
        meta["num_images"] = len(by_image)
        meta["num_success"] = sum(1 for r in reports.values() if r.get("status") == "ok")
        meta["num_failed"] = sum(1 for r in reports.values() if r.get("status") != "ok")
        save_json(self.output_root / "run_metadata.json", meta)
        return {"reports": reports, "metadata": meta}


def _process_image_worker(config: ConsensusConfig, image_id: str, path_map: dict[str, str]) -> dict[str, Any]:
    builder = ConsensusMaskBuilder(config)
    return builder.process_image(image_id, {k: Path(v) for k, v in path_map.items()})

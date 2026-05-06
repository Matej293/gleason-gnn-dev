from __future__ import annotations

import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from . import metrics
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
    boundary_disagreement_penalty,
    choose_hard_mask,
    confidence_uncertainty_maps,
    gpu_info,
    make_ignore_mask,
    normalize_probs,
)
from .qc import QCConfig, class_stats, decide_rater_status, fragmentation_stats, qc_flags_for_rater
from .staple import StapleConfig, run_multiclass_one_vs_rest_staple


@dataclass
class ConsensusConfig:
    dataset_root: str = "data"
    output_root: str = "data/consensus"
    raters: list[str] = field(default_factory=lambda: ["p1", "p2", "p3", "p4", "p5", "p6"])
    num_classes: int = NUM_CLASSES
    enable_gpu: bool = True
    strict_ignore: bool = False
    workers: int = 1

    qc: QCConfig = field(default_factory=QCConfig)
    staple: StapleConfig = field(default_factory=StapleConfig)
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

    def _run_consensus(self, sem_maps: dict[str, np.ndarray], statuses: dict[str, str], weights: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        kept = [r for r, s in statuses.items() if s != "exclude"]
        if not kept:
            raise RuntimeError("No valid raters after QC")

        masks = [sem_maps[r] for r in kept]
        reliable_flags = [weights[r] >= 0.7 for r in kept]

        if len(masks) == 1:
            hard = masks[0].astype(np.uint8)
            probs = np.full((self.config.num_classes, *hard.shape), 0.05 / (self.config.num_classes - 1), dtype=np.float32)
            for c in range(self.config.num_classes):
                probs[c][hard == c] = 0.95
            ignore = np.ones_like(hard, dtype=np.uint8)
            return hard, probs, ignore

        probs_raw = run_multiclass_one_vs_rest_staple(masks, self.config.num_classes, self.config.staple)
        probs = normalize_probs(probs_raw, epsilon=self.config.post.epsilon, use_gpu=self.config.enable_gpu)
        probs = apply_grade5_safeguard(probs, masks, reliable_flags, self.config.post.grade5_floor)
        if not np.isfinite(probs).all():
            raise RuntimeError("Non-finite probabilities detected after normalization/safeguard")

        hard = choose_hard_mask(probs)
        conf, unc = confidence_uncertainty_maps(probs, use_gpu=self.config.enable_gpu)
        if self.config.post.apply_boundary_penalty:
            conf = boundary_disagreement_penalty(hard, masks, conf, self.config.post.boundary_dilate_px)

        ignore = make_ignore_mask(conf, n_raters=len(masks), strict=self.config.strict_ignore)
        return hard, probs, ignore

    def process_image(self, image_id: str, by_rater_paths: dict[str, Path]) -> dict[str, Any]:
        out_dir = self.output_root / image_id
        out_dir.mkdir(parents=True, exist_ok=True)

        sem_maps, report = self._process_raters(by_rater_paths)
        invalids = report["invalid_labels"]

        try:
            qc, statuses, weights = self._qc_and_status(
                sem_maps, invalids, report.get("available_pathologist_ids", sorted(by_rater_paths))
            )
            hard, probs, ignore = self._run_consensus(sem_maps, statuses, weights)

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
                            self.config.post.ignore_conf_threshold_strict
                            if self.config.strict_ignore
                            else self.config.post.ignore_conf_threshold_loose
                        ),
                        "grade5_floor": self.config.post.grade5_floor,
                    },
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
            # STAPLE is CPU-bound; parallelize across images when not using GPU.
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

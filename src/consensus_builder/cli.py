from __future__ import annotations

import argparse

from .pipeline import ConsensusConfig, ConsensusMaskBuilder


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Gleason2019 consensus mask builder (STAPLE + QC)")
    p.add_argument("--dataset-root", default="data", help="Dataset root with Maps*_T and image folders")
    p.add_argument("--output-root", default="data/consensus", help="Output root")
    p.add_argument("--disable-gpu", action="store_true", help="Disable optional GPU acceleration")
    p.add_argument("--strict-ignore", action="store_true", help="Use stricter ignore mask threshold")
    p.add_argument(
        "--target-ignore-tissue-frac",
        type=float,
        default=0.05,
        help="Target maximum ignored fraction over inferred tissue pixels.",
    )
    p.add_argument(
        "--target-ignore-total-frac",
        type=float,
        default=0.12,
        help="Target maximum ignored fraction over all pixels.",
    )
    p.add_argument("--ignore-threshold-min", type=float, default=0.05, help="Minimum ignore confidence threshold.")
    p.add_argument("--ignore-threshold-max", type=float, default=0.35, help="Maximum ignore confidence threshold.")
    p.add_argument(
        "--auto-calibrate-ignore-threshold",
        dest="auto_calibrate_ignore_threshold",
        action="store_true",
        help="Auto-calibrate ignore threshold per image to reduce ignored tissue.",
    )
    p.add_argument(
        "--disable-auto-calibrate-ignore-threshold",
        dest="auto_calibrate_ignore_threshold",
        action="store_false",
        help="Disable per-image auto-calibration of ignore threshold.",
    )
    p.set_defaults(auto_calibrate_ignore_threshold=True)
    p.add_argument("--disable-boundary-penalty", action="store_true", help="Disable boundary disagreement confidence penalty.")
    p.add_argument("--boundary-dilate-px", type=int, default=1, help="Boundary dilation size in pixels.")
    p.add_argument("--edge-smooth-open-px", type=int, default=0, help="Opening iterations for class-wise hard-mask cleanup.")
    p.add_argument("--edge-smooth-close-px", type=int, default=1, help="Closing iterations for class-wise hard-mask cleanup.")
    p.add_argument("--remove-small-islands-px", type=int, default=64, help="Remove class components smaller than this size.")
    p.add_argument("--fill-small-holes-px", type=int, default=64, help="Fill class holes smaller than this size.")
    p.add_argument("--workers", type=int, default=1, help="Parallel workers across images (CPU mode)")
    p.add_argument("--low-loo-dice", type=float, default=0.35, help="QC threshold for low leave-one-out dice")
    p.add_argument("--grade5-floor", type=float, default=0.08, help="Minimum grade-5 prob floor when safeguarded")
    p.add_argument(
        "--consensus-fusion-mode",
        type=str,
        default="staple_unweighted",
        choices=["staple_unweighted", "weighted"],
        help="Consensus fusion mode.",
    )
    p.add_argument("--ignore-threshold-loose", type=float, default=0.30, help="Loose ignore confidence threshold")
    p.add_argument("--ignore-threshold-strict", type=float, default=0.50, help="Strict ignore confidence threshold")
    p.add_argument(
        "--single-rater-ignore-policy",
        type=str,
        default="confidence_mask",
        choices=["all_ignore", "confidence_mask"],
        help="Single-rater ignore behavior.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = ConsensusConfig(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        enable_gpu=not args.disable_gpu,
        strict_ignore=args.strict_ignore,
        workers=max(1, args.workers),
        consensus_fusion_mode=args.consensus_fusion_mode,
        ignore_threshold_loose=float(args.ignore_threshold_loose),
        ignore_threshold_strict=float(args.ignore_threshold_strict),
        target_ignore_tissue_frac=float(args.target_ignore_tissue_frac),
        target_ignore_total_frac=float(args.target_ignore_total_frac),
        ignore_threshold_min=float(args.ignore_threshold_min),
        ignore_threshold_max=float(args.ignore_threshold_max),
        auto_calibrate_ignore_threshold=bool(args.auto_calibrate_ignore_threshold),
        disable_boundary_penalty=bool(args.disable_boundary_penalty),
        single_rater_ignore_policy=args.single_rater_ignore_policy,
    )
    cfg.qc.low_loo_dice = args.low_loo_dice
    cfg.post.grade5_floor = args.grade5_floor
    cfg.post.boundary_dilate_px = int(max(0, args.boundary_dilate_px))
    cfg.post.edge_smooth_open_px = int(max(0, args.edge_smooth_open_px))
    cfg.post.edge_smooth_close_px = int(max(0, args.edge_smooth_close_px))
    cfg.post.remove_small_islands_px = int(max(0, args.remove_small_islands_px))
    cfg.post.fill_small_holes_px = int(max(0, args.fill_small_holes_px))

    if cfg.enable_gpu and cfg.workers > 1:
        print("GPU enabled; forcing workers=1 to avoid CUDA multiprocessing contention")
        cfg.workers = 1

    builder = ConsensusMaskBuilder(cfg)
    result = builder.process_all()
    m = result["metadata"]
    print(f"Done. Images={m['num_images']} success={m['num_success']} failed={m['num_failed']}")


if __name__ == "__main__":
    main()

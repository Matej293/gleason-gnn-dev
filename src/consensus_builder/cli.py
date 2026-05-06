from __future__ import annotations

import argparse

from .pipeline import ConsensusConfig, ConsensusMaskBuilder


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Gleason2019 consensus mask builder (STAPLE + QC)")
    p.add_argument("--dataset-root", default="data", help="Dataset root with Maps*_T and image folders")
    p.add_argument("--output-root", default="data/consensus", help="Output root")
    p.add_argument("--disable-gpu", action="store_true", help="Disable optional GPU acceleration")
    p.add_argument("--strict-ignore", action="store_true", help="Use stricter ignore mask threshold")
    p.add_argument("--workers", type=int, default=1, help="Parallel workers across images (CPU mode)")
    p.add_argument("--low-loo-dice", type=float, default=0.35, help="QC threshold for low leave-one-out dice")
    p.add_argument("--grade5-floor", type=float, default=0.08, help="Minimum grade-5 prob floor when safeguarded")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = ConsensusConfig(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        enable_gpu=not args.disable_gpu,
        strict_ignore=args.strict_ignore,
        workers=max(1, args.workers),
    )
    cfg.qc.low_loo_dice = args.low_loo_dice
    cfg.post.grade5_floor = args.grade5_floor

    if cfg.enable_gpu and cfg.workers > 1:
        print("GPU enabled; forcing workers=1 to avoid CUDA multiprocessing contention")
        cfg.workers = 1

    builder = ConsensusMaskBuilder(cfg)
    result = builder.process_all()
    m = result["metadata"]
    print(f"Done. Images={m['num_images']} success={m['num_success']} failed={m['num_failed']}")


if __name__ == "__main__":
    main()

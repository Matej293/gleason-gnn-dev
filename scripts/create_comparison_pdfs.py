#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


IMAGE_EXTS = ("*.png", "*.jpg", "*.jpeg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create high-resolution PDFs for baseline-comparison visualization runs. "
            "By default, each PDF is named with its run directory (timestamp) to keep association clear."
        )
    )
    parser.add_argument(
        "--comparison-dir",
        required=True,
        type=str,
        help="Baseline comparison root directory (e.g. outputs/gnn_runs/<ts>_baseline_comparison)",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        type=str,
        help="Specific viz run directory under viz_<split> (e.g. .../viz_test/20260512_134003).",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Split folder to scan in bulk mode: viz_<split>.",
    )
    parser.add_argument(
        "--include-cases",
        dest="include_cases",
        action="store_true",
        help="Include images from the cases/ subdirectory.",
    )
    parser.add_argument(
        "--no-include-cases",
        dest="include_cases",
        action="store_false",
        help="Exclude images from the cases/ subdirectory.",
    )
    parser.set_defaults(include_cases=True)
    parser.add_argument(
        "--dpi",
        default=100,
        type=int,
        help="PDF rendering DPI used with native pixel sizing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing target PDF if present.",
    )
    return parser.parse_args()


def _collect_images(run_dir: Path, include_cases: bool) -> list[Path]:
    root_imgs: list[Path] = []
    for ext in IMAGE_EXTS:
        root_imgs.extend(sorted(p for p in run_dir.glob(ext) if p.is_file()))

    case_imgs: list[Path] = []
    if include_cases:
        cases_dir = run_dir / "cases"
        for ext in IMAGE_EXTS:
            case_imgs.extend(sorted(p for p in cases_dir.glob(ext) if p.is_file()))

    return root_imgs + case_imgs


def _target_pdf_path(run_dir: Path) -> Path:
    return run_dir / f"{run_dir.name}_comparison_run_images_fullres.pdf"


def _write_pdf(images: list[Path], out_pdf: Path, dpi: int) -> None:
    with PdfPages(out_pdf) as pdf:
        for image_path in images:
            arr = mpimg.imread(image_path)
            h, w = arr.shape[:2]
            fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(arr)
            ax.axis("off")
            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)


def _resolve_run_dirs(comparison_dir: Path, split: str, run_dir_arg: str | None) -> list[Path]:
    if run_dir_arg is not None:
        run_dir = Path(run_dir_arg)
        if not run_dir.exists() or not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory does not exist or is not a directory: {run_dir}")
        return [run_dir]

    viz_root = comparison_dir / f"viz_{split}"
    if not viz_root.exists() or not viz_root.is_dir():
        raise FileNotFoundError(f"Visualization root not found: {viz_root}")

    run_dirs = sorted([p for p in viz_root.iterdir() if p.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under: {viz_root}")
    return run_dirs


def main() -> None:
    args = parse_args()
    comparison_dir = Path(args.comparison_dir)
    if not comparison_dir.exists() or not comparison_dir.is_dir():
        raise FileNotFoundError(f"Comparison directory does not exist or is not a directory: {comparison_dir}")

    run_dirs = _resolve_run_dirs(comparison_dir, args.split, args.run_dir)

    built = 0
    skipped = 0
    for run_dir in run_dirs:
        images = _collect_images(run_dir, include_cases=args.include_cases)
        if not images:
            print(f"[skip] No images found for run: {run_dir}")
            skipped += 1
            continue

        out_pdf = _target_pdf_path(run_dir)
        if out_pdf.exists() and not args.overwrite:
            print(f"[skip] PDF exists (use --overwrite): {out_pdf}")
            skipped += 1
            continue

        _write_pdf(images, out_pdf, dpi=args.dpi)
        print(f"[ok] {run_dir.name} -> {out_pdf} (pages={len(images)})")
        built += 1

    print(f"Done. Built={built}, Skipped={skipped}, TotalRuns={len(run_dirs)}")


if __name__ == "__main__":
    main()

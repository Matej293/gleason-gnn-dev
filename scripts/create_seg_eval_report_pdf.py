#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


KEY_METRICS = [
    "macro_dice",
    "miou",
    "sensitivity",
    "precision",
    "dice_benign",
    "dice_g3",
    "dice_g4",
    "dice_g5",
    "iou_benign",
    "iou_g3",
    "iou_g4",
    "iou_g5",
    "num_test_samples",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create eval-only segmentation comparison PDF for two runs.")
    p.add_argument("--run-a", required=True, type=str, help="First run directory.")
    p.add_argument("--run-b", required=True, type=str, help="Second run directory.")
    p.add_argument("--out", default=None, type=str, help="Output PDF path.")
    p.add_argument("--dpi", default=120, type=int)
    return p.parse_args()


def _load_eval_summary(run_dir: Path) -> dict:
    path = run_dir / "evaluation_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing evaluation summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _eval_images(run_dir: Path) -> dict[str, Path]:
    eval_dir = run_dir / "eval_viz"
    if not eval_dir.exists():
        return {}
    out: dict[str, Path] = {}
    for p in sorted(eval_dir.glob("*.png")):
        # "001_slide001_core009.png" -> "slide001_core009"
        stem = p.stem
        case_id = stem.split("_", 1)[1] if "_" in stem else stem
        out[case_id] = p
    return out


def _fmt(v: object) -> str:
    if isinstance(v, (int, float)):
        if abs(float(v)) < 1e-6:
            return f"{float(v):.2e}"
        return f"{float(v):.6f}"
    return str(v)


def _summary_page(pdf: PdfPages, run_a: Path, run_b: Path, s_a: dict, s_b: dict) -> None:
    a_agg = s_a.get("aggregate", {})
    b_agg = s_b.get("aggregate", {})

    fig = plt.figure(figsize=(11.69, 8.27), dpi=120)  # A4 landscape
    ax = fig.add_axes([0.04, 0.04, 0.92, 0.92])
    ax.axis("off")

    lines = [
        "Segmentation Evaluation Report (eval-only)",
        "",
        f"Run A: {run_a.name}",
        f"Checkpoint A: {Path(str(s_a.get('checkpoint', 'N/A'))).name}",
        f"Run B: {run_b.name}",
        f"Checkpoint B: {Path(str(s_b.get('checkpoint', 'N/A'))).name}",
        "",
        "Metric                          Run A            Run B            Delta (B-A)",
        "-" * 78,
    ]

    for k in KEY_METRICS:
        av = a_agg.get(k)
        bv = b_agg.get(k)
        delta = None
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            delta = float(bv) - float(av)
        lines.append(
            f"{k:<30} {_fmt(av):>14} {_fmt(bv):>14} {_fmt(delta) if delta is not None else 'N/A':>14}"
        )

    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=10)
    pdf.savefig(fig)
    plt.close(fig)


def _panel_pages(pdf: PdfPages, run_a: Path, run_b: Path, imgs_a: dict[str, Path], imgs_b: dict[str, Path], dpi: int) -> None:
    common = sorted(set(imgs_a.keys()) & set(imgs_b.keys()))
    for case_id in common:
        pa = imgs_a[case_id]
        pb = imgs_b[case_id]
        a = mpimg.imread(pa)
        b = mpimg.imread(pb)

        # Keep large readable pages
        fig = plt.figure(figsize=(16, 8), dpi=dpi)
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax1.imshow(a)
        ax2.imshow(b)
        ax1.axis("off")
        ax2.axis("off")
        ax1.set_title(f"{run_a.name} | {pa.name}", fontsize=9)
        ax2.set_title(f"{run_b.name} | {pb.name}", fontsize=9)
        fig.suptitle(case_id, fontsize=12)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    run_a = Path(args.run_a).resolve()
    run_b = Path(args.run_b).resolve()
    s_a = _load_eval_summary(run_a)
    s_b = _load_eval_summary(run_b)

    imgs_a = _eval_images(run_a)
    imgs_b = _eval_images(run_b)

    default_out = run_b / f"eval_report_{run_a.name}_vs_{run_b.name}.pdf"
    out_pdf = Path(args.out).resolve() if args.out else default_out
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_pdf) as pdf:
        _summary_page(pdf, run_a, run_b, s_a, s_b)
        _panel_pages(pdf, run_a, run_b, imgs_a, imgs_b, dpi=args.dpi)

    print(f"Saved eval report PDF: {out_pdf}")
    print(f"Common eval panels: {len(set(imgs_a.keys()) & set(imgs_b.keys()))}")


if __name__ == "__main__":
    main()

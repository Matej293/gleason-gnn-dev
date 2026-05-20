from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def visualize_example(
    image_path: str | Path,
    rater_masks: dict[str, np.ndarray],
    majority_vote: np.ndarray,
    staple_hard: np.ndarray,
    uncertainty: np.ndarray,
    disagreement: np.ndarray,
    out_path: str | Path,
) -> None:
    image = np.array(Image.open(image_path))
    raters = sorted(rater_masks)

    cols = max(3, len(raters) + 2)
    fig, axes = plt.subplots(2, cols, figsize=(4 * cols, 8))

    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    for i, r in enumerate(raters, start=1):
        axes[0, i].imshow(rater_masks[r], vmin=0, vmax=3, cmap="viridis")
        axes[0, i].set_title(r)
        axes[0, i].axis("off")

    for i in range(len(raters) + 1, cols):
        axes[0, i].axis("off")

    axes[1, 0].imshow(majority_vote, vmin=0, vmax=3, cmap="viridis")
    axes[1, 0].set_title("Majority Vote")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(staple_hard, vmin=0, vmax=3, cmap="viridis")
    axes[1, 1].set_title("STAPLE Hard")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(uncertainty, vmin=0, vmax=1, cmap="magma")
    axes[1, 2].set_title("Uncertainty")
    axes[1, 2].axis("off")

    if cols > 3:
        axes[1, 3].imshow(disagreement, vmin=0, vmax=1, cmap="inferno")
        axes[1, 3].set_title("Disagreement")
        axes[1, 3].axis("off")

    for i in range(4, cols):
        axes[1, i].axis("off")

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

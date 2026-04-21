from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


def save_comparison_panel(
    original: Image.Image,
    mask: Image.Image,
    corrupted: Image.Image,
    prediction: Image.Image,
    out_path: Path,
    title: str,
) -> None:
    """Save a 4-column qualitative comparison panel."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Mask")
    axes[1].axis("off")

    axes[2].imshow(corrupted)
    axes[2].set_title("Corrupted")
    axes[2].axis("off")

    axes[3].imshow(prediction)
    axes[3].set_title("Inpainted")
    axes[3].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

"""Dataset utilities for loading images and generating inpainting masks.

Provides functions to:
- List valid image files from a directory
- Generate center-rectangle and irregular brush-stroke masks
- Apply masks to create corrupted (masked-out) images
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageDraw


ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def list_images(image_dir: Path) -> List[Path]:
    """Return all valid image files from the directory (non-recursive)."""
    if not image_dir.exists():
        return []
    return sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in ALLOWED_SUFFIXES and p.is_file()]
    )


def generate_center_mask(width: int, height: int, ratio: float = 0.35) -> Image.Image:
    """Generate a white center rectangle on black background.

    White pixels (255) indicate the region to be inpainted.
    """
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    # Compute centered bounding box based on the given ratio of image dimensions
    box_w = int(width * ratio)
    box_h = int(height * ratio)
    left = (width - box_w) // 2
    top = (height - box_h) // 2
    draw.rectangle([left, top, left + box_w, top + box_h], fill=255)
    return mask


def generate_irregular_mask(width: int, height: int, strokes: int = 6, seed: int | None = None) -> Image.Image:
    """Generate irregular brush-like mask using random strokes.

    Draws multiple random line segments with varying widths to simulate
    free-form masks commonly used in inpainting benchmarks.

    Pass a fixed *seed* to produce the same mask across runs.
    """
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    rng = np.random.default_rng(seed)

    for _ in range(strokes):
        x1, y1 = int(rng.integers(0, width)), int(rng.integers(0, height))
        x2, y2 = int(rng.integers(0, width)), int(rng.integers(0, height))
        line_width = int(rng.integers(18, 56))
        draw.line((x1, y1, x2, y2), fill=255, width=line_width)
    return mask


def apply_mask(image: Image.Image, mask: Image.Image, fill_value: int = 0) -> Image.Image:
    """Create corrupted image where masked area is replaced by fill_value.

    Pixels where the mask exceeds 127 (i.e. white region) are set to
    fill_value, simulating the missing/damaged region for inpainting.
    """
    image_np = np.array(image.convert("RGB"), dtype=np.uint8)
    mask_np = np.array(mask.convert("L"), dtype=np.uint8)
    corrupted = image_np.copy()
    corrupted[mask_np > 127] = fill_value
    return Image.fromarray(corrupted)

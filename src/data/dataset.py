from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageDraw


ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass
class SampleItem:
    image_path: Path
    image: Image.Image
    mask: Image.Image
    corrupted: Image.Image


def list_images(image_dir: Path) -> List[Path]:
    """Return all valid image files from the directory (non-recursive)."""
    if not image_dir.exists():
        return []
    return sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in ALLOWED_SUFFIXES and p.is_file()]
    )


def generate_center_mask(width: int, height: int, ratio: float = 0.35) -> Image.Image:
    """Generate a white center rectangle on black background."""
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    box_w = int(width * ratio)
    box_h = int(height * ratio)
    left = (width - box_w) // 2
    top = (height - box_h) // 2
    draw.rectangle([left, top, left + box_w, top + box_h], fill=255)
    return mask


def generate_irregular_mask(width: int, height: int, strokes: int = 6) -> Image.Image:
    """Generate irregular brush-like mask using random strokes."""
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    rng = np.random.default_rng()

    for _ in range(strokes):
        x1, y1 = int(rng.integers(0, width)), int(rng.integers(0, height))
        x2, y2 = int(rng.integers(0, width)), int(rng.integers(0, height))
        line_width = int(rng.integers(18, 56))
        draw.line((x1, y1, x2, y2), fill=255, width=line_width)
    return mask


def apply_mask(image: Image.Image, mask: Image.Image, fill_value: int = 0) -> Image.Image:
    """Create corrupted image where masked area is replaced by fill_value."""
    image_np = np.array(image.convert("RGB"), dtype=np.uint8)
    mask_np = np.array(mask.convert("L"), dtype=np.uint8)
    corrupted = image_np.copy()
    corrupted[mask_np > 127] = fill_value
    return Image.fromarray(corrupted)


def load_samples(
    image_dir: Path,
    image_size: Tuple[int, int],
    mask_type: str,
) -> Iterable[SampleItem]:
    """Yield loaded samples with generated masks and corrupted images."""
    files = list_images(image_dir)
    for image_path in files:
        image = Image.open(image_path).convert("RGB").resize(image_size)
        width, height = image.size

        if mask_type == "center":
            mask = generate_center_mask(width, height)
        elif mask_type == "irregular":
            mask = generate_irregular_mask(width, height)
        else:
            raise ValueError(f"Unsupported mask_type: {mask_type}")

        corrupted = apply_mask(image, mask)
        yield SampleItem(image_path=image_path, image=image, mask=mask, corrupted=corrupted)

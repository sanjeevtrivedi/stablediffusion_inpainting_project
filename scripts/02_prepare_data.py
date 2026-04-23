"""
Prepare dataset: split images and generate masks.

Usage:
    python scripts/02_prepare_data.py --data-dir data/samples --output-dir data/splits
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from src.data.dataset import generate_center_mask, generate_irregular_mask, list_images
from PIL import Image


MASK_TYPES = ["center", "irregular"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare dataset splits and pre-save masks")
    parser.add_argument("--data-dir", type=Path, default=Path("data/samples"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_splits(files: list, train_ratio: float, val_ratio: float, seed: int) -> tuple:
    """Split files into train, val, test."""
    rng = random.Random(seed)
    shuffled = files[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


def main() -> None:
    args = parse_args()
    files = list_images(args.data_dir)
    if not files:
        raise RuntimeError(f"No images found in {args.data_dir}")

    train, val, test = make_splits(files, args.train_ratio, args.val_ratio, args.seed)
    splits = {"train": train, "val": val, "test": test}

    manifest_rows = []

    for split_name, split_files in splits.items():
        for mask_type in MASK_TYPES:
            # Create output dirs
            img_dir = args.output_dir / split_name / "images"
            mask_dir = args.output_dir / split_name / "masks" / mask_type
            img_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)

            for src_path in split_files:
                image = Image.open(src_path).convert("RGB").resize((args.image_size, args.image_size))
                dest_img = img_dir / src_path.name
                if not dest_img.exists():
                    image.save(dest_img)

                # Generate and save mask
                if mask_type == "center":
                    mask = generate_center_mask(args.image_size, args.image_size)
                else:
                    mask = generate_irregular_mask(args.image_size, args.image_size)

                mask_path = mask_dir / src_path.name
                mask.save(mask_path)

                manifest_rows.append(
                    {
                        "split": split_name,
                        "mask_type": mask_type,
                        "image": str(dest_img),
                        "mask": str(mask_path),
                    }
                )

    # Write manifest CSV
    manifest_path = args.output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "mask_type", "image", "mask"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nDataset prepared in: {args.output_dir}")
    print(f"  train: {len(train)} images")
    print(f"  val:   {len(val)} images")
    print(f"  test:  {len(test)} images")
    print(f"  mask types: {MASK_TYPES}")
    print(f"  manifest: {manifest_path}")


if __name__ == "__main__":
    main()

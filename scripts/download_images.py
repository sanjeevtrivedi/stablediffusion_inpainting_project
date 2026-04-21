"""Download sample images from picsum.photos.

Features:
- Configurable number of images to download (`--count`, default 10)
- Duplicate protection by filename (existing names are skipped)
- Predictable naming: sample_01.jpg, sample_02.jpg, ...

Example:
    python scripts/download_images.py --count 10 --output-dir data/samples
"""
from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download images from picsum.photos")
    parser.add_argument("--count", type=int, default=10, help="Number of new images to download")
    parser.add_argument("--output-dir", type=Path, default=Path("data/samples"), help="Output folder")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=100,
        help="Numeric offset used in picsum seed",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="Starting filename index (sample_XX.jpg)",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.2,
        help="Small delay between requests",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be greater than 0")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    idx = max(1, args.start_index)

    while downloaded < args.count:
        filename = f"sample_{idx:02d}.jpg"
        dest = args.output_dir / filename

        # Do not overwrite existing filenames.
        if dest.exists():
            skipped += 1
            print(f"Skipping existing: {filename}")
            idx += 1
            continue

        url = f"https://picsum.photos/seed/{args.seed_offset + idx}/{args.width}/{args.height}"
        urllib.request.urlretrieve(url, dest)
        downloaded += 1
        print(f"Downloaded {filename} ({downloaded}/{args.count})")

        idx += 1
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    print("\nDownload complete")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Downloaded: {downloaded}")
    print(f"  Skipped existing names: {skipped}")


if __name__ == "__main__":
    main()

"""
Download sample images from picsum.photos.

Usage:
    python scripts/01_download_images.py
"""
from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download images from picsum.photos")
    # Number of images to download
    parser.add_argument("--count", type=int, default=3, help="Number of images to download")
    # Directory where images will be saved
    parser.add_argument("--output-dir", type=Path, default=Path("data/samples"), help="Directory to save downloaded images")
    # Width of each image in pixels
    parser.add_argument("--width", type=int, default=512, help="Width of images in pixels")
    # Height of each image in pixels
    parser.add_argument("--height", type=int, default=512, help="Height of images in pixels")
    # Seed offset for image randomness (ensures unique images)
    parser.add_argument("--seed-offset", type=int, default=100, help="Seed offset for image randomness")
    # Starting index for image filenames
    parser.add_argument("--start-index", type=int, default=1, help="Starting index for image filenames")
    # Seconds to sleep between downloads (to avoid rate limiting)
    parser.add_argument("--sleep-seconds", type=float, default=0.2, help="Seconds to sleep between downloads")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Validate that the count is positive
    if args.count <= 0:
        raise ValueError("--count must be > 0")

    # Ensure the output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0  # Number of images successfully downloaded
    skipped = 0     # Number of images skipped because they already exist
    idx = max(1, args.start_index)  # Start index for naming images

    # Loop until the required number of images are downloaded
    while downloaded < args.count:
        filename = f"sample_{idx:02d}.jpg"  # Construct filename
        dest = args.output_dir / filename    # Full path for the image

        # Skip download if file already exists
        if dest.exists():
            skipped += 1
            print(f"Skipping existing: {filename}")
            idx += 1
            continue

        # Construct the URL for the image with unique seed, width, and height
        url = f"https://picsum.photos/seed/{args.seed_offset + idx}/{args.width}/{args.height}"
        try:
            # Download the image from the URL
            urllib.request.urlretrieve(url, dest)
            downloaded += 1
            print(f"Downloaded {filename} ({downloaded}/{args.count})")
            idx += 1
            # Sleep between downloads to avoid rate limiting
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
        except Exception as e:
            # Handle download errors gracefully
            print(f"Error downloading {filename}: {e}")
            idx += 1

    print("\nDownload complete")
    print(f"  Downloaded: {downloaded}")
    print(f"  Skipped existing: {skipped}")
    print(f"  Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()

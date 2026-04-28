"""
Aerial Road Inpainting Pipeline
================================
Removes vehicles and their shadows from aerial imagery.

Usage:
    python main.py --detect-only   # detect vehicles, save output_detections.png, stop
    python main.py                 # full pipeline: detect + inpaint + metrics
"""

import argparse
import logging
import os
import ssl

import certifi
import numpy as np
from PIL import Image, ImageDraw

import config
import detection
import inpainting
import masking
import metrics

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def save_detections(aerial: Image.Image, dets: list[dict], path: str) -> None:
    img  = aerial.copy()
    draw = ImageDraw.Draw(img)
    for d in dets:
        draw.rectangle([d["x1"], d["y1"], d["x2"], d["y2"]], outline=(255, 0, 0), width=3)
        draw.rectangle([d["x1"], d["y1"] - 18, d["x1"] + 80, d["y1"]], fill=(255, 0, 0))
        draw.text((d["x1"] + 2, d["y1"] - 16),
                  f"{d['label']} {d['conf']:.2f}", fill=(255, 255, 255))
    img.save(path)
    log.info("Saved: %s", path)


def save_mask_overlay(aerial: Image.Image, mask: np.ndarray, path: str) -> None:
    arr = np.array(aerial).copy()
    arr[mask > 0] = [255, 0, 0]
    Image.fromarray(arr).save(path)
    log.info("Saved: %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aerial Road Inpainting Pipeline")
    parser.add_argument("--detect-only", action="store_true",
                        help="Run detection only and exit. Review output_detections.png before inpainting.")
    args = parser.parse_args()

    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)

    # Load image
    log.info("Loading: %s", config.AERIAL_IMAGE)
    aerial = Image.open(config.AERIAL_IMAGE).convert("RGB")
    log.info("Size: %s", aerial.size)

    # Detect vehicles
    dets = detection.detect_vehicles(aerial)
    if not dets:
        log.error("No vehicles detected.")
        return

    save_detections(aerial, dets,
                    os.path.join(config.ARTIFACTS_DIR, "output_detections.png"))

    if args.detect_only:
        log.info("Detection-only mode complete.")
        log.info("Review output_detections.png — if correct, run without --detect-only to inpaint.")
        return

    # Build mask
    mask = masking.build_combined_mask(aerial, dets)

    save_mask_overlay(aerial, mask,
                      os.path.join(config.ARTIFACTS_DIR, "output_mask.png"))

    # Inpaint
    inpainted = inpainting.inpaint(aerial, mask)
    inpainted.save(os.path.join(config.ARTIFACTS_DIR, "output_inpainted.png"))
    log.info("Saved: output_inpainted.png")

    # Metrics
    metrics.compute_and_save(
        aerial, inpainted, mask,
        os.path.join(config.ARTIFACTS_DIR, "metrics.txt")
    )

    log.info("Pipeline complete. Outputs in: %s", config.ARTIFACTS_DIR)


if __name__ == "__main__":
    main()

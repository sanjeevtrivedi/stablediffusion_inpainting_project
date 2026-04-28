"""
Mask building for the aerial road inpainting pipeline.

Builds binary masks from:
  - Vehicle bounding boxes (dilated)
  - Cast shadows within expanded vehicle zones
"""

import logging

import cv2
import numpy as np
from PIL import Image

import config

log = logging.getLogger(__name__)


def build_vehicle_mask(detections: list[dict], image_shape: tuple) -> np.ndarray:
    H, W = image_shape
    mask = np.zeros((H, W), dtype=np.uint8)
    for d in detections:
        mask[d["y1"]:d["y2"], d["x1"]:d["x2"]] = 255
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.VEHICLE_DILATE_PX * 2 + 1,) * 2)
    mask = cv2.dilate(mask, k)
    log.info("Vehicle mask: %d px", (mask > 0).sum())
    return mask


def build_shadow_mask(aerial: Image.Image, detections: list[dict]) -> np.ndarray:
    """
    Finds dark shadow pixels strictly within the expanded zone of each vehicle.
    Never touches pixels outside vehicle-adjacent areas.
    """
    arr  = np.array(aerial)
    H, W = arr.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)

    for d in detections:
        x1 = max(0, d["x1"] - config.SHADOW_EXPAND_PX)
        y1 = max(0, d["y1"] - config.SHADOW_EXPAND_PX)
        x2 = min(W, d["x2"] + config.SHADOW_EXPAND_PX)
        y2 = min(H, d["y2"] + config.SHADOW_EXPAND_PX)
        zone = arr[y1:y2, x1:x2]
        shadow = (
            (zone[:, :, 0] < config.SHADOW_DARK_THR) &
            (zone[:, :, 1] < config.SHADOW_DARK_THR) &
            (zone[:, :, 2] < config.SHADOW_DARK_THR)
        )
        mask[y1:y2, x1:x2][shadow] = 255

    if mask.sum() > 0:
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.SHADOW_DILATE_PX * 2 + 1,) * 2)
        mask = cv2.dilate(mask, k)

    log.info("Shadow mask: %d px", (mask > 0).sum())
    return mask


def build_combined_mask(aerial: Image.Image, detections: list[dict]) -> np.ndarray:
    H, W = np.array(aerial).shape[:2]
    vehicle_mask = build_vehicle_mask(detections, (H, W))
    shadow_mask  = build_shadow_mask(aerial, detections)
    combined     = cv2.bitwise_or(vehicle_mask, shadow_mask)
    pct = (combined > 0).sum() / combined.size * 100
    log.info("Combined mask: %.3f%% of image", pct)
    return combined

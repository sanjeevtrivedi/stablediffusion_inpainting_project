"""
Vehicle detection using multi-scale tiled YOLOv8.

Runs DETECT_PASSES confidence levels and merges via NMS.
Size and aspect ratio filters remove false positives.
"""

import logging

import numpy as np
from PIL import Image
from ultralytics import YOLO

import config

log = logging.getLogger(__name__)


def tile_coords(dim: int, tile: int, overlap: int) -> list[int]:
    step = tile - overlap
    coords = list(range(0, dim - tile + 1, step))
    if not coords or coords[-1] + tile < dim:
        coords.append(max(0, dim - tile))
    return sorted(set(coords))


def nms(dets: list[dict], iou_thr: float) -> list[dict]:
    if not dets:
        return []
    boxes  = np.array([[d["x1"], d["y1"], d["x2"], d["y2"]] for d in dets], dtype=np.float32)
    scores = np.array([d["conf"] for d in dets])
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas  = (x2 - x1) * (y2 - y1)
    order  = scores.argsort()[::-1]
    keep   = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thr]
    return [dets[i] for i in keep]


def detect_vehicles(aerial: Image.Image) -> list[dict]:
    """
    Multi-scale iterative detection.
    Runs three confidence passes and merges via NMS.
    Applies size and aspect ratio filters to remove false positives.
    """
    model    = YOLO(config.YOLO_WEIGHTS)
    W, H     = aerial.size
    img_area = W * H
    all_raw  = []

    for conf in config.DETECT_PASSES:
        pass_raw = []
        for scale in config.DETECT_SCALES:
            for y in tile_coords(H, scale, config.DETECT_OVERLAP):
                for x in tile_coords(W, scale, config.DETECT_OVERLAP):
                    tile = aerial.crop((x, y, min(x + scale, W), min(y + scale, H)))
                    for box in model(tile, conf=conf, verbose=False)[0].boxes:
                        if int(box.cls) not in config.VEHICLE_CLASSES:
                            continue
                        bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
                        gx1, gy1 = x + bx1, y + by1
                        gx2, gy2 = min(x + bx2, W), min(y + by2, H)
                        area   = (gx2 - gx1) * (gy2 - gy1)
                        aspect = (gx2 - gx1) / max(gy2 - gy1, 1)
                        if not (config.MIN_VEHICLE_AREA_FRAC * img_area <= area <= config.MAX_VEHICLE_AREA_FRAC * img_area):
                            continue
                        if not (config.MIN_ASPECT <= aspect <= config.MAX_ASPECT):
                            continue
                        # Reject detections inside exclusion zones
                        cx, cy = (gx1 + gx2) // 2, (gy1 + gy2) // 2
                        if any(ex1 <= cx <= ex2 and ey1 <= cy <= ey2
                               for ex1, ey1, ex2, ey2 in config.EXCLUSION_ZONES):
                            continue
                        pass_raw.append({
                            "x1": gx1, "y1": gy1, "x2": gx2, "y2": gy2,
                            "conf": float(box.conf),
                            "label": config.VEHICLE_CLASSES[int(box.cls)],
                        })
        log.info("Pass conf=%.2f: %d raw detections", conf, len(pass_raw))
        all_raw.extend(pass_raw)

    dets = nms(all_raw, config.NMS_IOU_THR)

    # Add manual detections for vehicles YOLO cannot detect
    for x1, y1, x2, y2, label in config.MANUAL_DETECTIONS:
        dets.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                     "conf": 1.0, "label": label})
        log.info("Manual detection added: %s box=(%d,%d,%d,%d)", label, x1, y1, x2, y2)

    log.info("After NMS: %d final detections", len(dets))
    for d in sorted(dets, key=lambda x: (x["y1"], x["x1"])):
        log.info("  %s conf=%.3f box=(%d,%d,%d,%d)",
                 d["label"], d["conf"], d["x1"], d["y1"], d["x2"], d["y2"])
    return dets

"""
Configuration for the aerial road inpainting pipeline.
All tunable parameters are defined here.
"""

import os
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR      = os.path.dirname(__file__)
AERIAL_IMAGE  = os.path.join(BASE_DIR, "artifacts", "input_aerial.png")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
YOLO_WEIGHTS  = os.path.join(ARTIFACTS_DIR, "yolov8n.pt")

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Three passes at decreasing confidence — catches vehicles missed at higher thresholds
DETECT_PASSES  = [0.20, 0.10, 0.06]
DETECT_SCALES  = [512, 640, 800]
DETECT_OVERLAP = 120

# NMS IoU threshold
NMS_IOU_THR = 0.25

# Vehicle size filter (fraction of image area)
# Filters noise (too small) and non-vehicle false positives (too large)
MIN_VEHICLE_AREA_FRAC = 0.000005
MAX_VEHICLE_AREA_FRAC = 0.002    # 5280px (largest real vehicle) passes, 6966px (house FP) fails

# Exclusion zones — confirmed false positives for this image [x1,y1,x2,y2]
# Add off-road detections here to permanently exclude them
EXCLUSION_ZONES = [
    (1400, 450, 1510, 540),  # off-road bus falsely detected
    (1310, 480, 1420, 560),  # off-road truck falsely detected
]

# Aspect ratio — vehicles are wider than tall in aerial view
MIN_ASPECT = 1.30
MAX_ASPECT = 5.0

# Manual detections — vehicles confirmed on road that YOLO cannot detect
# Format: (x1, y1, x2, y2, label)
# Add coordinates from debug_row700_zoom.png inspection
MANUAL_DETECTIONS = [
    # (x1, y1, x2, y2, 'car'),  # example — fill in after inspecting debug_row700_zoom.png
]

# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------
VEHICLE_DILATE_PX = 12

# Shadow detection — only within expanded zone around each vehicle
SHADOW_EXPAND_PX = 35
SHADOW_DARK_THR  = 65
SHADOW_DILATE_PX = 6

# ---------------------------------------------------------------------------
# Inpainting
# ---------------------------------------------------------------------------
INPAINT_TILE      = 512
INPAINT_OVERLAP   = 64
INPAINT_STEPS     = 35
GUIDANCE_SCALE    = 7.5
INPAINT_PROMPT    = (
    "aerial top-down satellite view of empty asphalt road, "
    "clean road surface, road markings, no vehicles, "
    "uniform tarmac texture, photorealistic"
)
INPAINT_NEG_PROMPT = (
    "cars, trucks, vehicles, people, shadows, "
    "blurry, distorted, artifacts"
)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE  = torch.float32

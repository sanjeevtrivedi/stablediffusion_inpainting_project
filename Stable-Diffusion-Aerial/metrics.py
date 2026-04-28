"""
Pixel integrity metrics.

Verifies that only masked pixels were modified in the inpainted output.
"""

import logging

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


def compute_and_save(original: Image.Image, inpainted: Image.Image,
                     mask: np.ndarray, path: str) -> dict:
    orig = np.array(original).astype(np.int32)
    inp  = np.array(inpainted).astype(np.int32)
    diff = np.abs(orig - inp)

    mb  = mask > 0
    nmb = ~mb
    changed_outside = (diff.sum(axis=2) > 0) & nmb
    passed = changed_outside.sum() == 0

    metrics = {
        "total_pixels":                int(mask.size),
        "masked_pixels":               int(mb.sum()),
        "masked_pixels_pct":           round(mb.sum() / mask.size * 100, 3),
        "pixels_changed_outside_mask": int(changed_outside.sum()),
        "max_diff_outside_mask":       int(diff[nmb].max()),
        "pixels_changed_inside_mask":  int((diff.sum(axis=2)[mb] > 0).sum()),
        "mean_diff_inside_mask":       round(float(diff[mb].mean()), 3),
        "max_diff_inside_mask":        int(diff[mb].max()),
        "passed":                      passed,
    }

    lines = [
        "=" * 60,
        "INPAINTING PIXEL INTEGRITY REPORT",
        "=" * 60,
        f"Total pixels               : {metrics['total_pixels']:,}",
        f"Masked pixels              : {metrics['masked_pixels']:,} ({metrics['masked_pixels_pct']}%)",
        "",
        "INTEGRITY CHECK",
        "-" * 40,
        f"Non-masked pixels identical: {passed}",
        f"Pixels changed outside mask: {metrics['pixels_changed_outside_mask']}",
        f"Max diff outside mask      : {metrics['max_diff_outside_mask']}",
        "",
        "INPAINTED REGION",
        "-" * 40,
        f"Pixels changed inside mask : {metrics['pixels_changed_inside_mask']:,}",
        f"Mean diff inside mask      : {metrics['mean_diff_inside_mask']}",
        f"Max diff inside mask       : {metrics['max_diff_inside_mask']}",
        "",
        "=" * 60,
        "RESULT: " + (
            "PASS - Only vehicle pixels were modified."
            if passed else
            "FAIL - Non-vehicle pixels were modified."
        ),
        "=" * 60,
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines))

    for line in lines:
        log.info(line)

    return metrics

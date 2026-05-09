from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
import lpips
from PIL import Image
from skimage.metrics import structural_similarity


@dataclass
class MetricResult:
    image_name: str
    psnr: float
    ssim: float
    lpips_val: float = 0.0
    mask_psnr: float = 0.0
    mask_ssim: float = 0.0
    mask_lpips: float = 0.0


def _to_np_rgb(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32)


def compute_psnr(target: Image.Image, prediction: Image.Image, max_pixel: float = 255.0) -> float:
    target_np = _to_np_rgb(target)
    pred_np = _to_np_rgb(prediction)
    mse = np.mean((target_np - pred_np) ** 2)
    if mse == 0:
        return float("inf")
    return float(20.0 * np.log10(max_pixel / np.sqrt(mse)))


def compute_ssim(target: Image.Image, prediction: Image.Image) -> float:
    target_np = _to_np_rgb(target).astype(np.uint8)
    pred_np = _to_np_rgb(prediction).astype(np.uint8)
    return float(
        structural_similarity(
            target_np,
            pred_np,
            channel_axis=2,
            data_range=255,
        )
    )


_lpips_model = None


def _get_lpips_model() -> lpips.LPIPS:
    global _lpips_model
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net="alex").eval()
    return _lpips_model


def compute_lpips(target: Image.Image, prediction: Image.Image) -> float:
    model = _get_lpips_model()
    target_np = _to_np_rgb(target) / 255.0
    pred_np = _to_np_rgb(prediction) / 255.0
    # LPIPS expects tensors in [-1, 1] with shape (1, 3, H, W)
    target_t = torch.from_numpy(target_np).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    pred_t = torch.from_numpy(pred_np).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    with torch.no_grad():
        score = model(target_t, pred_t)
    return float(score.item())


# ── Mask-region metrics ──────────────────────────────────────────────

def _mask_to_bool(mask: Image.Image) -> np.ndarray:
    """Convert a PIL mask to a boolean array (True = masked / inpainted region)."""
    return np.asarray(mask.convert("L"), dtype=np.float32) > 127.0


def compute_psnr_masked(
    target: Image.Image,
    prediction: Image.Image,
    mask: Image.Image,
    max_pixel: float = 255.0,
) -> float:
    """PSNR computed only over pixels where *mask* is white (inpainted region)."""
    target_np = _to_np_rgb(target)
    pred_np = _to_np_rgb(prediction)
    mask_bool = _mask_to_bool(mask)  # (H, W)
    # Broadcast mask to (H, W, 3)
    mask_3d = np.stack([mask_bool] * 3, axis=-1)
    diff = (target_np[mask_3d] - pred_np[mask_3d]) ** 2
    if diff.size == 0:
        return float("inf")
    mse = np.mean(diff)
    if mse == 0:
        return float("inf")
    return float(20.0 * np.log10(max_pixel / np.sqrt(mse)))


def compute_ssim_masked(
    target: Image.Image,
    prediction: Image.Image,
    mask: Image.Image,
) -> float:
    """SSIM computed only over the bounding-box of the masked region.

    We crop both images to the tight bounding box of the mask so that
    the SSIM windowing operates on the region of interest.  Pixels
    outside the mask but inside the bounding box are included in the
    windowing (unavoidable with block-based SSIM), but the crop still
    focuses the metric on the inpainted area.
    """
    mask_bool = _mask_to_bool(mask)
    rows = np.any(mask_bool, axis=1)
    cols = np.any(mask_bool, axis=0)
    if not rows.any():
        return 1.0  # empty mask → perfect match by convention
    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]

    target_crop = _to_np_rgb(target).astype(np.uint8)[r_min:r_max + 1, c_min:c_max + 1]
    pred_crop = _to_np_rgb(prediction).astype(np.uint8)[r_min:r_max + 1, c_min:c_max + 1]

    # win_size must be odd and <= smallest spatial dim of the crop
    min_side = min(target_crop.shape[0], target_crop.shape[1])
    win_size = min(7, min_side)
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        win_size = 3
        # Pad crops so that the window fits
        pad_h = max(0, win_size - target_crop.shape[0])
        pad_w = max(0, win_size - target_crop.shape[1])
        target_crop = np.pad(target_crop, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
        pred_crop = np.pad(pred_crop, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")

    return float(
        structural_similarity(
            target_crop,
            pred_crop,
            channel_axis=2,
            data_range=255,
            win_size=win_size,
        )
    )


def compute_lpips_masked(
    target: Image.Image,
    prediction: Image.Image,
    mask: Image.Image,
) -> float:
    """LPIPS computed by zeroing out non-mask pixels in both images.

    This isolates the perceptual difference to the inpainted region.
    """
    model = _get_lpips_model()
    target_np = _to_np_rgb(target) / 255.0
    pred_np = _to_np_rgb(prediction) / 255.0
    mask_bool = _mask_to_bool(mask)
    mask_3d = np.stack([mask_bool] * 3, axis=-1).astype(np.float32)
    # Zero out non-mask pixels so LPIPS only sees the inpainted region
    target_masked = target_np * mask_3d
    pred_masked = pred_np * mask_3d
    target_t = torch.from_numpy(target_masked).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    pred_t = torch.from_numpy(pred_masked).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    with torch.no_grad():
        score = model(target_t, pred_t)
    return float(score.item())


def evaluate_pairs(
    image_names: Iterable[str],
    targets: Iterable[Image.Image],
    predictions: Iterable[Image.Image],
) -> List[MetricResult]:
    results: List[MetricResult] = []
    for image_name, target, prediction in zip(image_names, targets, predictions):
        results.append(
            MetricResult(
                image_name=image_name,
                psnr=compute_psnr(target, prediction),
                ssim=compute_ssim(target, prediction),
            )
        )
    return results


def summarize_metrics(results: List[MetricResult]) -> dict:
    if not results:
        return {"count": 0, "mean_psnr": 0.0, "mean_ssim": 0.0}
    return {
        "count": len(results),
        "mean_psnr": float(np.mean([r.psnr for r in results])),
        "mean_ssim": float(np.mean([r.ssim for r in results])),
    }


def save_metrics_csv(results: List[MetricResult], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("image_name,psnr,ssim,lpips,mask_psnr,mask_ssim,mask_lpips\n")
        for r in results:
            f.write(
                f"{r.image_name},{r.psnr:.6f},{r.ssim:.6f},{r.lpips_val:.6f},"
                f"{r.mask_psnr:.6f},{r.mask_ssim:.6f},{r.mask_lpips:.6f}\n"
            )

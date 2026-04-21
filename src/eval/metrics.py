from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


@dataclass
class MetricResult:
    image_name: str
    psnr: float
    ssim: float


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
        f.write("image_name,psnr,ssim\n")
        for r in results:
            f.write(f"{r.image_name},{r.psnr:.6f},{r.ssim:.6f}\n")

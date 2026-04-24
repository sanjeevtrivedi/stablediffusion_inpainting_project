"""
Run Level 3 SOTA inpainting.

Uses ControlNet + inpainting and RePaint-style resampling (optional).

Usage:
    python scripts/run_level3.py --data-dir data/samples --output-dir outputs/level3
    python scripts/run_level3.py --no-controlnet --data-dir data/samples
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

from src.data.dataset import load_samples
from src.eval.metrics import compute_psnr, compute_ssim, evaluate_pairs, save_metrics_csv, summarize_metrics
from src.eval.visualize import save_comparison_panel
from src.models.level3_advanced import Level3Config, Level3SotaPipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Level 3 SOTA inpainting")
    parser.add_argument("--data-dir", type=Path, default=Path("data/samples"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/level3"))
    parser.add_argument("--mask-type", type=str, default="center", choices=["center", "irregular"])
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--prompt", type=str, default="A natural and realistic completion")
    parser.add_argument("--negative-prompt", type=str, default="blurry, distorted, low quality")
    parser.add_argument("--guidance-scale", type=float, default=9.0)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--repaint-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-controlnet", action="store_true")
    parser.add_argument("--no-repaint", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "panels").mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = Level3Config(
        device=device,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_steps,
        seed=args.seed,
        use_controlnet=not args.no_controlnet,
        use_repaint_resampling=not args.no_repaint,
        repaint_resample_steps=args.repaint_steps,
    )
    pipeline = Level3SotaPipeline(config)

    image_names, targets, predictions = [], [], []

    for sample in load_samples(args.data_dir, (args.image_size, args.image_size), args.mask_type):
        result = pipeline.run(
            prompt=args.prompt,
            image=sample.corrupted,
            mask=sample.mask,
            negative_prompt=args.negative_prompt,
        )
        psnr = compute_psnr(sample.image, result)
        ssim = compute_ssim(sample.image, result)

        image_names.append(sample.image_path.name)
        targets.append(sample.image)
        predictions.append(result)

        save_comparison_panel(
            original=sample.image,
            mask=sample.mask,
            corrupted=sample.corrupted,
            prediction=result,
            out_path=args.output_dir / "panels" / f"{sample.image_path.stem}_panel.png",
            title=f"Level 3 | {sample.image_path.name} | PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}",
        )
        result.save(args.output_dir / "predictions" / sample.image_path.name)

    if not image_names:
        raise RuntimeError(f"No images found in {args.data_dir}")

    results = evaluate_pairs(image_names, targets, predictions)
    save_metrics_csv(results, args.output_dir / "metrics.csv")
    summary = summarize_metrics(results)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Level 3 complete:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


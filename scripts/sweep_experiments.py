"""
Experiment sweep runner.

Sweeps over CFG scale and DDIM step counts for Level 1 (and optionally Level 2),
collecting PSNR/SSIM per configuration and writing a consolidated comparison table.

Usage:
    python scripts/sweep_experiments.py --data-dir data/samples
    python scripts/sweep_experiments.py --level 2 --data-dir data/samples
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List

import torch

from src.data.dataset import load_samples
from src.eval.metrics import evaluate_pairs, summarize_metrics


# --------------------------------------------------------------------------
# Sweep grid
# --------------------------------------------------------------------------
CFG_SCALES = [5.0, 7.5, 10.0]
DDIM_STEPS = [20, 30, 50]
MASK_TYPES = ["center", "irregular"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CFG / DDIM hyperparameter sweep")
    parser.add_argument("--data-dir", type=Path, default=Path("data/samples"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sweep"))
    parser.add_argument("--level", type=int, default=1, choices=[1, 2])
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--prompt", type=str, default="A natural and realistic completion")
    parser.add_argument("--negative-prompt", type=str, default="blurry, distorted, low quality")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_single(
    level: int,
    data_dir: Path,
    image_size: int,
    mask_type: str,
    prompt: str,
    negative_prompt: str,
    cfg: float,
    steps: int,
    seed: int,
    device: str,
):
    """Run a single (cfg, steps, mask_type) configuration and return mean metrics."""
    if level == 1:
        from src.models.level1_sd_inpaint import Level1Config, StableDiffusionInpaintingLevel1
        model = StableDiffusionInpaintingLevel1(
            Level1Config(
                model_id="runwayml/stable-diffusion-inpainting",
                guidance_scale=cfg,
                num_inference_steps=steps,
                ddim_eta=0.0,
                device=device,
            )
        )
        def infer(sample):
            return model.inpaint(
                prompt=prompt,
                image=sample.corrupted,
                mask=sample.mask,
                negative_prompt=negative_prompt,
                seed=seed,
            )
    else:
        from src.models.level2_finetune_ldm import Level2Config, Level2ProductionPipeline
        model = Level2ProductionPipeline(
            Level2Config(
                device=device,
                guidance_scale=cfg,
                num_inference_steps=steps,
                seed=seed,
            )
        )
        def infer(sample):
            return model.inpaint(
                prompt=prompt,
                image=sample.corrupted,
                mask=sample.mask,
                negative_prompt=negative_prompt,
            )

    names, targets, predictions = [], [], []
    for sample in load_samples(data_dir, (image_size, image_size), mask_type):
        names.append(sample.image_path.name)
        targets.append(sample.image)
        predictions.append(infer(sample))

    results = evaluate_pairs(names, targets, predictions)
    return summarize_metrics(results)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows: List[dict] = []
    total = len(CFG_SCALES) * len(DDIM_STEPS) * len(MASK_TYPES)
    run = 0

    for mask_type in MASK_TYPES:
        for cfg in CFG_SCALES:
            for steps in DDIM_STEPS:
                run += 1
                label = f"L{args.level}_cfg{cfg}_steps{steps}_{mask_type}"
                print(f"[{run}/{total}] {label} ...", flush=True)

                summary = run_single(
                    level=args.level,
                    data_dir=args.data_dir,
                    image_size=args.image_size,
                    mask_type=mask_type,
                    prompt=args.prompt,
                    negative_prompt=args.negative_prompt,
                    cfg=cfg,
                    steps=steps,
                    seed=args.seed,
                    device=device,
                )
                rows.append({
                    "level": args.level,
                    "mask_type": mask_type,
                    "cfg_scale": cfg,
                    "ddim_steps": steps,
                    "mean_psnr": summary["mean_psnr"],
                    "mean_ssim": summary["mean_ssim"],
                    "count": summary["count"],
                })
                print(f"  PSNR={summary['mean_psnr']:.2f}  SSIM={summary['mean_ssim']:.4f}")

    # Save sweep table
    csv_path = args.output_dir / "sweep_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    json_path = args.output_dir / "sweep_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    # Best config
    best = max(rows, key=lambda r: r["mean_psnr"])
    print(f"\nSweep complete. Results saved to {csv_path}")
    print(f"Best config (PSNR): {json.dumps(best, indent=2)}")


if __name__ == "__main__":
    main()

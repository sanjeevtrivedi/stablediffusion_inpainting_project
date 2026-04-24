"""
Run Level 2 production inpainting.

Supports inference-only mode (default) or optional LoRA fine-tune before inference.

Usage (inference only):
    python scripts/run_level2.py --data-dir data/splits/test/images \
        --mask-dir data/splits/test/masks/center \
        --output-dir outputs/level2

Usage (fine-tune then infer):
    python scripts/run_level2.py --finetune \
        --train-dir data/splits/train/images \
        --data-dir data/splits/test/images \
        --mask-dir data/splits/test/masks/center
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

from src.data.dataset import list_images, load_samples
from src.eval.metrics import compute_psnr, compute_ssim, evaluate_pairs, save_metrics_csv, summarize_metrics
from src.eval.visualize import save_comparison_panel
from src.models.level2_finetune_ldm import Level2Config, Level2ProductionPipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Level 2 production inpainting")
    parser.add_argument("--data-dir", type=Path, default=Path("data/samples"))
    parser.add_argument("--mask-dir", type=Path, default=None,
                        help="Pre-saved masks directory (optional; generated if omitted)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/level2"))
    parser.add_argument("--mask-type", type=str, default="center", choices=["center", "irregular"])
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--prompt", type=str, default="A natural and realistic completion")
    parser.add_argument("--negative-prompt", type=str, default="blurry, distorted, low quality")
    parser.add_argument("--guidance-scale", type=float, default=8.0)
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-boundary-smoothing", action="store_true")
    parser.add_argument("--finetune", action="store_true",
                        help="Run LoRA fine-tune before inference")
    parser.add_argument("--train-dir", type=Path, default=None)
    parser.add_argument("--train-steps", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "panels").mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = Level2Config(
        device=device,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_steps,
        seed=args.seed,
        use_boundary_smoothing=not args.no_boundary_smoothing,
        train_steps=args.train_steps,
    )
    pipeline = Level2ProductionPipeline(config)

    # Optional fine-tune
    if args.finetune:
        train_files = list_images(args.train_dir)
        if not train_files:
            raise RuntimeError(f"No images found in {args.train_dir}")
        pipeline.train(
            image_paths=train_files,
            prompts=[args.prompt] * len(train_files),
        )

    # Inference loop
    image_names, targets, predictions = [], [], []

    for sample in load_samples(args.data_dir, (args.image_size, args.image_size), args.mask_type):
        prediction = pipeline.inpaint(
            prompt=args.prompt,
            image=sample.corrupted,
            mask=sample.mask,
            negative_prompt=args.negative_prompt,
        )
        psnr = compute_psnr(sample.image, prediction)
        ssim = compute_ssim(sample.image, prediction)

        image_names.append(sample.image_path.name)
        targets.append(sample.image)
        predictions.append(prediction)

        save_comparison_panel(
            original=sample.image,
            mask=sample.mask,
            corrupted=sample.corrupted,
            prediction=prediction,
            out_path=args.output_dir / "panels" / f"{sample.image_path.stem}_panel.png",
            title=f"Level 2 | {sample.image_path.name} | PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}",
        )
        prediction.save(args.output_dir / "predictions" / sample.image_path.name)

    if not image_names:
        raise RuntimeError(f"No images found in {args.data_dir}")

    results = evaluate_pairs(image_names, targets, predictions)
    save_metrics_csv(results, args.output_dir / "metrics.csv")
    summary = summarize_metrics(results)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Level 2 complete:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


"""
Main inpainting pipeline: load images, inpaint, evaluate.

Usage:
    python scripts/03_run_inpainting.py --data-dir data/samples --output-dir outputs
    python scripts/03_run_inpainting.py --data-dir data/samples --mask-type irregular --guidance-scale 9.0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline
from PIL import Image

from src.data.dataset import generate_center_mask, generate_irregular_mask, apply_mask, list_images
from src.eval.metrics import compute_psnr, compute_ssim, save_metrics_csv, MetricResult
from src.eval.visualize import save_comparison_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inpainting pipeline")
    parser.add_argument("--data-dir", type=Path, default=Path("data/samples"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model-id", type=str, default="runwayml/stable-diffusion-inpainting")
    parser.add_argument("--prompt", type=str, default="A realistic and coherent scene completion")
    parser.add_argument("--negative-prompt", type=str, default="blurry, distorted, low quality")
    parser.add_argument("--mask-type", type=str, default="center", choices=["center", "irregular"])
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--ddim-eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Setup device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Create output directories
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "panels").mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model_id}")
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionInpaintPipeline.from_pretrained(args.model_id, torch_dtype=dtype)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)

    # Get image files
    image_files = list_images(args.data_dir)
    if not image_files:
        raise RuntimeError(f"No images found in {args.data_dir}")

    print(f"Found {len(image_files)} images. Starting inpainting...")

    image_names = []
    targets = []
    predictions = []
    results = []

    for idx, image_path in enumerate(image_files, 1):
        print(f"  [{idx}/{len(image_files)}] Processing {image_path.name}...")

        # Load and prepare image
        image = Image.open(image_path).convert("RGB").resize((args.image_size, args.image_size))

        # Generate mask
        if args.mask_type == "center":
            mask = generate_center_mask(args.image_size, args.image_size)
        else:
            mask = generate_irregular_mask(args.image_size, args.image_size)

        # Create corrupted version
        corrupted = apply_mask(image, mask, fill_value=255)

        # Run inpainting
        generator = torch.Generator(device=device).manual_seed(args.seed + idx)
        output = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            image=corrupted,
            mask_image=mask,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_steps,
            eta=args.ddim_eta,
            generator=generator,
        )
        prediction = output.images[0]

        # Compute metrics
        psnr = compute_psnr(image, prediction)
        ssim = compute_ssim(image, prediction)
        results.append(MetricResult(image_name=image_path.name, psnr=psnr, ssim=ssim))

        # Save outputs
        prediction.save(args.output_dir / "predictions" / image_path.name)

        save_comparison_panel(
            original=image,
            mask=mask,
            corrupted=corrupted,
            prediction=prediction,
            out_path=args.output_dir / "panels" / f"{image_path.stem}_panel.png",
            title=f"{image_path.name} | PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}",
        )

        image_names.append(image_path.name)
        targets.append(image)
        predictions.append(prediction)

    # Save metrics
    save_metrics_csv(results, args.output_dir / "metrics.csv")

    mean_psnr = sum(r.psnr for r in results) / len(results) if results else 0
    mean_ssim = sum(r.ssim for r in results) / len(results) if results else 0

    summary = {
        "count": len(results),
        "mean_psnr": float(mean_psnr),
        "mean_ssim": float(mean_ssim),
        "mask_type": args.mask_type,
        "guidance_scale": args.guidance_scale,
        "num_steps": args.num_steps,
        "model": args.model_id,
        "device": device,
    }

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print(f"  Mean PSNR: {mean_psnr:.4f} dB")
    print(f"  Mean SSIM: {mean_ssim:.6f}")
    print(f"  Images: {len(results)}")
    print(f"  Mask type: {args.mask_type}")
    print(f"  Output directory: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

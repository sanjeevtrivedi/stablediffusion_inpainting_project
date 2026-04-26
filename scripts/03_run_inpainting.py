"""
Main inpainting pipeline: load images, inpaint, evaluate.

Supports two modes:
  1. Standard Stable Diffusion Inpainting  (StableDiffusionInpaintPipeline)
  2. ControlNet Inpainting                 (StableDiffusionControlNetInpaintPipeline)

Both modes auto-select the correct device (CUDA > MPS > CPU) and dtype
(float16 on CUDA, float32 on MPS/CPU).  Outputs are written to separate
sub-directories (outputs/standard/ and outputs/controlnet/)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STANDARD SD INPAINTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Minimal run (defaults: center mask, guidance=7.5, 50 steps):
    python scripts/03_run_inpainting.py

Custom data and output directories:
    python scripts/03_run_inpainting.py \
        --data-dir data/samples \
        --output-dir outputs

Irregular mask, higher guidance, more steps:
    python scripts/03_run_inpainting.py \
        --mask-type irregular \
        --guidance-scale 9.0 \
        --num-steps 75

Use an SD 2.x inpainting model at 768 px:
    python scripts/03_run_inpainting.py \
        --model-id stabilityai/stable-diffusion-2-inpainting \
        --image-size 768 \
        --guidance-scale 8.0

Custom prompt and seed for reproducibility:
    python scripts/03_run_inpainting.py \
        --prompt "A lush green meadow with soft sunlight" \
        --negative-prompt "blurry, distorted, low quality, watermark" \
        --seed 1234

Stochastic DDIM (eta=1.0 ≈ DDPM):
    python scripts/03_run_inpainting.py \
        --ddim-eta 1.0 \
        --num-steps 100

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTROLNET INPAINTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Default ControlNet (lllyasviel/control_v11p_sd15_inpaint + SD 1.5 base):
    python scripts/03_run_inpainting.py --use-controlnet

Explicit model IDs (same as defaults, shown for clarity):
    python scripts/03_run_inpainting.py \
        --use-controlnet \
        --model-id runwayml/stable-diffusion-inpainting \
        --controlnet-model-id lllyasviel/control_v11p_sd15_inpaint

Irregular mask with ControlNet:
    python scripts/03_run_inpainting.py \
        --use-controlnet \
        --mask-type irregular \
        --guidance-scale 8.0

ControlNet with more inference steps and a descriptive prompt:
    python scripts/03_run_inpainting.py \
        --use-controlnet \
        --prompt "A cozy living room interior, photorealistic" \
        --negative-prompt "blurry, artifacts, watermark" \
        --num-steps 80 \
        --guidance-scale 9.0 \
        --seed 99

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT LAYOUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
outputs/
  standard/          ← written by standard SD runs
    predictions/     ← inpainted images (one per input)
    panels/          ← 4-column comparison panels (original/mask/corrupted/inpainted)
    metrics.csv      ← per-image PSNR and SSIM
    summary.json     ← aggregate stats and run configuration
  controlnet/        ← written by ControlNet runs (same structure)
    predictions/
    panels/
    metrics.csv
    summary.json

KEY PARAMETERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  --image-size       Target resolution; must be a multiple of 8.
                     SD 1.x models are trained at 512, SD 2.x at 768. (default: 512)
  --guidance-scale   Classifier-free guidance strength [1.0–20.0].
                     Higher = stronger prompt adherence, less diversity. (default: 7.5)
  --num-steps        DDIM denoising steps [20–150 recommended]. (default: 50)
  --ddim-eta         0.0 = deterministic DDIM; 1.0 = stochastic DDPM. (default: 0.0)
  --mask-type        'center' (fixed rectangle) or 'irregular' (random strokes). (default: center)
  --seed             Base seed for reproducibility; each image uses seed + index. (default: 42)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import (
    ControlNetModel,
    DDIMScheduler,
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionInpaintPipeline,
)
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    # ControlNet options
    parser.add_argument(
        "--use-controlnet",
        action="store_true",
        help="Use ControlNet inpainting instead of the vanilla SD inpainting pipeline.",
    )
    parser.add_argument(
        "--controlnet-model-id",
        type=str,
        default="lllyasviel/control_v11p_sd15_inpaint",
        help="HuggingFace model ID for the ControlNet weights (used with --use-controlnet).",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate all CLI arguments before any model or data is loaded.

    Hard errors call sys.exit(1); soft issues are printed as warnings and
    execution continues.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # ------------------------------------------------------------------ #
    # image_size                                                           #
    # ------------------------------------------------------------------ #
    if args.image_size < 64:
        errors.append(
            f"--image-size {args.image_size} is too small; minimum meaningful size is 64."
        )
    elif args.image_size % 8 != 0:
        errors.append(
            f"--image-size {args.image_size} is not a multiple of 8. "
            "The SD VAE requires spatial dimensions divisible by 8."
        )
    elif args.image_size not in (512, 768):
        warnings.append(
            f"--image-size {args.image_size}: SD inpainting models are trained at 512 (SD 1.x) "
            "or 768 (SD 2.x). Other sizes may reduce quality."
        )

    # ------------------------------------------------------------------ #
    # num_steps                                                            #
    # ------------------------------------------------------------------ #
    if args.num_steps < 1:
        errors.append(f"--num-steps must be >= 1, got {args.num_steps}.")
    elif args.num_steps < 20:
        warnings.append(
            f"--num-steps {args.num_steps} is low. DDIM needs >=20 steps for coherent "
            "inpainting; fewer may produce blurry or incoherent results."
        )
    elif args.num_steps > 150:
        warnings.append(
            f"--num-steps {args.num_steps} is high. Beyond ~100 steps, DDIM gives "
            "diminishing quality returns and significantly increases run time."
        )

    # ------------------------------------------------------------------ #
    # guidance_scale                                                       #
    # ------------------------------------------------------------------ #
    if args.guidance_scale < 1.0:
        errors.append(
            f"--guidance-scale {args.guidance_scale} is below 1.0. "
            "Values < 1.0 invert classifier-free guidance and will produce incoherent output."
        )
    elif args.guidance_scale > 20.0:
        warnings.append(
            f"--guidance-scale {args.guidance_scale} is very high. "
            "Values above 20 typically cause oversaturation and loss of fine detail."
        )

    # ------------------------------------------------------------------ #
    # ddim_eta                                                             #
    # ------------------------------------------------------------------ #
    if not (0.0 <= args.ddim_eta <= 1.0):
        errors.append(
            f"--ddim-eta {args.ddim_eta} is out of range; must be in [0.0, 1.0]. "
            "0.0 = fully deterministic DDIM, 1.0 = DDPM-equivalent stochastic sampling."
        )

    # ------------------------------------------------------------------ #
    # seed                                                                 #
    # ------------------------------------------------------------------ #
    if args.seed < 0:
        errors.append(
            f"--seed {args.seed} is negative. PyTorch Generator requires a non-negative seed."
        )

    # ------------------------------------------------------------------ #
    # prompt                                                               #
    # ------------------------------------------------------------------ #
    if not args.prompt.strip():
        errors.append(
            "--prompt is empty. Stable Diffusion requires a text prompt to guide inpainting."
        )

    # ------------------------------------------------------------------ #
    # data_dir                                                             #
    # ------------------------------------------------------------------ #
    if not args.data_dir.exists():
        errors.append(f"--data-dir '{args.data_dir}' does not exist.")
    elif not args.data_dir.is_dir():
        errors.append(f"--data-dir '{args.data_dir}' is not a directory.")

    # ------------------------------------------------------------------ #
    # ControlNet / base-model version compatibility                        #
    # ------------------------------------------------------------------ #
    if args.use_controlnet:
        base = args.model_id.lower()
        ctrl = args.controlnet_model_id.lower()
        base_is_sd2 = any(k in base for k in ("stable-diffusion-2", "sd-2", "sd2", "v2"))
        ctrl_is_sd1 = any(k in ctrl for k in ("v11", "sd15", "control_v1", "sd1"))
        if base_is_sd2 and ctrl_is_sd1:
            warnings.append(
                "The ControlNet model appears to be SD 1.x-based while the base model looks "
                "like SD 2.x. This combination is architecturally incompatible and will likely "
                "produce errors or garbage output. Use a matching SD 2.x ControlNet or an "
                "SD 1.5 base model (e.g. runwayml/stable-diffusion-inpainting)."
            )

    # ------------------------------------------------------------------ #
    # Report                                                               #
    # ------------------------------------------------------------------ #
    if warnings:
        print("\n[CONFIG WARNINGS]")
        for w in warnings:
            print(f"  WARNING: {w}")
        print()

    if errors:
        print("\n[CONFIG ERRORS]")
        for e in errors:
            print(f"  ERROR:   {e}")
        print()
        sys.exit(1)

    if not warnings:
        print("[Config OK] All parameters validated successfully.")


def make_controlnet_condition(image: Image.Image, mask: Image.Image) -> torch.Tensor:
    """Build the ControlNet conditioning tensor for the inpainting ControlNet.

    Masked pixels are set to -1 (out-of-range sentinel) so the model treats
    them as the region to be filled.  The tensor has shape (1, 3, H, W) with
    values in [-1, 1].
    """
    image_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    mask_np = np.array(mask.convert("L")).astype(np.float32) / 255.0
    image_np[mask_np > 0.5] = -1.0
    control = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    return control


def main() -> None:
    args = parse_args()
    validate_args(args)

    # Setup device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # float16 only on CUDA; MPS and CPU require float32
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Variant-specific output sub-directory so that standard and ControlNet runs
    # never overwrite each other's predictions / panels / metrics.
    variant_tag = "controlnet" if args.use_controlnet else "standard"
    run_output_dir = args.output_dir / variant_tag

    # Create output directories
    run_output_dir.mkdir(parents=True, exist_ok=True)
    (run_output_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (run_output_dir / "panels").mkdir(parents=True, exist_ok=True)

    # Load model
    if args.use_controlnet:
        print(f"Loading ControlNet weights: {args.controlnet_model_id}")
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model_id, torch_dtype=dtype)
        print(f"Loading base model: {args.model_id}")
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            args.model_id,
            controlnet=controlnet,
            torch_dtype=dtype,
            use_safetensors=False,
        )
    else:
        print(f"Loading model: {args.model_id}")
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            args.model_id, torch_dtype=dtype, use_safetensors=False
        )

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    # Silence the NSFW safety checker on MPS/CPU to avoid device mismatch errors
    if device in ("mps", "cpu"):
        pipe.safety_checker = None

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
        if args.use_controlnet:
            control_image = make_controlnet_condition(image, mask)
            output = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                image=corrupted,
                mask_image=mask,
                control_image=control_image,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_steps,
                eta=args.ddim_eta,
                generator=generator,
            )
        else:
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
        prediction.save(run_output_dir / "predictions" / image_path.name)

        save_comparison_panel(
            original=image,
            mask=mask,
            corrupted=corrupted,
            prediction=prediction,
            out_path=run_output_dir / "panels" / f"{image_path.stem}_panel.png",
            title=f"{image_path.name} | PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}",
        )

        image_names.append(image_path.name)
        targets.append(image)
        predictions.append(prediction)

    # Save metrics
    save_metrics_csv(results, run_output_dir / "metrics.csv")

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
        "use_controlnet": args.use_controlnet,
        "controlnet_model": args.controlnet_model_id if args.use_controlnet else None,
        "device": device,
    }

    with (run_output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print(f"  Mode: {'ControlNet' if args.use_controlnet else 'Standard SD Inpainting'}")
    print(f"  Mean PSNR: {mean_psnr:.4f} dB")
    print(f"  Mean SSIM: {mean_ssim:.6f}")
    print(f"  Images: {len(results)}")
    print(f"  Mask type: {args.mask_type}")
    print(f"  Output directory: {run_output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

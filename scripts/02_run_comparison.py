"""
Side-by-side comparison: Standard SD Inpainting vs ControlNet Inpainting.

For each input image the script generates the same mask, runs both
pipelines, and saves the two predictions into outputs/comparison/ with
prefixed filenames:

    sd-stepsNN-<original_name>   - Standard Stable Diffusion inpainting
    cn-stepsNN-<original_name>   - ControlNet inpainting

Usage examples:
    python scripts/02_run_comparison.py #default is 20 steps
    python scripts/02_run_comparison.py --num-steps 50 

"""

# Enable future annotations for type hints (Python 3.7+ compatibility)
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


# Standard logging and warning modules
import logging
import warnings

import cv2
import numpy as np
import torch
from diffusers import (
    ControlNetModel,
    DDIMScheduler,
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionInpaintPipeline,
)
from PIL import Image

# Suppress general warnings
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# Add project root to sys.path for module imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import (
    generate_center_mask,
    generate_irregular_mask,
    apply_mask,
    list_images,
)
from src.eval.metrics import (
    compute_psnr,
    compute_ssim,
    compute_lpips,
    compute_psnr_masked,
    compute_ssim_masked,
    compute_lpips_masked,
    save_metrics_csv,
    MetricResult,
)
from src.eval.visualize import save_comparison_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Standard SD Inpainting vs ControlNet Inpainting")

    # ── I/O paths ─────────────────────────────────────────────────────
    parser.add_argument("--data-dir", type=Path, default=Path("data/samples"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/comparison"))

    # ── Model identifiers ─────────────────────────────────────────────
    # SD model: used as the base inpainting backbone for BOTH pipelines.
    # ControlNet model: auxiliary structure-guidance network layered on
    # top of the SD model; only used by the ControlNet pipeline.
    parser.add_argument("--sd-model-id", type=str, default="runwayml/stable-diffusion-inpainting")
    parser.add_argument("--controlnet-model-id", type=str, default="lllyasviel/control_v11p_sd15_canny")

    # ── Text conditioning (shared by both pipelines) ──────────────────
    # The prompt steers the denoising process via cross-attention in the
    # U-Net.  The negative prompt discourages undesirable attributes
    # during classifier-free guidance.
    parser.add_argument("--prompt", type=str, default="A realistic and coherent scene completion")
    parser.add_argument("--negative-prompt", type=str, default="blurry, distorted, low quality")

    # ── Mask & image configuration ────────────────────────────────────
    # mask-type: determines the shape of the region to inpaint.
    #   "center"    – rectangular hole in the centre (good for benchmarking)
    #   "irregular" – random brush strokes (closer to real-world damage)
    # The same mask is fed to both SD and ControlNet for a fair comparison.
    parser.add_argument("--mask-type", type=str, default="irregular", choices=["center", "irregular"])
    # image-size: both pipelines resize inputs to this square resolution.
    # SD inpainting expects 512×512; larger sizes increase VRAM usage.
    parser.add_argument("--image-size", type=int, default=512)

    # ── Sampling parameters (shared by both pipelines) ────────────────
    # guidance-scale: classifier-free guidance weight.  Higher values
    # make the output follow the prompt more closely but may reduce
    # diversity.  Applied identically to SD and ControlNet pipelines.
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    # num-steps: number of DDIM denoising steps.  More steps improve
    # quality at the cost of inference time; affects both pipelines.
    parser.add_argument("--num-steps", type=int, default=20)
    # ddim-eta: stochasticity parameter for the DDIM scheduler.
    # 0.0 = fully deterministic; 1.0 = equivalent to DDPM.
    parser.add_argument("--ddim-eta", type=float, default=0.0)
    # seed: base random seed.  Each image uses (seed + image_index) so
    # both pipelines receive the same initial noise for fair comparison.
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def make_controlnet_condition(image: Image.Image) -> Image.Image:
    """Build a Canny edge map as the ControlNet conditioning image.

    Edges are computed from the original (unmasked) image so ControlNet
    has structural context for the full scene, including the region to
    fill.  The edge map guides the model to produce inpainting that is
    consistent with the surrounding structure (lines, contours, object
    boundaries).

    Returns a PIL Image (RGB) as expected by the diffusers ControlNet
    pipeline.
    """
    # Convert PIL image to numpy array (RGB)
    image_np = np.array(image.convert("RGB"))
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Auto-threshold using the median pixel value (Otsu-style)
    median = np.median(gray)
    low = int(max(0, 0.66 * median))
    high = int(min(255, 1.33 * median))

    # Compute Canny edges and convert to 3-channel RGB for ControlNet
    edges = cv2.Canny(gray, low, high)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    # Return as PIL Image (RGB)
    return Image.fromarray(edges_rgb)

def make_controlnet_condition_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Build a Canny edge map only for the masked (inpaint) region.

    Edges are computed from the full image but zeroed outside the mask,
    so ControlNet receives structural guidance only within the region to
    fill.  This can help the model focus on reconstructing structure
    inside the hole without being influenced by surrounding edges.

    Returns a PIL Image (RGB) – same type as make_controlnet_condition()
    so the two functions are interchangeable at the call site.
    """
    image_np = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    median = np.median(gray)
    low = int(max(0, 0.66 * median))
    high = int(min(255, 1.33 * median))

    edges = cv2.Canny(gray, low, high)

    # Zero out edges outside the mask – keep only masked region edges
    mask_np = np.array(mask.convert("L"))
    edges[mask_np > 128] = 0

    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)

def main() -> None:

    args = parse_args()

    # Check if the data directory exists
    if not args.data_dir.exists():
        sys.exit(f"ERROR: --data-dir '{args.data_dir}' does not exist.")

    # Select device: CUDA (GPU) preferred, then MPS (Apple Silicon), else CPU
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    # Use float16 for CUDA, float32 otherwise (saves memory on GPU)
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Prepare output directories for results, panels, masks, and canny edges
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (out_dir / "panels").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    (out_dir / "canny").mkdir(parents=True, exist_ok=True)

    # Load the standard Stable Diffusion inpainting pipeline
    print(f"Loading Standard SD model: {args.sd_model_id}")
    sd_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.sd_model_id, torch_dtype=dtype, use_safetensors=False
    )
    # Use DDIM scheduler for deterministic sampling
    sd_pipe.scheduler = DDIMScheduler.from_config(sd_pipe.scheduler.config)
    sd_pipe.to(device)
    # Disable safety checker on non-CUDA devices for compatibility
    if device in ("mps", "cpu"):
        sd_pipe.safety_checker = None

    # Load the ControlNet inpainting pipeline (adds structure guidance)
    print(f"Loading ControlNet weights: {args.controlnet_model_id}")
    controlnet = ControlNetModel.from_pretrained(args.controlnet_model_id, torch_dtype=dtype)
    print(f"Loading ControlNet base model: {args.sd_model_id}")
    cn_pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        args.sd_model_id,
        controlnet=controlnet,
        torch_dtype=dtype,
        use_safetensors=False,
    )
    cn_pipe.scheduler = DDIMScheduler.from_config(cn_pipe.scheduler.config)
    cn_pipe.to(device)
    if device in ("mps", "cpu"):
        cn_pipe.safety_checker = None

    # List all images in the data directory
    image_files = list_images(args.data_dir)
    if not image_files:
        raise RuntimeError(f"No images found in {args.data_dir}")

    print(f"Found {len(image_files)} images. Starting comparison...\n")

    # Store metric results for each pipeline
    sd_results: list[MetricResult] = []
    cn_results: list[MetricResult] = []

    # Process each image one by one
    for idx, image_path in enumerate(image_files, 1):
        print(f"  [{idx}/{len(image_files)}] Processing {image_path.name}...")

        # Load image and resize to target size
        image = Image.open(image_path).convert("RGB").resize(
            (args.image_size, args.image_size)
        )

        # Generate a mask (center or irregular) with deterministic seed for reproducibility
        if args.mask_type == "center":
            mask = generate_center_mask(args.image_size, args.image_size)
        else:
            mask = generate_irregular_mask(args.image_size, args.image_size, seed=args.seed + idx)

        # Apply mask to the image (simulate missing region)
        corrupted = apply_mask(image, mask, fill_value=255)

        # Use the same random seed for both pipelines for fair comparison
        seed = args.seed + idx

        # --- Standard SD inpainting ---
        # Create a torch random generator for reproducibility
        gen_sd = torch.Generator(device=device).manual_seed(seed)
        # Measure inference time
        sd_t0 = time.perf_counter()
        # Run the SD inpainting pipeline
        sd_output = sd_pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            image=corrupted,
            mask_image=mask,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_steps,
            eta=args.ddim_eta,
            generator=gen_sd,
        )
        sd_time = time.perf_counter() - sd_t0
        # Get the predicted image
        sd_pred = sd_output.images[0]

        # --- ControlNet inpainting ---
        # Create a torch random generator for reproducibility
        gen_cn = torch.Generator(device=device).manual_seed(seed)
        # Generate Canny edge map as control image
        control_image = make_controlnet_condition(image)
        # Measure inference time
        cn_t0 = time.perf_counter()
        # Run the ControlNet inpainting pipeline
        cn_output = cn_pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            image=corrupted,
            mask_image=mask,
            control_image=control_image,
            controlnet_conditioning_scale=0.5,  # Adjust strength of ControlNet guidance, for inpaiting it is between 0.3 to 0.7
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_steps,
            eta=args.ddim_eta,
            generator=gen_cn,
        )
        cn_time = time.perf_counter() - cn_t0
        # Get the predicted image
        cn_pred = cn_output.images[0]

        # --- Compute metrics for both outputs ---
        # Standard SD metrics
        sd_psnr = compute_psnr(image, sd_pred)
        sd_ssim = compute_ssim(image, sd_pred)
        sd_lpips = compute_lpips(image, sd_pred)
        sd_mask_psnr = compute_psnr_masked(image, sd_pred, mask)
        sd_mask_ssim = compute_ssim_masked(image, sd_pred, mask)
        sd_mask_lpips = compute_lpips_masked(image, sd_pred, mask)
        sd_results.append(
            MetricResult(
                image_name=f"sd-steps{args.num_steps}-{image_path.name}",
                psnr=sd_psnr, ssim=sd_ssim, lpips_val=sd_lpips,
                mask_psnr=sd_mask_psnr, mask_ssim=sd_mask_ssim, mask_lpips=sd_mask_lpips,
                time_sec=sd_time,
            )
        )

        # ControlNet metrics
        cn_psnr = compute_psnr(image, cn_pred)
        cn_ssim = compute_ssim(image, cn_pred)
        cn_lpips = compute_lpips(image, cn_pred)
        cn_mask_psnr = compute_psnr_masked(image, cn_pred, mask)
        cn_mask_ssim = compute_ssim_masked(image, cn_pred, mask)
        cn_mask_lpips = compute_lpips_masked(image, cn_pred, mask)
        cn_results.append(
            MetricResult(
                image_name=f"cn-steps{args.num_steps}-{image_path.name}",
                psnr=cn_psnr, ssim=cn_ssim, lpips_val=cn_lpips,
                mask_psnr=cn_mask_psnr, mask_ssim=cn_mask_ssim, mask_lpips=cn_mask_lpips,
                time_sec=cn_time,
            )
        )

        # Save predictions, mask, and Canny edge map for inspection
        sd_pred.save(out_dir / "predictions" / f"sd-steps{args.num_steps}-{image_path.name}")
        cn_pred.save(out_dir / "predictions" / f"cn-steps{args.num_steps}-{image_path.name}")
        mask.save(out_dir / "masks" / f"{image_path.stem}_mask.png")
        control_image.save(out_dir / "canny" / f"{image_path.stem}_canny.png")

        # Save side-by-side comparison panels for both pipelines
        save_comparison_panel(
            original=image,
            mask=mask,
            corrupted=corrupted,
            prediction=sd_pred,
            out_path=out_dir / "panels" / f"sd-steps{args.num_steps}-{image_path.stem}_panel.png",
            title=f"SD | steps={args.num_steps} | {image_path.name} | PSNR: {sd_psnr:.2f}, SSIM: {sd_ssim:.4f}, LPIPS: {sd_lpips:.4f}"
                  f" | Mask PSNR: {sd_mask_psnr:.2f}, SSIM: {sd_mask_ssim:.4f}, LPIPS: {sd_mask_lpips:.4f}",
        )
        save_comparison_panel(
            original=image,
            mask=mask,
            corrupted=corrupted,
            prediction=cn_pred,
            out_path=out_dir / "panels" / f"cn-steps{args.num_steps}-{image_path.stem}_panel.png",
            title=f"CN | steps={args.num_steps} | {image_path.name} | PSNR: {cn_psnr:.2f}, SSIM: {cn_ssim:.4f}, LPIPS: {cn_lpips:.4f}"
                  f" | Mask PSNR: {cn_mask_psnr:.2f}, SSIM: {cn_mask_ssim:.4f}, LPIPS: {cn_mask_lpips:.4f}",
        )

        # Print metrics for this image
        print(
            f"    SD  → PSNR: {sd_psnr:.2f} dB, SSIM: {sd_ssim:.4f}, LPIPS: {sd_lpips:.4f}"
            f" | Mask PSNR: {sd_mask_psnr:.2f}, SSIM: {sd_mask_ssim:.4f}, LPIPS: {sd_mask_lpips:.4f}"
            f" | {sd_time:.2f}s\n"
            f"    CN  → PSNR: {cn_psnr:.2f} dB, SSIM: {cn_ssim:.4f}, LPIPS: {cn_lpips:.4f}"
            f" | Mask PSNR: {cn_mask_psnr:.2f}, SSIM: {cn_mask_ssim:.4f}, LPIPS: {cn_mask_lpips:.4f}"
            f" | {cn_time:.2f}s"
        )

    # ── Aggregate metrics ─────────────────────────────────────────────
    # Write per-image CSV and JSON summary with mean metrics for both
    # pipelines, along with the experiment configuration.
    all_results = sd_results + cn_results
    # Save metrics CSV with steps in filename
    metrics_filename = f"metrics-steps{args.num_steps}.csv"
    save_metrics_csv(all_results, out_dir / metrics_filename)

    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    summary = {
        "count": len(image_files),
        "standard": {
            "mean_psnr": _mean([r.psnr for r in sd_results]),
            "mean_ssim": _mean([r.ssim for r in sd_results]),
            "mean_lpips": _mean([r.lpips_val for r in sd_results]),
            "mean_mask_psnr": _mean([r.mask_psnr for r in sd_results]),
            "mean_mask_ssim": _mean([r.mask_ssim for r in sd_results]),
            "mean_mask_lpips": _mean([r.mask_lpips for r in sd_results]),
            "mean_time_sec": _mean([r.time_sec for r in sd_results]),
            "total_time_sec": sum(r.time_sec for r in sd_results),
        },
        "controlnet": {
            "mean_psnr": _mean([r.psnr for r in cn_results]),
            "mean_ssim": _mean([r.ssim for r in cn_results]),
            "mean_lpips": _mean([r.lpips_val for r in cn_results]),
            "mean_mask_psnr": _mean([r.mask_psnr for r in cn_results]),
            "mean_mask_ssim": _mean([r.mask_ssim for r in cn_results]),
            "mean_mask_lpips": _mean([r.mask_lpips for r in cn_results]),
            "mean_time_sec": _mean([r.time_sec for r in cn_results]),
            "total_time_sec": sum(r.time_sec for r in cn_results),
        },
        "config": {
            "mask_type": args.mask_type,
            "guidance_scale": args.guidance_scale,
            "num_steps": args.num_steps,
            "sd_model": args.sd_model_id,
            "controlnet_model": args.controlnet_model_id,
            "image_size": args.image_size,
            "seed": args.seed,
            "device": device,
        },
    }

    # Save summary JSON with steps in filename
    summary_filename = f"summary-steps{args.num_steps}.json"
    with (out_dir / summary_filename).open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ── Print summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Comparison complete!")
    print(f"  Images processed : {len(image_files)}")
    print(f"  Mask type        : {args.mask_type}")
    print(f"  Standard SD      : PSNR {summary['standard']['mean_psnr']:.4f} dB, "
          f"SSIM {summary['standard']['mean_ssim']:.6f}, "
          f"LPIPS {summary['standard']['mean_lpips']:.6f}, "
          f"Time {summary['standard']['mean_time_sec']:.2f}s/img ({summary['standard']['total_time_sec']:.1f}s total)")
    print(f"    Mask region    : PSNR {summary['standard']['mean_mask_psnr']:.4f} dB, "
          f"SSIM {summary['standard']['mean_mask_ssim']:.6f}, "
          f"LPIPS {summary['standard']['mean_mask_lpips']:.6f}")
    print(f"  ControlNet       : PSNR {summary['controlnet']['mean_psnr']:.4f} dB, "
          f"SSIM {summary['controlnet']['mean_ssim']:.6f}, "
          f"LPIPS {summary['controlnet']['mean_lpips']:.6f}, "
          f"Time {summary['controlnet']['mean_time_sec']:.2f}s/img ({summary['controlnet']['total_time_sec']:.1f}s total)")
    print(f"    Mask region    : PSNR {summary['controlnet']['mean_mask_psnr']:.4f} dB, "
          f"SSIM {summary['controlnet']['mean_mask_ssim']:.6f}, "
          f"LPIPS {summary['controlnet']['mean_mask_lpips']:.6f}")
    print(f"  Output directory : {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

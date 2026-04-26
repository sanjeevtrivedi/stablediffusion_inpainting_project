"""
LoRA fine-tuning on ControlNet for inpainting, then inference with saved adapters.

End-to-end flow
───────────────
1. Download Flowers102 via torchvision (auto-downloaded on first run)
2. Fine-tune ControlNet attention blocks with LoRA adapters
3. Save LoRA adapters to disk
4. Reload adapters and run inference on held-out demo samples
5. Save per-image PSNR / SSIM metrics and comparison panels

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BASIC USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Minimal run (all defaults — auto-selects best available device):
    python scripts/04_run_finetuning.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEVICE SELECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Force a CUDA GPU (Linux / Windows with NVIDIA card):
    python scripts/04_run_finetuning.py --device cuda

Force Apple Silicon MPS (Mac M-series):
    python scripts/04_run_finetuning.py --device mps

Force CPU (slow — for testing / environments without a GPU):
    python scripts/04_run_finetuning.py --device cpu

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRAINING CONFIGURATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Quick smoke-test (few steps, few samples):
    python scripts/04_run_finetuning.py \
        --train-steps 20 \
        --train-samples 16 \
        --demo-samples 2

Longer run for better convergence:
    python scripts/04_run_finetuning.py \
        --train-steps 500 \
        --train-samples 256 \
        --demo-samples 10

Use the full dataset (no limit) with mixed masks:
    python scripts/04_run_finetuning.py \
        --dataset-limit 0 \
        --mask-type mixed \
        --train-steps 300

Train with only center masks:
    python scripts/04_run_finetuning.py --mask-type center

Train with only irregular masks:
    python scripts/04_run_finetuning.py --mask-type irregular

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATA AND OUTPUT PATHS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Custom dataset root and output directory:
    python scripts/04_run_finetuning.py \
        --dataset-root data/finetune_dataset \
        --output-dir outputs/my_lora_run

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMBINED EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Full GPU run with custom paths and reproducible seed:
    python scripts/04_run_finetuning.py \
        --device cuda \
        --dataset-root data/finetune_dataset \
        --output-dir outputs/finetune_gpu \
        --train-steps 300 \
        --train-samples 200 \
        --demo-samples 8 \
        --mask-type irregular \
        --seed 7

Apple Silicon MPS run with moderate settings:
    python scripts/04_run_finetuning.py \
        --device mps \
        --train-steps 150 \
        --train-samples 128 \
        --demo-samples 6 \
        --image-size 512 \
        --seed 42

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT LAYOUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
outputs/finetune_controlnet/   (or --output-dir)
  controlnet_lora/             ← saved LoRA adapter weights
  demo_inference/
    predictions/               ← inpainted images
    panels/                    ← 4-column comparison panels
    metrics.csv                ← per-image PSNR and SSIM
    summary.json               ← aggregate inference metrics
  run_summary.json             ← full run configuration and results

KEY PARAMETERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  --device           cuda | mps | cpu  (default: auto-select)
  --train-steps      LoRA gradient update steps [20–2000]. (default: 100)
  --train-samples    Training images drawn from the dataset [16+]. (default: 128)
  --demo-samples     Images used for post-training inference. (default: 6)
  --dataset-limit    Cap on total dataset images loaded (0 = no limit). (default: 200)
  --mask-type        center | irregular | mixed. (default: mixed)
  --image-size       Must be a multiple of 8; SD 1.x trained at 512. (default: 512)
  --seed             Global seed for reproducibility. (default: 42)
  --dataset-root     Path where Flowers102 is downloaded. (default: data/finetune_dataset)
  --output-dir       Root output directory for this run. (default: outputs/finetune_controlnet)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import (
    ControlNetModel,
    DDIMScheduler,
    StableDiffusionControlNetInpaintPipeline,
)
from peft import LoraConfig
from PIL import Image
from torchvision.datasets import Flowers102

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import apply_mask, generate_center_mask, generate_irregular_mask
from src.eval.metrics import MetricResult, compute_psnr, compute_ssim, save_metrics_csv
from src.eval.visualize import save_comparison_panel


DEFAULT_MODEL_ID = "runwayml/stable-diffusion-inpainting"
DEFAULT_CONTROLNET_ID = "lllyasviel/control_v11p_sd15_inpaint"
DEFAULT_DATASET_LIMIT = 200
DEFAULT_BATCH_SIZE = 1
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_LORA_RANK = 4
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_NUM_INFERENCE_STEPS = 40
DEFAULT_DDIM_ETA = 0.0
DEFAULT_CONTROLNET_SCALE = 1.0


# -----------------------------------------------------------------------------
# Dataset Helpers
# -----------------------------------------------------------------------------


def load_flowers_dataset(root: Path, split: str, download: bool) -> Flowers102:
    return Flowers102(
        root=str(root),
        split=split,
        download=download,
    )


def choose_indices(total: int, max_samples: int, seed: int) -> list[int]:
    indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(indices)
    if max_samples > 0:
        return indices[: min(max_samples, total)]
    return indices


def make_prompt(dataset: Flowers102, label: int) -> str:
    class_names = [name.replace("_", " ").lower() for name in dataset.classes]
    idx = int(label) - 1
    class_name = class_names[idx] if 0 <= idx < len(class_names) else "flower"
    return f"a high quality photo of a {class_name}"


def make_mask(width: int, height: int, mask_type: str, rng: random.Random) -> Image.Image:
    if mask_type == "center":
        return generate_center_mask(width, height)
    if mask_type == "irregular":
        return generate_irregular_mask(width, height)
    if rng.random() < 0.5:
        return generate_center_mask(width, height)
    return generate_irregular_mask(width, height)


def prepare_sample(
    dataset: Flowers102,
    sample_idx: int,
    image_size: int,
    mask_type: str,
    rng: random.Random,
) -> dict[str, Image.Image | str]:
    image, label = dataset[sample_idx]
    image = image.convert("RGB").resize((image_size, image_size))
    mask = make_mask(image_size, image_size, mask_type, rng)
    corrupted = apply_mask(image, mask, fill_value=255)
    control_image = make_control_image(corrupted)

    return {
        "image": image,
        "mask": mask,
        "corrupted": corrupted,
        "control_image": control_image,
        "prompt": make_prompt(dataset, int(label)),
        "name": f"flower_{sample_idx:05d}.png",
    }


def make_training_batch(
    dataset: Flowers102,
    indices: list[int],
    image_size: int,
    mask_type: str,
    batch_size: int,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    batch_items = [
        prepare_sample(dataset, rng.choice(indices), image_size, mask_type, rng)
        for _ in range(batch_size)
    ]

    images = torch.stack([pil_to_image_tensor(item["image"]) for item in batch_items])
    masks = torch.stack([pil_to_mask_tensor(item["mask"]) for item in batch_items])
    control_images = torch.stack([pil_to_control_tensor(item["control_image"]) for item in batch_items])
    prompts = [str(item["prompt"]) for item in batch_items]
    return images, masks, control_images, prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple demo: fine-tune a ControlNet LoRA adapter, save it, and use it for inference"
    )

    parser.add_argument("--dataset-root", type=Path, default=Path("data/finetune_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/finetune_controlnet"))
    parser.add_argument(
        "--dataset-limit",
        type=int,
        default=DEFAULT_DATASET_LIMIT,
        help="Max number of dataset images to use (0 = all)",
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--mask-type", type=str, default="mixed", choices=["center", "irregular", "mixed"])
    parser.add_argument("--train-samples", type=int, default=128)
    parser.add_argument("--demo-samples", type=int, default=6)
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help=(
            "Force a specific device. Omit to auto-select: "
            "CUDA (GPU) > MPS (Apple Silicon) > CPU."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate all CLI arguments before any model or data is loaded.

    Hard errors call sys.exit(1); soft issues are printed as warnings
    and execution continues.
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
            f"--image-size {args.image_size}: SD 1.x ControlNet is trained at 512 px; "
            "other sizes may reduce quality."
        )

    # ------------------------------------------------------------------ #
    # train_steps                                                          #
    # ------------------------------------------------------------------ #
    if args.train_steps < 1:
        errors.append(f"--train-steps must be >= 1, got {args.train_steps}.")
    elif args.train_steps < 20:
        warnings.append(
            f"--train-steps {args.train_steps} is very low. LoRA adapters typically "
            "need at least 50–100 steps to learn anything meaningful."
        )
    elif args.train_steps > 2000:
        warnings.append(
            f"--train-steps {args.train_steps} is high for a small LoRA fine-tune. "
            "Risk of overfitting on the training samples."
        )

    # ------------------------------------------------------------------ #
    # train_samples                                                        #
    # ------------------------------------------------------------------ #
    if args.train_samples < 0:
        errors.append(
            f"--train-samples must be >= 0 (0 = use all available), got {args.train_samples}."
        )
    elif args.train_samples == 0:
        warnings.append(
            "--train-samples 0: the entire training split will be used. "
            "This may be slow to iterate through."
        )
    elif args.train_samples < 16:
        warnings.append(
            f"--train-samples {args.train_samples} is very small. "
            "Fewer than 16 samples risk producing an overfit or degenerate adapter."
        )

    # ------------------------------------------------------------------ #
    # demo_samples                                                         #
    # ------------------------------------------------------------------ #
    if args.demo_samples < 0:
        errors.append(
            f"--demo-samples must be >= 0 (0 = use all available), got {args.demo_samples}."
        )
    elif args.demo_samples == 0:
        warnings.append(
            "--demo-samples 0: all test-split images will be used for inference. "
            "This may be slow."
        )

    # ------------------------------------------------------------------ #
    # dataset_limit                                                        #
    # ------------------------------------------------------------------ #
    if args.dataset_limit < 0:
        errors.append(
            f"--dataset-limit must be >= 0 (0 = no limit), got {args.dataset_limit}."
        )
    elif args.dataset_limit == 0:
        warnings.append(
            "--dataset-limit 0: the full dataset will be loaded. "
            "This may be slow on first download."
        )

    # ------------------------------------------------------------------ #
    # seed                                                                 #
    # ------------------------------------------------------------------ #
    if args.seed < 0:
        errors.append(
            f"--seed {args.seed} is negative. PyTorch Generator requires a non-negative seed."
        )

    # ------------------------------------------------------------------ #
    # dataset_root                                                         #
    # ------------------------------------------------------------------ #
    if args.dataset_root.exists() and not args.dataset_root.is_dir():
        errors.append(
            f"--dataset-root '{args.dataset_root}' exists but is not a directory."
        )

    # ------------------------------------------------------------------ #
    # train_samples vs demo_samples cross-check                           #
    # ------------------------------------------------------------------ #
    if args.train_samples > 0 and args.demo_samples > args.train_samples:
        warnings.append(
            f"--demo-samples ({args.demo_samples}) > --train-samples ({args.train_samples}). "
            "Evaluating on more images than were trained on can skew comparison results."
        )

    # ------------------------------------------------------------------ #
    # device                                                              #
    # ------------------------------------------------------------------ #
    if args.device == "cuda" and not torch.cuda.is_available():
        errors.append(
            "--device cuda requested but no CUDA-capable GPU was found on this machine."
        )
    if args.device == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        errors.append(
            "--device mps requested but MPS is not available on this machine "
            "(requires Apple Silicon Mac with macOS 12.3+)."
        )
    if args.device == "cpu":
        warnings.append(
            "--device cpu: training LoRA on CPU is extremely slow and not recommended. "
            "Use --device cuda (GPU) or --device mps (Apple Silicon) if available."
        )

    # ------------------------------------------------------------------ #
    # LoRA hyper-parameter sanity (module-level constants)                #
    # ------------------------------------------------------------------ #
    if DEFAULT_LORA_RANK < 1:
        errors.append(f"DEFAULT_LORA_RANK must be >= 1, got {DEFAULT_LORA_RANK}.")
    elif DEFAULT_LORA_RANK > 64:
        warnings.append(
            f"DEFAULT_LORA_RANK={DEFAULT_LORA_RANK} is high for ControlNet LoRA. "
            "Ranks above 32 rarely improve quality and increase VRAM/memory usage."
        )
    if DEFAULT_LEARNING_RATE <= 0:
        errors.append(f"DEFAULT_LEARNING_RATE must be > 0, got {DEFAULT_LEARNING_RATE}.")
    elif DEFAULT_LEARNING_RATE > 1e-2:
        warnings.append(
            f"DEFAULT_LEARNING_RATE={DEFAULT_LEARNING_RATE} is very high for LoRA. "
            "Typical range is 1e-5 to 1e-4; higher values may destabilise training."
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


# -----------------------------------------------------------------------------
# Image / Tensor Helpers
# -----------------------------------------------------------------------------


def get_device(requested: str | None = None) -> str:
    """Return the device string to use for all tensors.

    If *requested* is provided the value is trusted (validated earlier by
    validate_args).  Otherwise the best available device is selected
    automatically: CUDA GPU > Apple MPS > CPU.
    """
    if requested is not None:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_control_image(image: Image.Image) -> Image.Image:
    """Build a 3-channel Canny edge map for ControlNet conditioning."""
    gray = np.asarray(image.convert("L"))

    try:
        import cv2  # type: ignore

        edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    except Exception:
        gx = np.abs(np.diff(gray.astype(np.float32), axis=1, prepend=gray[:, :1]))
        gy = np.abs(np.diff(gray.astype(np.float32), axis=0, prepend=gray[:1, :]))
        edges = np.clip(gx + gy, 0, 255).astype(np.uint8)

    edges_rgb = np.stack([edges, edges, edges], axis=-1)
    return Image.fromarray(edges_rgb)


def pil_to_image_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def pil_to_mask_tensor(mask: Image.Image) -> torch.Tensor:
    arr = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def pil_to_control_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


# -----------------------------------------------------------------------------
# Model Helpers
# -----------------------------------------------------------------------------



def encode_images_to_latents(pipe: StableDiffusionControlNetInpaintPipeline, pixel_values: torch.Tensor) -> torch.Tensor:
    """Convert input images into VAE latents used by the diffusion model."""
    with torch.no_grad():
        latents = pipe.vae.encode(pixel_values).latent_dist.sample()
    return latents * pipe.vae.config.scaling_factor


def add_training_noise(
    pipe: StableDiffusionControlNetInpaintPipeline,
    latents: torch.Tensor,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample random timesteps and noise, then corrupt the clean latents."""
    noise = torch.randn_like(latents)
    timesteps = torch.randint(
        0,
        pipe.scheduler.config.num_train_timesteps,
        (latents.shape[0],),
        device=device,
        dtype=torch.long,
    )
    noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
    return noisy_latents, noise, timesteps


def build_inpaint_model_input(latents: torch.Tensor, noisy_latents: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Build the 9-channel inpainting input: noisy latents + mask + known-region latents."""
    mask_latent = F.interpolate(mask, size=latents.shape[-2:])
    masked_latents = latents * (1.0 - mask_latent)
    return torch.cat([noisy_latents, mask_latent, masked_latents], dim=1)


def encode_prompts(
    pipe: StableDiffusionControlNetInpaintPipeline,
    prompts: list[str],
    device: str,
) -> torch.Tensor:
    """Tokenize prompts and get text embeddings once per batch."""
    with torch.no_grad():
        text_inputs = pipe.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipe.tokenizer.model_max_length,
        )
        input_ids = text_inputs.input_ids.to(device)
        return pipe.text_encoder(input_ids)[0]


def get_training_target(
    pipe: StableDiffusionControlNetInpaintPipeline,
    latents: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    if pipe.scheduler.config.prediction_type == "epsilon":
        return noise
    if pipe.scheduler.config.prediction_type == "v_prediction":
        return pipe.scheduler.get_velocity(latents, noise, timesteps)
    raise ValueError(f"Unsupported prediction type: {pipe.scheduler.config.prediction_type}")


# -----------------------------------------------------------------------------
# Train / Inference
# -----------------------------------------------------------------------------


def train_lora(args: argparse.Namespace, device: str) -> Path:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    lora_dir = output_dir / "controlnet_lora"

    print("Loading dataset for fine-tuning...")
    train_dataset = load_flowers_dataset(args.dataset_root, split="train", download=True)
    available_indices = choose_indices(len(train_dataset), args.dataset_limit, args.seed)
    if not available_indices:
        raise RuntimeError("Training dataset is empty.")
    if args.train_samples > 0:
        train_indices = available_indices[: min(args.train_samples, len(available_indices))]
    else:
        train_indices = available_indices
    data_rng = random.Random(args.seed)

    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading base model: {DEFAULT_MODEL_ID}")
    print(f"Loading ControlNet: {DEFAULT_CONTROLNET_ID}")

    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        DEFAULT_MODEL_ID,
        controlnet=ControlNetModel.from_pretrained(
            DEFAULT_CONTROLNET_ID,
            torch_dtype=dtype,
            use_safetensors=False,
        ),
        torch_dtype=dtype,
        use_safetensors=False,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    if device in ("mps", "cpu"):
        pipe.safety_checker = None

    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    pipe.controlnet.requires_grad_(False)

    lora_config = LoraConfig(
        r=DEFAULT_LORA_RANK,
        lora_alpha=DEFAULT_LORA_RANK,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    pipe.controlnet.add_adapter(lora_config)

    trainable_params = [p for p in pipe.controlnet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=DEFAULT_LEARNING_RATE)

    print(f"Starting LoRA fine-tune for {args.train_steps} steps...")
    for step in range(1, args.train_steps + 1):
        pixel_values, mask, control_image, prompts = make_training_batch(
            dataset=train_dataset,
            indices=train_indices,
            image_size=args.image_size,
            mask_type=args.mask_type,
            batch_size=DEFAULT_BATCH_SIZE,
            rng=data_rng,
        )
        pixel_values = pixel_values.to(device=device, dtype=dtype)
        mask = mask.to(device=device, dtype=dtype)
        control_image = control_image.to(device=device, dtype=dtype)

        # Encode images into latent space because diffusion training happens on latents, not pixels.
        latents = encode_images_to_latents(pipe, pixel_values)

        # Add random noise at random timesteps; the model learns to predict this corruption.
        noisy_latents, noise, timesteps = add_training_noise(pipe, latents, device)
        model_input = build_inpaint_model_input(latents, noisy_latents, mask)
        encoder_hidden_states = encode_prompts(pipe, list(prompts), device)

        # ControlNet reads the edge map and produces extra residuals that guide the UNet.
        # ControlNet conv_in expects 4-channel noisy latents, not the 9-channel inpainting input.
        down_block_res_samples, mid_block_res_sample = pipe.controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=control_image,
            conditioning_scale=DEFAULT_CONTROLNET_SCALE,
            return_dict=False,
        )

        noise_pred = pipe.unet(
            model_input,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
        ).sample

        target = get_training_target(pipe, latents, noise, timesteps)

        loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % 10 == 0 or step == args.train_steps:
            print(f"  step {step}/{args.train_steps} | loss={loss.item():.6f}")

    lora_dir.mkdir(parents=True, exist_ok=True)
    pipe.controlnet.save_lora_adapter(lora_dir)
    print(f"Saved ControlNet LoRA adapters to: {lora_dir}")

    return lora_dir


def run_inference_with_saved_lora(args: argparse.Namespace, lora_dir: Path, device: str) -> dict[str, float]:
    demo_output = args.output_dir / "demo_inference"
    predictions_dir = demo_output / "predictions"
    panels_dir = demo_output / "panels"
    demo_output.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    panels_dir.mkdir(parents=True, exist_ok=True)

    demo_dataset = load_flowers_dataset(args.dataset_root, split="test", download=True)
    demo_indices = choose_indices(len(demo_dataset), args.dataset_limit, args.seed + 7)
    if not demo_indices:
        raise RuntimeError("Demo dataset is empty.")
    if args.demo_samples > 0:
        demo_indices = demo_indices[: min(args.demo_samples, len(demo_indices))]
    demo_rng = random.Random(args.seed + 7)

    dtype = torch.float16 if device == "cuda" else torch.float32

    print("Reloading base ControlNet and loading saved LoRA adapters for inference...")
    controlnet = ControlNetModel.from_pretrained(
        DEFAULT_CONTROLNET_ID,
        torch_dtype=dtype,
        use_safetensors=False,
    )
    controlnet.load_lora_adapter(lora_dir, prefix=None)

    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        DEFAULT_MODEL_ID,
        controlnet=controlnet,
        torch_dtype=dtype,
        use_safetensors=False,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    if device in ("mps", "cpu"):
        pipe.safety_checker = None

    results: list[MetricResult] = []

    for idx, sample_idx in enumerate(demo_indices):
        item = prepare_sample(
            dataset=demo_dataset,
            sample_idx=sample_idx,
            image_size=args.image_size,
            mask_type=args.mask_type,
            rng=demo_rng,
        )
        image = item["image"]
        mask = item["mask"]
        corrupted = item["corrupted"]
        control_image = item["control_image"]
        prompt = item["prompt"]
        name = item["name"]

        generator = torch.Generator(device=device).manual_seed(args.seed + idx)

        out = pipe(
            prompt=prompt,
            image=corrupted,
            mask_image=mask,
            control_image=control_image,
            guidance_scale=DEFAULT_GUIDANCE_SCALE,
            controlnet_conditioning_scale=DEFAULT_CONTROLNET_SCALE,
            num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS,
            eta=DEFAULT_DDIM_ETA,
            generator=generator,
        )
        prediction = out.images[0]

        psnr = compute_psnr(image, prediction)
        ssim = compute_ssim(image, prediction)
        results.append(MetricResult(image_name=name, psnr=psnr, ssim=ssim))

        prediction.save(predictions_dir / name)
        save_comparison_panel(
            original=image,
            mask=mask,
            corrupted=corrupted,
            prediction=prediction,
            out_path=panels_dir / f"{Path(name).stem}_panel.png",
            title=f"{name} | PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}",
        )

        print(f"  demo [{idx + 1}/{len(demo_indices)}] {name} | PSNR={psnr:.3f} SSIM={ssim:.4f}")

    save_metrics_csv(results, demo_output / "metrics.csv")

    mean_psnr = float(sum(r.psnr for r in results) / len(results)) if results else 0.0
    mean_ssim = float(sum(r.ssim for r in results) / len(results)) if results else 0.0

    summary = {
        "count": len(results),
        "mean_psnr": mean_psnr,
        "mean_ssim": mean_ssim,
    }
    with (demo_output / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    args = parse_args()
    validate_args(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device(args.device)
    print(f"Using device: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    lora_dir = train_lora(args, device)
    demo_summary = run_inference_with_saved_lora(args, lora_dir, device)

    run_summary = {
        "model_id": DEFAULT_MODEL_ID,
        "controlnet_id": DEFAULT_CONTROLNET_ID,
        "dataset": "Flowers102 (torchvision)",
        "dataset_root": str(args.dataset_root),
        "dataset_limit": args.dataset_limit,
        "train_samples": args.train_samples,
        "train_steps": args.train_steps,
        "lora_rank": DEFAULT_LORA_RANK,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "lora_dir": str(lora_dir),
        "demo_samples": args.demo_samples,
        "demo_mean_psnr": demo_summary["mean_psnr"],
        "demo_mean_ssim": demo_summary["mean_ssim"],
        "device": device,
    }

    with (args.output_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    print("\n" + "=" * 72)
    print("Fine-tuning + inference demo complete")
    print(f"LoRA directory: {lora_dir}")
    print(f"Demo mean PSNR: {demo_summary['mean_psnr']:.4f}")
    print(f"Demo mean SSIM: {demo_summary['mean_ssim']:.6f}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()

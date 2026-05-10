# Stable Diffusion Inpainting – Comparison Project

A comparative evaluation of **Standard Stable Diffusion Inpainting** vs **ControlNet-guided Inpainting** using quantitative metrics (PSNR, SSIM, LPIPS) computed on both the whole image and the masked region.

## Project Structure

```
├── scripts/
│   ├── 01_download_images.py      # Download sample images from picsum.photos
│   └── 02_run_comparison.py       # Run SD vs ControlNet comparison pipeline
├── src/
│   ├── data/
│   │   └── dataset.py             # Image loading, mask generation, corruption
│   └── eval/
│       ├── metrics.py             # PSNR, SSIM, LPIPS (whole-image & masked)
│       └── visualize.py           # Side-by-side comparison panel generation
├── data/
│   └── samples/                   # Downloaded input images
├── outputs/
│   ├── standard/                  # Standard SD results (metrics + panels)
│   ├── controlnet/                # ControlNet results (metrics + panels)
│   └── comparison/                # Side-by-side comparison results
├── requirements.txt
└── README.md
```

## Setup

**Prerequisites:** Python 3.10+, pip

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `torch`, `torchvision` | Tensor operations, GPU acceleration |
| `diffusers` | Stable Diffusion & ControlNet pipelines |
| `transformers` | Model tokenizers and encoders |
| `accelerate` | Efficient model loading |
| `Pillow` | Image I/O and manipulation |
| `numpy` | Numerical operations |
| `scikit-image` | SSIM computation |
| `lpips` | Learned perceptual similarity metric |
| `matplotlib` | Comparison panel visualisation |
| `opencv-python` | Image processing utilities |

## Usage

### 1. Download Sample Images

```bash
python scripts/01_download_images.py --count 10 --output-dir data/samples
```

| Flag | Default | Description |
|------|---------|-------------|
| `--count` | 10 | Number of images to download |
| `--output-dir` | `data/samples` | Destination directory |
| `--width` | 512 | Image width |
| `--height` | 512 | Image height |
| `--seed-offset` | 100 | Seed for deterministic image selection |
| `--start-index` | 1 | Starting filename index |

### 2. Run Comparison Pipeline

```bash
python scripts/02_run_comparison.py
```

This runs both pipelines on every image in `data/samples/` and saves results to `outputs/comparison/`.

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | `data/samples` | Input image directory |
| `--output-dir` | `outputs/comparison` | Output directory |
| `--mask-type` | `center` | Mask type: `center` or `irregular` |
| `--num-steps` | 50 | DDIM denoising steps |
| `--guidance-scale` | 7.5 | Classifier-free guidance scale |
| `--image-size` | 512 | Resize images to this resolution |
| `--seed` | 42 | Base random seed |
| `--sd-model-id` | `runwayml/stable-diffusion-inpainting` | SD model identifier |
| `--controlnet-model-id` | `lllyasviel/control_v11p_sd15_inpaint` | ControlNet model identifier |

**Output files:**

- `predictions/sd-*.jpg` and `predictions/cn-*.jpg` – inpainted images
- `panels/sd-*_panel.png` and `panels/cn-*_panel.png` – 4-column comparison panels (Original | Mask | Corrupted | Inpainted)
- `metrics.csv` – per-image PSNR, SSIM, LPIPS (whole + mask region)
- `summary.json` – aggregated mean metrics and experiment configuration

## Evaluation Metrics

| Metric | Measures | Range | Better |
|--------|----------|-------|--------|
| **PSNR** | Pixel-level fidelity | 0 – ∞ dB | Higher |
| **SSIM** | Structural similarity (luminance, contrast, structure) | 0 – 1 | Higher |
| **LPIPS** | Perceptual similarity via deep features (AlexNet) | 0 – 1 | Lower |

Each metric is computed in two variants:
- **Whole-image**: evaluates the entire reconstructed image against the original
- **Mask-region**: restricts evaluation to the inpainted area only

## Models Used

- **Standard SD Inpainting**: [`runwayml/stable-diffusion-inpainting`](https://huggingface.co/runwayml/stable-diffusion-inpainting) – fine-tuned SD 1.5 for mask-conditioned inpainting
- **ControlNet Inpainting**: [`lllyasviel/control_v11p_sd15_inpaint`](https://huggingface.co/lllyasviel/control_v11p_sd15_inpaint) – ControlNet v1.1 adding structural guidance to SD inpainting

Both pipelines use a **DDIM scheduler** for deterministic, eta-controlled sampling.

## Device Support

The pipeline auto-selects the best available device:
- **CUDA** (NVIDIA GPU) – uses float16 for speed
- **MPS** (Apple Silicon) – uses float32
- **CPU** – fallback, uses float32

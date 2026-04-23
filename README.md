# Diffusion Inpainting Project

## Problem Statement

Image inpainting using diffusion models. Given an image with missing regions (marked by a mask), generate realistic content inside masked regions while preserving consistency with the unmasked context.

## Quick Start

```bash
# 1. Download 20 sample images
python scripts/01_download_images.py --count 20

# 2. Prepare dataset (split + generate masks)
python scripts/02_prepare_data.py

# 3. Run inpainting pipeline
python scripts/03_run_inpainting.py --data-dir data/samples --mask-type center
```

**Output:** Predictions, visual panels, PSNR/SSIM metrics in `outputs/`

## Dataset

- **Source:** picsum.photos (20 random images @ 512×512)
- **Masks:** Center (fixed rectangle) and Irregular (random strokes)
- **Evaluation:** Ground truth = original image; no corrupted ground truth needed

## Method

Single baseline: Stable Diffusion Inpainting + Classifier-Free Guidance (CFG) + DDIM sampling.

- **Model:** runwayml/stable-diffusion-inpainting (pretrained)
- **Config:** CFG=7.5, steps=50 (default)
- **No training required**

## Evaluation Metrics

| Metric | Measures |
|---|---|
| **PSNR** | Peak Signal-to-Noise Ratio (dB). Higher = better pixel fidelity. Typical range: 20–28 dB |
| **SSIM** | Structural Similarity Index. Range [0,1]. Higher = better perceptual match |
| **Visual panels** | 4-column grid: Original | Mask | Corrupted | Inpainted |

## Outputs

- **predictions/** — Per-image inpainted results
- **panels/** — Visual comparison grids with metrics
- **metrics.csv** — Per-image PSNR and SSIM
- **summary.json** — Mean metrics + run configuration

## Repository Structure

```
project-root/
├── README.md                 # This file
├── MID_SEMESTER.md           # Mid-semester requirements (all 5 items)
├── project-details.md        # Task description & references
├── requirements.txt          # Python dependencies
├── data/
│   ├── samples/              # Downloaded images (auto-created)
│   └── splits/               # Train/val/test splits + masks (auto-created)
├── src/
│   ├── data/dataset.py       # Load images, generate masks, corrupt
│   └── eval/
│       ├── metrics.py        # Compute PSNR, SSIM
│       └── visualize.py      # 4-panel comparison grids
├── scripts/
│   ├── 01_download_images.py # Download from picsum.photos
│   ├── 02_prepare_data.py    # Split dataset, generate masks
│   └── 03_run_inpainting.py  # Main inpainting + evaluation
└── outputs/                  # Results (auto-created)
    ├── predictions/          # Inpainted images
    ├── panels/               # Visual comparison grids
    ├── metrics.csv           # Per-image metrics
    └── summary.json          # Mean metrics + config
```

## Setup & Installation

```bash
# 1. Create and activate Python environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Running the Simplified Workflow

### Step 1: Download sample images

```bash
python scripts/01_download_images.py --count 20 --output-dir data/samples
```

Important options:

| Argument | Default | Description |
|---|---|---|
| `--count` | `10` | Number of images to download |
| `--output-dir` | `data/samples` | Folder for downloaded images |
| `--width` | `512` | Image width |
| `--height` | `512` | Image height |

### Step 2: Prepare dataset and masks

```bash
python scripts/02_prepare_data.py --data-dir data/samples --output-dir data/splits
```

This script:
- creates train, validation, and test folders,
- copies resized images,
- generates `center` and `irregular` masks,
- writes `data/splits/manifest.csv`.

### Step 3: Run inpainting and evaluation

```bash
python scripts/03_run_inpainting.py \
  --data-dir data/samples \
  --output-dir outputs \
  --mask-type center \
  --guidance-scale 7.5 \
  --num-steps 50
```

Important options:

| Argument | Default | Description |
|---|---|---|
| `--mask-type` | `center` | `center` or `irregular` |
| `--guidance-scale` | `7.5` | Classifier-free guidance scale |
| `--num-steps` | `50` | DDIM inference steps |
| `--seed` | `42` | Random seed for reproducibility |

## Example Experiments

Baseline:

```bash
python scripts/03_run_inpainting.py --data-dir data/samples --mask-type center
```

Irregular masks:

```bash
python scripts/03_run_inpainting.py --data-dir data/samples --mask-type irregular
```

Faster ablation:

```bash
python scripts/03_run_inpainting.py --data-dir data/samples --mask-type center --num-steps 30 --guidance-scale 9.0
```

## Result Files

After a run, the `outputs/` folder contains:

- `predictions/` — inpainted images
- `panels/` — 4-panel comparison grids
- `metrics.csv` — per-image PSNR and SSIM
- `summary.json` — mean metrics and run configuration

## Notes

- This simplified project uses a single pretrained inpainting pipeline.
- There is no training, LoRA fine-tuning, ControlNet, or RePaint stage.
- The focus is on dataset preparation, inpainting, experimentation, evaluation, and visualization.
- Mid-semester documentation is in `MID_SEMESTER.md`.




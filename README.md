# Diffusion Inpainting Project

This project implements image inpainting with diffusion models in 3 levels of complexity.

- Level 1 (Quick win): Stable Diffusion Inpainting with CFG + mask and DDIM sampling
- Level 2 (Production): Fine-tuned latent diffusion, mask-aware conditioning, prompt + image guidance, boundary smoothing
- Level 3 (SOTA): ControlNet + inpainting, RePaint resampling, DiT backbone, plug-and-play guidance

The current implementation includes a fully runnable Level 1 pipeline, shared data/evaluation modules, and clear extension stubs for Levels 2 and 3.

## Repository Structure

```text
.
|- configs/
|  |- level1.yaml
|  |- level2.yaml
|  \- level3.yaml
|- data/
|  \- samples/                  # Add a few sample images here
|- outputs/                     # Predictions, panels, and metrics
|- reports/
|  \- results_summary.md
|- scripts/
|  |- evaluate.py
|  |- run_level1.py
|  |- run_level2.py
|  \- run_level3.py
|- src/
|  |- data/dataset.py
|  |- eval/metrics.py
|  |- eval/visualize.py
|  |- models/level1_sd_inpaint.py
|  |- models/level2_finetune_ldm.py
|  \- models/level3_advanced.py
\- requirements.txt
```

## Setup

1. Create and activate a Python environment (recommended: Python 3.10+).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add a small set of images to `data/samples`.

## Run

### Level 1 (Implemented)

```bash
python scripts/run_level1.py --data-dir data/samples --output-dir outputs/level1
```

You can control prompt, mask type, DDIM steps, and CFG scale:

```bash
python scripts/run_level1.py \
	--prompt "A realistic scene completion" \
	--mask-type irregular \
	--num-steps 40 \
	--guidance-scale 8.0
```

### Evaluate predictions separately

```bash
python scripts/evaluate.py --targets data/samples --predictions outputs/level1/predictions --output outputs/eval
```

### Level 2 and Level 3

`scripts/run_level2.py` and `scripts/run_level3.py` are extension entrypoints and currently contain explicit `NotImplementedError` scaffolds to keep implementation simple and organized.

## Dataset Plan (Small-Scale)

Use a small subset to avoid large-scale training cost:

- Option A: CelebA-HQ subset (easy to evaluate structure and texture)
- Option B: Places2 subset (more diverse scenes)

Suggested minimal split for presentation:

- Train: 100-300 images (for Level 2 prototype fine-tuning)
- Validation: 20-50 images
- Test: 20-50 images

Mask strategies used in this project:

- Center mask
- Irregular brush mask

## Evaluation Metrics

The project uses PSNR, SSIM, and visual panels.

For target image $x$ and prediction $\hat{x}$ with $N$ pixels:

$$
	ext{MSE} = \frac{1}{N}\sum_{i=1}^{N}(x_i - \hat{x}_i)^2
$$

$$
	ext{PSNR} = 20\log_{10}(MAX_I) - 10\log_{10}(\text{MSE})
$$

SSIM is computed channel-wise with luminance, contrast, and structure terms (implemented through `skimage.metrics.structural_similarity`).

## Mathematics by Level

### Level 1: Stable Diffusion Inpainting + CFG + DDIM

Forward diffusion process:

$$
q(x_t\mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)
$$

Noise prediction objective (standard denoising loss):

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{x_0,\epsilon,t}\left[\|\epsilon - \epsilon_\theta(x_t, t, c)\|_2^2\right]
$$

Classifier-Free Guidance (CFG):

$$
\hat{\epsilon} = \epsilon_\theta(x_t, t, \varnothing) + s\left(\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \varnothing)\right)
$$

where $s$ is guidance scale.

Inpainting uses a binary mask $m$ (1 for region to regenerate), where denoising is constrained by known context outside mask.

DDIM update (deterministic when $\eta=0$):

$$
x_{t-1} = \sqrt{\alpha_{t-1}}\hat{x}_0 + \sqrt{1-\alpha_{t-1}}\hat{\epsilon}
$$

### Level 2: Production Fine-Tuned Latent Diffusion

Latent diffusion operates in latent space $z$ via VAE encoder/decoder:

$$
z = \mathcal{E}(x), \quad \hat{x} = \mathcal{D}(z)
$$

Training objective in latent space:

$$
\mathcal{L}_{\text{LDM}} = \mathbb{E}_{z_0,\epsilon,t}\left[\|\epsilon - \epsilon_\theta(z_t, t, c, m)\|_2^2\right]
$$

Mask-aware conditioning introduces explicit mask $m$ and corrupted latent context $z \odot (1-m)$.

Boundary smoothing blends generated and known pixels near mask edge:

$$
x_{\text{blend}} = w \odot x_{\text{gen}} + (1-w) \odot x_{\text{context}}
$$

where $w$ is a soft edge map from mask dilation/blur.

### Level 3: SOTA Extensions

ControlNet conditions denoising on additional structural inputs $u$ (edges, segmentation, depth):

$$
\epsilon_\theta(x_t, t, c, u)
$$

RePaint resampling repeatedly re-noises and denoises masked regions to improve consistency between known and unknown regions.

DiT (Diffusion Transformer) replaces UNet-style denoiser with a transformer denoiser over latent tokens:

$$
\epsilon_\theta(\text{tokens}(z_t), t, c)
$$

Plug-and-play guidance adds external gradient-based guidance at inference without full retraining.

## Suggested Experiments

1. Level 1 baseline grid search:
- CFG scale: 5.0, 7.5, 10.0
- DDIM steps: 20, 30, 50
- Mask type: center vs irregular

2. Level 2 ablation:
- With vs without mask-aware conditioning
- With vs without boundary smoothing
- Prompt-only vs prompt + image guidance

3. Level 3 constrained trial:
- Base Level 2 vs +ControlNet vs +RePaint

Report mean PSNR/SSIM and qualitative panels for each setting.

## Presentation Checklist

- Problem statement and motivation
- Level-wise method diagram
- Metric table (PSNR/SSIM/runtime)
- Visual quality comparison panels
- Failure analysis and insights
- Future work and constraints

## References (Papers and Articles)

1. Ho et al., "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
2. Song et al., "Denoising Diffusion Implicit Models" (ICLR 2021)
3. Ho and Salimans, "Classifier-Free Diffusion Guidance" (2021)
4. Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models" (CVPR 2022)
5. Lugmayr et al., "RePaint: Inpainting using Denoising Diffusion Probabilistic Models" (CVPR 2022)
6. Zhang et al., "Adding Conditional Control to Text-to-Image Diffusion Models (ControlNet)" (ICCV 2023)
7. Peebles and Xie, "Scalable Diffusion Models with Transformers (DiT)" (ICCV 2023)

Official libraries and practical docs:

- Hugging Face Diffusers documentation (Stable Diffusion inpainting pipelines)
- CompVis / Stability AI resources for Stable Diffusion checkpoints


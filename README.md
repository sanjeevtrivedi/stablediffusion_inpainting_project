# Diffusion Inpainting Project

Image inpainting with diffusion models — fully implemented across 3 levels of increasing capability.

| Level | Name | Core techniques | Status |
|-------|------|----------------|--------|
| 1 | Quick Win | Stable Diffusion Inpainting, CFG, DDIM | ✅ Fully implemented |
| 2 | Production | Latent diffusion LoRA fine-tune, mask-aware conditioning, boundary smoothing | ✅ Fully implemented |
| 3 | SOTA | ControlNet + inpainting, RePaint resampling, DiT backbone, PnP guidance | ✅ Fully implemented |

## Repository Structure

```text
.
|- configs/
|  |- level1.yaml               # CFG scale, DDIM steps, model ID
|  |- level2.yaml               # LoRA settings, boundary smoothing flag
|  \- level3.yaml               # ControlNet, RePaint, DiT flags
|- data/
|  |- samples/                  # Raw downloaded images (10 × 512×512)
|  \- splits/                   # Created by prepare_dataset.py
|     |- train/images/
|     |- val/images/
|     |- test/images/
|     |- */masks/center/
|     |- */masks/irregular/
|     \- manifest.csv
|- outputs/                     # Per-level predictions, panels, metrics
|  |- level1/
|  |- level2/
|  |- level3/
|  \- sweep/
|- reports/
|  \- results_summary.md        # Fill after experiments
|- scripts/
|  |- prepare_dataset.py        # Split + pre-save masks
|  |- run_level1.py
|  |- run_level2.py
|  |- run_level3.py
|  |- evaluate.py               # Stand-alone evaluation against GT
|  \- sweep_experiments.py      # CFG × DDIM steps grid search
|- src/
|  |- data/dataset.py           # Image loading, mask generation, sample yield
|  |- eval/metrics.py           # PSNR, SSIM, CSV export
|  |- eval/visualize.py         # 4-panel comparison plots
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

3. Images are already in `data/samples` (10 × 512×512 downloaded via `picsum.photos`).

4. Prepare dataset splits and pre-save masks (optional but recommended before running experiments):

```bash
python scripts/prepare_dataset.py --data-dir data/samples --output-dir data/splits
```

This creates `train/val/test` folders, generates center and irregular masks for every image, and writes `data/splits/manifest.csv`.

## Run

### Step 0 — Prepare dataset

```bash
python scripts/prepare_dataset.py
```

### Level 1 — Quick Win (Stable Diffusion + CFG + DDIM)

Inference only, no training required. Downloads `runwayml/stable-diffusion-inpainting` on first run.

```bash
python scripts/run_level1.py --data-dir data/samples --output-dir outputs/level1
```

Key options:

```bash
python scripts/run_level1.py \
  --prompt "A realistic and coherent scene completion" \
  --negative-prompt "blurry, distorted, artifact" \
  --mask-type irregular \
  --guidance-scale 7.5 \
  --num-steps 30 \
  --ddim-eta 0.0 \
  --seed 42
```

| Argument | Default | Description |
|---|---|---|
| `--mask-type` | `center` | `center` or `irregular` |
| `--guidance-scale` | `7.5` | CFG scale $s$ |
| `--num-steps` | `30` | DDIM denoising steps |
| `--ddim-eta` | `0.0` | 0 = deterministic, 1 = stochastic |

Outputs saved to `outputs/level1/`: `predictions/`, `panels/`, `metrics.csv`, `summary.json`.

---

### Level 2 — Production (LoRA fine-tune + mask-aware conditioning + boundary smoothing)

**Inference only** (uses base SD weights with Level 2 conditioning):

```bash
python scripts/run_level2.py --data-dir data/samples --output-dir outputs/level2
```

**With LoRA fine-tune** on the training split before inference:

```bash
python scripts/run_level2.py \
  --finetune \
  --train-dir data/splits/train/images \
  --train-steps 200 \
  --data-dir data/samples \
  --output-dir outputs/level2
```

Key options:

```bash
python scripts/run_level2.py \
  --guidance-scale 8.0 \
  --num-steps 40 \
  --no-boundary-smoothing        # disable to ablate smoothing
```

| Argument | Default | Description |
|---|---|---|
| `--finetune` | off | Run LoRA fine-tune before inference |
| `--train-dir` | — | Directory of training images (required with `--finetune`) |
| `--train-steps` | `200` | LoRA training steps |
| `--no-boundary-smoothing` | off | Disable Gaussian edge blending |

LoRA weights are saved to `outputs/level2/lora/` for reuse.

---

### Level 3 — SOTA (ControlNet + RePaint resampling)

**Default** — ControlNet inpainting + RePaint resampling:

```bash
python scripts/run_level3.py --data-dir data/samples --output-dir outputs/level3
```

**Without ControlNet** (RePaint only):

```bash
python scripts/run_level3.py --no-controlnet --data-dir data/samples
```

**Without RePaint** (ControlNet only):

```bash
python scripts/run_level3.py --no-repaint --data-dir data/samples
```

Key options:

```bash
python scripts/run_level3.py \
  --guidance-scale 9.0 \
  --num-steps 50 \
  --repaint-steps 5 \
  --mask-type center
```

| Argument | Default | Description |
|---|---|---|
| `--no-controlnet` | off | Disable ControlNet structural conditioning |
| `--no-repaint` | off | Disable RePaint resampling |
| `--repaint-steps` | `5` | Number of RePaint resample iterations per image |
| `--guidance-scale` | `9.0` | CFG scale |

ControlNet uses `lllyasviel/control_v11p_sd15_inpaint`. If no control image is provided, Canny edges are auto-computed from the input image.

---

### Evaluate predictions against ground truth

```bash
python scripts/evaluate.py \
  --targets data/samples \
  --predictions outputs/level1/predictions \
  --output outputs/eval/level1
```

---

### Experiment sweep (CFG × DDIM grid)

Sweeps CFG ∈ {5.0, 7.5, 10.0} × DDIM steps ∈ {20, 30, 50} × mask types = **18 runs**, then prints the best config.

```bash
python scripts/sweep_experiments.py --data-dir data/samples
python scripts/sweep_experiments.py --level 2 --data-dir data/samples
```

Results written to `outputs/sweep/sweep_results.csv` and `sweep_results.json`.

## Dataset (Small-Scale)

10 sample images are pre-downloaded to `data/samples` (512×512 JPEGs from `picsum.photos`).

For a more rigorous experiment, replace with a named subset:

| Dataset | Size | Notes |
|---|---|---|
| CelebA-HQ subset | 100–300 images | Clean faces; easy qualitative evaluation |
| Places2 subset | 100–300 images | Diverse scenes; harder but realistic |
| Paris StreetView | ~15K (use 200) | Structured architecture; good for structural masks |

Suggested split for presentation:

| Split | Count | Purpose |
|---|---|---|
| train | 100–300 | LoRA fine-tune (Level 2) |
| val | 20–50 | Hyperparameter selection |
| test | 20–50 | Final metric reporting |

Mask types:

| Type | Description | Use case |
|---|---|---|
| Center | Fixed rectangle (35% of image area) | Controlled, reproducible |
| Irregular | Random brush strokes (6 strokes) | Realistic degradation |

## Evaluation Metrics

The project uses PSNR, SSIM, and visual comparison panels (`src/eval/metrics.py`, `src/eval/visualize.py`).

For target image $x$ and prediction $\hat{x}$ with $N$ pixels and maximum pixel value $MAX_I = 255$:

$$
\text{MSE} = \frac{1}{N}\sum_{i=1}^{N}(x_i - \hat{x}_i)^2
$$

$$
\text{PSNR} = 20\log_{10}(MAX_I) - 10\log_{10}(\text{MSE}) \quad [\text{dB}]
$$

Higher PSNR = better pixel fidelity. Typical good inpainting: 25–35 dB.

SSIM decomposes perceptual similarity into luminance $l$, contrast $c$, and structure $s$:

$$
\text{SSIM}(x, \hat{x}) = l(x,\hat{x})^\alpha \cdot c(x,\hat{x})^\beta \cdot s(x,\hat{x})^\gamma
$$

where $\alpha = \beta = \gamma = 1$ in the standard formulation. Range: [0, 1]; higher is better.

Each run outputs:
- `metrics.csv` — per-image PSNR and SSIM
- `summary.json` — mean PSNR and SSIM over the test set
- `panels/` — 4-column visual grids: Original | Mask | Corrupted | Inpainted

## Mathematics by Level

### Level 1: Stable Diffusion Inpainting + CFG + DDIM

**Forward diffusion process** (gradually adds Gaussian noise over $T$ steps):

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\, \sqrt{1-\beta_t}\,x_{t-1},\, \beta_t I\right)
$$

This allows direct sampling at any step $t$ using the cumulative schedule $\bar{\alpha}_t = \prod_{s=1}^{t}(1-\beta_s)$:

$$
q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\, \sqrt{\bar{\alpha}_t}\,x_0,\, (1-\bar{\alpha}_t) I\right)
$$

**Reverse denoising objective** (train a UNet $\epsilon_\theta$ to predict the added noise):

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{x_0,\epsilon,t}\!\left[\|\epsilon - \epsilon_\theta(x_t, t, c)\|_2^2\right]
$$

**Classifier-Free Guidance (CFG)** jointly trains conditioned and null-conditioned predictions, then blends at inference with scale $s$:

$$
\hat{\epsilon} = \epsilon_\theta(x_t, t, \varnothing) + s\!\left(\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \varnothing)\right)
$$

Higher $s$ → stronger prompt adherence; typical range 5–10.

**Inpainting mask conditioning**: the mask $m \in \{0,1\}^{H \times W}$ (1 = unknown) is concatenated channel-wise with $x_t$ and the known context so the UNet sees where to regenerate.

**DDIM update** (deterministic path when $\eta = 0$):

$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\underbrace{\left(\frac{x_t - \sqrt{1-\bar{\alpha}_t}\,\hat{\epsilon}}{\sqrt{\bar{\alpha}_t}}\right)}_{\hat{x}_0\text{ prediction}} + \sqrt{1-\bar{\alpha}_{t-1}}\,\hat{\epsilon}
$$

Setting $\eta > 0$ adds stochastic noise, recovering DDPM behavior at $\eta = 1$.

---

### Level 2: Production — Latent Diffusion + LoRA Fine-Tune + Boundary Smoothing

**Latent diffusion** performs the entire diffusion process in a compressed latent space via a pretrained VAE:

$$
z = \mathcal{E}(x) \in \mathbb{R}^{h \times w \times d}, \qquad \hat{x} = \mathcal{D}(\hat{z})
$$

Training loss now operates on latent noise rather than pixel noise:

$$
\mathcal{L}_{\text{LDM}} = \mathbb{E}_{z_0,\epsilon,t}\!\left[\|\epsilon - \epsilon_\theta(z_t, t, c, m)\|_2^2\right]
$$

**Mask-aware conditioning**: the corrupted latent $z_{\text{ctx}} = z \odot (1 - m_z)$ and downsampled mask $m_z$ are concatenated to the noisy latent as extra input channels, giving the UNet explicit spatial context:

$$
\text{input} = \left[z_t \;\|\; m_z \;\|\; z_{\text{ctx}}\right] \in \mathbb{R}^{h \times w \times (d + 1 + d)}
$$

**LoRA fine-tuning** injects low-rank adapters into the UNet cross-attention weight matrices rather than updating all parameters:

$$
W' = W + \Delta W = W + BA, \qquad B \in \mathbb{R}^{d \times r},\; A \in \mathbb{R}^{r \times k},\; r \ll \min(d, k)
$$

Only $A$ and $B$ are trained (rank $r = 4$ by default), reducing trainable parameters by ~99% versus full fine-tuning.

**Boundary smoothing** computes a soft weight map $w$ by Gaussian-blurring the dilated mask, then composites generaated and original pixels:

$$
x_{\text{out}} = w \odot x_{\text{gen}} + (1 - w) \odot x_{\text{orig}}
$$

where $w \in [0, 1]^{H \times W}$ is high (≈1) at the mask centre and tapers smoothly to 0 at known boundaries, eliminating hard seam artifacts.

**Prompt + image guidance** blends text-guided generation with an image reference by running both and averaging predicted $\hat{x}_0$ estimates (or by adjusting the conditioning tensor in the UNet cross-attention layers).

---

### Level 3: SOTA — ControlNet + RePaint + DiT + PnP Guidance

**ControlNet** adds a trainable copy of the UNet encoder conditioned on a structural input $u$ (Canny edges, depth maps, segmentation). Its output feature maps are added back to the frozen UNet decoder at each resolution:

$$
\epsilon_\theta(x_t, t, c, u) = \epsilon_{\text{UNet}}(x_t, t, c) + \lambda \cdot \epsilon_{\text{CN}}(x_t, t, c, u)
$$

This allows pixel-level structural constraints (e.g. preserve edges and object layout in the inpainted region) without retraining the base model.

**RePaint resampling** forces the known region to remain consistent at every denoising step. At each DDIM step $t$, after generating $x_{t-1}^{\text{gen}}$, the known region is replaced by the forward-noised original:

$$
x_{t-1}[\text{known}] = \sqrt{\bar{\alpha}_{t-1}}\,x_0 + \sqrt{1-\bar{\alpha}_{t-1}}\,\epsilon
$$
$$
x_{t-1}[\text{unknown}] = x_{t-1}^{\text{gen}}
$$

This is repeated $r$ times per timestep to integrate the boundary information into the unknown region's trajectory, improving coherence. In this implementation $r = 5$ RePaint passes are run over the full trajectory.

**DiT (Diffusion Transformer)** replaces the UNet with a transformer that treats the latent as a sequence of flattened tokens:

$$
\epsilon_\theta\!\left(\text{tokens}(z_t),\, t,\, c\right) = \text{Transformer}\!\left(\text{patchify}(z_t), t, c\right)
$$

Scaling laws favour DiT over UNet at higher resolutions and larger model sizes. Enabled via `use_dit_backbone=True` in `Level3Config` when a DiT checkpoint is available.

**Plug-and-play (PnP) guidance** steers the denoising trajectory at inference using an external differentiable objective $\mathcal{E}(x_0)$ (e.g. perceptual loss, CLIP similarity) without retraining:

$$
\tilde{\epsilon} = \hat{\epsilon} - \sqrt{1 - \bar{\alpha}_t}\,\nabla_{x_t}\mathcal{E}(\hat{x}_0)
$$

This acts as a gradient correction on the noise prediction and requires no additional training.

## Suggested Experiments

### Level 1 — Baseline grid search (automated via sweep script)

| Variable | Values |
|---|---|
| CFG scale | 5.0, 7.5, 10.0 |
| DDIM steps | 20, 30, 50 |
| Mask type | center, irregular |

```bash
python scripts/sweep_experiments.py --data-dir data/samples
```

### Level 2 — Ablation study

Run each variant and compare PSNR/SSIM:

```bash
# Without boundary smoothing
python scripts/run_level2.py --no-boundary-smoothing --data-dir data/samples --output-dir outputs/level2_no_smooth

# With boundary smoothing (default)
python scripts/run_level2.py --data-dir data/samples --output-dir outputs/level2_smooth

# With LoRA fine-tune + smoothing
python scripts/run_level2.py --finetune --train-dir data/splits/train/images \
  --data-dir data/samples --output-dir outputs/level2_lora
```

### Level 3 — Component comparison

```bash
# RePaint resampling only (no ControlNet)
python scripts/run_level3.py --no-controlnet --data-dir data/samples --output-dir outputs/level3_repaint

# ControlNet only (no RePaint)
python scripts/run_level3.py --no-repaint --data-dir data/samples --output-dir outputs/level3_controlnet

# Full stack (ControlNet + RePaint)
python scripts/run_level3.py --data-dir data/samples --output-dir outputs/level3_full
```

### Cross-level comparison table (fill after running experiments)

| Level | Variant | Mean PSNR (dB) | Mean SSIM | Notes |
|---|---|---|---|---|
| 1 | CFG=7.5, steps=30, center mask | — | — | Baseline |
| 1 | CFG=7.5, steps=30, irregular mask | — | — | Harder mask |
| 2 | +boundary smoothing | — | — | Seam quality |
| 2 | +LoRA fine-tune | — | — | Domain adaptation |
| 3 | +ControlNet | — | — | Structural control |
| 3 | +RePaint | — | — | Boundary coherence |
| 3 | Full stack | — | — | Best quality |

Update `reports/results_summary.md` with your final numbers.

## Presentation Checklist

- Problem statement and motivation
- Level-wise method diagram
- Metric table (PSNR/SSIM/runtime)
- Visual quality comparison panels
- Failure analysis and insights
- Future work and constraints

## References (Papers and Articles)

### Foundational diffusion papers (all levels)

1. **DDPM** — Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020.
  https://arxiv.org/abs/2006.11239
  *Establishes the forward/reverse diffusion framework used in Level 1.*

2. **DDIM** — Song et al., "Denoising Diffusion Implicit Models", ICLR 2021.
  https://arxiv.org/abs/2010.02502
  *Introduces deterministic DDIM sampling (η=0) enabling faster inference with fewer steps.*

3. **CFG** — Ho and Salimans, "Classifier-Free Diffusion Guidance", NeurIPS Workshop 2021.
  https://arxiv.org/abs/2207.12598
  *Derives the joint unconditional/conditional training used for guidance scale s.*

### Level 2 — Latent diffusion and fine-tuning

4. **LDM** — Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models", CVPR 2022.
  https://arxiv.org/abs/2112.10752
  *Core paper for latent diffusion; introduces VAE compression enabling 512×512 generation.*

5. **LoRA** — Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022.
  https://arxiv.org/abs/2106.09685
  *Low-rank adapter fine-tuning method applied to UNet cross-attention in Level 2.*

6. **Stable Diffusion Inpainting** — Rombach et al., runway/stable-diffusion-inpainting model card.
  https://huggingface.co/runwayml/stable-diffusion-inpainting
  *Pretrained checkpoint with mask-channel conditioning used across Level 1 and 2.*

### Level 3 — SOTA extensions

7. **RePaint** — Lugmayr et al., "RePaint: Inpainting using Denoising Diffusion Probabilistic Models", CVPR 2022.
  https://arxiv.org/abs/2201.09865
  *Introduces per-step resampling of the known region for improved boundary coherence.*

8. **ControlNet** — Zhang et al., "Adding Conditional Control to Text-to-Image Diffusion Models", ICCV 2023.
  https://arxiv.org/abs/2302.05543
  *Learnable structural conditioning (Canny, depth, segmentation) on frozen SD models.*

9. **DiT** — Peebles and Xie, "Scalable Diffusion Models with Transformers", ICCV 2023.
  https://arxiv.org/abs/2212.09748
  *Transformer-based denoiser replacing UNet; scales better at high resolution.*

10. **Plug-and-Play Guidance** — Tumanyan et al., "Plug-and-Play Diffusion Features for Text-Driven Image-to-Image Translation", CVPR 2023.
   https://arxiv.org/abs/2211.12572
   *Inference-time gradient steering of diffusion trajectories without retraining.*

### Libraries and practical resources

- Hugging Face Diffusers — SD inpainting, ControlNet, DDIM scheduler implementations.
  https://huggingface.co/docs/diffusers
- CompVis / Stability AI — original LDM codebase and model weights.
  https://github.com/CompVis/latent-diffusion
- ControlNet official repository — training and inference reference.
  https://github.com/lllyasviel/ControlNet
- scikit-image SSIM — SSIM metric implementation used in `src/eval/metrics.py`.
  https://scikit-image.org/docs/stable/api/skimage.metrics.html


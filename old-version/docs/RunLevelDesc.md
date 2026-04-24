## run_level1.py

This program is demonstrating a baseline image inpainting experiment: “If I take a normal image, remove part of it with a synthetic mask, and run an off-the-shelf Stable Diffusion inpainting model, how well can it reconstruct the missing content?”

The flow in run_level1.py is:

1. It loads sample images and creates an artificial missing region for each one via dataset.py.
2. It builds a Level 1 inpainting model wrapper around the pretrained Diffusers Stable Diffusion inpainting pipeline in level1_sd_inpaint.py.
3. For each corrupted image, it asks the model to fill the masked area using a text prompt in run_level1.py.
4. It computes PSNR and SSIM against the original image in metrics.py and metrics.py.
5. It saves a side-by-side panel showing original, mask, corrupted input, and inpainted output using visualize.py, then writes per-image predictions plus aggregate summary files.

So the script is not training a model. It is evaluating a pretrained model as a baseline. In project terms, “Level 1” appears to mean: use standard Stable Diffusion inpainting with configurable prompt, guidance scale, steps, and mask type, then measure the quality of the completed image.

What it is trying to demonstrate, more specifically:

- Qualitative performance:
  Can the model generate visually coherent missing content?
- Quantitative performance:
  How close is the generated image to the ground truth, using PSNR and SSIM?
- Baseline establishment:
  This gives you a reference point before moving to more advanced levels such as fine-tuning or more specialized methods.

One important nuance: the metrics are computed on the whole image, not only the masked region, because metrics.py compares full original and full prediction images. That makes sense for a quick baseline, but it can make scores look better than a masked-region-only evaluation since large unmodified areas may already match closely.

If you want, I can also explain:
1. Why the title now includes PSNR and SSIM for each panel.
2. How Level 1 differs conceptually from the later levels in this project.
3. Whether this evaluation setup is scientifically strong or just a practical demo baseline.

---

## run_level2.py

## What the program demonstrates

Level 2 is the "production" tier of an academic inpainting experiment. It builds on Level 1 (plain SD baseline) by answering a more refined question:

> Can we get better inpainting results by (a) giving the model explicit spatial awareness of the mask, (b) optionally adapting the model with lightweight fine-tuning on the target domain, and (c) smoothing out boundary artifacts at the edges of the filled region?

It is not just running inference — it is testing whether three specific techniques improve over the Level 1 baseline.

---

## The three technical additions over Level 1

### 1. Mask-aware conditioning
During fine-tuning, instead of passing only the noisy latent to the UNet, it concatenates three channels:
```
model_input = [noisy_latent | mask_latent | masked_image_latent]
```
This gives the UNet an explicit map of *where* pixels are missing and *what the surrounding context looks like* in latent space, rather than relying on the model to infer it from the corrupted image alone. This directly implements the latent diffusion objective described in the model docstring:

$$\mathcal{L} = \mathbb{E}_{z_0, \epsilon, t}\left[\|\epsilon - \epsilon_\theta(z_t, t, c, m)\|^2\right]$$

where $m$ is the mask in latent space.

### 2. LoRA fine-tuning (optional)
Fine-tuning is performed via **LoRA (Low-Rank Adaptation)** — not full retraining. The steps in level2_finetune_ldm.py:

- LoRA adapter layers (`LoRAAttnProcessor`, rank=4 by default) are attached to every **cross-attention and self-attention layer** of the SD UNet.
- Only those adapter weights are trained via AdamW (`lr=1e-4`).
- The VAE and text encoder are frozen (`requires_grad_(False)`).
- Each training step picks one image from the training set (cycling by `step % len(image_paths)`), encodes it to latent space, adds random noise at a random timestep, and trains the UNet to predict the noise — the standard denoising diffusion objective.
- Training runs for `--train-steps` (default 200) steps and saves the LoRA weights to `outputs/level2/lora/`.

This is essentially a lightweight DreamBooth-style fine-tune: the model learns the visual style/domain of your specific dataset without needing thousands of images or hours of compute.

### 3. Boundary smoothing
After inference, the hard mask edge is softened:
$$x_{out} = w \cdot x_{gen} + (1 - w) \cdot x_{original}$$
where $w$ is a Gaussian-blurred version of the mask (`blur_radius=12` px). This prevents visible seams at the inpainted boundary — a known artifact of hard mask compositing.

---

## How to run it and on what dataset

The workspace already has split data in splits. The correct paths are:

**Inference only (no fine-tuning):**
```bash
python scripts/run_level2.py \
    --data-dir data/splits/test/images \
    --mask-dir data/splits/test/masks/center \
    --output-dir outputs/level2
```

**Fine-tune on training set, then infer on test set:**
```bash
python scripts/run_level2.py \
    --finetune \
    --train-dir data/splits/train/images \
    --data-dir data/splits/test/images \
    --mask-dir data/splits/test/masks/center \
    --output-dir outputs/level2
```

**Irregular masks variant:**
```bash
python scripts/run_level2.py \
    --finetune \
    --train-dir data/splits/train/images \
    --data-dir data/splits/test/images \
    --mask-type irregular \
    --output-dir outputs/level2_irregular
```

Key flags:
| Flag | Default | Purpose |
|---|---|---|
| `--finetune` | off | Run LoRA training before inference |
| `--train-steps` | 200 | LoRA gradient steps |
| `--guidance-scale` | 8.0 | CFG strength (higher = more prompt-adherent) |
| `--num-steps` | 40 | DDIM denoising steps |
| `--no-boundary-smoothing` | off | Disable the Gaussian blend post-step |
| `--mask-type` | center | `center` or `irregular` |

**Dataset:** The project is designed for a small dataset (IIT Bombay coursework — the project details explicitly say "avoid large datasets"). The splits directory already has `train/`, `val/`, and `test/` splits. For LoRA fine-tuning, even 50–300 images in `train/images/` is sufficient.



## One important note about `run_level2.py` vs Level 1

Unlike Level 1 (which was patched to compute per-image PSNR/SSIM in the loop), Level 2 still uses `evaluate_pairs` after the loop, so the panel titles only say `Level 2 | {filename}` without metrics. If you want the same per-image PSNR/SSIM in the panels as Level 1 now has, that same change can be applied here.


---

**What run_level3.py demonstrates**
run_level3.py is a Level-3 inference experiment showing “advanced sampling-time inpainting” with:
1. ControlNet-guided inpainting (optional), configured in level3_advanced.py.
2. RePaint-style resampling (optional), implemented in level3_advanced.py.
3. Standard SD inpainting fallback when ControlNet is disabled, in level3_advanced.py.

The script loops over masked samples, generates outputs, saves prediction images and 4-panel visualizations, then computes aggregate PSNR/SSIM via metrics.py.

**How fine-tuning is performed**
It is not performed in this program.
1. There is no training loop, optimizer, or checkpoint save/load in run_level3.py or level3_advanced.py.
2. Level 3 here is inference-only with stronger conditioning/sampling strategies.
3. If you want fine-tuning, that is in Level 2 (LoRA) via level2_finetune_ldm.py.

**How to run it**
From workspace root:
```bash
python old-version/scripts/run_level3.py --data-dir data/samples --output-dir outputs/level3
```

Common variants:
```bash
python old-version/scripts/run_level3.py --data-dir data/samples --mask-type irregular
python old-version/scripts/run_level3.py --no-controlnet --data-dir data/samples
python old-version/scripts/run_level3.py --no-repaint --data-dir data/samples
```

**What dataset to use**
For this script, use a folder of RGB images as `--data-dir` (it generates masks on the fly through `load_samples`):
1. Quick demo: samples.
2. Evaluation split: images (recommended for reporting final metrics).
3. It supports both `center` and `irregular` synthetic masks from dataset.py and dataset.py.

If you want, I can also give you a “best-practice” command set to run Level 1/2/3 on the same test split so their PSNR/SSIM are directly comparable.

---


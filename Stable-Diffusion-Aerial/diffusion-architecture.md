# Stable Diffusion Inpainting — Internal Architecture

This document explains how the three neural networks inside `runwayml/stable-diffusion-inpainting`
are wired together and how they fit into the broader pipeline defined in [`inpainting.py`](inpainting.py).

---

## The Three Neural Networks

| Network | Role |
|---------|------|
| **VAE** (Variational Autoencoder) | Compresses image → latent space, decodes latent → pixels |
| **UNet** | Core denoising network — fills masked region guided by context + prompt |
| **CLIP Text Encoder** | Converts prompt text → embeddings that steer the UNet |

All three are loaded in a single call in [`inpainting.py`](inpainting.py#L20):

```python
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=config.DTYPE,
    safety_checker=None,
).to(config.DEVICE)
```

---

## End-to-End Architecture Diagram

```
tile_img (512×512 RGB)  +  tile_mask  +  prompt text
         │                     │               │
         ▼                     │               ▼
   ┌───────────┐               │      ┌─────────────────┐
   │    VAE    │               │      │  CLIP Text      │
   │  Encoder  │               │      │  Encoder        │
   └───────────┘               │      └─────────────────┘
         │                     │               │
         │ latent z            │               │ text embeddings
         │ (64×64×4)           │               │ (77×768)
         ▼                     ▼               ▼
   ┌─────────────────────────────────────────────────┐
   │                   UNet                          │
   │  Input:  noisy latent  +  masked latent         │
   │          +  mask (downsampled)                  │
   │          +  text embeddings (cross-attention)   │
   │                                                 │
   │  Runs 35 denoising steps (INPAINT_STEPS)        │
   │  CFG scale 7.5 doubles each step:               │
   │    noise = uncond + 7.5 × (cond - uncond)       │
   └─────────────────────────────────────────────────┘
         │
         │ clean latent z'
         ▼
   ┌───────────┐
   │    VAE    │
   │  Decoder  │
   └───────────┘
         │
         │ result (512×512 RGB)
         ▼
   pixel-level copy in inpaint() ← only masked pixels taken
```

---

## Step-by-Step Walkthrough

Everything below happens inside a single `pipe(...)` call in [`inpainting.py`](inpainting.py#L44),
once per active tile (tiles where the mask is non-empty).

---

### Step 1 — CLIP Text Encoder

The prompt string defined in [`config.py`](config.py#L65) is tokenised and encoded into a
**77 × 768 tensor** of text embeddings:

```
"aerial top-down satellite view of empty asphalt road,
 clean road surface, road markings, no vehicles,
 uniform tarmac texture, photorealistic"
```

This runs **once per tile**, before denoising starts. The embeddings are injected into every
layer of the UNet via **cross-attention**, which is how the prompt continuously steers what
gets generated.

The negative prompt (from [`config.py`](config.py#L70)) is encoded the same way and used
during Classifier-Free Guidance (see Step 3).

---

### Step 2 — VAE Encoder

The 512×512 tile is compressed by the VAE Encoder into a compact **64×64×4 latent tensor** `z`.
Working in latent space (rather than pixel space) is what makes Stable Diffusion fast — the
UNet operates on a tensor 64× smaller than the original image.

The mask is simultaneously downsampled to **64×64**. The masked region in the latent is
replaced with **random Gaussian noise** — this is the starting point for denoising.

```
512×512 RGB  →  VAE Encoder  →  64×64×4 latent z
                                masked region = N(0,1) noise
```

---

### Step 3 — UNet Denoising (35 steps)

This is the core of the model. The UNet runs for `INPAINT_STEPS = 35` iterations
(set in [`config.py`](config.py#L60)), progressively removing noise from the masked latent.

**Inputs at each step:**

| Input | Shape | Description |
|-------|-------|-------------|
| Noisy latent | 64×64×4 | Current noisy estimate of the masked region |
| Masked latent | 64×64×4 | Original image latent with mask zeroed out |
| Downsampled mask | 64×64×1 | Binary mask indicating which region to fill |
| Text embeddings | 77×768 | CLIP output — injected via cross-attention |

The masked latent and mask are **concatenated as extra channels** to the noisy latent before
being fed into the UNet. This is the architectural difference between `runwayml/stable-diffusion-inpainting`
and base SD — it was fine-tuned with this masked-latent conditioning so the UNet learns to
hallucinate content that blends naturally with the surrounding context.

**Classifier-Free Guidance (CFG):**

At each of the 35 steps, the UNet runs **twice** — once conditioned on the prompt, once
unconditioned (empty prompt). The outputs are blended using the guidance scale `w = 7.5`
from [`config.py`](config.py#L63):

```
predicted_noise = uncond + 7.5 × (cond − uncond)
```

A higher `w` means stronger prompt adherence. At `7.5` it balances prompt fidelity against
image naturalness — the road texture looks realistic while still matching "asphalt, no vehicles".

---

### Step 4 — VAE Decoder

After 35 denoising steps, the clean latent `z'` is decoded back to pixel space by the
VAE Decoder:

```
64×64×4 clean latent z'  →  VAE Decoder  →  512×512 RGB result
```

---

### Step 5 — Pixel-Level Copy

Back in [`inpainting.py`](inpainting.py#L53), the SD result is **not** used wholesale.
Only the masked pixels are taken from the SD output — all non-masked pixels are copied
directly from the original:

```python
m = tile_mask > 0
out_arr[y:y + TILE, x:x + TILE][m]  = result[m]   # SD output  → masked pixels only
# non-masked pixels already in out_arr via .copy() at the start  ← unchanged
```

This pixel-level surgical copy is what guarantees the [`metrics.py`](metrics.py) **PASS**:
`Pixels changed outside mask = 0`.

---

## Why This Model Specifically

`runwayml/stable-diffusion-inpainting` is a **fine-tuned variant of SD v1.5**, not the base
model. The key difference is in how the UNet was trained:

| | Base SD v1.5 | SD Inpainting |
|--|--|--|
| UNet input channels | 4 | **9** (4 noisy + 4 masked original + 1 mask) |
| Training objective | Denoise full image | Denoise masked region conditioned on visible context |
| Inpainting method | Post-hoc (RePaint) | Native — mask is a first-class input |

Because the mask is injected **inside the UNet** as extra input channels (not post-processed),
the model learns to hallucinate content that is contextually aware of the surrounding road
surface — producing seamless asphalt texture rather than a blurry patch.

---

## Parameter Reference

All values below are defined in [`config.py`](config.py):

| Parameter | Value | Effect |
|-----------|-------|--------|
| `INPAINT_STEPS` | `35` | Number of UNet denoising iterations per tile |
| `GUIDANCE_SCALE` | `7.5` | CFG weight — higher = stronger prompt adherence |
| `INPAINT_TILE` | `512` | Tile size fed to SD (matches native SD resolution) |
| `INPAINT_OVERLAP` | `64` | Overlap between tiles to avoid seam artifacts |
| `INPAINT_PROMPT` | `"aerial top-down..."` | Positive conditioning for the CLIP encoder |
| `INPAINT_NEG_PROMPT` | `"cars, trucks..."` | Negative conditioning — suppresses unwanted content |
| `DEVICE` | `"mps"` | Apple Silicon GPU (falls back to CPU) |
| `DTYPE` | `torch.float32` | Full precision — required for MPS stability |

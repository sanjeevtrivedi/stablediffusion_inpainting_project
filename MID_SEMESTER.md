# Mid-Semester Report

## 1. Understanding of the Problem Statement

This project solves image inpainting using diffusion models.

Given:
- an input image with missing or corrupted region,
- a binary mask where white indicates missing region and black indicates known region,

the goal is to generate realistic content inside the masked region while preserving consistency with unmasked context (structure, texture, color, and semantics).

### Project Objective
Build a single, reproducible inpainting baseline using Stable Diffusion Inpainting + Classifier-Free Guidance + DDIM sampling.

Scope:
- Use pretrained model (no training required)
- Small dataset (10–30 images from picsum.photos)
- Two mask types: center (fixed rectangle) and irregular (random strokes)
- Evaluate on PSNR, SSIM, and visual quality

### Constraints
- Limited compute (CPU or single GPU)
- No fine-tuning or multi-level complexity
- Reproducible pipeline with clear outputs

## 2. Literature Survey

### Reference 1: RePaint: Inpainting using Denoising Diffusion Probabilistic Models
**Link:** https://arxiv.org/abs/2201.09865

**Key Contributions:**
- Uses pretrained unconditional diffusion model without retraining
- Applies per-step resampling of known regions to enforce consistency
- Works on diverse mask types (center, irregular, extreme masks)

**Relevance to This Project:**
- Validates using pretrained models for inpainting without mask-specific training
- Demonstrates sampling-time customization (no model retraining needed)
- Supports our single-model, inference-only strategy

### Reference 2: How to Customize any Diffusion Models for Inpainting
**Link:** https://medium.com/@aromalma/how-to-customize-any-diffusion-models-for-inpainting-178111b239cd

**Key Concepts:**
- Sampling pipeline customization enables inpainting without retraining
- At each DDIM step: blend forward-noised reference with reverse generation
- Mask blurring reduces visible seams
- Formula: $x_{t-1} = \text{denoise}(x_t \cdot m + \text{noised\_ref} \cdot (1-m), t)$

**Relevance to This Project:**
- Directly matches our implementation strategy
- Explains why Stable Diffusion Inpainting works without additional training
- Justifies focus on inference-time optimization (prompts, CFG, step count)

## 3. Approach

**Single Baseline Pipeline: Stable Diffusion Inpainting with Classifier-Free Guidance and DDIM Sampling**

### Method Overview
1. Use pretrained Stable Diffusion Inpainting model (`runwayml/stable-diffusion-inpainting`)
2. Apply classifier-free guidance (CFG) with scale $s = 7.5$ (controllable via CLI)
3. Use DDIM scheduler with 50 inference steps (controllable)
4. No fine-tuning; direct inference on test images

### Denoising Objective

The pretrained model predicts added noise at each denoising step:

$$
\mathcal{L} = \mathbb{E}_{x_0,\epsilon,t}[\|\epsilon - \epsilon_\theta(x_t, t, c, m)\|^2]
$$

where:
- $x_t$ is the noisy latent at step $t$
- $c$ is text conditioning (prompt)
- $m$ is the inpainting mask (concatenated as input channels)

### Sampling Process
1. Load image and apply mask (white fill = unknown region)
2. Pass corrupted image + mask + prompt through Stable Diffusion pipeline
3. CFG steers generation toward the text prompt
4. DDIM deterministic sampling (eta=0) ensures reproducibility
5. Output: inpainted image with filled masked region

### Why This Approach
- **Fast:** No training; uses pretrained model weights
- **Practical:** Stable Diffusion is proven for diverse inpainting tasks
- **Reproducible:** Fixed seed + deterministic DDIM sampling
- **Aligned with Literature:** Directly implements the Medium article's inpainting customization strategy
- **Simple:** Clear CLI parameters (CFG, steps) for easy experimentation

## 4. Dataset Selection

### Selected Dataset Strategy
A lightweight custom dataset is used to fit compute constraints:
- **Source:** Images downloaded from picsum.photos (deterministic, reproducible)
- **Size:** 10–30 images (default 20 for experimentation)
- **Resolution:** 512×512 pixels (standard for Stable Diffusion)
- **Masks:** Two types generated per image:
  - **Center mask:** Fixed white rectangle (35% of image area)
  - **Irregular mask:** Random brush strokes (6 strokes, 18–56 pixel width)
- **No train/val/test split needed:** All images used for inference evaluation

### Why This Is Suitable
- Small size fits limited compute constraints
- Picsum.photos provides diverse, copyright-free images
- Reproducible (deterministic seed for mask generation)
- Two mask types allow robustness comparison (center vs. irregular)
- Ground truth: original image (no manual annotations required)

## 5. Evaluation Metrics

The project evaluates outputs using following quantitative measures and by seeing visually i.e. qualitative.

### 1. PSNR (Peak Signal-to-Noise Ratio)
Measures pixel-level reconstruction fidelity between original and inpainted image.

$$
\text{PSNR} = 20\log_{10}\left(\frac{\text{MAX}}{\sqrt{\text{MSE}}}\right)
$$

**Interpretation:** Higher is better. Typical good inpainting: 20–28 dB.

### 2. SSIM (Structural Similarity Index)
Measures structural and perceptual similarity, better aligns with human perception than PSNR alone.

**Interpretation:** Range [0, 1]. Higher is better. Typical range: 0.65–0.75 for inpainting.

### 3. Visual Quality Assessment
4-panel visual grids for each image showing:
- **Original:** Ground truth image
- **Mask:** Binary mask (white = unknown region)
- **Corrupted:** Input with mask applied (white fill)
- **Inpainted:** Model output

**Qualitative checks:**
- Semantic plausibility  like: does the content "make sense"?
- Failure modes - blurriness, color mismatch, temporal inconsistency

## 5. Diffusion Inpainting Taxonomy
```text
Diffusion Inpainting
│
├── 1. Conditioning Strategy
│   ├── Training-time (mask-aware models)
│   └── Inference-time (RePaint, plug-and-play)
│
├── 2. Diffusion Space
│   ├── Pixel (DDPM)
│   ├── Latent (LDM)
│   ├── Multi-scale
│   └── Hybrid (latent + pixel refinement)
│
├── 3. Mask Representation
│   ├── Binary mask
│   ├── Soft/blurred mask
│   ├── Progressive masks
│   └── Learned mask embeddings
│
├── 4. Constraint Enforcement
│   ├── Hard (pixel preservation)
│   ├── Soft (loss-based)
│   ├── Hybrid
│   └── Projection-based constraints
│
├── 5. Region Interaction
│   ├── Independent generation
│   ├── Step-wise blending
│   └── Joint modeling (full image)
│
├── 6. Guidance Mechanism
│   ├── None (unconditional)
│   ├── Text (CFG)
│   ├── Structural (edges, depth)
│   └── Semantic (segmentation, layout)
│
├── 7. Sampling Strategy
│   ├── Standard diffusion
│   ├── Resampling (RePaint)
│   ├── Posterior sampling
│   └── Deterministic (DDIM)
│
├── 8. Model Type
│   ├── Unconditional diffusion
│   ├── Conditional diffusion (mask-aware)
│   ├── Fine-tuned inpainting models
│   └── Plug-and-play methods
│
└── 9. Training Objective (optional axis)
    ├── Noise prediction (DDPM loss)
    ├── Mask-weighted loss
    ├── Perceptual / adversarial loss
    └── CLIP-aligned loss
```

## 6. Paper Comparisons: RePaint vs Stable Diffusion Inpainting vs ControlNet

| Dimension | RePaint | Stable Diffusion Inpainting | ControlNet (with Inpainting) |
|-----------|---------|-----------------------------|------------------------------|
| Conditioning Strategy | Inference-time only (no retraining) | Training-time (mask-aware finetuned model) | Training-time + strong conditional control |
| Diffusion Space | Pixel space (DDPM) | Latent space (LDM) | Latent space (LDM) |
| Mask Handling | Binary mask + iterative resampling | Binary mask concatenated as input | Mask + additional control signals |
| Constraint Enforcement | Hard constraint via resampling of known pixels | Hybrid (implicit via training + CFG) | Hybrid (strong control via conditioning maps) |
| Region Interaction | Step-wise blending (known + generated) | Joint modeling in latent space | Joint modeling with external control |
| Guidance | None (unconditional) | Text guidance (CFG) | Structural + semantic guidance (edges, depth, pose) |
| Sampling Strategy | Resampling (key contribution) | DDIM / standard schedulers | DDIM / advanced schedulers |
| Model Type | Unconditional pretrained diffusion | Conditional pretrained inpainting model | Conditional model with auxiliary control network |
| Strengths | No training required, flexible | Fast, practical, high-quality outputs | Precise structural control, high fidelity |
| Weaknesses | Slow, less semantic control | Limited structural control | More complex, heavier compute |

### Key Insights
- **RePaint** demonstrates that *sampling alone* can enable inpainting without retraining.
- **Stable Diffusion Inpainting** (your baseline) provides the best trade-off between quality and simplicity.
- **ControlNet** extends diffusion with *explicit structural guidance*, making it ideal for controlled generation tasks.

## References
1. Andreas Lugmayr et al., RePaint: Inpainting using Denoising Diffusion Probabilistic Models, arXiv:2201.09865, 2022. https://arxiv.org/abs/2201.09865
2. Aromal M A, How to Customize any Diffusion Models for Inpainting, Medium, 2024. https://medium.com/@aromalma/how-to-customize-any-diffusion-models-for-inpainting-178111b239cd

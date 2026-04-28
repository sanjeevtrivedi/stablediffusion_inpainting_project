"""
Tiled Stable Diffusion inpainting.

Only masked pixels are replaced — all other pixels are bit-for-bit
identical to the input image.
"""

import logging

import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

import config
from detection import tile_coords

log = logging.getLogger(__name__)


def load_pipeline() -> StableDiffusionInpaintPipeline:
    log.info("Loading SD inpainting pipeline")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=config.DTYPE,
        safety_checker=None,
    ).to(config.DEVICE)
    pipe.enable_attention_slicing()
    return pipe


def inpaint(aerial: Image.Image, mask: np.ndarray) -> Image.Image:
    pipe    = load_pipeline()
    img_arr = np.array(aerial)
    out_arr = img_arr.copy()
    H, W    = img_arr.shape[:2]

    ys = tile_coords(H, config.INPAINT_TILE, config.INPAINT_OVERLAP)
    xs = tile_coords(W, config.INPAINT_TILE, config.INPAINT_OVERLAP)
    active = [(y, x) for y in ys for x in xs
              if mask[y:y + config.INPAINT_TILE, x:x + config.INPAINT_TILE].sum() > 0]

    log.info("Inpainting %d / %d tiles", len(active), len(ys) * len(xs))

    for idx, (y, x) in enumerate(active):
        tile_img  = Image.fromarray(img_arr[y:y + config.INPAINT_TILE, x:x + config.INPAINT_TILE])
        tile_mask = mask[y:y + config.INPAINT_TILE, x:x + config.INPAINT_TILE]

        result = np.array(pipe(
            prompt=config.INPAINT_PROMPT,
            negative_prompt=config.INPAINT_NEG_PROMPT,
            image=tile_img,
            mask_image=Image.fromarray(tile_mask),
            num_inference_steps=config.INPAINT_STEPS,
            guidance_scale=config.GUIDANCE_SCALE,
            strength=0.99,
        ).images[0])

        m = tile_mask > 0
        out_arr[y:y + config.INPAINT_TILE, x:x + config.INPAINT_TILE][m] = result[m]
        log.info("[%d/%d] Tile (%d,%d) — %d px replaced", idx + 1, len(active), y, x, m.sum())

    return Image.fromarray(out_arr)

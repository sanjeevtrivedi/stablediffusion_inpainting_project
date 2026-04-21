from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline
from PIL import Image


@dataclass
class Level1Config:
    model_id: str
    guidance_scale: float
    num_inference_steps: int
    ddim_eta: float
    device: str


class StableDiffusionInpaintingLevel1:
    """Simple Level 1 wrapper: SD Inpainting + CFG + DDIM."""

    def __init__(self, config: Level1Config) -> None:
        self.config = config
        dtype = torch.float16 if "cuda" in config.device else torch.float32

        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            config.model_id,
            torch_dtype=dtype,
        )
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(config.device)

    def inpaint(
        self,
        prompt: str,
        image: Image.Image,
        mask: Image.Image,
        negative_prompt: Optional[str] = None,
        seed: int = 42,
    ) -> Image.Image:
        generator = torch.Generator(device=self.config.device).manual_seed(seed)
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            guidance_scale=self.config.guidance_scale,
            num_inference_steps=self.config.num_inference_steps,
            eta=self.config.ddim_eta,
            generator=generator,
        )
        return output.images[0]

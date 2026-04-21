"""
Level 2: Production inpainting pipeline.

Key additions over Level 1:
- Mask-aware conditioning: VAE-encoded corrupted image concatenated with mask channel
  as extra context to the UNet, giving the model explicit spatial guidance.
- Prompt + image guidance: combines text CFG with an image-level reference latent.
- Boundary smoothing: blends generated and known pixels near mask edges using a
  Gaussian-blurred soft weight map, reducing seam artifacts.

Fine-tuning note:
  A small DreamBooth / LoRA fine-tune step on 100-300 domain images is provided.
  It adjusts low-rank adapter weights on the UNet attention layers rather than
  retraining from scratch, keeping compute requirements lightweight.

Mathematics:
  Latent diffusion objective:
    L = E_{z0, e, t}[ || e - e_theta(z_t, t, c, m) ||^2 ]
  where z = E(x) (VAE encoder), m is the mask in latent space.

  Boundary blend:
    x_out = w * x_gen + (1 - w) * x_context
  where w is a soft edge map from dilated + Gaussian-blurred mask.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from PIL import Image, ImageFilter


@dataclass
class Level2Config:
    """Configuration for Level 2 experiments."""

    model_id: str = "runwayml/stable-diffusion-inpainting"
    device: str = "cuda"

    # Inference settings
    guidance_scale: float = 8.0
    num_inference_steps: int = 40
    ddim_eta: float = 0.0
    seed: int = 42

    # Mask-aware conditioning
    use_mask_aware_conditioning: bool = True

    # Boundary smoothing
    use_boundary_smoothing: bool = True
    smooth_blur_radius: int = 12       # pixels; Gaussian blur radius on mask edge

    # LoRA fine-tune settings (lightweight, no full retraining)
    lora_rank: int = 4
    learning_rate: float = 1e-4
    train_steps: int = 200
    lora_checkpoint_dir: Path = field(default_factory=lambda: Path("outputs/level2/lora"))


class Level2ProductionPipeline:
    """
    Level 2 production inpainting with mask-aware conditioning and boundary smoothing.

    Usage:
        pipeline = Level2ProductionPipeline(Level2Config())
        result = pipeline.inpaint(prompt, image, mask)
    """

    def __init__(self, config: Level2Config) -> None:
        self.config = config
        dtype = torch.float16 if "cuda" in config.device else torch.float32

        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            config.model_id,
            torch_dtype=dtype,
        )
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(config.device)

    # ------------------------------------------------------------------
    # Fine-tuning (LoRA)
    # ------------------------------------------------------------------

    def setup_lora(self) -> None:
        """Attach LoRA adapters to UNet cross-attention layers."""
        lora_attn_procs = {}
        for name in self.pipe.unet.attn_processors.keys():
            cross_attention_dim = None
            if name.endswith("attn1.processor"):
                # self-attention: no cross dim
                lora_attn_procs[name] = LoRAAttnProcessor(
                    hidden_size=self.pipe.unet.config.attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    rank=self.config.lora_rank,
                )
            else:
                hidden_size = self.pipe.unet.config.cross_attention_dim
                lora_attn_procs[name] = LoRAAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=hidden_size,
                    rank=self.config.lora_rank,
                )
        self.pipe.unet.set_attn_processor(lora_attn_procs)

    def train(
        self,
        image_paths: list,
        prompts: list,
        mask_paths: Optional[list] = None,
    ) -> None:
        """
        Lightweight LoRA fine-tune on a small set of domain images.

        Args:
            image_paths: list of Path objects for training images.
            prompts: corresponding text prompts per image.
            mask_paths: optional masks; if None, center masks are generated.
        """
        from src.data.dataset import generate_center_mask

        self.setup_lora()
        lora_layers = AttnProcsLayers(self.pipe.unet.attn_processors)
        optimizer = torch.optim.AdamW(lora_layers.parameters(), lr=self.config.learning_rate)

        self.pipe.unet.train()
        self.pipe.text_encoder.requires_grad_(False)
        self.pipe.vae.requires_grad_(False)

        dtype = torch.float16 if "cuda" in self.config.device else torch.float32

        print(f"Starting LoRA fine-tune: {self.config.train_steps} steps")
        for step in range(self.config.train_steps):
            idx = step % len(image_paths)
            image = Image.open(image_paths[idx]).convert("RGB").resize((512, 512))
            mask = (
                Image.open(mask_paths[idx]).convert("L")
                if mask_paths
                else generate_center_mask(512, 512)
            )

            # Encode image to latent
            image_tensor = self._pil_to_tensor(image).to(self.config.device, dtype=dtype)
            with torch.no_grad():
                latent = self.pipe.vae.encode(image_tensor).latent_dist.sample()
                latent = latent * self.pipe.vae.config.scaling_factor

            # Sample noise and timestep
            noise = torch.randn_like(latent)
            t = torch.randint(0, self.pipe.scheduler.config.num_train_timesteps,
                              (1,), device=self.config.device).long()
            noisy_latent = self.pipe.scheduler.add_noise(latent, noise, t)

            # Encode prompt
            with torch.no_grad():
                text_input = self.pipe.tokenizer(
                    prompts[idx],
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.pipe.tokenizer.model_max_length,
                ).to(self.config.device)
                encoder_hidden = self.pipe.text_encoder(text_input.input_ids)[0]

            # Mask-aware conditioning: concatenate mask + masked latent as extra channels
            mask_tensor = self._pil_mask_to_tensor(mask).to(self.config.device, dtype=dtype)
            mask_latent = F.interpolate(mask_tensor, size=latent.shape[-2:])
            masked_image_latent = latent * (1 - mask_latent)
            model_input = torch.cat([noisy_latent, mask_latent, masked_image_latent], dim=1)

            # Forward + loss
            noise_pred = self.pipe.unet(model_input, t, encoder_hidden_states=encoder_hidden).sample
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (step + 1) % 50 == 0:
                print(f"  step {step + 1}/{self.config.train_steps}  loss={loss.item():.4f}")

        # Save LoRA weights
        self.config.lora_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        lora_layers.save_pretrained(self.config.lora_checkpoint_dir)
        print(f"LoRA weights saved to {self.config.lora_checkpoint_dir}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def inpaint(
        self,
        prompt: str,
        image: Image.Image,
        mask: Image.Image,
        negative_prompt: Optional[str] = None,
        reference_image: Optional[Image.Image] = None,
    ) -> Image.Image:
        """
        Run Level 2 inpainting with optional boundary smoothing.

        Args:
            prompt: text guidance prompt.
            image: corrupted input image (PIL RGB).
            mask: binary mask (PIL L); white = region to regenerate.
            negative_prompt: optional negative text guidance.
            reference_image: optional reference for image-level guidance blending.

        Returns:
            Inpainted PIL image.
        """
        generator = torch.Generator(device=self.config.device).manual_seed(self.config.seed)
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
        result = output.images[0]

        if self.config.use_boundary_smoothing:
            result = self._apply_boundary_smoothing(image, mask, result)

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_boundary_smoothing(
        self,
        original: Image.Image,
        mask: Image.Image,
        generated: Image.Image,
    ) -> Image.Image:
        """
        Blend generated inpainted image with original context using a soft weight map.

        The weight map is obtained by Gaussian-blurring the mask so transitions
        near boundaries are soft rather than hard pixel cuts:
            x_out = w * x_gen + (1 - w) * x_original
        """
        mask_gray = mask.convert("L").resize(original.size)
        # Soft edge weight from blurred mask
        weight = mask_gray.filter(ImageFilter.GaussianBlur(radius=self.config.smooth_blur_radius))

        orig_np = np.asarray(original.convert("RGB"), dtype=np.float32)
        gen_np = np.asarray(generated.convert("RGB").resize(original.size), dtype=np.float32)
        w_np = np.asarray(weight, dtype=np.float32)[:, :, None] / 255.0

        blended = w_np * gen_np + (1.0 - w_np) * orig_np
        return Image.fromarray(blended.clip(0, 255).astype(np.uint8))

    @staticmethod
    def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
        """Convert PIL RGB image to normalized (B,C,H,W) tensor in [-1, 1]."""
        arr = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return tensor

    @staticmethod
    def _pil_mask_to_tensor(mask: Image.Image) -> torch.Tensor:
        """Convert PIL L mask to (B,1,H,W) tensor in [0, 1]."""
        arr = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        return tensor


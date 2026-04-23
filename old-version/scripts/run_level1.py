from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data.dataset import load_samples
from src.eval.metrics import evaluate_pairs, save_metrics_csv, summarize_metrics
from src.eval.visualize import save_comparison_panel
from src.models.level1_sd_inpaint import Level1Config, StableDiffusionInpaintingLevel1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Level 1 SD inpainting baseline")
    parser.add_argument("--data-dir", type=Path, default=Path("data/samples"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/level1"))
    parser.add_argument("--model-id", type=str, default="runwayml/stable-diffusion-inpainting")
    parser.add_argument("--prompt", type=str, default="A natural and realistic completion")
    parser.add_argument("--negative-prompt", type=str, default="blurry, distorted, low quality")
    parser.add_argument("--mask-type", type=str, default="center", choices=["center", "irregular"])
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--ddim-eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "panels").mkdir(parents=True, exist_ok=True)

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    
    model = StableDiffusionInpaintingLevel1(
        Level1Config(
            model_id=args.model_id,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_steps,
            ddim_eta=args.ddim_eta,
            device=device,
        )
    )

    image_names = []
    targets = []
    predictions = []

    for sample in load_samples(args.data_dir, (args.image_size, args.image_size), args.mask_type):
        prediction = model.inpaint(
            prompt=args.prompt,
            image=sample.corrupted,
            mask=sample.mask,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
        )

        image_names.append(sample.image_path.name)
        targets.append(sample.image)
        predictions.append(prediction)

        save_comparison_panel(
            original=sample.image,
            mask=sample.mask,
            corrupted=sample.corrupted,
            prediction=prediction,
            out_path=args.output_dir / "panels" / f"{sample.image_path.stem}_panel.png",
            title=f"Level 1 | {sample.image_path.name}",
        )

        prediction.save(args.output_dir / "predictions" / sample.image_path.name)

    if not image_names:
        raise RuntimeError(
            f"No images found in {args.data_dir}. Add sample images before running Level 1."
        )

    results = evaluate_pairs(image_names, targets, predictions)
    save_metrics_csv(results, args.output_dir / "metrics.csv")

    summary = summarize_metrics(results)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Level 1 complete:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

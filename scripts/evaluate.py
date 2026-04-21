from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from src.eval.metrics import evaluate_pairs, save_metrics_csv, summarize_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predictions against ground truth")
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/eval"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    image_names = []
    targets = []
    predictions = []

    for pred_file in sorted(args.predictions.iterdir()):
        if not pred_file.is_file():
            continue
        target_file = args.targets / pred_file.name
        if not target_file.exists():
            continue

        image_names.append(pred_file.name)
        targets.append(Image.open(target_file).convert("RGB"))
        predictions.append(Image.open(pred_file).convert("RGB"))

    results = evaluate_pairs(image_names, targets, predictions)
    save_metrics_csv(results, args.output / "metrics.csv")

    summary = summarize_metrics(results)
    with (args.output / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

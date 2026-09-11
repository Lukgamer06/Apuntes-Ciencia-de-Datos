"""Run a reproducible YOLOv8 random search and keep the three best models.

Each .pt file contains the learned weights and biases. The companion txt file
records the exact hyperparameters, metrics, and source checkpoint.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

from ia import build_dataset

SEARCH_SPACE: dict[str, list[Any]] = {
    "lr0": [0.0005, 0.001, 0.005, 0.01],
    "lrf": [0.01, 0.05, 0.1],
    "weight_decay": [0.0001, 0.0005, 0.001],
    "hsv_h": [0.01, 0.02, 0.03],
    "hsv_s": [0.5, 0.7, 0.9],
    "hsv_v": [0.3, 0.5, 0.7],
    "degrees": [0.0, 5.0, 10.0],
    "scale": [0.3, 0.5, 0.7],
    "fliplr": [0.0, 0.5],
    "mosaic": [0.5, 1.0],
    "mixup": [0.0, 0.1],
    "dropout": [0.0, 0.1, 0.2],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).parent / "Data")
    parser.add_argument("--dataset-root", type=Path, default=Path(__file__).parent / "yolo_dataset")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "top_models")
    return parser.parse_args()


def sample_config(generator: random.Random) -> dict[str, Any]:
    return {name: generator.choice(values) for name, values in SEARCH_SPACE.items()}


def save_artifact(source: Path, destination: Path, config: dict[str, Any], score: float, trial: int) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination / "best.pt")
    metadata = {
        "trial": trial,
        "metric": "mAP50-95",
        "score": score,
        "hyperparameters": config,
        "note": "best.pt includes YOLO learned weights and biases",
    }
    (destination / "configuration.txt").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("Instala Placas/requirements.txt antes de entrenar") from error

    dataset_root, _ = build_dataset(args.data_root, args.dataset_root, 0.20, args.seed, False)
    args.output.mkdir(parents=True, exist_ok=True)
    generator = random.Random(args.seed)
    results: list[dict[str, Any]] = []

    for trial in range(1, args.trials + 1):
        config = sample_config(generator)
        run_name = f"random_trial_{trial:03d}"
        model = YOLO(args.weights)
        model.train(
            data=str(dataset_root / "dataset.yaml"), epochs=args.epochs, imgsz=args.imgsz,
            batch=args.batch, device=args.device, workers=4, patience=20, seed=args.seed + trial,
            project=str(args.output / "runs"), name=run_name, exist_ok=True,
            optimizer="AdamW", amp=True, **config,
        )
        metrics = model.val(data=str(dataset_root / "dataset.yaml"), split="val", imgsz=args.imgsz, device=args.device)
        score = float(metrics.box.map)
        best_path = Path(model.trainer.best)
        results.append({"trial": trial, "score": score, "weights": str(best_path), "hyperparameters": config})
        print(f"Trial {trial}/{args.trials}: mAP50-95={score:.5f}")

    results.sort(key=lambda item: item["score"], reverse=True)
    (args.output / "search_results.json").write_text(json.dumps(results, indent=2) + "\n")
    for rank, result in enumerate(results[:3], 1):
        save_artifact(Path(result["weights"]), args.output / f"model_{rank}", result["hyperparameters"], result["score"], result["trial"])
    if results:
        shutil.copy2(args.output / "model_1" / "best.pt", Path(__file__).parent / "best.pt")
    print(f"Guardados {min(3, len(results))} modelos en {args.output}")


if __name__ == "__main__":
    main()

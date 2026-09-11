"""Prepare and retrain a YOLOv8 detection model from mixed datasets.

The source data is expected at Data/images and Data/labels. Original files are
never changed: labels with polygon coordinates are converted to bounding boxes
inside a generated dataset directory.
"""

from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image


# ------------------------------ Editable settings -------------------------
DATA_ROOT = Path(__file__).parent / "Data"
OUTPUT_ROOT = Path(__file__).parent / "yolo_dataset"
MODEL_WEIGHTS = "yolov8n.pt"
RUNS_ROOT = Path(__file__).parent / "runs"
RUN_NAME = "placas_yolov8"

EPOCHS = 100
IMAGE_SIZE = 640
BATCH_SIZE = 16
DEVICE = "0"  # Use "cpu", "0", or "0,1".
WORKERS = 4
PATIENCE = 30
SEED = 42
VALIDATION_RATIO = 0.20
FORCE_REBUILD_DATASET = False

# Use the real names in the same order as the source class IDs. If omitted,
# names class_0, class_1, ... are generated from the largest observed ID.
CLASS_NAMES: list[str] | None = None

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_arguments() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
	parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
	parser.add_argument("--weights", default=MODEL_WEIGHTS)
	parser.add_argument("--epochs", type=int, default=EPOCHS)
	parser.add_argument("--imgsz", type=int, default=IMAGE_SIZE)
	parser.add_argument("--batch", type=int, default=BATCH_SIZE)
	parser.add_argument("--device", default=DEVICE)
	parser.add_argument("--workers", type=int, default=WORKERS)
	parser.add_argument("--patience", type=int, default=PATIENCE)
	parser.add_argument("--val-ratio", type=float, default=VALIDATION_RATIO)
	parser.add_argument("--seed", type=int, default=SEED)
	parser.add_argument("--run-name", default=RUN_NAME)
	parser.add_argument("--force-rebuild", action="store_true")
	return parser.parse_args()


def read_yolo_label(label_path: Path) -> tuple[list[str], int | None]:
	"""Validate a label and normalize boxes/polygons to YOLO boxes."""
	normalized: list[str] = []
	maximum_class: int | None = None

	for line_number, line in enumerate(label_path.read_text().splitlines(), 1):
		values = line.split()
		if not values:
			continue
		if len(values) < 5 or (len(values) > 5 and len(values) % 2 == 0):
			raise ValueError(
				f"{label_path.name}:{line_number}: expected 5 box values "
				"or class plus an even number of polygon coordinates"
			)

		try:
			class_id = int(values[0])
			coordinates = [float(value) for value in values[1:]]
		except ValueError as error:
			raise ValueError(f"{label_path.name}:{line_number}: non-numeric value") from error

		if class_id < 0 or any(value < 0 or value > 1 for value in coordinates):
			raise ValueError(f"{label_path.name}:{line_number}: values must be in [0, 1]")

		if len(coordinates) == 4:
			center_x, center_y, width, height = coordinates
		else:
			x_values = coordinates[::2]
			y_values = coordinates[1::2]
			x_min, x_max = min(x_values), max(x_values)
			y_min, y_max = min(y_values), max(y_values)
			center_x = (x_min + x_max) / 2
			center_y = (y_min + y_max) / 2
			width = x_max - x_min
			height = y_max - y_min

		if width <= 0 or height <= 0:
			raise ValueError(f"{label_path.name}:{line_number}: empty bounding box")

		normalized.append(
			f"{class_id} {center_x:.8f} {center_y:.8f} {width:.8f} {height:.8f}"
		)
		maximum_class = class_id if maximum_class is None else max(maximum_class, class_id)

	return normalized, maximum_class


def build_dataset(data_root: Path, output_root: Path, val_ratio: float, seed: int, force: bool) -> tuple[Path, list[str]]:
	image_root = data_root / "images"
	label_root = data_root / "labels"
	if not image_root.is_dir() or not label_root.is_dir():
		raise FileNotFoundError("The data root must contain images/ and labels/ folders")
	if not 0 < val_ratio < 1:
		raise ValueError("--val-ratio must be between 0 and 1")

	pairs = []
	class_ids = Counter()
	invalid_files = []
	label_paths = {path.stem: path for path in label_root.glob("*.txt")}

	for image_path in sorted(image_root.iterdir()):
		if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
			continue
		label_path = label_paths.get(image_path.stem)
		if label_path is None:
			invalid_files.append(f"missing label: {image_path.name}")
			continue
		try:
			normalized, maximum_class = read_yolo_label(label_path)
			if maximum_class is not None:
				for row in normalized:
					class_ids[int(row.split()[0])] += 1
			with Image.open(image_path) as image:
				image.verify()
			pairs.append((image_path, normalized))
		except (OSError, ValueError) as error:
			invalid_files.append(f"invalid pair: {image_path.name} ({error})")

	if not pairs:
		raise RuntimeError("No valid image/label pairs were found")
	if force and output_root.exists():
		shutil.rmtree(output_root)
	if output_root.exists() and (output_root / "dataset.yaml").exists():
		return output_root, make_class_names(class_ids)

	random_generator = random.Random(seed)
	random_generator.shuffle(pairs)
	validation_count = max(1, round(len(pairs) * val_ratio))
	splits = {"val": pairs[:validation_count], "train": pairs[validation_count:]}
	if not splits["train"]:
		raise RuntimeError("The dataset needs at least two valid pairs")

	for split, split_pairs in splits.items():
		image_destination = output_root / "images" / split
		label_destination = output_root / "labels" / split
		image_destination.mkdir(parents=True, exist_ok=True)
		label_destination.mkdir(parents=True, exist_ok=True)
		for image_path, normalized in split_pairs:
			shutil.copy2(image_path, image_destination / image_path.name)
			(label_destination / f"{image_path.stem}.txt").write_text(
				"\n".join(normalized) + ("\n" if normalized else "")
			)

	names = make_class_names(class_ids)
	yaml_lines = [
		f"path: {output_root.resolve().as_posix()}",
		"train: images/train",
		"val: images/val",
		f"nc: {len(names)}",
		"names:",
	] + [f"  {index}: {name}" for index, name in enumerate(names)]
	(output_root / "dataset.yaml").write_text("\n".join(yaml_lines) + "\n")
	print(f"Valid pairs: {len(pairs)}; train: {len(splits['train'])}; val: {len(splits['val'])}")
	print(f"Skipped files: {len(invalid_files)}")
	if invalid_files:
		(output_root / "validation_errors.txt").write_text("\n".join(invalid_files) + "\n")
		print(f"Details written to {output_root / 'validation_errors.txt'}")
	return output_root, names


def make_class_names(class_ids: Counter[int]) -> list[str]:
	if CLASS_NAMES is not None:
		if any(class_id >= len(CLASS_NAMES) for class_id in class_ids):
			raise ValueError("CLASS_NAMES does not contain all observed class IDs")
		return CLASS_NAMES
	return [f"class_{class_id}" for class_id in range(max(class_ids, default=-1) + 1)]


def train(dataset_root: Path, weights: str, args: argparse.Namespace) -> None:
	try:
		from ultralytics import YOLO  # type: ignore[import-not-found]
	except ImportError as error:
		raise RuntimeError(
			"Install dependencies first with: python -m pip install -r Placas/requirements.txt"
		) from error

	model = YOLO(weights)
	model.train(
		data=str(dataset_root / "dataset.yaml"),
		epochs=args.epochs,
		imgsz=args.imgsz,
		batch=args.batch,
		device=args.device,
		workers=args.workers,
		patience=args.patience,
		seed=args.seed,
		project=str(RUNS_ROOT),
		name=args.run_name,
		exist_ok=True,
	)


def main() -> None:
	args = parse_arguments()
	dataset_root, names = build_dataset(
		args.data_root, args.output_root, args.val_ratio, args.seed, args.force_rebuild
	)
	print(f"Classes ({len(names)}): {names}")
	train(dataset_root, args.weights, args)


if __name__ == "__main__":
	main()

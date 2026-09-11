"""FastAPI service for license-plate detection and OCR."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import easyocr
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = BASE_DIR / "models" / "best.pt"
MODEL_PATH = Path(os.getenv("PLATE_MODEL", str(DEFAULT_WEIGHTS)))
OCR_LANGUAGES = [language.strip() for language in os.getenv("OCR_LANGUAGES", "en").split(",")]
CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONFIDENCE", "0.35"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))

app = FastAPI(title="Plate Reader API", version="1.0.0")
app.add_middleware(
	CORSMiddleware,
	allow_origins=[origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",")],
	allow_methods=["*"],
	allow_headers=["*"],
)

_model: YOLO | None = None
_reader: easyocr.Reader | None = None


def get_model() -> YOLO:
	global _model
	if _model is None:
		if not MODEL_PATH.exists():
			raise HTTPException(
				status_code=503,
				detail=f"No se encontro el peso del modelo en {MODEL_PATH}. Configura PLATE_MODEL.",
			)
		_model = YOLO(str(MODEL_PATH))
	return _model


def get_reader() -> easyocr.Reader:
	global _reader
	if _reader is None:
		_reader = easyocr.Reader(OCR_LANGUAGES, gpu=os.getenv("EASYOCR_GPU", "false").lower() == "true")
	return _reader


def decode_image(contents: bytes) -> np.ndarray:
	if not contents or len(contents) > MAX_UPLOAD_BYTES:
		raise HTTPException(status_code=413, detail="La imagen esta vacia o supera el tamano permitido.")
	image = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
	if image is None:
		raise HTTPException(status_code=400, detail="El archivo no es una imagen valida.")
	return image


def clean_text(text: str) -> str:
	return "".join(character for character in text.upper() if character.isalnum())


def recognize_plate(crop: np.ndarray) -> tuple[str, float]:
	if crop.size == 0:
		return "", 0.0
	gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
	gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
	processed = cv2.bilateralFilter(gray, 7, 50, 50)
	result = get_reader().readtext(processed, detail=1, paragraph=False)
	parts = [clean_text(item[1]) for item in result if clean_text(item[1])]
	text = "".join(parts)
	confidence = max((float(item[2]) for item in result), default=0.0)
	return text, confidence


def predict(image: np.ndarray) -> list[dict[str, Any]]:
	result = get_model().predict(source=image, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
	if result.boxes is None:
		return []
	plates: list[dict[str, Any]] = []
	height, width = image.shape[:2]
	for box in result.boxes:
		x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
		x1, x2 = max(0, x1), min(width, x2)
		y1, y2 = max(0, y1), min(height, y2)
		text, ocr_confidence = recognize_plate(image[y1:y2, x1:x2])
		class_id = int(box.cls[0])
		plates.append({
			"text": text,
			"confidence": round(float(box.conf[0]), 4),
			"ocr_confidence": round(ocr_confidence, 4),
			"class_id": class_id,
			"class_name": result.names.get(class_id, str(class_id)),
			"box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
		})
	return plates


@app.get("/health")
def health() -> dict[str, str | bool]:
	return {"status": "ok", "model_loaded": _model is not None, "model_path": str(MODEL_PATH)}


@app.post("/predict")
async def predict_plate(file: UploadFile = File(...)) -> dict[str, Any]:
	if file.content_type and not file.content_type.startswith("image/"):
		raise HTTPException(status_code=415, detail="Envie un archivo de imagen.")
	try:
		image = decode_image(await file.read())
		plates = predict(image)
	except HTTPException:
		raise
	except Exception as error:
		raise HTTPException(status_code=500, detail=f"Error durante la inferencia: {error}") from error
	return {"filename": file.filename, "width": image.shape[1], "height": image.shape[0], "plates": plates}

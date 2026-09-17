"""FastAPI service for license-plate detection and OCR."""

from __future__ import annotations

import os
import logging
import asyncio
import json
import re
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from pathlib import Path
from typing import Any

import cv2
import easyocr
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("plate-reader")

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = BASE_DIR / "models" / "best.pt"
MODEL_PATH = Path(os.getenv("PLATE_MODEL", str(DEFAULT_WEIGHTS)))
OCR_LANGUAGES = [language.strip() for language in os.getenv("OCR_LANGUAGES", "en").split(",")]
CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONFIDENCE", "0.35"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))
OCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
MAX_INFERENCE_DIMENSION = int(os.getenv("MAX_INFERENCE_DIMENSION", "1600"))
MAX_DETECTIONS = int(os.getenv("MAX_DETECTIONS", "3"))
CROMA_URL = "https://api.croma.run/co/simit/account-status/v1"
CROMA_TIMEOUT_SECONDS = float(os.getenv("CROMA_TIMEOUT_SECONDS", "20"))
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
PLATE_PATTERN = re.compile(r"^[A-Z]{3}(?:\d{3}|\d{2}[A-Z])$")

app = FastAPI(title="Plate Reader API", version="1.0.0")
app.add_middleware(
	CORSMiddleware,
	allow_origins=[origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",")],
	allow_methods=["*"],
	allow_headers=["*"],
)

_model: YOLO | None = None
_reader: easyocr.Reader | None = None


def get_croma_api_key() -> str:
	return os.getenv("CROMA_API_KEY", "").strip()


def get_model() -> YOLO:
	global _model
	if _model is None:
		if not MODEL_PATH.exists():
			raise HTTPException(
				status_code=503,
				detail=f"No se encontro el peso del modelo en {MODEL_PATH}. Configura PLATE_MODEL.",
			)
		logger.info("Cargando modelo YOLO desde %s", MODEL_PATH)
		_model = YOLO(str(MODEL_PATH))
		logger.info("Modelo YOLO cargado")
	return _model


def get_reader() -> easyocr.Reader:
	global _reader
	if _reader is None:
		logger.info("Cargando EasyOCR con idiomas %s", OCR_LANGUAGES)
		_reader = easyocr.Reader(OCR_LANGUAGES, gpu=os.getenv("EASYOCR_GPU", "false").lower() == "true")
		logger.info("EasyOCR cargado")
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
	filtered = cv2.bilateralFilter(gray, 7, 50, 50)
	variants = [
		gray,
		cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
	]

	best_text = ""
	best_confidence = 0.0
	for variant in variants:
		results = get_reader().readtext(
			variant,
			detail=1,
			paragraph=False,
			allowlist=OCR_ALLOWLIST,
			mag_ratio=1.0,
		)
		items = [(clean_text(item[1]), float(item[2])) for item in results if clean_text(item[1])]
		candidates: list[tuple[str, float]] = []
		for index, (text, confidence) in enumerate(items):
			if PLATE_PATTERN.fullmatch(text):
				candidates.append((text, confidence))
			if index + 1 < len(items):
				joined = text + items[index + 1][0]
				if PLATE_PATTERN.fullmatch(joined):
					candidates.append((joined, min(confidence, items[index + 1][1])))
		if candidates:
			text, confidence = max(candidates, key=lambda candidate: (candidate[1], candidate[0]))
			if confidence > best_confidence:
				best_text, best_confidence = text, confidence

	return best_text, best_confidence


def resize_for_inference(image: np.ndarray) -> np.ndarray:
	height, width = image.shape[:2]
	longest_side = max(height, width)
	if longest_side <= MAX_INFERENCE_DIMENSION:
		return image
	scale = MAX_INFERENCE_DIMENSION / longest_side
	return cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)


def query_simit_sync(plate: str) -> dict[str, Any]:
	croma_api_key = get_croma_api_key()
	if not croma_api_key:
		logger.error("Consulta Croma omitida para %s: CROMA_API_KEY no esta configurada", plate)
		return {"error": "CROMA_API_KEY no esta configurada en el servidor."}
	logger.info("Consultando Croma para la placa %s", plate)
	request = Request(
		CROMA_URL,
		data=json.dumps({"document_number": plate}).encode("utf-8"),
		headers={
			"Authorization": f"Bearer {croma_api_key}",
			"Content-Type": "application/json",
		},
		method="POST",
	)
	try:
		with urlopen(request, timeout=CROMA_TIMEOUT_SECONDS) as response:
			result = json.loads(response.read().decode("utf-8"))
			logger.info("Croma respondio HTTP %s para %s", response.status, plate)
			return result
	except HTTPError as error:
		body = error.read().decode("utf-8", errors="replace")
		logger.warning("Croma respondio HTTP %s para %s: %s", error.code, plate, body[:500])
		return {"error": f"Croma respondio HTTP {error.code}", "details": body[:500]}
	except (URLError, TimeoutError, json.JSONDecodeError) as error:
		logger.warning("No se pudo consultar Croma para %s: %s", plate, error)
		return {"error": "No fue posible consultar SIMIT en este momento."}


async def query_simit(plate: str) -> dict[str, Any]:
	return await asyncio.to_thread(query_simit_sync, plate)


def require_database() -> str:
	if not DATABASE_URL:
		raise HTTPException(status_code=503, detail="DATABASE_URL no esta configurada en el servidor.")
	return DATABASE_URL


def initialize_database() -> None:
	if not DATABASE_URL:
		return
	with psycopg.connect(DATABASE_URL) as connection:
		connection.execute(
			"""
			CREATE TABLE IF NOT EXISTS plate_history (
				plate VARCHAR(6) PRIMARY KEY,
				simit_data JSONB,
				last_checked_at TIMESTAMPTZ,
				created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
			)
			"""
		)


def store_plate(plate: str, simit_data: dict[str, Any] | None = None) -> None:
	if not DATABASE_URL:
		logger.warning("No se guardo %s: DATABASE_URL no esta configurada", plate)
		return
	simit_json = Jsonb(simit_data) if simit_data is not None else None
	checked_at = datetime.now(timezone.utc) if simit_data is not None else None
	with psycopg.connect(DATABASE_URL) as connection:
		connection.execute(
			"""
			INSERT INTO plate_history (plate, simit_data, last_checked_at)
			VALUES (%s, %s, %s)
			ON CONFLICT (plate) DO UPDATE SET
				simit_data = COALESCE(EXCLUDED.simit_data, plate_history.simit_data),
				last_checked_at = COALESCE(EXCLUDED.last_checked_at, plate_history.last_checked_at)
			""",
			(plate, simit_json, checked_at),
		)



async def query_and_store(plate: str) -> None:
	try:
		await asyncio.to_thread(store_plate, plate)
		simit_data = await query_simit(plate)
		await asyncio.to_thread(store_plate, plate, simit_data)
		logger.info("Placa %s guardada con respuesta SIMIT", plate)
	except Exception:
		logger.exception("No se pudo guardar la placa %s", plate)


def list_plates() -> list[dict[str, Any]]:
	with psycopg.connect(require_database(), row_factory=dict_row) as connection:
		return list(connection.execute(
			"SELECT plate, simit_data, last_checked_at, created_at FROM plate_history ORDER BY created_at DESC"
		).fetchall())


def delete_plate(plate: str) -> bool:
	with psycopg.connect(require_database()) as connection:
		result = connection.execute("DELETE FROM plate_history WHERE plate = %s", (plate,))
		return result.rowcount > 0


def refresh_plates_sync() -> list[str]:
	with psycopg.connect(require_database(), row_factory=dict_row) as connection:
		return [row["plate"] for row in connection.execute("SELECT plate FROM plate_history").fetchall()]


def predict(image: np.ndarray) -> list[dict[str, Any]]:
	image = resize_for_inference(image)
	logger.info("Iniciando deteccion YOLO para imagen %sx%s", image.shape[1], image.shape[0])
	result = get_model().predict(
		source=image,
		conf=CONFIDENCE_THRESHOLD,
		imgsz=640,
		max_det=MAX_DETECTIONS,
		verbose=False,
	)[0]
	if result.boxes is None:
		logger.info("YOLO no devolvio cajas")
		return []
	plates: list[dict[str, Any]] = []
	height, width = image.shape[:2]
	for box in result.boxes:
		x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
		x1, x2 = max(0, x1), min(width, x2)
		y1, y2 = max(0, y1), min(height, y2)
		text, ocr_confidence = recognize_plate(image[y1:y2, x1:x2])
		logger.info("Placa detectada: caja=%s texto=%s", (x1, y1, x2, y2), text or "<vacio>")
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
	database_connected = False
	if DATABASE_URL:
		try:
			with psycopg.connect(DATABASE_URL, connect_timeout=5) as connection:
				connection.execute("SELECT 1")
			database_connected = True
		except Exception as error:
			logger.warning("La base de datos no esta disponible: %s", error)
	return {
		"status": "ok",
		"model_loaded": _model is not None,
		"model_path": str(MODEL_PATH),
		"croma_api_key_configured": bool(get_croma_api_key()),
		"database_configured": bool(DATABASE_URL),
		"database_connected": database_connected,
	}


@app.post("/predict")
async def predict_plate(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> dict[str, Any]:
	logger.info("Peticion /predict recibida: nombre=%s tipo=%s", file.filename, file.content_type)
	if file.content_type and not file.content_type.startswith("image/"):
		raise HTTPException(status_code=415, detail="Envie un archivo de imagen.")
	try:
		contents = await file.read()
		logger.info("Imagen recibida: %s bytes", len(contents))
		image = decode_image(contents)
		plates = predict(image)
		for plate in plates:
			if plate["text"]:
				background_tasks.add_task(query_and_store, plate["text"])
	except HTTPException:
		raise
	except Exception as error:
		raise HTTPException(status_code=500, detail=f"Error durante la inferencia: {error}") from error
	logger.info("Peticion /predict terminada: %s placas", len(plates))
	return {"filename": file.filename, "plates": [{"text": plate["text"]} for plate in plates if plate["text"]]}


@app.on_event("startup")
def startup() -> None:
	try:
		initialize_database()
		logger.info("Base de datos lista")
	except Exception as error:
		logger.error("No se pudo inicializar la base de datos: %s", error)


@app.get("/history")
async def history() -> dict[str, Any]:
	return {"items": await asyncio.to_thread(list_plates)}


@app.delete("/history/{plate}")
async def remove_history(plate: str) -> dict[str, str]:
	plate = clean_text(plate)
	if not PLATE_PATTERN.fullmatch(plate):
		raise HTTPException(status_code=400, detail="La placa debe tener formato ABC123 o ABC12D.")
	deleted = await asyncio.to_thread(delete_plate, plate)
	if not deleted:
		raise HTTPException(status_code=404, detail="La placa no existe en el historial.")
	return {"message": f"Placa {plate} eliminada."}


@app.post("/history/refresh")
async def refresh_history() -> dict[str, Any]:
	plates = await asyncio.to_thread(refresh_plates_sync)
	await asyncio.gather(*(query_and_store(plate) for plate in plates))
	return {"updated": len(plates)}

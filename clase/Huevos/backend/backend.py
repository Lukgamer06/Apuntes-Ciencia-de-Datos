"""API de inferencia en vivo para el detector de huevos."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from threading import Lock
from urllib.request import Request, urlopen

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("egg-detector")

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = BASE_DIR / "models" / "best.pt"
MODEL_PATH = Path(os.getenv("EGG_MODEL", str(DEFAULT_MODEL)))
CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONFIDENCE", "0.35"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(2 * 1024 * 1024)))
MAX_DETECTIONS = int(os.getenv("MAX_DETECTIONS", "20"))
ESP32_URL = os.getenv("ESP32_URL", "").rstrip("/")
ESP32_TIMEOUT = float(os.getenv("ESP32_TIMEOUT", "2.0"))

app = FastAPI(title="Egg Live Detector API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",")],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model: YOLO | None = None
_model_lock = Lock()
motor_enabled = False


class MotorCommand(BaseModel):
    enabled: bool


def get_model() -> YOLO:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                if not MODEL_PATH.exists():
                    raise HTTPException(
                        status_code=503,
                        detail=(
                            f"No se encontro el modelo entrenado en {MODEL_PATH}. "
                            "Guarda alli best.pt o configura EGG_MODEL."
                        ),
                    )
                logger.info("Cargando modelo YOLO desde %s", MODEL_PATH)
                _model = YOLO(str(MODEL_PATH))
    return _model


def decode_image(contents: bytes) -> np.ndarray:
    if not contents or len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="El frame esta vacio o supera el tamano permitido.")
    image = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="El frame recibido no es una imagen valida.")
    return image


def infer(image: np.ndarray) -> list[dict[str, object]]:
    result = get_model().predict(
        source=image,
        conf=CONFIDENCE_THRESHOLD,
        imgsz=640,
        max_det=MAX_DETECTIONS,
        verbose=False,
    )[0]
    if result.boxes is None:
        return []

    height, width = image.shape[:2]
    detections: list[dict[str, object]] = []
    for box in result.boxes:
        x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
        class_id = int(box.cls[0])
        names = result.names
        class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
        detections.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(float(box.conf[0]), 4),
                "box": {
                    "x1": max(0, min(x1, width)),
                    "y1": max(0, min(y1, height)),
                    "x2": max(0, min(x2, width)),
                    "y2": max(0, min(y2, height)),
                },
            }
        )
    return detections


def send_motor_command(enabled: bool) -> str:
    """Envia la orden al ESP32 cuando ESP32_URL esta configurada.

    Sin ESP32_URL funciona en modo simulacion para poder probar la app.
    El ESP32 debe exponer POST /motor con {"enabled": true|false}.
    """
    if not ESP32_URL:
        logger.info("Motor %s (simulacion; ESP32_URL no configurada)", "ON" if enabled else "OFF")
        return "simulado"

    request = Request(
        f"{ESP32_URL}/motor",
        data=json.dumps({"enabled": enabled}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=ESP32_TIMEOUT) as response:
            response.read()
        return "esp32"
    except Exception as error:
        logger.warning("No se pudo enviar la orden al ESP32: %s", error)
        raise HTTPException(status_code=502, detail="No se pudo comunicar con el ESP32.") from error


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_exists": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH),
        "motor_transport": "esp32" if ESP32_URL else "simulado",
        "motor_enabled": motor_enabled,
    }


@app.post("/predict-frame")
async def predict_frame(file: UploadFile = File(...)) -> dict[str, object]:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="El frame debe ser una imagen JPEG o PNG.")
    try:
        image = decode_image(await file.read())
        detections = await asyncio.to_thread(infer, image)
        return {
            "width": image.shape[1],
            "height": image.shape[0],
            "detections": detections,
        }
    except HTTPException:
        raise
    except Exception as error:
        logger.exception("Fallo durante la inferencia")
        raise HTTPException(status_code=500, detail=f"Error durante la inferencia: {error}") from error


@app.post("/motor")
async def set_motor(command: MotorCommand) -> dict[str, object]:
    global motor_enabled
    transport = await asyncio.to_thread(send_motor_command, command.enabled)
    motor_enabled = command.enabled
    return {"enabled": motor_enabled, "transport": transport}

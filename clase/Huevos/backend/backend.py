"""API de inferencia en vivo para el detector de huevos."""

from __future__ import annotations

import asyncio
import logging
import os
import time
import unicodedata
from pathlib import Path
from threading import Lock

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
DEFAULT_CONDITION_MODEL = BASE_DIR / "models" / "best1.pt"
CONDITION_MODEL_PATH = Path(os.getenv("EGG_CONDITION_MODEL", str(DEFAULT_CONDITION_MODEL)))
CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONFIDENCE", "0.35"))
CONDITION_TIMEOUT_SECONDS = float(os.getenv("CONDITION_TIMEOUT_SECONDS", "10"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(2 * 1024 * 1024)))
MAX_DETECTIONS = int(os.getenv("MAX_DETECTIONS", "20"))
CLASS_NAMES = {0: "sano", 1: "roto"}
DAMAGED_CLASS_NAMES = ("roto", "danado", "broken", "damaged", "cracked", "defect")
HEALTHY_CLASS_NAMES = ("sano", "healthy", "intacto")
SERVO_HOME_ANGLE = int(os.getenv("SERVO_HOME_ANGLE", "90"))
SERVO_HEALTHY_ANGLE = int(os.getenv("SERVO_HEALTHY_ANGLE", "0"))
SERVO_DAMAGED_ANGLE = int(os.getenv("SERVO_DAMAGED_ANGLE", "180"))
SERVO_ANGLES = {0, 90, 180}
if not {SERVO_HOME_ANGLE, SERVO_HEALTHY_ANGLE, SERVO_DAMAGED_ANGLE} <= SERVO_ANGLES:
    raise ValueError("Los angulos del servo deben ser 0, 90 o 180.")
if CONDITION_TIMEOUT_SECONDS <= 0:
    raise ValueError("CONDITION_TIMEOUT_SECONDS debe ser mayor que cero.")

app = FastAPI(title="Egg Live Detector API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",")],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model: YOLO | None = None
_condition_model: YOLO | None = None
_model_lock = Lock()
_condition_model_lock = Lock()
_condition_inference_lock = Lock()
motor_enabled = False
last_servo_state: str | None = None
servo_angle = SERVO_HOME_ANGLE


class MotorCommand(BaseModel):
    enabled: bool


class ConditionModelBusy(Exception):
    pass


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


def get_condition_model() -> YOLO:
    global _condition_model
    if _condition_model is None:
        with _condition_model_lock:
            if _condition_model is None:
                if not CONDITION_MODEL_PATH.exists():
                    raise HTTPException(
                        status_code=503,
                        detail=(
                            f"No se encontro el modelo de estado en {CONDITION_MODEL_PATH}. "
                            "Guarda alli best1.pt o configura EGG_CONDITION_MODEL."
                        ),
                    )
                logger.info("Cargando modelo de estado YOLO desde %s", CONDITION_MODEL_PATH)
                _condition_model = YOLO(str(CONDITION_MODEL_PATH))
    return _condition_model


def decode_image(contents: bytes) -> np.ndarray:
    if not contents or len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="El frame esta vacio o supera el tamano permitido.")
    image = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="El frame recibido no es una imagen valida.")
    return image


def detect_eggs(image: np.ndarray) -> list[tuple[int, int, int, int, float]]:
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
    eggs = []
    for box in result.boxes:
        x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
        x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
        y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
        if x2 > x1 and y2 > y1:
            eggs.append((x1, y1, x2, y2, float(box.conf[0])))
    return eggs


def is_damaged_class(class_id: int, names: dict | list) -> bool:
    class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
    normalized_name = unicodedata.normalize("NFKD", str(class_name).lower())
    normalized_name = "".join(character for character in normalized_name if not unicodedata.combining(character))
    if any(name in normalized_name for name in HEALTHY_CLASS_NAMES):
        return False
    if any(name in normalized_name for name in DAMAGED_CLASS_NAMES):
        return True
    return class_id == 1


def classify_egg_crops(crops: list[np.ndarray]) -> list[tuple[bool, float]]:
    if not _condition_inference_lock.acquire(blocking=False):
        raise ConditionModelBusy
    try:
        model = get_condition_model()
        results = model.predict(
            source=crops,
            conf=CONFIDENCE_THRESHOLD,
            imgsz=640,
            max_det=MAX_DETECTIONS,
            verbose=False,
        )
    finally:
        _condition_inference_lock.release()

    classifications = []
    for result in results:
        probabilities = getattr(result, "probs", None)
        if probabilities is not None:
            class_id = int(probabilities.top1)
            confidence = float(probabilities.top1conf)
            classifications.append((is_damaged_class(class_id, result.names), confidence))
            continue

        result_boxes = result.boxes
        if result_boxes is None:
            classifications.append((False, 0.0))
            continue

        damaged_confidences = [
            float(box.conf[0])
            for box in result_boxes
            if is_damaged_class(int(box.cls[0]), result.names)
        ]
        classifications.append((bool(damaged_confidences), max(damaged_confidences, default=0.0)))
    return classifications


async def infer(image: np.ndarray) -> list[dict[str, object]]:
    eggs = await asyncio.to_thread(detect_eggs, image)
    if not eggs:
        return []

    crops = [image[y1:y2, x1:x2] for x1, y1, x2, y2, _ in eggs]
    try:
        classifications = await asyncio.wait_for(
            asyncio.to_thread(classify_egg_crops, crops),
            timeout=CONDITION_TIMEOUT_SECONDS,
        )
    except ConditionModelBusy:
        logger.warning("best1 sigue procesando un frame anterior; se consideran sanos los huevos actuales")
        classifications = [(False, 0.0)] * len(eggs)
    except asyncio.TimeoutError:
        logger.warning(
            "best1 supero %.1f segundos; se consideran sanos los huevos de este frame",
            CONDITION_TIMEOUT_SECONDS,
        )
        classifications = [(False, 0.0)] * len(eggs)

    height, width = image.shape[:2]
    detections = []
    for (x1, y1, x2, y2, egg_confidence), (damaged, condition_confidence) in zip(eggs, classifications):
        class_id = 1 if damaged else 0
        condition = CLASS_NAMES[class_id]
        detections.append(
            {
                "class_id": class_id,
                "class_name": condition,
                "condition": condition,
                "confidence": round(egg_confidence, 4),
                "condition_confidence": round(condition_confidence, 4),
                "box": {
                    "x1": max(0, min(x1, width)),
                    "y1": max(0, min(y1, height)),
                    "x2": max(0, min(x2, width)),
                    "y2": max(0, min(y2, height)),
                },
            }
        )
    return detections


def update_servo(detections: list[dict[str, object]]) -> None:
    global last_servo_state, servo_angle
    detected_state = None
    if detections:
        detected_state = "roto" if any(item["condition"] == "roto" for item in detections) else "sano"

    if detected_state is None:
        last_servo_state = None
        servo_angle = SERVO_HOME_ANGLE
        return

    if detected_state == last_servo_state:
        return

    servo_angle = SERVO_DAMAGED_ANGLE if detected_state == "roto" else SERVO_HEALTHY_ANGLE
    last_servo_state = detected_state
    logger.info("Clasificacion %s: servo %d grados (disponible en GET /servo)", detected_state, servo_angle)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_exists": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH),
        "condition_model_loaded": _condition_model is not None,
        "condition_model_exists": CONDITION_MODEL_PATH.exists(),
        "condition_model_path": str(CONDITION_MODEL_PATH),
        "condition_timeout_seconds": CONDITION_TIMEOUT_SECONDS,
        "motor_transport": "polling",
        "motor_enabled": motor_enabled,
        "servo_angle": servo_angle,
        "servo_home_angle": SERVO_HOME_ANGLE,
        "servo_healthy_angle": SERVO_HEALTHY_ANGLE,
        "servo_damaged_angle": SERVO_DAMAGED_ANGLE,
    }


@app.get("/servo")
def get_servo() -> dict[str, int]:
    return {"angle": servo_angle}


@app.post("/predict-frame")
async def predict_frame(file: UploadFile = File(...)) -> dict[str, object]:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="El frame debe ser una imagen JPEG o PNG.")
    started_at = time.perf_counter()
    try:
        image = decode_image(await file.read())
        detections = await infer(image)
        await asyncio.to_thread(update_servo, detections)
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        logger.info(
            "Inferencia: %sx%s, %d deteccion(es), %.0f ms",
            image.shape[1],
            image.shape[0],
            len(detections),
            elapsed_ms,
        )
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
    global motor_enabled, servo_angle
    angle = SERVO_DAMAGED_ANGLE if command.enabled else SERVO_HEALTHY_ANGLE
    servo_angle = angle
    motor_enabled = command.enabled
    return {"enabled": motor_enabled, "angle": angle, "transport": "polling"}

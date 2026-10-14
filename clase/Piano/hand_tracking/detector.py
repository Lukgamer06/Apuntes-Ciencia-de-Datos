"""MediaPipe Hands adapter."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from config import VideoConfig
from hand_tracking.models import HandObservation, Landmark

LOGGER = logging.getLogger(__name__)


class HandDetector:
    """Detect up to two hands and return all 21 normalized landmarks per hand."""

    def __init__(self, config: VideoConfig) -> None:
        model_path = Path(__file__).resolve().parents[1] / "assets" / "hand_landmarker.task"
        if not model_path.is_file():
            raise FileNotFoundError(f"MediaPipe hand landmark model not found: {model_path}")
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=config.max_hands,
            min_hand_detection_confidence=config.detection_confidence,
            min_hand_presence_confidence=config.detection_confidence,
            min_tracking_confidence=config.tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def detect(self, frame_bgr: cv2.Mat) -> tuple[HandObservation, ...]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = max(time.monotonic_ns() // 1_000_000, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        observations: list[HandObservation] = []
        for hand_landmarks, handedness in zip(
            result.hand_landmarks, result.handedness, strict=True
        ):
            side = handedness[0].category_name
            points = tuple(
                Landmark(point.x, point.y, point.z) for point in hand_landmarks
            )
            observations.append(HandObservation(side, points))
        return tuple(observations)

    def close(self) -> None:
        self._landmarker.close()
        LOGGER.debug("MediaPipe hand detector closed")

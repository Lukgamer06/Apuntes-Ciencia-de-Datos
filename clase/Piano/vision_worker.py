"""Threaded webcam capture and hand inference, publishing only the latest frame."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import cv2

from config import VideoConfig
from hand_tracking.detector import HandDetector
from hand_tracking.models import HandObservation

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class VisionSnapshot:
    frame: cv2.Mat
    hands: tuple[HandObservation, ...]
    fps: float
    sequence: int


class VisionWorker:
    def __init__(self, config: VideoConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._snapshot: VisionSnapshot | None = None
        self._sequence = 0
        self._error: BaseException | None = None
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, name="vision-camera", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def latest(self) -> VisionSnapshot | None:
        with self._lock:
            snapshot = self._snapshot
            if snapshot is None:
                return None
            return VisionSnapshot(snapshot.frame.copy(), snapshot.hands, snapshot.fps, snapshot.sequence)

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"Vision worker failed: {self._error}") from self._error

    def close(self) -> None:
        self._stopped.set()
        self._thread.join(timeout=3)
        if self._thread.is_alive():
            LOGGER.error("Vision worker did not stop within timeout")

    def _run(self) -> None:
        capture: cv2.VideoCapture | None = None
        detector: HandDetector | None = None
        try:
            capture = cv2.VideoCapture(self._config.camera_index)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
            capture.set(cv2.CAP_PROP_FPS, self._config.target_fps)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not capture.isOpened():
                raise RuntimeError(f"Could not open camera index {self._config.camera_index}.")
            detector = HandDetector(self._config)
            period = 1.0 / self._config.target_fps
            previous = time.perf_counter()
            smoothed_fps = 0.0
            while not self._stopped.is_set():
                started = time.perf_counter()
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Camera stopped returning frames.")
                frame = cv2.flip(frame, 1)
                now = time.perf_counter()
                instantaneous = 1.0 / max(now - previous, 1e-6)
                smoothed_fps = instantaneous if smoothed_fps == 0 else 0.85 * smoothed_fps + 0.15 * instantaneous
                previous = now
                hands = detector.detect(frame)
                with self._lock:
                    self._sequence += 1
                    self._snapshot = VisionSnapshot(frame, hands, smoothed_fps, self._sequence)
                remaining = period - (time.perf_counter() - started)
                if remaining > 0:
                    self._stopped.wait(remaining)
        except Exception as exc:
            self._error = exc
            LOGGER.exception("Vision worker failed")
        finally:
            if detector is not None:
                detector.close()
            if capture is not None:
                capture.release()
            LOGGER.info("Vision worker stopped")

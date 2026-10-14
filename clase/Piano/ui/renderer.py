"""Camera overlays for hand landmarks, gesture labels and a visual keyboard."""

from __future__ import annotations

import math
import time

import cv2

from config import AppConfig
from hand_tracking.gesture_state_manager import GestureFrameState
from hand_tracking.models import HandObservation
from piano.keyboard import VirtualKeyboard
from piano.note_mapper import FIST_GESTURE, GestureNoteMapper

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


class Renderer:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._keyboard = VirtualKeyboard(config.keyboard)
        self._note_mapper = GestureNoteMapper(
            config.keyboard.start_octave, config.gesture.velocity
        )
        self._started = time.perf_counter()
        self._highlight_until: dict[int, float] = {}

    def trigger_notes(self, notes: tuple[int, ...]) -> None:
        expiry = time.monotonic() + self._config.gesture.visual_hold_ms / 1000
        for note in notes:
            self._highlight_until[note % 12] = expiry

    def draw(
        self,
        frame: cv2.Mat,
        hands: tuple[HandObservation, ...],
        gestures: GestureFrameState,
        active_notes: set[int],
        fps: float,
    ) -> cv2.Mat:
        output = frame.copy()
        now = time.monotonic()
        for pitch_class, expiry in tuple(self._highlight_until.items()):
            if expiry <= now:
                del self._highlight_until[pitch_class]

        phase = math.sin((time.perf_counter() - self._started) * 2.0)
        self._keyboard.draw(
            output,
            {note % 12 for note in active_notes},
            set(self._highlight_until),
            phase,
        )
        height, width = output.shape[:2]
        for hand in hands:
            if len(hand.landmarks) != 21:
                continue
            points = [
                (round(point.x * width), round(point.y * height))
                for point in hand.landmarks
            ]
            for start, end in HAND_CONNECTIONS:
                cv2.line(output, points[start], points[end], (210, 225, 220), 2, cv2.LINE_AA)
            for point in points:
                cv2.circle(output, point, 3, (255, 235, 80), -1, cv2.LINE_AA)

        for finger in gestures.fingers:
            x, y = round(finger.x * width), round(finger.y * height)
            color = (45, 255, 55) if finger.active else (255, 235, 90)
            cv2.circle(output, (x, y), 8 if finger.active else 5, color, -1, cv2.LINE_AA)
            label = self._note_mapper.note_name_for(finger.hand, finger.finger)
            cv2.putText(
                output,
                f"{label} {finger.finger}",
                (x + 8, y - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                color,
                2 if finger.active else 1,
                cv2.LINE_AA,
            )

        self._draw_status(output, gestures, fps, len(active_notes), width, height)
        return output

    def _draw_status(
        self,
        frame: cv2.Mat,
        gestures: GestureFrameState,
        fps: float,
        note_count: int,
        width: int,
        height: int,
    ) -> None:
        panel_top = self._config.keyboard.top + self._config.keyboard.height + 12
        cv2.rectangle(frame, (12, panel_top), (min(width - 12, 270), panel_top + 48),
                      (20, 28, 32), -1)
        for row, side in enumerate(("Left", "Right")):
            closed = side in gestures.closed_fists
            fist_note = self._note_mapper.note_name_for(side, FIST_GESTURE)
            label = "IZQ" if side == "Left" else "DER"
            text = f"PUÑO {label}: {fist_note}  {'CERRADO' if closed else 'ABIERTO'}"
            cv2.putText(
                frame,
                text,
                (19, panel_top + 18 + row * 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (40, 255, 55) if closed else (225, 235, 235),
                2 if closed else 1,
                cv2.LINE_AA,
            )
        cv2.rectangle(frame, (width - 285, height - 40), (width - 12, height - 10),
                      (18, 25, 30), -1)
        cv2.putText(
            frame,
            f"{fps:4.1f} FPS | {note_count} notas | Q: salir",
            (width - 277, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (160, 255, 235),
            1,
            cv2.LINE_AA,
        )

"""Geometry and rendering for a centered, single-octave visual keyboard."""

from __future__ import annotations

from dataclasses import dataclass

import cv2

from config import KeyboardConfig
from piano.note_mapper import NOTE_NAMES

WHITE_PITCHES = (0, 2, 4, 5, 7, 9, 11)
BLACK_KEYS = ((1, 0.5), (3, 1.5), (6, 3.5), (8, 4.5), (10, 5.5))


@dataclass(frozen=True, slots=True)
class KeyRect:
    pitch_class: int
    x1: int
    x2: int
    y1: int
    y2: int
    black: bool


class VirtualKeyboard:
    """Draw twelve pitch-class keys; this visual layer performs no hit testing."""

    def __init__(self, config: KeyboardConfig) -> None:
        self.config = config

    def keys(self, frame_width: int) -> tuple[KeyRect, ...]:
        width = max(12, round(frame_width * self.config.width_ratio))
        left = (frame_width - width) // 2
        white_width = width / 7
        y1 = self.config.top
        y2 = y1 + self.config.height
        output = [
            KeyRect(
                pitch,
                round(left + index * white_width),
                round(left + (index + 1) * white_width),
                y1,
                y2,
                False,
            )
            for index, pitch in enumerate(WHITE_PITCHES)
        ]
        for pitch, center_in_whites in BLACK_KEYS:
            center = left + center_in_whites * white_width
            half_width = white_width * 0.30
            output.append(
                KeyRect(
                    pitch,
                    round(center - half_width),
                    round(center + half_width),
                    y1,
                    y1 + round(self.config.height * 0.62),
                    True,
                )
            )
        return tuple(sorted(output, key=lambda key: (key.black, key.x1)))

    def draw(
        self,
        frame: cv2.Mat,
        active_notes: set[int],
        highlighted_notes: set[int],
        animation_phase: float,
    ) -> None:
        _, width = frame.shape[:2]
        keys = self.keys(width)
        overlay = frame.copy()
        glow = (animation_phase + 1.0) * 0.5
        for key in (item for item in keys if not item.black):
            active = (key.pitch_class in active_notes
                      or key.pitch_class in highlighted_notes)
            fill = (52, 70, 54) if active else (222, 229, 232)
            edge = (35, 255, 45) if active else (245, 248, 250)
            cv2.rectangle(overlay, (key.x1, key.y1), (key.x2, key.y2), fill, -1)
            cv2.rectangle(frame, (key.x1, key.y1), (key.x2, key.y2), edge, 2 if active else 1)
            cv2.putText(
                frame,
                NOTE_NAMES[key.pitch_class],
                (key.x1 + 4, key.y2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (20, 25, 25) if not active else (70, 255, 70),
                1,
                cv2.LINE_AA,
            )
            if active:
                cv2.rectangle(
                    frame,
                    (key.x1 - 2, key.y1 - 2),
                    (key.x2 + 2, key.y2 + 2),
                    (int(30 + 45 * glow), 255, 40),
                    2,
                )
        for key in (item for item in keys if item.black):
            active = (key.pitch_class in active_notes
                      or key.pitch_class in highlighted_notes)
            fill = (25, 75, 28) if active else (30, 35, 38)
            edge = (25, 255, 35) if active else (220, 230, 235)
            cv2.rectangle(overlay, (key.x1, key.y1), (key.x2, key.y2), fill, -1)
            cv2.rectangle(frame, (key.x1, key.y1), (key.x2, key.y2), edge, 2 if active else 1)
            cv2.putText(
                frame,
                NOTE_NAMES[key.pitch_class],
                (key.x1 + 2, key.y2 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (80, 255, 90) if active else (240, 245, 245),
                1,
                cv2.LINE_AA,
            )
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

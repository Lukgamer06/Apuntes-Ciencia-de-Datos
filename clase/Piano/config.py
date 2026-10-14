"""Validated application configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class VideoConfig:
    camera_index: int = 0
    width: int = 1280
    height: int = 720
    target_fps: int = 30
    max_hands: int = 2
    detection_confidence: float = 0.65
    tracking_confidence: float = 0.6


@dataclass(frozen=True, slots=True)
class KeyboardConfig:
    start_octave: int = 3
    top: int = 28
    height: int = 150
    width_ratio: float = 0.78


@dataclass(frozen=True, slots=True)
class GestureConfig:
    flexion_angle_degrees: float = 145.0
    extension_angle_degrees: float = 160.0
    debounce_ms: int = 100
    stable_frames: int = 2
    velocity: int = 100
    visual_hold_ms: int = 220


@dataclass(frozen=True, slots=True)
class FilterConfig:
    landmark_window_size: int = 3


@dataclass(frozen=True, slots=True)
class AudioConfig:
    soundfont: str | None = None
    sample_rate: int = 44100
    audio_driver: str | None = None
    bank: int = 0
    preset: int = 0


@dataclass(frozen=True, slots=True)
class AppConfig:
    video: VideoConfig = VideoConfig()
    keyboard: KeyboardConfig = KeyboardConfig()
    gesture: GestureConfig = GestureConfig()
    filters: FilterConfig = FilterConfig()
    audio: AudioConfig = AudioConfig()
    window_name: str = "Air Piano"
    log_level: str = "INFO"


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section '{name}' must be a mapping.")
    return value


def load_config(path: str | Path) -> AppConfig:
    """Load YAML settings and reject invalid values before opening devices."""
    config_path = Path(path).expanduser().resolve()
    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            raw = yaml.safe_load(config_file) or {}
    except OSError as exc:
        raise ValueError(f"Cannot read configuration file '{config_path}': {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("The configuration root must be a mapping.")

    video = VideoConfig(**_section(raw, "video"))
    keyboard = KeyboardConfig(**_section(raw, "keyboard"))
    gesture = GestureConfig(**_section(raw, "gesture"))
    filters = FilterConfig(**_section(raw, "filters"))
    audio = AudioConfig(**_section(raw, "audio"))
    if audio.soundfont:
        soundfont_path = Path(audio.soundfont).expanduser()
        if not soundfont_path.is_absolute():
            soundfont_path = config_path.parent / soundfont_path
        audio = AudioConfig(
            soundfont=str(soundfont_path.resolve()),
            sample_rate=audio.sample_rate,
            audio_driver=audio.audio_driver,
            bank=audio.bank,
            preset=audio.preset,
        )

    if video.width < 320 or video.height < 240 or video.target_fps < 1:
        raise ValueError("Video width/height and target_fps must be positive and usable.")
    if not 1 <= video.max_hands <= 2:
        raise ValueError("video.max_hands must be 1 or 2.")
    for name, value in (
        ("video.detection_confidence", video.detection_confidence),
        ("video.tracking_confidence", video.tracking_confidence),
    ):
        if not 0 < value <= 1:
            raise ValueError(f"{name} must be in the interval (0, 1].")
    if keyboard.start_octave < 0 or keyboard.start_octave > 8:
        raise ValueError("keyboard.start_octave must be between 0 and 8.")
    if keyboard.height < 80 or keyboard.top < 0:
        raise ValueError("Keyboard top or height is invalid.")
    if keyboard.top + keyboard.height > video.height:
        raise ValueError("Keyboard geometry must fit within the configured video dimensions.")
    if not 0.2 <= keyboard.width_ratio <= 1:
        raise ValueError("keyboard.width_ratio must be in the interval [0.2, 1].")
    if not 0 < gesture.flexion_angle_degrees < gesture.extension_angle_degrees < 180:
        raise ValueError("Flexion and extension angles must be ordered within (0, 180).")
    if gesture.debounce_ms < 100:
        raise ValueError("gesture.debounce_ms must be at least 100.")
    if gesture.stable_frames < 1 or gesture.visual_hold_ms < 0:
        raise ValueError("Gesture stable_frames must be positive and visual_hold_ms non-negative.")
    if not 1 <= gesture.velocity <= 127:
        raise ValueError("Gesture velocity must be 1..127.")
    if filters.landmark_window_size < 1:
        raise ValueError("filters.landmark_window_size must be positive.")
    if audio.sample_rate < 8000 or audio.bank < 0 or audio.preset < 0:
        raise ValueError("Audio sample rate/bank/preset is invalid.")
    if raw.get("log_level", "INFO").upper() not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ValueError("log_level must be a standard Python logging level.")

    return AppConfig(
        video=video,
        keyboard=keyboard,
        gesture=gesture,
        filters=filters,
        audio=audio,
        window_name=str(raw.get("window_name", "Air Piano")),
        log_level=str(raw.get("log_level", "INFO")).upper(),
    )

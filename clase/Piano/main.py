"""Run the real-time Air Piano application."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2

from audio.midi_engine import MidiEngine
from config import load_config
from hand_tracking.gesture_state_manager import GestureFrameState, GestureStateManager
from piano.note_mapper import GestureNoteMapper
from ui.renderer import Renderer
from vision_worker import VisionWorker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a virtual piano with webcam hand gestures.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    return parser.parse_args()


def run(config_path: Path) -> None:
    config = load_config(config_path)
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)
    gesture_manager = GestureStateManager(config.gesture, config.filters)
    note_mapper = GestureNoteMapper(config.keyboard.start_octave, config.gesture.velocity)
    audio = MidiEngine(config.audio)
    vision = VisionWorker(config.video)
    renderer = Renderer(config)
    active_notes: set[int] = set()
    gesture_state = GestureFrameState((), (), frozenset(), frozenset())
    last_sequence = 0
    audio_started = False
    vision_started = False

    try:
        audio.start()
        audio_started = True
        vision.start()
        vision_started = True
        cv2.namedWindow(config.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(config.window_name, config.video.width, config.video.height)
        logger.info("Air Piano ready. Press Q or Esc to quit.")
        while True:
            vision.raise_if_failed()
            snapshot = vision.latest()
            if snapshot is not None:
                if snapshot.sequence != last_sequence:
                    gesture_state = gesture_manager.update(snapshot.hands)
                    events = note_mapper.map_events(gesture_state.events)
                    audio.send(events)
                    renderer.trigger_notes(
                        tuple(event.note for event in events if event.is_on)
                    )
                    active_notes = note_mapper.active_notes(
                        gesture_state.active_gestures
                    )
                    last_sequence = snapshot.sequence
                image = renderer.draw(
                    snapshot.frame,
                    snapshot.hands,
                    gesture_state,
                    active_notes,
                    snapshot.fps,
                )
                cv2.imshow(config.window_name, image)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        try:
            if audio_started:
                try:
                    audio.send(note_mapper.map_events(gesture_manager.release_all()))
                except RuntimeError:
                    logger.exception("Unable to send final MIDI note-offs")
        finally:
            if vision_started:
                vision.close()
            if audio_started:
                audio.close()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        run(parse_args().config)
    except Exception:
        logging.basicConfig(level=logging.ERROR, format="%(asctime)s %(levelname)s %(message)s")
        logging.exception("Air Piano could not start")
        raise SystemExit(1)

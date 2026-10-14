from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import cv2
import numpy as np

from audio.midi_engine import MidiEngine
from config import AppConfig, AudioConfig, FilterConfig, GestureConfig, KeyboardConfig
from filters.smoothing import ExponentialMovingAverage, MovingAverage
from hand_tracking.gesture_state_manager import (
    FIST_GESTURE,
    FingerGestureState,
    GestureFrameState,
    GestureStateManager,
    GestureTransition,
)
from hand_tracking.models import HandObservation, Landmark
from piano.keyboard import VirtualKeyboard
from piano.note_mapper import GestureNoteMapper, NoteEvent, NOTE_NAMES
from ui.renderer import Renderer


class NoteMapperTests(unittest.TestCase):
    def test_keyboard_shows_one_centered_chromatic_octave(self) -> None:
        config = KeyboardConfig()
        keys = VirtualKeyboard(config).keys(1280)
        self.assertEqual(len(keys), 12)
        self.assertEqual({key.pitch_class for key in keys}, set(range(12)))
        self.assertEqual(min(key.x1 for key in keys), round(1280 * (1 - config.width_ratio) / 2))
        self.assertEqual(max(key.x2 for key in keys), round(1280 * (1 + config.width_ratio) / 2))

    def test_gestures_map_to_exactly_one_fixed_octave(self) -> None:
        mapper = GestureNoteMapper(octave=3)
        gestures = (
            *(("Left", finger) for finger in ("thumb", "index", "middle", "ring", "pinky")),
            *(("Right", finger) for finger in ("thumb", "index", "middle", "ring", "pinky")),
            ("Left", FIST_GESTURE),
            ("Right", FIST_GESTURE),
        )
        transitions = tuple(GestureTransition(hand, gesture, True) for hand, gesture in gestures)
        events = mapper.map_events(transitions)
        self.assertEqual([event.note for event in events], list(range(48, 60)))
        self.assertEqual(
            [mapper.note_name_for(hand, gesture) for hand, gesture in gestures],
            list(NOTE_NAMES),
        )


class FilterTests(unittest.TestCase):
    def test_ema_suppresses_single_sample_jitter(self) -> None:
        filter_ = ExponentialMovingAverage(0.25)
        self.assertAlmostEqual(filter_.update(0.0), 0.0)
        self.assertAlmostEqual(filter_.update(1.0), 0.25)

    def test_moving_average_uses_recent_window(self) -> None:
        filter_ = MovingAverage(2)
        filter_.update(1.0)
        self.assertEqual(filter_.update(3.0), 2.0)
        self.assertEqual(filter_.update(5.0), 4.0)


class RendererTests(unittest.TestCase):
    def test_renderer_draws_skeleton_fingertip_labels_and_keyboard(self) -> None:
        renderer = Renderer(AppConfig())
        frame = cast(cv2.Mat, np.zeros((720, 1280, 3), dtype=np.uint8))
        hand = self.make_hand("Right", set())
        gesture_state = GestureFrameState(
            (),
            (FingerGestureState("Right", "index", 0.6, 0.2, False, False),),
            frozenset(),
            frozenset(),
        )
        output = renderer.draw(frame, (hand,), gesture_state, set(), 30.0)
        self.assertGreater(int(output[576, 640].sum()), 0)
        self.assertEqual(len(renderer._keyboard.keys(1280)), 12)

    @staticmethod
    def make_hand(handedness: str, flexed: set[str]) -> HandObservation:
        return GestureStateTests.hand(handedness, flexed)


class GestureStateTests(unittest.TestCase):
    @staticmethod
    def hand(handedness: str, flexed: set[str]) -> HandObservation:
        points = [Landmark(0.5, 0.5, 0.0) for _ in range(21)]
        points[0] = Landmark(0.5, 0.8, 0.0)
        points[1] = Landmark(0.5, 0.7, 0.0)
        thumb_mcp = (0.42, 0.64)
        thumb_ip = (0.38, 0.58)
        thumb_tip = (0.46, 0.64) if "thumb" in flexed else (0.34, 0.52)
        for index, (x, y) in zip((2, 3, 4), (thumb_mcp, thumb_ip, thumb_tip), strict=True):
            points[index] = Landmark(x, y, 0.0)
        for finger, (mcp, pip, tip) in zip(
            ("index", "middle", "ring", "pinky"),
            ((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)),
            strict=True,
        ):
            x = 0.35 + (mcp - 5) * 0.025
            points[mcp] = Landmark(x, 0.6, 0.0)
            points[pip] = Landmark(x, 0.4, 0.0)
            points[tip] = Landmark(x + 0.08, 0.45, 0.0) if finger in flexed else Landmark(x, 0.2, 0.0)
        return HandObservation(handedness, tuple(points))

    @staticmethod
    def manager(
        clock: list[float] | None = None,
        *,
        stable_frames: int = 1,
        smoothing_window: int = 1,
        debounce_ms: int = 100,
    ) -> GestureStateManager:
        return GestureStateManager(
            GestureConfig(stable_frames=stable_frames, debounce_ms=debounce_ms),
            FilterConfig(landmark_window_size=smoothing_window),
            clock=(lambda: clock[0]) if clock is not None else (lambda: 0.0),
        )

    def test_each_finger_has_a_fixed_note_and_requires_release_to_rearm(self) -> None:
        mapper = GestureNoteMapper(3)
        fingers = ("thumb", "index", "middle", "ring", "pinky")
        for side, first_note in (("Left", 48), ("Right", 53)):
            for offset, finger in enumerate(fingers):
                manager = self.manager()
                manager.update((self.hand(side, set()),))
                bent = manager.update((self.hand(side, {finger}),))
                events = mapper.map_events(bent.events)
                self.assertEqual(events, (NoteEvent(first_note + offset, True, 100),))
                self.assertEqual(manager.update((self.hand(side, {finger}),)).events, ())
                extended = manager.update((self.hand(side, set()),))
                self.assertEqual(
                    mapper.map_events(extended.events),
                    (NoteEvent(first_note + offset, False, 0),),
                )

    def test_fist_activates_after_four_flexed_fingers_and_releases_on_open(self) -> None:
        mapper = GestureNoteMapper(3)
        four_fingers = {"thumb", "index", "middle", "ring"}
        for side, pitch in (("Left", 58), ("Right", 59)):
            manager = self.manager()
            manager.update((self.hand(side, set()),))
            closed = manager.update((self.hand(side, four_fingers),))
            self.assertEqual(
                mapper.map_events(closed.events), (NoteEvent(pitch, True, 100),)
            )
            self.assertEqual(closed.closed_fists, {side})
            self.assertEqual(manager.update((self.hand(side, four_fingers),)).events, ())
            opened = manager.update((self.hand(side, set()),))
            self.assertEqual(
                mapper.map_events(opened.events), (NoteEvent(pitch, False, 0),)
            )

    def test_hand_first_seen_as_fist_must_open_before_it_can_trigger(self) -> None:
        mapper = GestureNoteMapper(3)
        manager = self.manager()
        closed_hand = self.hand("Left", {"thumb", "index", "middle", "ring"})
        self.assertEqual(manager.update((closed_hand,)).events, ())
        self.assertEqual(manager.update((self.hand("Left", set()),)).events, ())
        closed = manager.update((closed_hand,))
        self.assertEqual(mapper.map_events(closed.events), (NoteEvent(58, True, 100),))

    def test_three_flexed_fingers_do_not_make_a_fist(self) -> None:
        manager = self.manager()
        manager.update((self.hand("Left", set()),))
        update = manager.update((self.hand("Left", {"thumb", "index", "middle"}),))
        self.assertFalse(update.closed_fists)
        self.assertEqual(
            {(event.hand, event.gesture) for event in update.events},
            {("Left", "thumb"), ("Left", "index"), ("Left", "middle")},
        )

    def test_stability_filter_rejects_a_one_frame_flexion(self) -> None:
        manager = self.manager(stable_frames=2)
        manager.update((self.hand("Right", set()),))
        self.assertEqual(
            manager.update((self.hand("Right", {"index"}),)).events, ()
        )
        self.assertEqual(manager.update((self.hand("Right", set()),)).events, ())

    def test_debounce_requires_one_hundred_milliseconds_between_retriggers(self) -> None:
        clock = [1.0]
        manager = self.manager(clock, debounce_ms=100)
        index_extended = self.hand("Right", set())
        index_flexed = self.hand("Right", {"index"})
        manager.update((index_extended,))
        self.assertTrue(manager.update((index_flexed,)).events[0].is_on)
        manager.update((index_extended,))
        clock[0] = 1.05
        self.assertEqual(manager.update((index_flexed,)).events, ())
        manager.update((index_extended,))
        clock[0] = 1.10
        self.assertTrue(manager.update((index_flexed,)).events[0].is_on)

    def test_active_notes_are_released_when_a_hand_disappears(self) -> None:
        mapper = GestureNoteMapper(3)
        manager = self.manager()
        manager.update((self.hand("Left", set()),))
        pressed = manager.update((self.hand("Left", {"index"}),))
        self.assertEqual(mapper.map_events(pressed.events), (NoteEvent(49, True, 100),))
        missing = manager.update(())
        self.assertEqual(mapper.map_events(missing.events), (NoteEvent(49, False, 0),))
        self.assertFalse(missing.active_gestures)

    def test_gesture_detection_is_independent_of_hand_position_and_scale(self) -> None:
        mapper = GestureNoteMapper(3)
        outputs = []
        for dx, dy, scale in ((0.0, 0.0, 1.0), (0.2, -0.1, 1.5)):
            manager = self.manager()
            open_hand = self.hand("Left", set())
            bent_hand = self.hand("Left", {"index"})
            def transform(hand: HandObservation) -> HandObservation:
                points = tuple(
                    Landmark(dx + point.x * scale, dy + point.y * scale, point.z * scale)
                    for point in hand.landmarks
                )
                return HandObservation(hand.handedness, points)
            manager.update((transform(open_hand),))
            outputs.append(mapper.map_events(manager.update((transform(bent_hand),)).events))
        self.assertEqual(outputs[0], outputs[1])


class MidiEngineTests(unittest.TestCase):
    def test_duplicate_pitches_are_reference_counted_on_audio_thread(self) -> None:
        calls: list[tuple[str, int]] = []

        class FakeSynth:
            def __init__(self, samplerate: int) -> None:
                self.samplerate = samplerate

            def sfload(self, filename: str, reset_presets: bool = True) -> int:
                return 1

            def program_select(self, channel: int, soundfont_id: int, bank: int, preset: int) -> None:
                return None

            def start(self, driver: str | None = None) -> None:
                return None

            def noteon(self, channel: int, key: int, velocity: int) -> None:
                calls.append(("on", key))

            def noteoff(self, channel: int, key: int) -> None:
                calls.append(("off", key))

            def delete(self) -> None:
                return None

        with tempfile.NamedTemporaryFile(suffix=".sf2") as soundfont:
            fake_module = SimpleNamespace(Synth=FakeSynth)
            with patch.dict("sys.modules", {"fluidsynth": fake_module}):
                engine = MidiEngine(AudioConfig(soundfont=soundfont.name))
                engine.start()
                engine.send(
                    (
                        NoteEvent(60, True, 90),
                        NoteEvent(60, True, 100),
                        NoteEvent(60, False),
                        NoteEvent(60, False),
                    )
                )
                engine.close()
        self.assertEqual(calls, [("on", 60), ("off", 60)])

    def test_missing_midi_output_falls_back_to_polyphonic_synth(self) -> None:
        calls: list[tuple[str, int, int]] = []

        class FakeMidi:
            @staticmethod
            def init() -> None:
                return None

            @staticmethod
            def get_default_output_id() -> int:
                return -1

            @staticmethod
            def quit() -> None:
                return None

        class FakePygameSynth:
            def __init__(self, sample_rate: int) -> None:
                self.sample_rate = sample_rate

            def note_on(self, note: int, velocity: int) -> None:
                calls.append(("on", note, velocity))

            def note_off(self, note: int) -> None:
                calls.append(("off", note, 0))

            def close(self) -> None:
                return None

        fake_pygame = SimpleNamespace(midi=FakeMidi)
        with (
            patch.dict("sys.modules", {"pygame": fake_pygame, "pygame.midi": FakeMidi}),
            patch("audio.midi_engine._PygameSynth", FakePygameSynth),
        ):
            engine = MidiEngine(AudioConfig())
            engine.start()
            engine.send(
                (
                    NoteEvent(60, True, 90),
                    NoteEvent(60, True, 100),
                    NoteEvent(64, True, 100),
                    NoteEvent(60, False),
                    NoteEvent(60, False),
                    NoteEvent(64, False),
                )
            )
            engine.close()

        self.assertEqual(
            calls,
            [("on", 60, 90), ("on", 64, 100), ("off", 60, 0), ("off", 64, 0)],
        )


if __name__ == "__main__":
    unittest.main()

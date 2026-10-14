"""Fixed gesture-to-pitch mapping for one chromatic octave."""

from __future__ import annotations

from dataclasses import dataclass

from hand_tracking.gesture_state_manager import FIST_GESTURE, GestureTransition

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
PITCH_BY_GESTURE: dict[tuple[str, str], int] = {
    **{
        ("Left", finger): pitch
        for finger, pitch in zip(
            ("thumb", "index", "middle", "ring", "pinky"), range(5), strict=True
        )
    },
    **{
        ("Right", finger): pitch
        for finger, pitch in zip(
            ("thumb", "index", "middle", "ring", "pinky"), range(5, 10), strict=True
        )
    },
    ("Left", FIST_GESTURE): 10,
    ("Right", FIST_GESTURE): 11,
}


@dataclass(frozen=True, slots=True)
class NoteEvent:
    note: int
    is_on: bool
    velocity: int = 0


class GestureNoteMapper:
    """Map named gesture edges to MIDI notes without using image coordinates."""

    def __init__(self, octave: int, velocity: int = 100) -> None:
        if not 0 <= octave <= 8:
            raise ValueError("octave must be between 0 and 8.")
        if not 1 <= velocity <= 127:
            raise ValueError("velocity must be between 1 and 127.")
        self._octave = octave
        self._velocity = velocity

    @staticmethod
    def pitch_class_for(hand: str, gesture: str) -> int:
        try:
            return PITCH_BY_GESTURE[(hand, gesture)]
        except KeyError as exc:
            raise ValueError(f"Unsupported gesture: {hand}:{gesture}") from exc

    @classmethod
    def note_name_for(cls, hand: str, gesture: str) -> str:
        return NOTE_NAMES[cls.pitch_class_for(hand, gesture)]

    def note_for(self, hand: str, gesture: str) -> int:
        pitch_class = self.pitch_class_for(hand, gesture)
        return (self._octave + 1) * 12 + pitch_class

    def map_events(
        self, transitions: tuple[GestureTransition, ...]
    ) -> tuple[NoteEvent, ...]:
        return tuple(
            NoteEvent(
                self.note_for(event.hand, event.gesture),
                event.is_on,
                self._velocity if event.is_on else 0,
            )
            for event in transitions
        )

    def active_notes(
        self, gestures: frozenset[tuple[str, str]]
    ) -> set[int]:
        return {self.note_for(hand, gesture) for hand, gesture in gestures}

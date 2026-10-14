"""Smoothed finger-flexion and fist transitions from MediaPipe landmarks."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from config import FilterConfig, GestureConfig
from hand_tracking.models import HandObservation, Landmark

FINGER_JOINTS: dict[str, tuple[int, int, int]] = {
    "thumb": (2, 3, 4),
    "index": (5, 6, 8),
    "middle": (9, 10, 12),
    "ring": (13, 14, 16),
    "pinky": (17, 18, 20),
}
FINGER_TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
FIST_GESTURE = "fist"


@dataclass(frozen=True, slots=True)
class GestureTransition:
    hand: str
    gesture: str
    is_on: bool


@dataclass(frozen=True, slots=True)
class FingerGestureState:
    hand: str
    finger: str
    x: float
    y: float
    flexed: bool
    active: bool


@dataclass(frozen=True, slots=True)
class GestureFrameState:
    events: tuple[GestureTransition, ...]
    fingers: tuple[FingerGestureState, ...]
    closed_fists: frozenset[str]
    active_gestures: frozenset[tuple[str, str]]


@dataclass(slots=True)
class _HandMemory:
    history: deque[tuple[Landmark, ...]]
    flexed: dict[str, bool | None] = field(
        default_factory=lambda: dict.fromkeys(FINGER_JOINTS)
    )
    candidates: dict[str, tuple[bool, int]] = field(default_factory=dict)
    active_fingers: set[str] = field(default_factory=set)
    fist_closed: bool | None = None
    active_fist: bool = False
    fist_candidate: tuple[bool, int] | None = None


class GestureStateManager:
    """Turn robust finger/fist state changes into gesture transitions."""

    def __init__(
        self,
        gesture: GestureConfig,
        filters: FilterConfig,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._gesture = gesture
        self._filters = filters
        self._clock = clock
        self._hands: dict[str, _HandMemory] = {}
        self._last_activation: dict[tuple[str, str], float] = {}

    def update(self, hands: tuple[HandObservation, ...]) -> GestureFrameState:
        events: list[GestureTransition] = []
        finger_states: list[FingerGestureState] = []
        seen_hands: set[str] = set()
        closed_fists: set[str] = set()
        now = self._clock()

        for hand in hands:
            if hand.handedness not in {"Left", "Right"} or len(hand.landmarks) != 21:
                continue
            side = hand.handedness
            seen_hands.add(side)
            memory = self._hands.get(side)
            is_new_hand = memory is None
            if memory is None:
                memory = _HandMemory(deque(maxlen=self._filters.landmark_window_size))
                self._hands[side] = memory
            memory.history.append(hand.landmarks)
            landmarks = self._smoothed_landmarks(memory.history)
            angles = {
                finger: self._joint_angle(landmarks, joints)
                for finger, joints in FINGER_JOINTS.items()
            }
            changed: list[tuple[str, bool]] = []

            for finger, angle in angles.items():
                previous = memory.flexed[finger]
                if previous is None:
                    memory.flexed[finger] = angle <= self._gesture.flexion_angle_degrees
                    continue
                if is_new_hand:
                    continue
                target = previous
                if not previous and angle <= self._gesture.flexion_angle_degrees:
                    target = True
                elif previous and angle >= self._gesture.extension_angle_degrees:
                    target = False
                if target == previous:
                    memory.candidates.pop(finger, None)
                    continue
                candidate, count = memory.candidates.get(finger, (target, 0))
                count = count + 1 if candidate == target else 1
                if count >= self._gesture.stable_frames:
                    memory.flexed[finger] = target
                    memory.candidates.pop(finger, None)
                    changed.append((finger, target))
                else:
                    memory.candidates[finger] = (target, count)

            flexed_count = sum(value is True for value in memory.flexed.values())
            observed_fist = flexed_count >= 4
            if memory.fist_closed is None:
                memory.fist_closed = observed_fist
            elif is_new_hand:
                memory.fist_closed = observed_fist
            else:
                self._update_fist_state(
                    side, observed_fist, memory, events, now
                )

            if memory.fist_closed:
                closed_fists.add(side)
                for finger in tuple(memory.active_fingers):
                    self._deactivate(side, finger, memory, events)
            else:
                for finger, is_flexed in changed:
                    if is_flexed:
                        self._activate(side, finger, memory, events, now)
                    else:
                        self._deactivate(side, finger, memory, events)

            for finger, tip_index in FINGER_TIPS.items():
                tip = landmarks[tip_index]
                finger_states.append(
                    FingerGestureState(
                        side,
                        finger,
                        tip.x,
                        tip.y,
                        memory.flexed[finger] is True,
                        finger in memory.active_fingers,
                    )
                )

        for side in tuple(self._hands):
            if side not in seen_hands:
                self._release_hand(side, events, self._hands[side])
                del self._hands[side]

        active = frozenset(
            (side, finger)
            for side, memory in self._hands.items()
            for finger in memory.active_fingers
        )
        active |= frozenset(
            (side, FIST_GESTURE)
            for side, memory in self._hands.items()
            if memory.active_fist
        )
        return GestureFrameState(
            tuple(events), tuple(finger_states), frozenset(closed_fists), active
        )

    def _update_fist_state(
        self,
        side: str,
        observed: bool,
        memory: _HandMemory,
        events: list[GestureTransition],
        now: float,
    ) -> None:
        if observed == memory.fist_closed:
            memory.fist_candidate = None
            return
        candidate, count = memory.fist_candidate or (observed, 0)
        count = count + 1 if candidate == observed else 1
        if count < self._gesture.stable_frames:
            memory.fist_candidate = (observed, count)
            return
        memory.fist_closed = observed
        memory.fist_candidate = None
        if observed:
            for finger in tuple(memory.active_fingers):
                self._deactivate(side, finger, memory, events)
            self._activate(side, FIST_GESTURE, memory, events, now)
        elif memory.active_fist:
            events.append(GestureTransition(side, FIST_GESTURE, False))
            memory.active_fist = False

    def _activate(
        self,
        side: str,
        gesture: str,
        memory: _HandMemory,
        events: list[GestureTransition],
        now: float,
    ) -> None:
        identity = (side, gesture)
        if now - self._last_activation.get(identity, float("-inf")) < self._gesture.debounce_ms / 1000:
            return
        self._last_activation[identity] = now
        if gesture == FIST_GESTURE:
            memory.active_fist = True
        else:
            memory.active_fingers.add(gesture)
        events.append(GestureTransition(side, gesture, True))

    @staticmethod
    def _deactivate(
        side: str,
        gesture: str,
        memory: _HandMemory,
        events: list[GestureTransition],
    ) -> None:
        if gesture in memory.active_fingers:
            memory.active_fingers.remove(gesture)
            events.append(GestureTransition(side, gesture, False))

    @staticmethod
    def _release_hand(
        side: str, events: list[GestureTransition], memory: _HandMemory
    ) -> None:
        for finger in tuple(memory.active_fingers):
            memory.active_fingers.remove(finger)
            events.append(GestureTransition(side, finger, False))
        if memory.active_fist:
            memory.active_fist = False
            events.append(GestureTransition(side, FIST_GESTURE, False))

    def release_all(self) -> tuple[GestureTransition, ...]:
        events: list[GestureTransition] = []
        for side, memory in self._hands.items():
            self._release_hand(side, events, memory)
        self._hands.clear()
        return tuple(events)

    @staticmethod
    def _smoothed_landmarks(
        history: deque[tuple[Landmark, ...]]
    ) -> tuple[Landmark, ...]:
        count = len(history)
        return tuple(
            Landmark(
                sum(frame[index].x for frame in history) / count,
                sum(frame[index].y for frame in history) / count,
                sum(frame[index].z for frame in history) / count,
            )
            for index in range(21)
        )

    @staticmethod
    def _joint_angle(
        landmarks: tuple[Landmark, ...], joints: tuple[int, int, int]
    ) -> float:
        proximal, joint, distal = (landmarks[index] for index in joints)
        first = (
            proximal.x - joint.x,
            proximal.y - joint.y,
            proximal.z - joint.z,
        )
        second = (
            distal.x - joint.x,
            distal.y - joint.y,
            distal.z - joint.z,
        )
        first_length = math.sqrt(sum(value * value for value in first))
        second_length = math.sqrt(sum(value * value for value in second))
        if first_length < 1e-8 or second_length < 1e-8:
            return 180.0
        cosine = sum(a * b for a, b in zip(first, second, strict=True))
        cosine /= first_length * second_length
        return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

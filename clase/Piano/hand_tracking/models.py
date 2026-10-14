"""Immutable hand observations shared by the vision and gesture layers."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Landmark:
    x: float
    y: float
    z: float


@dataclass(frozen=True, slots=True)
class HandObservation:
    handedness: str
    landmarks: tuple[Landmark, ...]

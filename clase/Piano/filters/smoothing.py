"""Small stateful filters intended for per-landmark real-time signals."""

from __future__ import annotations

from collections import deque
from typing import Protocol


class ScalarFilter(Protocol):
    def update(self, value: float) -> float: ...


class ExponentialMovingAverage:
    def __init__(self, alpha: float) -> None:
        self._alpha = alpha
        self._value: float | None = None

    def update(self, value: float) -> float:
        if self._value is None:
            self._value = value
        else:
            self._value += self._alpha * (value - self._value)
        return self._value


class MovingAverage:
    def __init__(self, window: int) -> None:
        self._values: deque[float] = deque(maxlen=window)

    def update(self, value: float) -> float:
        self._values.append(value)
        return sum(self._values) / len(self._values)

"""Optional scalar Kalman filter for noisy fingertip motion."""

from __future__ import annotations


class ScalarKalmanFilter:
    def __init__(self, process_noise: float, measurement_noise: float) -> None:
        self._process_noise = process_noise
        self._measurement_noise = measurement_noise
        self._estimate: float | None = None
        self._error = 1.0

    def update(self, measurement: float) -> float:
        if self._estimate is None:
            self._estimate = measurement
            return measurement
        predicted_error = self._error + self._process_noise
        gain = predicted_error / (predicted_error + self._measurement_noise)
        self._estimate += gain * (measurement - self._estimate)
        self._error = (1.0 - gain) * predicted_error
        return self._estimate

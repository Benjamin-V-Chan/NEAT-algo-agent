"""Failure and anti-stagnation checks."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from neat_racer.config import TerminationSection
from neat_racer.simulation.car import CarState


@dataclass(slots=True)
class TerminationResult:
    done: bool
    reason: str | None = None


class TerminationManager:
    def __init__(self, cfg: TerminationSection) -> None:
        self.cfg = cfg
        self.progress_window: deque[float] = deque(maxlen=cfg.stagnation_window_steps)
        self.yaw_window: deque[float] = deque(maxlen=cfg.max_spin_window_steps)

    def update(self, car: CarState, progress_scalar: float) -> TerminationResult:
        self.progress_window.append(progress_scalar)
        self.yaw_window.append(abs(car.yaw_rate))

        if car.wall_contacts > self.cfg.max_wall_contacts:
            return TerminationResult(done=True, reason="too_many_wall_contacts")

        if car.speed <= 1e-4:
            car.stall_events += 1

        if self._is_stagnating():
            return TerminationResult(done=True, reason="stagnation")

        if self._is_spinning():
            return TerminationResult(done=True, reason="spin_detected")

        return TerminationResult(done=False)

    def _is_stagnating(self) -> bool:
        if len(self.progress_window) < self.progress_window.maxlen:
            return False
        progress_delta = self.progress_window[-1] - self.progress_window[0]
        return progress_delta < self.cfg.min_progress_delta

    def _is_spinning(self) -> bool:
        if len(self.yaw_window) < self.yaw_window.maxlen:
            return False
        mean_yaw = sum(self.yaw_window) / len(self.yaw_window)
        return mean_yaw > self.cfg.spin_angular_velocity_threshold

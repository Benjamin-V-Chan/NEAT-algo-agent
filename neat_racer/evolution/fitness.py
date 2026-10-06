"""Fitness shaping with curriculum-aware weight blending."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from neat_racer.config import FitnessSection, FitnessWeights
from neat_racer.utils.math_utils import clamp


@dataclass(slots=True)
class ActiveFitnessWeights:
    w_progress: float
    w_checkpoint: float
    w_lap: float
    w_speed: float
    w_time: float
    w_collision: float
    w_stall: float
    w_instability: float
    w_offline: float
    w_completion_bonus: float
    w_best_lap_bonus: float

    @classmethod
    def from_weights(cls, w: FitnessWeights) -> ActiveFitnessWeights:
        return cls(**asdict(w))


class CurriculumScheduler:
    def __init__(self, cfg: FitnessSection) -> None:
        self.cfg = cfg
        self.smoothed_completion_rate = 0.0

    def update(self, completion_rate: float) -> float:
        a = self.cfg.completion_rate_ema_alpha
        self.smoothed_completion_rate = (1 - a) * self.smoothed_completion_rate + a * completion_rate
        return self.alpha

    @property
    def alpha(self) -> float:
        target = max(self.cfg.completion_rate_target, 1e-9)
        return clamp(self.smoothed_completion_rate / target, 0.0, 1.0)

    def current_weights(self) -> ActiveFitnessWeights:
        early = asdict(self.cfg.early)
        late = asdict(self.cfg.late)
        alpha = self.alpha
        blended = {k: (1 - alpha) * early[k] + alpha * late[k] for k in early}
        return ActiveFitnessWeights(**blended)


class FitnessAccumulator:
    def __init__(self, weights: ActiveFitnessWeights) -> None:
        self.weights = weights
        self.prev_checkpoints = 0
        self.prev_laps = 0
        self.prev_stall_events = 0
        self.prev_wall_contacts = 0
        self.prev_steer = 0.0
        self.prev_throttle = 0.0
        # Highest progress scalar reached so far. Progress is only rewarded when the car sets a NEW
        # maximum, so re-covering ground it has already driven earns nothing. This is the key
        # anti-reward-hacking rule: oscillating back and forth (or driving circles) within a sector
        # no longer farms w_progress, because the progress scalar never exceeds its prior peak.
        self.max_progress = 0.0

    def step(self, features: dict) -> float:
        progress_scalar = float(features["progress_scalar"])
        progress_delta = max(0.0, progress_scalar - self.max_progress)
        self.max_progress = max(self.max_progress, progress_scalar)
        checkpoints_passed = int(features["checkpoints_passed"])
        laps_completed = int(features["laps_completed"])
        speed_norm = float(features["speed_norm"])
        time_norm = float(features["time_norm"])
        collision_flag = bool(features["collision_flag"])
        wall_contacts = int(features["wall_contacts"])
        stall_events = int(features["stagnation_events"])
        steer = float(features["steering_cmd"])
        throttle = float(features["throttle_cmd"])
        lap_completed_now = bool(features.get("lap_completed_now", False))

        checkpoint_delta = max(0, checkpoints_passed - self.prev_checkpoints)
        lap_delta = max(0, laps_completed - self.prev_laps)
        stall_delta = max(0, stall_events - self.prev_stall_events)
        wall_delta = max(0, wall_contacts - self.prev_wall_contacts)

        useful_speed = speed_norm * progress_delta
        control_instability = abs(steer - self.prev_steer) + abs(throttle - self.prev_throttle)

        lap_time = features.get("best_lap_time")
        lap_quality = 0.0
        if lap_completed_now and lap_time and lap_time > 0:
            lap_quality = 1.0 / (1.0 + float(lap_time))

        reward = (
            self.weights.w_progress * progress_delta
            + self.weights.w_checkpoint * checkpoint_delta
            + self.weights.w_lap * lap_delta
            + self.weights.w_speed * useful_speed
            - self.weights.w_time * time_norm
            - self.weights.w_collision * (1.0 if collision_flag else 0.0)
            - self.weights.w_stall * stall_delta
            - self.weights.w_instability * control_instability
            - self.weights.w_offline * wall_delta
            + self.weights.w_completion_bonus * (1.0 if lap_completed_now else 0.0)
            + self.weights.w_best_lap_bonus * lap_quality
        )

        self.prev_checkpoints = checkpoints_passed
        self.prev_laps = laps_completed
        self.prev_stall_events = stall_events
        self.prev_wall_contacts = wall_contacts
        self.prev_steer = steer
        self.prev_throttle = throttle

        return float(reward)

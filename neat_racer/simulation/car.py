"""Car state and vehicle integration model."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from neat_racer.config import CarSection
from neat_racer.utils.math_utils import clamp, clamp01


@dataclass(slots=True)
class CarControl:
    steer: float
    throttle: float
    brake: float


@dataclass(slots=True)
class CarState:
    pos: np.ndarray
    heading: float
    speed: float = 0.0
    acceleration: float = 0.0
    steering_cmd: float = 0.0
    yaw_rate: float = 0.0
    alive: bool = True
    crashed: bool = False
    wall_contacts: int = 0
    lap_count: int = 0
    progress_scalar: float = 0.0
    checkpoint_index: int = 0
    total_distance: float = 0.0
    max_speed: float = 0.0
    speed_integral: float = 0.0
    steps: int = 0
    stall_events: int = 0
    reverse_steps: int = 0
    steering_history: list[float] = field(default_factory=list)
    throttle_history: list[float] = field(default_factory=list)
    steering_jerk_accum: float = 0.0
    last_steer: float = 0.0

    def as_dict(self) -> dict:
        return {
            "x": float(self.pos[0]),
            "y": float(self.pos[1]),
            "heading": float(self.heading),
            "speed": float(self.speed),
            "acceleration": float(self.acceleration),
            "steering_cmd": float(self.steering_cmd),
            "yaw_rate": float(self.yaw_rate),
            "alive": bool(self.alive),
            "crashed": bool(self.crashed),
        }


@dataclass(slots=True)
class StepResult:
    prev_pos: np.ndarray
    current_pos: np.ndarray
    control: CarControl
    dt: float


def decode_network_outputs(outputs: list[float]) -> CarControl:
    if len(outputs) < 3:
        raise ValueError("Controller must output 3 values: steer, throttle, brake")
    steer_raw, throttle_raw, brake_raw = outputs[:3]
    steer = clamp(float(steer_raw), -1.0, 1.0)
    throttle = clamp01((float(throttle_raw) + 1.0) * 0.5)
    brake = clamp01((float(brake_raw) + 1.0) * 0.5)
    return CarControl(steer=steer, throttle=throttle, brake=brake)


def integrate_car_step(state: CarState, control: CarControl, car_cfg: CarSection, dt: float) -> StepResult:
    prev_pos = state.pos.copy()

    steer = clamp(control.steer, -1.0, 1.0)
    throttle = clamp01(control.throttle)
    brake = clamp01(control.brake)

    a_long = (
        car_cfg.acceleration_max * throttle
        - car_cfg.braking_max * brake
        - car_cfg.drag_coeff * state.speed
        - car_cfg.rolling_coeff * (1.0 if state.speed > 0 else 0.0)
    )
    next_speed = np.clip(state.speed + a_long * dt, 0.0, car_cfg.max_speed)

    delta = math.radians(car_cfg.max_steer_angle_deg) * steer
    speed_ratio = 0.0 if car_cfg.max_speed <= 1e-9 else next_speed / car_cfg.max_speed
    steer_gain = max(car_cfg.min_steer_gain, 1.0 - car_cfg.steer_speed_decay * speed_ratio)
    yaw_rate = (0.0 if abs(car_cfg.wheelbase) < 1e-9 else (next_speed / car_cfg.wheelbase) * math.tan(delta) * steer_gain)

    heading = state.heading + yaw_rate * dt
    dx = next_speed * math.cos(heading) * dt
    dy = next_speed * math.sin(heading) * dt

    state.pos = state.pos + np.array([dx, dy], dtype=float)
    state.heading = heading
    state.speed = float(next_speed)
    state.acceleration = float(a_long)
    state.steering_cmd = float(steer)
    state.yaw_rate = float(yaw_rate)

    dist = float(np.linalg.norm(state.pos - prev_pos))
    state.total_distance += dist
    state.max_speed = max(state.max_speed, state.speed)
    state.speed_integral += state.speed
    state.steps += 1

    state.steering_history.append(steer)
    state.throttle_history.append(throttle)
    state.steering_jerk_accum += abs(steer - state.last_steer)
    state.last_steer = steer

    return StepResult(prev_pos=prev_pos, current_pos=state.pos.copy(), control=control, dt=dt)

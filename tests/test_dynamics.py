from __future__ import annotations

import numpy as np

from neat_racer.config import CarSection
from neat_racer.simulation.car import CarControl, CarState, integrate_car_step


def test_car_accelerates_with_throttle() -> None:
    cfg = CarSection()
    state = CarState(pos=np.array([0.0, 0.0]), heading=0.0)

    control = CarControl(steer=0.0, throttle=1.0, brake=0.0)
    integrate_car_step(state, control, cfg, dt=0.1)

    assert state.speed > 0.0
    assert state.pos[0] > 0.0


def test_car_brakes_down() -> None:
    cfg = CarSection()
    state = CarState(pos=np.array([0.0, 0.0]), heading=0.0, speed=100.0)

    control = CarControl(steer=0.0, throttle=0.0, brake=1.0)
    integrate_car_step(state, control, cfg, dt=0.1)

    assert state.speed < 100.0


def test_steering_changes_heading() -> None:
    cfg = CarSection()
    state = CarState(pos=np.array([0.0, 0.0]), heading=0.0, speed=80.0)

    control = CarControl(steer=0.8, throttle=0.0, brake=0.0)
    integrate_car_step(state, control, cfg, dt=0.1)

    assert state.heading != 0.0

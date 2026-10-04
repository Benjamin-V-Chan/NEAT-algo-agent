from __future__ import annotations

import numpy as np

from neat_racer.config import CarSection, DynamicsSection
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


def test_default_dynamics_are_inert() -> None:
    cfg = CarSection()
    control = CarControl(steer=0.5, throttle=1.0, brake=0.0)

    base = CarState(pos=np.array([0.0, 0.0]), heading=0.0, speed=60.0)
    integrate_car_step(base, control, cfg, dt=0.1)

    withdyn = CarState(pos=np.array([0.0, 0.0]), heading=0.0, speed=60.0)
    integrate_car_step(withdyn, control, cfg, dt=0.1, dynamics=DynamicsSection())

    assert base.speed == withdyn.speed
    assert base.heading == withdyn.heading


def test_surface_grip_reduces_acceleration() -> None:
    cfg = CarSection()
    control = CarControl(steer=0.0, throttle=1.0, brake=0.0)

    full = CarState(pos=np.array([0.0, 0.0]), heading=0.0)
    integrate_car_step(full, control, cfg, dt=0.1, dynamics=DynamicsSection(surface_grip=1.0))

    slippery = CarState(pos=np.array([0.0, 0.0]), heading=0.0)
    integrate_car_step(slippery, control, cfg, dt=0.1, dynamics=DynamicsSection(surface_grip=0.4))

    assert slippery.speed < full.speed


def test_understeer_limits_yaw_at_high_speed() -> None:
    cfg = CarSection()
    control = CarControl(steer=1.0, throttle=0.0, brake=0.0)

    free = CarState(pos=np.array([0.0, 0.0]), heading=0.0, speed=220.0)
    integrate_car_step(free, control, cfg, dt=0.1, dynamics=DynamicsSection(understeer_enabled=False))

    limited = CarState(pos=np.array([0.0, 0.0]), heading=0.0, speed=220.0)
    integrate_car_step(
        limited, control, cfg, dt=0.1,
        dynamics=DynamicsSection(understeer_enabled=True, max_lateral_accel=300.0),
    )

    # The grip-limited car washes wide: it turns less than the full-grip car.
    assert abs(limited.yaw_rate) < abs(free.yaw_rate)

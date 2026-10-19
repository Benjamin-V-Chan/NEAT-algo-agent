from __future__ import annotations

from neat_racer.config import ExperimentConfig
from neat_racer.simulation.car import CarControl
from neat_racer.simulation.env import run_episode


def _cfg() -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.simulation.max_steps_per_episode = 300
    return cfg


def test_wall_contact_is_fatal_by_default(simple_track) -> None:
    cfg = _cfg()

    def controller(_obs):
        return CarControl(steer=-1.0, throttle=1.0, brake=0.0)

    episode = run_episode(cfg, simple_track, 0, 0, controller, lambda _f: 0.0)

    assert episode.crash_flag
    assert episode.wall_contacts == 1
    assert episode.crash_x is not None and episode.crash_y is not None


def test_wall_contact_not_fatal_when_disabled(simple_track) -> None:
    cfg = _cfg()
    cfg.termination.wall_contact_fatal = False

    def controller(_obs):
        return CarControl(steer=-1.0, throttle=1.0, brake=0.0)

    episode = run_episode(cfg, simple_track, 0, 0, controller, lambda _f: 0.0)

    steps_survived = episode.survival_time / cfg.simulation.dt
    assert steps_survived > 1
    assert episode.wall_contacts >= 1


def test_wall_contact_budget_terminates_with_crash(simple_track) -> None:
    cfg = _cfg()
    cfg.termination.wall_contact_fatal = False
    cfg.termination.max_wall_contacts = 1
    cfg.termination.stagnation_window_steps = 100000  # disable stagnation kill
    cfg.car.collision_speed_retain = 1.0  # keep bouncing off the wall

    def controller(_obs):
        return CarControl(steer=-1.0, throttle=1.0, brake=0.0)

    episode = run_episode(cfg, simple_track, 0, 0, controller, lambda _f: 0.0)

    assert episode.wall_contacts > cfg.termination.max_wall_contacts
    assert episode.crash_flag
    assert episode.crash_x is not None and episode.crash_y is not None


def test_episode_reports_progress_for_forward_driver(simple_track) -> None:
    cfg = _cfg()
    cfg.logging.save_per_step = True

    def controller(_obs):
        return CarControl(steer=0.0, throttle=1.0, brake=0.0)

    episode = run_episode(cfg, simple_track, 0, 0, controller, lambda _f: 0.0)

    assert episode.distance_traveled > 0.0
    assert episode.max_progress >= 0.0
    assert len(episode.per_step) == int(episode.survival_time / cfg.simulation.dt)

from __future__ import annotations

from neat_racer.config import FitnessSection
from neat_racer.evolution.fitness import CurriculumScheduler, FitnessAccumulator


def test_curriculum_alpha_increases_with_completion() -> None:
    cfg = FitnessSection()
    scheduler = CurriculumScheduler(cfg)
    a0 = scheduler.alpha
    scheduler.update(0.5)
    a1 = scheduler.alpha
    assert a1 >= a0


def test_fitness_accumulator_rewards_progress() -> None:
    scheduler = CurriculumScheduler(FitnessSection())
    weights = scheduler.current_weights()
    acc = FitnessAccumulator(weights)

    f0 = acc.step(
        {
            "progress_delta": 0.0,
            "checkpoints_passed": 0,
            "laps_completed": 0,
            "speed_norm": 0.0,
            "time_norm": 0.0,
            "collision_flag": False,
            "stagnation_events": 0,
            "wall_contacts": 0,
            "steering_cmd": 0.0,
            "throttle_cmd": 0.0,
            "brake_cmd": 0.0,
            "lap_completed_now": False,
            "best_lap_time": None,
        }
    )

    f1 = acc.step(
        {
            "progress_delta": 0.05,
            "checkpoints_passed": 1,
            "laps_completed": 0,
            "speed_norm": 0.6,
            "time_norm": 0.01,
            "collision_flag": False,
            "stagnation_events": 0,
            "wall_contacts": 0,
            "steering_cmd": 0.02,
            "throttle_cmd": 0.8,
            "brake_cmd": 0.0,
            "lap_completed_now": False,
            "best_lap_time": None,
        }
    )

    assert f1 > f0

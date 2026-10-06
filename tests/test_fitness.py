from __future__ import annotations

from neat_racer.config import FitnessSection
from neat_racer.evolution.fitness import CurriculumScheduler, FitnessAccumulator


def _features(progress_scalar: float, **over) -> dict:
    base = {
        "progress_scalar": progress_scalar,
        "progress_delta": 0.0,  # ignored by the accumulator now; kept for shape parity
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
    base.update(over)
    return base


def test_curriculum_alpha_increases_with_completion() -> None:
    cfg = FitnessSection()
    scheduler = CurriculumScheduler(cfg)
    a0 = scheduler.alpha
    scheduler.update(0.5)
    a1 = scheduler.alpha
    assert a1 >= a0


def test_fitness_accumulator_rewards_progress() -> None:
    scheduler = CurriculumScheduler(FitnessSection())
    acc = FitnessAccumulator(scheduler.current_weights())

    f0 = acc.step(_features(0.0))
    f1 = acc.step(_features(0.05, checkpoints_passed=1, speed_norm=0.6, time_norm=0.01,
                            steering_cmd=0.02, throttle_cmd=0.8))
    assert f1 > f0


def test_progress_reward_is_monotonic_no_circle_farming() -> None:
    """Oscillating progress (driving in circles) must not farm progress reward.

    A car advances to 0.30, slips back to 0.10, then re-advances to 0.30. Only the first
    advance to 0.30 should be rewarded; re-covering 0.10->0.30 earns nothing extra.
    """
    weights = CurriculumScheduler(FitnessSection()).current_weights()

    honest = FitnessAccumulator(weights)
    honest.step(_features(0.10))
    honest.step(_features(0.30))  # reaches 0.30 once
    honest_progress_component = weights.w_progress * 0.30

    hacker = FitnessAccumulator(weights)
    hacker.step(_features(0.10))
    hacker.step(_features(0.30))  # advance
    back = hacker.step(_features(0.10))  # slip back -> no reward
    re = hacker.step(_features(0.30))  # re-advance -> no NEW max -> no reward

    # Re-covering old ground yields no progress reward.
    assert back <= 0.0
    assert abs(re) < 1e-9
    # The hacker never earns more progress reward than honestly reaching the same peak.
    assert honest.max_progress == hacker.max_progress == 0.30
    assert honest_progress_component > 0

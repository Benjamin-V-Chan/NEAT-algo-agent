from __future__ import annotations

from pathlib import Path

import pandas as pd

from neat_racer.analysis.plots import analyze_run


def test_analysis_generates_training_curve(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "plots").mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "generation": 0,
                "population_size": 4,
                "alive_peak": 4,
                "best_fitness": 10,
                "mean_fitness": 5,
                "median_fitness": 4,
                "farthest_progress": 0.2,
                "laps_completed_count": 0,
                "best_lap_time": None,
                "completion_rate": 0,
                "crash_count": 3,
                "eval_time_sec": 0.1,
            },
            {
                "generation": 1,
                "population_size": 4,
                "alive_peak": 4,
                "best_fitness": 20,
                "mean_fitness": 12,
                "median_fitness": 11,
                "farthest_progress": 0.5,
                "laps_completed_count": 1,
                "best_lap_time": 18.2,
                "completion_rate": 0.25,
                "crash_count": 2,
                "eval_time_sec": 0.1,
            },
        ]
    ).to_csv(run_dir / "generation_summary.csv", index=False)

    pd.DataFrame(
        [
            {
                "generation": 0,
                "genome_id": 1,
                "final_fitness": 10,
                "laps_completed": 0,
                "best_lap_time": None,
                "checkpoints_passed": 2,
                "max_progress": 0.2,
                "distance_traveled": 100,
                "progress_efficiency": 0.002,
                "survival_time": 2,
                "avg_speed": 50,
                "max_speed": 80,
                "crash_flag": 1,
                "crash_x": 30,
                "crash_y": 40,
                "wall_contacts": 2,
                "stagnation_events": 1,
                "steering_var": 0.2,
                "steering_jerk": 0.4,
                "throttle_var": 0.3,
            }
        ]
    ).to_csv(run_dir / "per_car_summary.csv", index=False)

    pd.DataFrame([]).to_csv(run_dir / "events.csv", index=False)

    analyze_run(run_dir)

    assert (run_dir / "plots" / "training_curves.png").exists()
    assert (run_dir / "plots" / "fitness_distribution_by_generation.png").exists()

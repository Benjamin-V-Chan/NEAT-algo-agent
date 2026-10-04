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


def test_analysis_generates_evolution_and_curriculum_plots(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "plots").mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "generation": g,
                "population_size": 8,
                "alive_peak": 8,
                "best_fitness": 10 * (g + 1),
                "mean_fitness": 5 * (g + 1),
                "median_fitness": 4 * (g + 1),
                "farthest_progress": 0.2 * (g + 1),
                "laps_completed_count": g,
                "best_lap_time": None if g == 0 else 20.0 - g,
                "completion_rate": 0.1 * g,
                "crash_count": 5 - g,
                "eval_time_sec": 0.1,
                "num_species": 1 + g,
                "best_nodes": 3 + g,
                "best_connections": 39 + g,
                "mean_nodes": 3.0 + 0.2 * g,
                "mean_connections": 39.0 - 0.3 * g,
            }
            for g in range(4)
        ]
    ).to_csv(run_dir / "generation_summary.csv", index=False)

    pd.DataFrame(
        [
            {
                "generation": 0, "genome_id": 1, "final_fitness": 12, "laps_completed": 0,
                "best_lap_time": None, "checkpoints_passed": 2, "max_progress": 0.3,
                "distance_traveled": 120, "progress_efficiency": 0.0025, "survival_time": 3,
                "avg_speed": 55, "max_speed": 90, "crash_flag": 0, "crash_x": None, "crash_y": None,
                "wall_contacts": 1, "stagnation_events": 0, "steering_var": 0.2,
                "steering_jerk": 0.3, "throttle_var": 0.25,
            },
            {
                "generation": 1, "genome_id": 2, "final_fitness": 30, "laps_completed": 1,
                "best_lap_time": 18.0, "checkpoints_passed": 6, "max_progress": 1.1,
                "distance_traveled": 300, "progress_efficiency": 0.0037, "survival_time": 9,
                "avg_speed": 70, "max_speed": 140, "crash_flag": 0, "crash_x": None, "crash_y": None,
                "wall_contacts": 0, "stagnation_events": 0, "steering_var": 0.1,
                "steering_jerk": 0.2, "throttle_var": 0.15,
            },
        ]
    ).to_csv(run_dir / "per_car_summary.csv", index=False)

    pd.DataFrame([]).to_csv(run_dir / "events.csv", index=False)

    analyze_run(run_dir)

    assert (run_dir / "plots" / "evolution_dynamics.png").exists()
    assert (run_dir / "plots" / "curriculum_progression.png").exists()
    assert (run_dir / "plots" / "correlation_heatmap.png").exists()

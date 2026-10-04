"""Global constants used across NEAT Racer modules."""

from __future__ import annotations

from pathlib import Path

PACKAGE_NAME = "neat_racer"
DEFAULT_EXPERIMENT_CONFIG = Path("configs/default_experiment.yaml")
DEFAULT_TRACK_PATH = Path("tracks/baseline_loop.json")
DEFAULT_OUTPUT_ROOT = Path("outputs")
DEFAULT_SAVED_ARTIFACTS_ROOT = Path("saved_artifacts")

GENERATION_SUMMARY_COLUMNS = [
    "generation",
    "population_size",
    "alive_peak",
    "best_fitness",
    "mean_fitness",
    "median_fitness",
    "farthest_progress",
    "laps_completed_count",
    "best_lap_time",
    "completion_rate",
    "crash_count",
    "eval_time_sec",
    # Evolution dynamics (speciation + network topology growth).
    "num_species",
    "best_nodes",
    "best_connections",
    "mean_nodes",
    "mean_connections",
]

PER_CAR_SUMMARY_COLUMNS = [
    "generation",
    "genome_id",
    "final_fitness",
    "laps_completed",
    "best_lap_time",
    "checkpoints_passed",
    "max_progress",
    "distance_traveled",
    "progress_efficiency",
    "survival_time",
    "avg_speed",
    "max_speed",
    "crash_flag",
    "crash_x",
    "crash_y",
    "wall_contacts",
    "stagnation_events",
    "steering_var",
    "steering_jerk",
    "throttle_var",
]

PER_STEP_COLUMNS = [
    "tick",
    "sim_time",
    "generation",
    "genome_id",
    "car_id",
    "x",
    "y",
    "heading",
    "speed",
    "accel",
    "steering_cmd",
    "throttle_cmd",
    "brake_cmd",
    "sensor_vector",
    "checkpoint_idx",
    "lap_count",
    "progress_scalar",
    "alive",
    "collision_flag",
    "event_code",
    "fitness_delta",
]

EVENT_COLUMNS = [
    "generation",
    "genome_id",
    "tick",
    "sim_time",
    "event_type",
    "event_value",
    "x",
    "y",
    "details",
]

NEAT_OUTPUT_COUNT = 3
REQUIRED_TRACK_KEYS = {
    "track_id",
    "name",
    "scale",
    "outer_boundary",
    "inner_boundary",
    "checkpoints",
    "start_finish",
    "spawn_pose",
    "metadata",
}

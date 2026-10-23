from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ExperimentSection:
    name: str = "baseline_neat_racer"
    description: str = "NEAT autonomous racing experiment"
    seed: int = 7


@dataclass(slots=True)
class RuntimeSection:
    mode: str = "headless"  # headless | live
    deterministic: bool = True
    max_generations: int = 50
    evaluate_laps_required: int = 1
    realtime_speed: float = 1.0
    # Parallel genome evaluation. 1 = serial (default). > 1 evaluates a
    # generation's episodes across that many worker processes (headless only);
    # episodes are deterministic given their inputs, so results match the serial
    # path for the default (noise-free) model.
    num_workers: int = 1
    # Training budget in environment steps (one simulated tick of one car). 0 = no step budget
    # (NEAT runs max_generations). NEAT stops starting new generations once it is reached; PPO
    # trains for the largest whole number of rollouts that fits inside it.
    max_env_steps: int = 0


@dataclass(slots=True)
class SimulationSection:
    dt: float = 0.05
    max_steps_per_episode: int = 2000
    track_file: str = "tracks/baseline_loop.json"
    # Optional multi-track training set. Empty = train on track_file only. NEAT scores each genome
    # by its mean fitness over these tracks; PPO pins one vectorized env to each track.
    train_tracks: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CarSection:
    wheelbase: float = 22.0
    collision_radius: float = 7.5
    max_speed: float = 240.0
    acceleration_max: float = 120.0
    braking_max: float = 180.0
    drag_coeff: float = 0.02
    rolling_coeff: float = 1.0
    max_steer_angle_deg: float = 34.0
    steer_speed_decay: float = 0.55
    min_steer_gain: float = 0.2
    # Fraction of speed retained when the car scrapes a wall (0 = full stop).
    collision_speed_retain: float = 0.35


@dataclass(slots=True)
class SensorSection:
    angles_deg: list[float] = field(
        default_factory=lambda: [-80, -45, -20, -8, 0, 8, 20, 45, 80]
    )
    max_range: float = 180.0
    include_alignment_feature: bool = True


@dataclass(slots=True)
class DynamicsSection:

    # Grip-limited cornering: when the car's lateral acceleration (v * yaw_rate)
    # exceeds the grip budget, the yaw rate is scaled down so the car washes out.
    understeer_enabled: bool = False
    max_lateral_accel: float = 900.0

    # Gaussian noise added to the normalized [0, 1] sensor rays fed to the controller
    # (telemetry/rendering rays stay clean). Reproducible under the run seed.
    sensor_noise_enabled: bool = False
    sensor_noise_std: float = 0.02

    # Global grip multiplier scaling tractive/braking force (and the lateral grip
    # budget). < 1.0 models a slippery surface, > 1.0 extra grip.
    surface_grip: float = 1.0


@dataclass(slots=True)
class TerminationSection:
    stagnation_window_steps: int = 160
    min_progress_delta: float = 0.015
    max_spin_window_steps: int = 120
    spin_angular_velocity_threshold: float = 6.0
    max_wall_contacts: int = 16
    # When True (default), the first wall contact ends the car immediately (a crash). When False,
    # contacts bleed speed + bounce the car back and count against max_wall_contacts.
    wall_contact_fatal: bool = True


@dataclass(slots=True)
class ProgressSection:
    checkpoint_radius: float = 10.0
    require_start_finish_after_checkpoints: bool = True


@dataclass(slots=True)
class FitnessWeights:
    w_progress: float = 900.0
    w_checkpoint: float = 120.0
    w_lap: float = 600.0
    w_speed: float = 0.25
    w_time: float = 0.2
    w_collision: float = 250.0
    w_stall: float = 40.0
    w_instability: float = 6.0
    w_offline: float = 18.0
    w_completion_bonus: float = 450.0
    w_best_lap_bonus: float = 120.0


@dataclass(slots=True)
class FitnessSection:
    early: FitnessWeights = field(default_factory=FitnessWeights)
    late: FitnessWeights = field(
        default_factory=lambda: FitnessWeights(
            w_progress=500.0,
            w_checkpoint=100.0,
            w_lap=780.0,
            w_speed=0.75,
            w_time=0.45,
            w_collision=280.0,
            w_stall=35.0,
            w_instability=9.0,
            w_offline=22.0,
            w_completion_bonus=650.0,
            w_best_lap_bonus=260.0,
        )
    )
    completion_rate_target: float = 0.35
    completion_rate_ema_alpha: float = 0.25


@dataclass(slots=True)
class NeatSection:
    population_size: int = 120
    activation_default: str = "tanh"
    hidden_nodes: int = 0
    survival_threshold: float = 0.2
    reset_on_extinction: bool = True


@dataclass(slots=True)
class PpoSection:
    # stable-baselines3 PPO hyperparameters. SB3 defaults except n_steps / batch_size.
    envs_per_track: int = 1
    n_steps: int = 1024
    batch_size: int = 256
    n_epochs: int = 10
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    log_std_init: float = 0.0
    net_arch: list[int] = field(default_factory=lambda: [64, 64])
    normalize_reward: bool = True
    torch_threads: int = 1


@dataclass(slots=True)
class StudySection:
    # NEAT-vs-PPO comparison protocol (see README). The training budget is runtime.max_env_steps
    # and the training tracks are simulation.train_tracks.
    seeds: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    holdout_tracks: list[str] = field(default_factory=list)
    # Out-of-family held-out tier (narrower, clockwise, hairpins). Scored as its own split.
    hard_holdout_tracks: list[str] = field(default_factory=list)
    eval_points: int = 30
    eval_max_steps: int = 1800
    # Start conditions per track: start 0 is the nominal spawn; the rest shift the car along the
    # start line (up to start_lateral_jitter x the line's half-length) and turn it (up to
    # start_heading_jitter_deg), drawn from start_seed so every policy faces the same starts.
    curve_eval_starts: int = 1
    final_eval_starts: int = 1
    start_lateral_jitter: float = 0.4
    start_heading_jitter_deg: float = 10.0
    start_seed: int = 0
    # Pre-registered primary comparisons: final_by_seed.csv columns compared NEAT vs PPO with an
    # exact Mann-Whitney test, Holm-corrected across this list.
    primary_metrics: list[str] = field(default_factory=list)
    max_parallel_runs: int = 10
    output_root: str = "outputs/study"


@dataclass(slots=True)
class LoggingSection:
    output_root: str = "outputs"
    save_per_step: bool = False
    per_step_format: str = "parquet"
    per_step_compression: str = "snappy"


@dataclass(slots=True)
class RenderingSection:
    enabled: bool = False
    width: int = 1400
    height: int = 900
    target_fps: int = 60
    # How many simulation ticks advance per rendered frame. The display stays at
    # target_fps; this is the "watch speed" (adjustable live with [ and ]). Higher
    # values fast-forward through a generation without dropping the frame rate.
    sim_ticks_per_frame: int = 3
    draw_sensors: bool = True
    draw_trails: bool = True


@dataclass(slots=True)
class AnalysisSection:
    enabled_post_train: bool = True
    crash_heatmap_bins: int = 36


@dataclass(slots=True)
class ReplaySection:
    fps: int = 60
    save_trajectory: bool = True


@dataclass(slots=True)
class ExperimentConfig:
    experiment: ExperimentSection = field(default_factory=ExperimentSection)
    runtime: RuntimeSection = field(default_factory=RuntimeSection)
    simulation: SimulationSection = field(default_factory=SimulationSection)
    car: CarSection = field(default_factory=CarSection)
    sensors: SensorSection = field(default_factory=SensorSection)
    dynamics: DynamicsSection = field(default_factory=DynamicsSection)
    termination: TerminationSection = field(default_factory=TerminationSection)
    progress: ProgressSection = field(default_factory=ProgressSection)
    fitness: FitnessSection = field(default_factory=FitnessSection)
    neat: NeatSection = field(default_factory=NeatSection)
    ppo: PpoSection = field(default_factory=PpoSection)
    study: StudySection = field(default_factory=StudySection)
    logging: LoggingSection = field(default_factory=LoggingSection)
    rendering: RenderingSection = field(default_factory=RenderingSection)
    analysis: AnalysisSection = field(default_factory=AnalysisSection)
    replay: ReplaySection = field(default_factory=ReplaySection)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConfigError(ValueError):
    pass


def _merge_dataclass(dc: Any, updates: dict[str, Any]) -> Any:
    for key, value in updates.items():
        if not hasattr(dc, key):
            raise ConfigError(f"Unknown config key: {type(dc).__name__}.{key}")
        current = getattr(dc, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(dc, key, value)
    return dc


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    return config_from_dict(raw)


def config_from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    cfg = ExperimentConfig()
    _merge_dataclass(cfg, raw)
    _validate_config(cfg)
    return cfg


def load_experiment_config_json(path: str | Path) -> ExperimentConfig:
    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return config_from_dict(payload)


def dump_config_json(config: ExperimentConfig, path: str | Path) -> None:
    import json

    Path(path).write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")


def _validate_config(cfg: ExperimentConfig) -> None:
    sensor_count = len(cfg.sensors.angles_deg)
    if sensor_count < 3:
        raise ConfigError("Sensor angles must include at least 3 rays.")
    if cfg.simulation.dt <= 0:
        raise ConfigError("simulation.dt must be positive")
    if cfg.runtime.max_generations <= 0:
        raise ConfigError("runtime.max_generations must be positive")
    if cfg.runtime.num_workers < 1:
        raise ConfigError("runtime.num_workers must be >= 1")
    if cfg.dynamics.surface_grip <= 0:
        raise ConfigError("dynamics.surface_grip must be positive")
    if cfg.dynamics.sensor_noise_std < 0:
        raise ConfigError("dynamics.sensor_noise_std must be non-negative")
    if cfg.dynamics.max_lateral_accel <= 0:
        raise ConfigError("dynamics.max_lateral_accel must be positive")
    if cfg.runtime.max_env_steps < 0:
        raise ConfigError("runtime.max_env_steps must be >= 0")
    if min(cfg.ppo.envs_per_track, cfg.ppo.n_steps, cfg.ppo.batch_size, cfg.ppo.n_epochs) < 1:
        raise ConfigError("ppo.envs_per_track, n_steps, batch_size and n_epochs must be >= 1")
    if not 0.0 < cfg.ppo.gamma <= 1.0:
        raise ConfigError("ppo.gamma must be in (0, 1]")
    if min(cfg.study.eval_points, cfg.study.eval_max_steps, cfg.study.max_parallel_runs) < 1:
        raise ConfigError("study.eval_points, eval_max_steps and max_parallel_runs must be >= 1")
    if min(cfg.study.curve_eval_starts, cfg.study.final_eval_starts) < 1:
        raise ConfigError("study.curve_eval_starts and final_eval_starts must be >= 1")
    if not 0.0 <= cfg.study.start_lateral_jitter < 1.0:
        raise ConfigError("study.start_lateral_jitter must be in [0, 1)")
    if not cfg.study.seeds:
        raise ConfigError("study.seeds must not be empty")


def training_track_files(cfg: ExperimentConfig) -> list[str]:
    return list(cfg.simulation.train_tracks) or [cfg.simulation.track_file]

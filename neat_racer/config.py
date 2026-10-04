"""Typed configuration loading for NEAT Racer."""

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


@dataclass(slots=True)
class SimulationSection:
    dt: float = 0.05
    max_steps_per_episode: int = 2000
    track_file: str = "tracks/baseline_loop.json"


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
    """Optional, off-by-default realism extensions to the base kinematic model.

    All defaults reproduce the original deterministic behavior exactly: understeer
    and sensor noise are disabled and ``surface_grip`` is 1.0 (full grip).
    """

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
    hud_panel_width: int = 360
    draw_sensors: bool = True
    draw_checkpoints: bool = True


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
    logging: LoggingSection = field(default_factory=LoggingSection)
    rendering: RenderingSection = field(default_factory=RenderingSection)
    analysis: AnalysisSection = field(default_factory=AnalysisSection)
    replay: ReplaySection = field(default_factory=ReplaySection)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConfigError(ValueError):
    """Raised when config files are malformed."""


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
    if cfg.dynamics.surface_grip <= 0:
        raise ConfigError("dynamics.surface_grip must be positive")
    if cfg.dynamics.sensor_noise_std < 0:
        raise ConfigError("dynamics.sensor_noise_std must be non-negative")
    if cfg.dynamics.max_lateral_accel <= 0:
        raise ConfigError("dynamics.max_lateral_accel must be positive")

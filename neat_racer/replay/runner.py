"""Champion replay utilities."""

from __future__ import annotations

import pickle
from pathlib import Path

import neat
import pandas as pd

from neat_racer.config import load_experiment_config_json
from neat_racer.simulation.env import network_controller, run_episode
from neat_racer.simulation.track import load_track
from neat_racer.telemetry.metrics_bus import MetricsBus
from neat_racer.visualization.renderer import LiveRenderer


class ReplayError(RuntimeError):
    """Raised when replay artifacts are incomplete."""


def replay_best_genome(run_dir: str | Path, live: bool = True) -> Path:
    run_path = Path(run_dir)
    config_path = run_path / "experiment_config.json"
    neat_cfg_path = run_path / "neat_config_used.txt"
    genome_path = run_path / "best_genome.pkl"

    if not config_path.exists() or not neat_cfg_path.exists() or not genome_path.exists():
        raise ReplayError(f"Missing replay artifact(s) in {run_path}")

    cfg = load_experiment_config_json(config_path)
    cfg.rendering.enabled = bool(live)
    cfg.runtime.mode = "live" if live else "headless"
    cfg.logging.save_per_step = True

    neat_cfg = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(neat_cfg_path),
    )

    with genome_path.open("rb") as fh:
        genome = pickle.load(fh)

    track = load_track(cfg.simulation.track_file)
    net = neat.nn.FeedForwardNetwork.create(genome, neat_cfg)

    renderer = None
    metrics = MetricsBus()
    if live:
        renderer = LiveRenderer(cfg.rendering, metrics)
        renderer.initialize(track)

    trajectory_rows = []

    def fitness_step(_: dict) -> float:
        return 0.0

    def render_step(track_obj, car, sensors, overlay):
        if renderer is None:
            return
        metrics.update_generation(generation=overlay.get("generation", 0))
        metrics.update_car(
            genome_id=overlay.get("genome_id", -1),
            speed=car.speed,
            steer=overlay.get("steer_cmd", car.steering_cmd),
            throttle=overlay.get("throttle_cmd", 0.0),
            brake=overlay.get("brake_cmd", 0.0),
            checkpoint=car.checkpoint_index,
            lap=car.lap_count,
            sensor_angles_deg=cfg.sensors.angles_deg,
            sensor_max_range=cfg.sensors.max_range,
        )
        renderer.draw_frame(track_obj, [car], 0, sensors, overlay)

    episode = run_episode(
        cfg,
        track,
        generation=0,
        genome_id=0,
        controller=network_controller(net.activate),
        fitness_step_fn=fitness_step,
        render_step=render_step if live else None,
    )

    for step in episode.per_step:
        trajectory_rows.append(
            {
                "tick": step.tick,
                "x": step.x,
                "y": step.y,
                "heading": step.heading,
                "speed": step.speed,
                "progress_scalar": step.progress_scalar,
            }
        )

    replay_path = run_path / "replays" / "best_genome_replay.csv"
    if trajectory_rows:
        pd.DataFrame(trajectory_rows).to_csv(replay_path, index=False)

    if renderer is not None:
        renderer.shutdown()

    return replay_path

"""Champion replay utilities."""

from __future__ import annotations

import contextlib
import pickle
from pathlib import Path

import neat
import pandas as pd

from neat_racer.config import load_experiment_config_json
from neat_racer.simulation.env import CarAgent, network_controller, run_population
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

    agent = CarAgent(
        cfg, track, generation=0, genome_id=0,
        controller=network_controller(net.activate), fitness_step_fn=lambda _f: 0.0,
    )

    def _overlay(agents, tick):
        a = agents[0]
        return {
            "generation": 0, "tick": tick,
            "alive": sum(1 for x in agents if x.alive), "total": len(agents),
            "gen_best_fitness": a.fitness_total, "run_best_fitness": a.fitness_total,
            "run_best_lap": a.progress_state.best_lap_time,
            "lead_speed": a.car.speed, "lead_lap": a.car.lap_count,
            "lead_checkpoint": a.car.checkpoint_index,
        }

    def render_all(track_obj, agents, tick):
        if renderer is None:
            return
        controls = renderer.poll_events()
        while controls.paused and not controls.quit_requested:
            renderer.draw_population(track_obj, agents, _overlay(agents, tick))
            controls = renderer.poll_events()
        if controls.quit_requested:
            raise KeyboardInterrupt("Quit requested by user")
        renderer.draw_population(track_obj, agents, _overlay(agents, tick))

    # A live quit (window close) raises KeyboardInterrupt; keep the partial trajectory.
    with contextlib.suppress(KeyboardInterrupt):
        run_population(cfg, track, [agent], render_all=render_all if live else None)

    for step in agent.per_step:
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

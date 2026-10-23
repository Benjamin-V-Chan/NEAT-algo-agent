from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from neat_racer.analysis.compare import compare_runs
from neat_racer.config import load_experiment_config
from neat_racer.constants import GENERATION_SUMMARY_COLUMNS, PER_CAR_SUMMARY_COLUMNS
from neat_racer.evolution.trainer import NeatTrainer
from neat_racer.utils.random_utils import seed_everything

TWO_TRACKS = ["tracks/oval_speedway.json", "tracks/twisty_circuit.json"]


def _small_cfg(tmp_path: Path, budget: int):
    cfg = load_experiment_config("configs/smoke_experiment.yaml")
    cfg.logging.output_root = str(tmp_path)
    cfg.simulation.train_tracks = list(TWO_TRACKS)
    cfg.simulation.max_steps_per_episode = 120
    cfg.neat.population_size = 12
    cfg.runtime.max_generations = 50
    cfg.runtime.max_env_steps = budget
    return cfg


def _train_neat(cfg, hook=None):
    seed_everything(cfg.experiment.seed)
    trainer = NeatTrainer(cfg, policy_hook=hook)
    return trainer, trainer.run()


def test_neat_multitrack_budget_and_hook(tmp_path: Path) -> None:
    cfg = _small_cfg(tmp_path, budget=3000)
    snapshots = []
    trainer, result = _train_neat(cfg, hook=snapshots.append)

    gen = pd.read_csv(result.run_dir / "generation_summary.csv")
    cars = pd.read_csv(result.run_dir / "per_car_summary.csv")
    assert list(gen.columns) == GENERATION_SUMMARY_COLUMNS
    assert list(cars.columns) == PER_CAR_SUMMARY_COLUMNS
    # Every genome drives every training track each generation.
    assert len(cars) == len(gen) * cfg.neat.population_size * len(TWO_TRACKS)
    assert set(cars["track_id"]) == {"oval_speedway", "twisty_circuit"}
    # The budget stops training at the first generation boundary at or past it.
    steps = gen["env_steps"].tolist()
    assert steps == sorted(steps)
    assert steps[-1] >= cfg.runtime.max_env_steps > steps[-2]
    assert steps[-1] == result.env_steps == int(cars["survival_time"].sum() / cfg.simulation.dt + 0.5)
    # One snapshot per generation, carrying the cumulative step count and a usable controller.
    assert [s.env_steps for s in snapshots] == steps
    controller = snapshots[-1].make_controller()
    assert controller is not None


def test_neat_multitrack_parallel_matches_serial(tmp_path: Path) -> None:
    serial_cfg = _small_cfg(tmp_path / "serial", budget=2500)
    parallel_cfg = _small_cfg(tmp_path / "parallel", budget=2500)
    parallel_cfg.runtime.num_workers = 2
    _, serial = _train_neat(serial_cfg)
    _, parallel = _train_neat(parallel_cfg)
    for name in ("per_car_summary.csv", "events.csv"):
        a = pd.read_csv(serial.run_dir / name)
        b = pd.read_csv(parallel.run_dir / name)
        pd.testing.assert_frame_equal(a, b)


def test_ppo_smoke_writes_neat_format_telemetry(tmp_path: Path) -> None:
    pytest.importorskip("stable_baselines3")
    from neat_racer.rl.train_ppo import PpoTrainer

    cfg = _small_cfg(tmp_path, budget=600)
    cfg.ppo.n_steps = 128
    cfg.ppo.batch_size = 64
    cfg.ppo.n_epochs = 1
    snapshots = []
    result = PpoTrainer(cfg, policy_hook=snapshots.append).run()

    run_dir = result.run_dir
    for name in ("generation_summary.csv", "per_car_summary.csv", "events.csv", "run_summary.json",
                 "experiment_config.json", "ppo_model.zip"):
        assert (run_dir / name).exists(), name
    gen = pd.read_csv(run_dir / "generation_summary.csv")
    assert list(gen.columns) == GENERATION_SUMMARY_COLUMNS
    # Whole rollouts only: 2 rollouts of 2 envs x 128 steps fit inside 600.
    assert result.env_steps == 512
    assert gen["env_steps"].tolist() == [256, 512]
    assert [s.env_steps for s in snapshots] == [0, 256, 512]
    # The existing comparison tool reads PPO runs like NEAT runs.
    summary = compare_runs([str(run_dir)])
    assert len(summary) == 1

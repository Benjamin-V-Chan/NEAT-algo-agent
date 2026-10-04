from __future__ import annotations

import math
from pathlib import Path

from neat_racer.config import ExperimentConfig
from neat_racer.evolution.trainer import NeatTrainer
from neat_racer.utils.random_utils import seed_everything


def _train(output_root: Path, *, seed: int, num_workers: int) -> float:
    cfg = ExperimentConfig()
    cfg.experiment.name = f"repro_w{num_workers}"
    cfg.experiment.seed = seed
    cfg.runtime.max_generations = 2
    cfg.runtime.num_workers = num_workers
    cfg.simulation.max_steps_per_episode = 150
    cfg.neat.population_size = 10
    cfg.analysis.enabled_post_train = False
    cfg.logging.save_per_step = False
    cfg.logging.output_root = str(output_root)

    seed_everything(seed)
    trainer = NeatTrainer(cfg)
    return trainer.run().best_fitness


def test_training_is_deterministic(tmp_path: Path) -> None:
    first = _train(tmp_path / "a", seed=11, num_workers=1)
    second = _train(tmp_path / "b", seed=11, num_workers=1)
    assert math.isfinite(first)
    assert first == second


def test_parallel_matches_serial(tmp_path: Path) -> None:
    serial = _train(tmp_path / "serial", seed=11, num_workers=1)
    parallel = _train(tmp_path / "parallel", seed=11, num_workers=2)
    assert math.isfinite(serial)
    # The default model is noise-free, so per-episode results are deterministic
    # given their inputs; parallel evaluation must reproduce the serial outcome.
    assert parallel == serial

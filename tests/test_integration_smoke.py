from __future__ import annotations

from pathlib import Path

from neat_racer.config import load_experiment_config
from neat_racer.evolution.trainer import NeatTrainer
from neat_racer.utils.random_utils import seed_everything


def test_smoke_training_creates_artifacts(tmp_path: Path) -> None:
    cfg = load_experiment_config("configs/smoke_experiment.yaml")
    cfg.logging.output_root = str(tmp_path)
    cfg.analysis.enabled_post_train = False
    cfg.rendering.enabled = False
    cfg.runtime.mode = "headless"

    seed_everything(cfg.experiment.seed)
    trainer = NeatTrainer(cfg)
    result = trainer.run()

    assert result.run_dir.exists()
    assert (result.run_dir / "generation_summary.csv").exists()
    assert (result.run_dir / "per_car_summary.csv").exists()
    assert (result.run_dir / "best_genome.pkl").exists()
    assert (result.run_dir / "run_summary.json").exists()

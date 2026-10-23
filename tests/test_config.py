from __future__ import annotations

import pytest

from neat_racer.config import (
    ConfigError,
    ExperimentConfig,
    config_from_dict,
    load_experiment_config,
    training_track_files,
)
from neat_racer.study.evaluation import eval_marks


@pytest.mark.parametrize(
    "path", ["configs/default_experiment.yaml", "configs/smoke_experiment.yaml", "configs/neat_vs_ppo.yaml"]
)
def test_configs_load(path) -> None:
    load_experiment_config(path)


def test_study_config_shares_one_budget_and_split() -> None:
    cfg = load_experiment_config("configs/neat_vs_ppo.yaml")
    assert cfg.runtime.max_env_steps == 3_000_000
    assert len(training_track_files(cfg)) == 5
    assert len(cfg.study.holdout_tracks) >= 2
    assert not set(cfg.study.holdout_tracks) & set(training_track_files(cfg))
    assert len(cfg.study.seeds) >= 5
    assert cfg.ppo.net_arch == [64, 64]


def test_old_configs_get_safe_defaults() -> None:
    cfg = ExperimentConfig()
    assert cfg.runtime.max_env_steps == 0  # no budget -> NEAT runs max_generations as before
    assert training_track_files(cfg) == [cfg.simulation.track_file]


def test_unknown_ppo_key_rejected() -> None:
    with pytest.raises(ConfigError):
        config_from_dict({"ppo": {"learning_rat": 1e-3}})


@pytest.mark.parametrize(
    "raw",
    [
        {"runtime": {"max_env_steps": -1}},
        {"ppo": {"batch_size": 0}},
        {"ppo": {"gamma": 1.5}},
        {"study": {"seeds": []}},
        {"study": {"eval_points": 0}},
    ],
)
def test_invalid_values_rejected(raw) -> None:
    with pytest.raises(ConfigError):
        config_from_dict(raw)


def test_eval_marks_span_budget() -> None:
    cfg = config_from_dict({"runtime": {"max_env_steps": 3000}, "study": {"eval_points": 3}})
    assert eval_marks(cfg) == [0, 1000, 2000, 3000]

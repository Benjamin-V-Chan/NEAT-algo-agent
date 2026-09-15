from __future__ import annotations

import numpy as np

from neat_racer.config import ExperimentConfig, SensorSection
from neat_racer.evolution.neat_config_builder import observation_size
from neat_racer.simulation.car import CarState
from neat_racer.simulation.sensors import build_observation, raycast_distances


def test_raycast_values_are_normalized(simple_track) -> None:
    cfg = SensorSection(angles_deg=[-45, 0, 45], max_range=200.0)
    origin = np.array([25.0, 25.0])
    distances = raycast_distances(simple_track, origin, heading=0.0, sensor_cfg=cfg)

    assert distances.shape == (3,)
    assert np.all(distances >= 0.0)
    assert np.all(distances <= 1.0)


def test_observation_size_matches_builder(simple_track) -> None:
    cfg = SensorSection(angles_deg=[-45, 0, 45], max_range=200.0, include_alignment_feature=True)
    car = CarState(pos=np.array([25.0, 25.0]), heading=0.0)
    obs = build_observation(simple_track, car, cfg, speed_norm=0.5, progress_delta=0.01)

    # rays + [speed, steer, progress_delta] + alignment feature
    assert obs.shape[0] == len(cfg.angles_deg) + 4


def test_observation_size_helper_agrees_with_config() -> None:
    exp = ExperimentConfig()
    exp.sensors.angles_deg = [-30, 0, 30]
    exp.sensors.include_alignment_feature = False
    assert observation_size(exp) == 3 + 3  # rays + [speed, steer, progress_delta]

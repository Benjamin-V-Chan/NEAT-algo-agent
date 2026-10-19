from __future__ import annotations

import numpy as np

from neat_racer.config import SensorSection
from neat_racer.simulation.car import CarState
from neat_racer.simulation.geometry import raycast_min_distances
from neat_racer.simulation.track import Track


def raycast_distances(track: Track, origin: np.ndarray, heading: float, sensor_cfg: SensorSection) -> np.ndarray:
    angles = np.radians(np.asarray(sensor_cfg.angles_deg, dtype=float)) + heading
    ray_dirs = np.column_stack([np.cos(angles), np.sin(angles)])
    seg_a, seg_b = track.wall_arrays
    dists = raycast_min_distances(np.asarray(origin, dtype=float), ray_dirs, seg_a, seg_b, sensor_cfg.max_range)
    return np.clip(dists / sensor_cfg.max_range, 0.0, 1.0)


def build_observation(
    track: Track,
    car: CarState,
    sensor_cfg: SensorSection,
    speed_norm: float,
    progress_delta: float,
    *,
    sensor_vector: np.ndarray | None = None,
    alignment: float | None = None,
) -> np.ndarray:
    if sensor_vector is None:
        sensor_vector = raycast_distances(track, car.pos, car.heading, sensor_cfg)
    extras = [
        speed_norm,
        float((car.steering_cmd + 1.0) * 0.5),
        float(np.clip(progress_delta, -1.0, 1.0)),
    ]

    if sensor_cfg.include_alignment_feature:
        if alignment is None:
            alignment = track.local_alignment_cos(car.pos, car.heading)
        extras.append(float((alignment + 1.0) * 0.5))

    return np.concatenate([sensor_vector, np.asarray(extras, dtype=float)])

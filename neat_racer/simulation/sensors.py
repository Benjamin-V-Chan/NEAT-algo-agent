"""Sensor generation for local car observations."""

from __future__ import annotations

import math

import numpy as np

from neat_racer.config import SensorSection
from neat_racer.simulation.car import CarState
from neat_racer.simulation.geometry import ray_segment_distance
from neat_racer.simulation.track import Track


def raycast_distances(track: Track, origin: np.ndarray, heading: float, sensor_cfg: SensorSection) -> np.ndarray:
    distances = []
    segments = track.wall_segments
    for angle_deg in sensor_cfg.angles_deg:
        ray_theta = heading + math.radians(angle_deg)
        ray_dir = np.array([math.cos(ray_theta), math.sin(ray_theta)], dtype=float)
        min_dist = sensor_cfg.max_range
        for seg in segments:
            d = ray_segment_distance(origin, ray_dir, seg.a, seg.b)
            if d is not None and d < min_dist:
                min_dist = d
        distances.append(min_dist)

    arr = np.asarray(distances, dtype=float)
    return np.clip(arr / sensor_cfg.max_range, 0.0, 1.0)


def build_observation(
    track: Track,
    car: CarState,
    sensor_cfg: SensorSection,
    speed_norm: float,
    progress_delta: float,
) -> np.ndarray:
    sensor_vector = raycast_distances(track, car.pos, car.heading, sensor_cfg)
    extras = [
        speed_norm,
        float((car.steering_cmd + 1.0) * 0.5),
        float(np.clip(progress_delta, -1.0, 1.0)),
    ]

    if sensor_cfg.include_alignment_feature:
        alignment = track.local_alignment_cos(car.pos, car.heading)
        extras.append(float((alignment + 1.0) * 0.5))

    return np.concatenate([sensor_vector, np.asarray(extras, dtype=float)])

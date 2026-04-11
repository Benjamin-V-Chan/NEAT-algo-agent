from __future__ import annotations

import numpy as np

from neat_racer.simulation.geometry import point_in_polygon, ray_segment_distance, segment_intersection


def test_segment_intersection_hits() -> None:
    p1 = np.array([0.0, 0.0])
    p2 = np.array([2.0, 2.0])
    q1 = np.array([0.0, 2.0])
    q2 = np.array([2.0, 0.0])
    hit, point = segment_intersection(p1, p2, q1, q2)
    assert hit
    assert point is not None
    assert np.allclose(point, np.array([1.0, 1.0]))


def test_ray_segment_distance() -> None:
    origin = np.array([0.0, 0.0])
    ray_dir = np.array([1.0, 0.0])
    a = np.array([5.0, -1.0])
    b = np.array([5.0, 1.0])
    d = ray_segment_distance(origin, ray_dir, a, b)
    assert d is not None
    assert abs(d - 5.0) < 1e-6


def test_point_in_polygon() -> None:
    poly = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])
    assert point_in_polygon(np.array([2.0, 2.0]), poly)
    assert not point_in_polygon(np.array([6.0, 2.0]), poly)

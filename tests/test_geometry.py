from __future__ import annotations

import numpy as np

from neat_racer.simulation.geometry import (
    min_point_segment_distance_vec,
    min_point_segment_distances_pop,
    point_in_polygon,
    point_in_polygon_vec,
    points_in_polygon_pop,
    raycast_min_distances,
    raycast_min_distances_pop,
    segment_intersection,
)


def test_segment_intersection_hits() -> None:
    p1 = np.array([0.0, 0.0])
    p2 = np.array([2.0, 2.0])
    q1 = np.array([0.0, 2.0])
    q2 = np.array([2.0, 0.0])
    hit, point = segment_intersection(p1, p2, q1, q2)
    assert hit
    assert point is not None
    assert np.allclose(point, np.array([1.0, 1.0]))


def test_point_in_polygon() -> None:
    poly = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])
    assert point_in_polygon(np.array([2.0, 2.0]), poly)
    assert not point_in_polygon(np.array([6.0, 2.0]), poly)


def _square_walls() -> tuple[np.ndarray, np.ndarray]:
    poly = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    seg_a = poly
    seg_b = np.roll(poly, -1, axis=0)
    return seg_a, seg_b


def test_raycast_single_matches_expected() -> None:
    seg_a, seg_b = _square_walls()
    origin = np.array([5.0, 5.0])
    ray_dirs = np.array([[1.0, 0.0], [0.0, 1.0]])  # +x hits x=10, +y hits y=10
    dists = raycast_min_distances(origin, ray_dirs, seg_a, seg_b, max_range=100.0)
    assert np.allclose(dists, [5.0, 5.0])


def test_raycast_pop_matches_single_agent() -> None:
    seg_a, seg_b = _square_walls()
    origins = np.array([[5.0, 5.0], [2.0, 2.0], [8.0, 6.0]])
    ray_dirs = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    batched = raycast_min_distances_pop(
        origins, np.broadcast_to(ray_dirs, (3, 3, 2)).copy(), seg_a, seg_b, max_range=100.0
    )
    for i, origin in enumerate(origins):
        single = raycast_min_distances(origin, ray_dirs, seg_a, seg_b, max_range=100.0)
        assert np.array_equal(batched[i], single)


def test_point_segment_distance_pop_matches_single() -> None:
    seg_a, seg_b = _square_walls()
    points = np.array([[5.0, 1.0], [1.0, 5.0], [5.0, 5.0]])
    batched = min_point_segment_distances_pop(points, seg_a, seg_b)
    for i, p in enumerate(points):
        assert batched[i] == min_point_segment_distance_vec(p, seg_a, seg_b)


def test_points_in_polygon_pop_matches_single() -> None:
    poly = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])
    points = np.array([[2.0, 2.0], [6.0, 2.0], [0.5, 3.5]])
    batched = points_in_polygon_pop(points, poly)
    for i, p in enumerate(points):
        assert bool(batched[i]) == point_in_polygon_vec(p, poly)

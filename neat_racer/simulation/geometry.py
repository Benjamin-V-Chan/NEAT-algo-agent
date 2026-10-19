from __future__ import annotations

from collections.abc import Iterable

import numpy as np

EPS = 1e-9


def as_np(point: Iterable[float]) -> np.ndarray:
    return np.asarray(point, dtype=float)


def segment_intersection(
    p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray
) -> tuple[bool, np.ndarray | None]:
    r = p2 - p1
    s = q2 - q1
    rxs = _cross2(r, s)
    qpxr = _cross2((q1 - p1), r)

    if abs(rxs) < EPS and abs(qpxr) < EPS:
        return False, None
    if abs(rxs) < EPS and abs(qpxr) >= EPS:
        return False, None

    t = _cross2((q1 - p1), s) / rxs
    u = _cross2((q1 - p1), r) / rxs

    if -EPS <= t <= 1.0 + EPS and -EPS <= u <= 1.0 + EPS:
        return True, p1 + t * r
    return False, None


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        intersects = ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / (y2 - y1 + EPS) + x1
        )
        if intersects:
            inside = not inside
    return inside


def point_to_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = np.dot(ab, ab)
    if denom < EPS:
        return float(np.linalg.norm(point - a))
    t = np.dot(point - a, ab) / denom
    t = float(np.clip(t, 0.0, 1.0))
    proj = a + t * ab
    return float(np.linalg.norm(point - proj))


def closest_segment_index(point: np.ndarray, polyline: np.ndarray) -> int:
    min_d = float("inf")
    min_i = 0
    for i in range(len(polyline) - 1):
        d = point_to_segment_distance(point, polyline[i], polyline[i + 1])
        if d < min_d:
            min_d = d
            min_i = i
    return min_i


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


# --- Vectorized hot paths ------------------------------------------------------
# Each of these evaluates every wall segment in one numpy call. The ``*_pop``
# variants add a leading "agent" axis so an entire generation is resolved in a
# single batched call per tick. For one agent they return exactly the same values
# as the single-agent helpers, so batching never changes the simulation outcome.


def raycast_min_distances(
    origin: np.ndarray, ray_dirs: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray, max_range: float
) -> np.ndarray:
    return raycast_min_distances_pop(
        origin[None, :], ray_dirs[None, :, :], seg_a, seg_b, max_range
    )[0]


def raycast_min_distances_pop(
    origins: np.ndarray, ray_dirs: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray, max_range: float
) -> np.ndarray:
    v2 = seg_b - seg_a  # (M, 2) segment direction
    v1 = origins[:, None, :] - seg_a[None, :, :]  # (A, M, 2) origin -> segment start
    # cross(v2, v1) per (agent, segment)
    cross_v2_v1 = v2[None, :, 0] * v1[:, :, 1] - v2[None, :, 1] * v1[:, :, 0]  # (A, M)

    v3 = np.stack([-ray_dirs[:, :, 1], ray_dirs[:, :, 0]], axis=-1)  # (A, R, 2) ray normal
    denom = np.einsum("arc,mc->arm", v3, v2)  # (A, R, M)
    safe = np.where(np.abs(denom) < EPS, np.nan, denom)
    t1 = cross_v2_v1[:, None, :] / safe  # distance along each ray (A, R, M)
    t2 = np.einsum("arc,amc->arm", v3, v1) / safe  # param along each segment (A, R, M)

    valid = (t1 >= -EPS) & (t2 >= -EPS) & (t2 <= 1.0 + EPS)
    dist = np.where(valid, np.maximum(0.0, t1), np.inf)
    nearest = dist.min(axis=2)  # (A, R)
    return np.minimum(nearest, max_range)


def min_point_segment_distance_vec(point: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray) -> float:
    return float(min_point_segment_distances_pop(point[None, :], seg_a, seg_b)[0])


def min_point_segment_distances_pop(points: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray) -> np.ndarray:
    ab = seg_b - seg_a  # (M, 2)
    denom = np.einsum("ij,ij->i", ab, ab)  # (M,)
    denom_safe = np.where(denom < EPS, 1.0, denom)
    ap = points[:, None, :] - seg_a[None, :, :]  # (A, M, 2)
    t = np.clip(np.einsum("amc,mc->am", ap, ab) / denom_safe[None, :], 0.0, 1.0)  # (A, M)
    proj = seg_a[None, :, :] + t[:, :, None] * ab[None, :, :]  # (A, M, 2)
    d2 = ((points[:, None, :] - proj) ** 2).sum(axis=2)  # (A, M)
    return np.sqrt(d2.min(axis=1))


def point_in_polygon_vec(point: np.ndarray, polygon: np.ndarray) -> bool:
    return bool(points_in_polygon_pop(point[None, :], polygon)[0])


def points_in_polygon_pop(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    x = points[:, 0][:, None]  # (A, 1)
    y = points[:, 1][:, None]  # (A, 1)
    x1 = polygon[:, 0][None, :]  # (1, P)
    y1 = polygon[:, 1][None, :]
    x2 = np.roll(x1, -1, axis=1)
    y2 = np.roll(y1, -1, axis=1)
    crosses = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1 + EPS) + x1)
    return (np.count_nonzero(crosses, axis=1) % 2) == 1

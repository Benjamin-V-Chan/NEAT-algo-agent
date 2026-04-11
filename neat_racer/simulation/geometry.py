"""Geometry primitives and collision helpers for the 2D simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

EPS = 1e-9


@dataclass(frozen=True, slots=True)
class Segment:
    a: np.ndarray
    b: np.ndarray


def as_np(point: Iterable[float]) -> np.ndarray:
    return np.asarray(point, dtype=float)


def segments_from_polygon(points: np.ndarray) -> list[Segment]:
    segs: list[Segment] = []
    for i in range(len(points)):
        segs.append(Segment(points[i], points[(i + 1) % len(points)]))
    return segs


def segment_intersection(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> tuple[bool, np.ndarray | None]:
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


def ray_segment_distance(ray_origin: np.ndarray, ray_dir: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray) -> float | None:
    v1 = ray_origin - seg_a
    v2 = seg_b - seg_a
    v3 = np.array([-ray_dir[1], ray_dir[0]], dtype=float)

    denom = np.dot(v2, v3)
    if abs(denom) < EPS:
        return None

    t1 = _cross2(v2, v1) / denom
    t2 = np.dot(v1, v3) / denom

    if t1 >= -EPS and -EPS <= t2 <= 1.0 + EPS:
        return float(max(0.0, t1))
    return None


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


def min_distance_to_segments(point: np.ndarray, segments: list[Segment]) -> float:
    return min(point_to_segment_distance(point, s.a, s.b) for s in segments)


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

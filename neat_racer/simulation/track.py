from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from neat_racer.constants import REQUIRED_TRACK_KEYS
from neat_racer.simulation.geometry import (
    as_np,
    closest_segment_index,
    min_point_segment_distance_vec,
    min_point_segment_distances_pop,
    point_in_polygon_vec,
    points_in_polygon_pop,
)


@dataclass(slots=True)
class Track:
    track_id: str
    name: str
    scale: float
    outer_boundary: np.ndarray
    inner_boundary: np.ndarray
    checkpoints: list[tuple[np.ndarray, np.ndarray]]
    start_finish: tuple[np.ndarray, np.ndarray]
    spawn_pos: np.ndarray
    spawn_heading: float
    metadata: dict
    _wall_seg_a: np.ndarray | None = field(default=None, compare=False, repr=False)
    _wall_seg_b: np.ndarray | None = field(default=None, compare=False, repr=False)
    _cl_seg_a: np.ndarray | None = field(default=None, compare=False, repr=False)
    _cl_seg_b: np.ndarray | None = field(default=None, compare=False, repr=False)

    @property
    def wall_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if self._wall_seg_a is None:
            oa, ob = self.outer_boundary, np.roll(self.outer_boundary, -1, axis=0)
            ia, ib = self.inner_boundary, np.roll(self.inner_boundary, -1, axis=0)
            self._wall_seg_a = np.vstack([oa, ia])
            self._wall_seg_b = np.vstack([ob, ib])
        return self._wall_seg_a, self._wall_seg_b

    @property
    def checkpoint_centers(self) -> np.ndarray:
        return np.asarray([(a + b) * 0.5 for a, b in self.checkpoints], dtype=float)

    @property
    def start_center(self) -> np.ndarray:
        a, b = self.start_finish
        return (a + b) * 0.5

    def is_inside_corridor(self, point: np.ndarray) -> bool:
        inside_outer = point_in_polygon_vec(point, self.outer_boundary)
        inside_inner = point_in_polygon_vec(point, self.inner_boundary)
        return bool(inside_outer and not inside_inner)

    def collision_at(self, point: np.ndarray, radius: float) -> bool:
        if not self.is_inside_corridor(point):
            return True
        seg_a, seg_b = self.wall_arrays
        return bool(min_point_segment_distance_vec(point, seg_a, seg_b) <= radius)

    def collision_mask(self, points: np.ndarray, radius: float) -> np.ndarray:
        points = np.atleast_2d(np.asarray(points, dtype=float))
        inside_outer = points_in_polygon_pop(points, self.outer_boundary)
        inside_inner = points_in_polygon_pop(points, self.inner_boundary)
        in_corridor = inside_outer & ~inside_inner
        seg_a, seg_b = self.wall_arrays
        near_wall = min_point_segment_distances_pop(points, seg_a, seg_b) <= radius
        return (~in_corridor) | near_wall

    def local_alignment_cos(self, point: np.ndarray, heading: float) -> float:
        centerline = self._centerline()
        idx = closest_segment_index(point, centerline)
        tangent = centerline[idx + 1] - centerline[idx]
        norm = np.linalg.norm(tangent)
        if norm < 1e-9:
            return 0.0
        tangent = tangent / norm
        heading_vec = np.array([np.cos(heading), np.sin(heading)], dtype=float)
        return float(np.clip(np.dot(heading_vec, tangent), -1.0, 1.0))

    @property
    def _centerline_segments(self) -> tuple[np.ndarray, np.ndarray]:
        if self._cl_seg_a is None:
            cl = self._centerline()
            self._cl_seg_a = cl[:-1]
            self._cl_seg_b = cl[1:]
        return self._cl_seg_a, self._cl_seg_b

    def local_alignment_cos_pop(self, points: np.ndarray, headings: np.ndarray) -> np.ndarray:
        from neat_racer.simulation.geometry import EPS

        points = np.atleast_2d(np.asarray(points, dtype=float))
        headings = np.asarray(headings, dtype=float).reshape(-1)
        seg_a, seg_b = self._centerline_segments
        ab = seg_b - seg_a  # (S, 2)
        denom = np.einsum("sc,sc->s", ab, ab)  # (S,)
        denom_safe = np.where(denom < EPS, 1.0, denom)
        ap = points[:, None, :] - seg_a[None, :, :]  # (A, S, 2)
        t = np.clip(np.einsum("asc,sc->as", ap, ab) / denom_safe[None, :], 0.0, 1.0)  # (A, S)
        proj = seg_a[None, :, :] + t[:, :, None] * ab[None, :, :]  # (A, S, 2)
        dist = np.sqrt(((points[:, None, :] - proj) ** 2).sum(axis=2))  # (A, S)
        idx = dist.argmin(axis=1)  # (A,)

        tangents = ab[idx]  # (A, 2)
        norms = np.linalg.norm(tangents, axis=1)  # (A,)
        safe_norms = np.where(norms < 1e-9, 1.0, norms)
        unit = tangents / safe_norms[:, None]
        heading_vec = np.stack([np.cos(headings), np.sin(headings)], axis=1)  # (A, 2)
        cos = np.clip(np.einsum("ac,ac->a", heading_vec, unit), -1.0, 1.0)
        return np.where(norms < 1e-9, 0.0, cos)

    def _centerline(self) -> np.ndarray:
        data = self.metadata.get("centerline")
        if data:
            return np.asarray(data, dtype=float)

        centers = self.checkpoint_centers
        if len(centers) < 2:
            return np.vstack([self.start_center, self.start_center + np.array([1.0, 0.0])])
        wrapped = np.vstack([centers, centers[0]])
        return wrapped


class TrackError(ValueError):
    pass


def load_track(path: str | Path) -> Track:
    track_path = Path(path)
    if not track_path.exists():
        raise FileNotFoundError(f"Track file not found: {track_path}")

    payload = json.loads(track_path.read_text(encoding="utf-8"))
    missing = REQUIRED_TRACK_KEYS - payload.keys()
    if missing:
        raise TrackError(f"Track file missing keys: {sorted(missing)}")

    outer = np.asarray(payload["outer_boundary"], dtype=float)
    inner = np.asarray(payload["inner_boundary"], dtype=float)
    if len(outer) < 3 or len(inner) < 3:
        raise TrackError("Track boundaries require at least 3 points each")

    checkpoints = []
    for gate in payload["checkpoints"]:
        if len(gate) != 2:
            raise TrackError("Each checkpoint must be a 2-point segment")
        checkpoints.append((as_np(gate[0]), as_np(gate[1])))

    start_finish_raw = payload["start_finish"]
    if len(start_finish_raw) != 2:
        raise TrackError("start_finish must contain exactly two points")

    spawn_pose = payload["spawn_pose"]
    return Track(
        track_id=str(payload["track_id"]),
        name=str(payload["name"]),
        scale=float(payload["scale"]),
        outer_boundary=outer,
        inner_boundary=inner,
        checkpoints=checkpoints,
        start_finish=(as_np(start_finish_raw[0]), as_np(start_finish_raw[1])),
        spawn_pos=np.array([float(spawn_pose["x"]), float(spawn_pose["y"])], dtype=float),
        spawn_heading=float(spawn_pose["heading"]),
        metadata=dict(payload.get("metadata", {})),
    )

"""Track representation and loading."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from neat_racer.constants import REQUIRED_TRACK_KEYS
from neat_racer.simulation.geometry import (
    Segment,
    as_np,
    closest_segment_index,
    min_distance_to_segments,
    point_in_polygon,
    segments_from_polygon,
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

    @property
    def outer_segments(self) -> list[Segment]:
        return segments_from_polygon(self.outer_boundary)

    @property
    def inner_segments(self) -> list[Segment]:
        return segments_from_polygon(self.inner_boundary)

    @property
    def wall_segments(self) -> list[Segment]:
        return self.outer_segments + self.inner_segments

    @property
    def checkpoint_centers(self) -> np.ndarray:
        return np.asarray([(a + b) * 0.5 for a, b in self.checkpoints], dtype=float)

    @property
    def start_center(self) -> np.ndarray:
        a, b = self.start_finish
        return (a + b) * 0.5

    def is_inside_corridor(self, point: np.ndarray) -> bool:
        inside_outer = point_in_polygon(point, self.outer_boundary)
        inside_inner = point_in_polygon(point, self.inner_boundary)
        return bool(inside_outer and not inside_inner)

    def collision_at(self, point: np.ndarray, radius: float) -> bool:
        if not self.is_inside_corridor(point):
            return True
        d_outer = min_distance_to_segments(point, self.outer_segments)
        d_inner = min_distance_to_segments(point, self.inner_segments)
        return bool(min(d_outer, d_inner) <= radius)

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
    """Raised when track definitions are malformed."""


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

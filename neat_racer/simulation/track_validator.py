"""Geometry sanity checks for track definitions.

:func:`validate_track` runs a battery of cheap checks against a loaded :class:`Track`
and returns a list of :class:`Issue` records. Each issue is either an ``"error"``
(something that will break training — e.g. a self-intersecting wall or a spawn stuck in
a wall) or a ``"warning"`` (suspicious but survivable — e.g. a checkpoint gate whose
midpoint sits just outside the drivable corridor, which the hand-authored baseline track
does yet still drives fine). :func:`has_errors` is the pass/fail gate used by the CLI and
the track-generation tooling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neat_racer.simulation.geometry import point_in_polygon, segment_intersection
from neat_racer.simulation.track import Track

__all__ = ["Issue", "validate_track", "has_errors", "format_issues"]


@dataclass(frozen=True, slots=True)
class Issue:
    severity: str  # "error" or "warning"
    message: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.message}"


def _self_intersects(polygon: np.ndarray) -> bool:
    """True if a closed polygon has any pair of non-adjacent edges that cross."""
    n = len(polygon)
    for i in range(n):
        a1 = polygon[i]
        a2 = polygon[(i + 1) % n]
        for j in range(i + 1, n):
            # Skip shared-endpoint neighbours (including the wrap-around pair).
            if j == (i + 1) % n or (j + 1) % n == i:
                continue
            b1 = polygon[j]
            b2 = polygon[(j + 1) % n]
            hit, _ = segment_intersection(a1, a2, b1, b2)
            if hit:
                return True
    return False


def validate_track(track: Track, *, collision_radius: float = 7.5) -> list[Issue]:
    """Return a list of :class:`Issue` records for ``track`` (empty means pristine)."""
    issues: list[Issue] = []

    def error(msg: str) -> None:
        issues.append(Issue("error", msg))

    def warning(msg: str) -> None:
        issues.append(Issue("warning", msg))

    if len(track.outer_boundary) < 3:
        error("outer_boundary needs at least 3 points")
    if len(track.inner_boundary) < 3:
        error("inner_boundary needs at least 3 points")
    if issues:
        return issues  # further checks assume usable polygons

    if _self_intersects(track.outer_boundary):
        error("outer_boundary self-intersects")
    if _self_intersects(track.inner_boundary):
        error("inner_boundary self-intersects")

    # The inner boundary (the hole) must sit entirely inside the outer boundary.
    outside_inner = sum(
        1 for p in track.inner_boundary if not point_in_polygon(p, track.outer_boundary)
    )
    if outside_inner:
        error(f"inner_boundary has {outside_inner} point(s) outside the outer boundary")

    # Spawn pose must be in the drivable corridor and clear of the walls.
    if not track.is_inside_corridor(track.spawn_pos):
        error("spawn pose is not inside the drivable corridor")
    elif track.collision_at(track.spawn_pos, collision_radius):
        error(f"spawn pose is within collision_radius ({collision_radius}) of a wall")

    if len(track.checkpoints) < 1:
        error("track needs at least one checkpoint")

    # Checkpoint / start-finish midpoints outside the corridor are suspicious but not
    # fatal: the gate is still a crossable segment spanning the corridor.
    for i, center in enumerate(track.checkpoint_centers):
        if not track.is_inside_corridor(center):
            warning(f"checkpoint {i} midpoint is outside the corridor")
    if not track.is_inside_corridor(track.start_center):
        warning("start/finish midpoint is outside the corridor")

    centerline = track.metadata.get("centerline")
    if centerline:
        outside = sum(
            1 for p in centerline if not track.is_inside_corridor(np.asarray(p, dtype=float))
        )
        if outside:
            warning(f"centerline has {outside} point(s) outside the corridor")

    return issues


def has_errors(issues: list[Issue]) -> bool:
    return any(issue.severity == "error" for issue in issues)


def format_issues(issues: list[Issue]) -> list[str]:
    return [str(issue) for issue in issues]

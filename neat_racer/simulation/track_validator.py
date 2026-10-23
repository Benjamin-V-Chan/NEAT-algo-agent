from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neat_racer.simulation.geometry import point_in_polygon, segment_intersection
from neat_racer.simulation.track import Track

__all__ = ["Issue", "validate_track", "has_errors", "format_issues", "corridor_span", "gate_wall_gaps"]


@dataclass(frozen=True, slots=True)
class Issue:
    severity: str  # "error" or "warning"
    message: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.message}"


def _self_intersects(polygon: np.ndarray) -> bool:
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


def corridor_span(track: Track, a: np.ndarray, b: np.ndarray) -> tuple[float, float] | None:
    a = np.asarray(a, dtype=float)
    d = np.asarray(b, dtype=float) - a
    seg_a, seg_b = track.wall_arrays
    e = seg_b - seg_a
    denom = d[0] * e[:, 1] - d[1] * e[:, 0]
    ok = np.abs(denom) > 1e-12
    diff = seg_a - a
    t = np.where(ok, (diff[:, 0] * e[:, 1] - diff[:, 1] * e[:, 0]) / np.where(ok, denom, 1.0), np.nan)
    u = np.where(ok, (diff[:, 0] * d[1] - diff[:, 1] * d[0]) / np.where(ok, denom, 1.0), np.nan)
    cuts = np.unique(t[ok & (u >= 0.0) & (u <= 1.0)])
    best, best_overlap = None, 0.0
    for t0, t1 in zip(cuts[:-1], cuts[1:], strict=True):
        if t1 - t0 < 1e-9 or not track.is_inside_corridor(a + d * (t0 + t1) * 0.5):
            continue
        overlap = min(t1, 1.0) - max(t0, 0.0)
        if overlap > best_overlap:
            best, best_overlap = (float(t0), float(t1)), overlap
    return best


def gate_wall_gaps(track: Track) -> list[tuple[str, float, float]]:
    gates = [(f"checkpoint {i}", a, b) for i, (a, b) in enumerate(track.checkpoints)]
    gates.append(("start/finish", *track.start_finish))
    out = []
    for name, a, b in gates:
        span = corridor_span(track, a, b)
        if span is None:
            out.append((name, np.inf, np.inf))
            continue
        length = float(np.linalg.norm(np.asarray(b, dtype=float) - np.asarray(a, dtype=float)))
        out.append((name, max(0.0, -span[0]) * length, max(0.0, span[1] - 1.0) * length))
    return out


def validate_track(track: Track, *, collision_radius: float = 7.5) -> list[Issue]:
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

    # Gates must span the corridor wall-to-wall. If a gate stops short of a wall by more than the
    # car's collision radius, a car can drive past it without crossing it and never earn progress.
    for name, gap_a, gap_b in gate_wall_gaps(track):
        gap = max(gap_a, gap_b)
        if gap > collision_radius:
            error(f"{name} stops {gap:.1f} px short of a wall (car can bypass it)")
        elif gap > 0.5:
            warning(f"{name} stops {gap:.1f} px short of a wall")

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

"""Procedural track construction.

Hand-authoring an annulus track (outer polygon, inner polygon, an ordered ring of
checkpoint gates, a start/finish line, a spawn pose and a centerline) by typing raw
coordinates is tedious and error-prone. This module builds all of that from a single
input: a closed *centerline* polyline plus a track half-width.

The centerline is resampled to even arc-length spacing, offset outward/inward by the
half-width with mitered joins to form the two boundaries, and sampled at evenly spaced
arc positions to place the start/finish line (at ``s = 0``) and the checkpoint gates.
Because every gate spans the full corridor and is laid down in travel order, the result
is always consistent with the progress/lap logic in :mod:`neat_racer.simulation.progress`.
"""

from __future__ import annotations

import numpy as np

__all__ = ["resample_closed", "build_annulus_track"]


def _polygon_area(points: np.ndarray) -> float:
    """Signed area of a closed polygon (positive for counter-clockwise)."""
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def resample_closed(points: np.ndarray, n: int) -> np.ndarray:
    """Resample a closed polyline to ``n`` points spaced evenly by arc length.

    ``points`` is the loop vertices without a repeated closing point; the returned
    array is likewise open (``n`` distinct points, first != last).
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        raise ValueError("A centerline needs at least 3 points")

    loop = np.vstack([pts, pts[0]])
    seg = np.linalg.norm(np.diff(loop, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg)])
    total = cumulative[-1]
    if total <= 0:
        raise ValueError("Degenerate centerline with zero length")

    targets = np.linspace(0.0, total, n, endpoint=False)
    out_x = np.interp(targets, cumulative, loop[:, 0])
    out_y = np.interp(targets, cumulative, loop[:, 1])
    return np.column_stack([out_x, out_y])


def _constant_width_offsets(centerline: np.ndarray, half_width: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (outer, inner) boundaries offset perpendicular to the centerline.

    Each boundary point is placed exactly ``half_width`` along the local unit normal (from a
    central-difference tangent), so the corridor has a **constant** perpendicular width of
    ``2 * half_width`` everywhere. ``outer`` is the side facing away from the loop centroid. For a
    smooth centerline whose curvature radius exceeds ``half_width``, neither boundary self-intersects.
    """
    centroid = centerline.mean(axis=0)
    nxt = np.roll(centerline, -1, axis=0)
    prv = np.roll(centerline, 1, axis=0)

    tangent = nxt - prv  # central difference -> smooth local direction
    length = np.linalg.norm(tangent, axis=1, keepdims=True)
    length = np.where(length < 1e-9, 1.0, length)
    unit_t = tangent / length
    # Left-hand unit normal of the tangent.
    normal = np.column_stack([-unit_t[:, 1], unit_t[:, 0]])

    # Orient every normal to point away from the centroid (outward), so width stays uniform.
    outward_sign = np.sign(np.sum(normal * (centerline - centroid), axis=1, keepdims=True))
    outward_sign = np.where(outward_sign == 0, 1.0, outward_sign)
    normal = normal * outward_sign

    outer = centerline + half_width * normal
    inner = centerline - half_width * normal
    return outer, inner


def build_annulus_track(
    track_id: str,
    name: str,
    centerline: np.ndarray,
    half_width: float,
    *,
    n_checkpoints: int = 10,
    samples: int = 400,
    difficulty: str = "medium",
    scale: float = 1.0,
) -> dict:
    """Build a track definition dict (ready to JSON-dump) from a closed centerline.

    ``centerline`` is the loop shape (open: first point not repeated). ``half_width`` is
    half the corridor width. ``n_checkpoints`` gates are placed in travel order between
    start/finish lines; the start/finish line sits at the ``s = 0`` sample.
    """
    if half_width <= 0:
        raise ValueError("half_width must be positive")
    if n_checkpoints < 1:
        raise ValueError("n_checkpoints must be >= 1")

    dense = resample_closed(centerline, samples)
    # Ensure counter-clockwise so travel direction (increasing index) is consistent.
    if _polygon_area(dense) < 0:
        dense = dense[::-1].copy()

    outer, inner = _constant_width_offsets(dense, half_width)

    n_gates = n_checkpoints + 1  # gate 0 is the start/finish line
    gate_indices = [round(k * samples / n_gates) % samples for k in range(n_gates)]

    def gate_at(idx: int) -> list[list[float]]:
        return [inner[idx].tolist(), outer[idx].tolist()]

    start_finish = gate_at(gate_indices[0])
    checkpoints = [gate_at(idx) for idx in gate_indices[1:]]

    spawn_idx = gate_indices[0]
    tangent = dense[(spawn_idx + 1) % samples] - dense[spawn_idx]
    heading = float(np.arctan2(tangent[1], tangent[0]))

    centerline_closed = np.vstack([dense, dense[0]])

    return {
        "track_id": track_id,
        "name": name,
        "scale": float(scale),
        "outer_boundary": outer.tolist(),
        "inner_boundary": inner.tolist(),
        "checkpoints": checkpoints,
        "start_finish": start_finish,
        "spawn_pose": {
            "x": float(dense[spawn_idx][0]),
            "y": float(dense[spawn_idx][1]),
            "heading": heading,
        },
        "metadata": {
            "difficulty": difficulty,
            "half_width": float(half_width),
            "generated": True,
            "centerline": centerline_closed.tolist(),
        },
    }

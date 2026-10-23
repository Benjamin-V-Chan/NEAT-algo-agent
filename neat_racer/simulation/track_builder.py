from __future__ import annotations

import numpy as np

__all__ = ["resample_closed", "build_annulus_track"]


def _polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def resample_closed(points: np.ndarray, n: int) -> np.ndarray:
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
    nxt = np.roll(centerline, -1, axis=0)
    prv = np.roll(centerline, 1, axis=0)

    tangent = nxt - prv  # central difference -> smooth local direction
    length = np.linalg.norm(tangent, axis=1, keepdims=True)
    length = np.where(length < 1e-9, 1.0, length)
    unit_t = tangent / length
    # Outward unit normal: the right-hand normal for a counter-clockwise loop, the left-hand one for
    # a clockwise loop. Using the winding (not the centroid) keeps deep lobes/hairpins correct.
    normal = np.column_stack([unit_t[:, 1], -unit_t[:, 0]])
    if _polygon_area(centerline) < 0:
        normal = -normal

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
    clockwise: bool = False,
) -> dict:
    if half_width <= 0:
        raise ValueError("half_width must be positive")
    if n_checkpoints < 1:
        raise ValueError("n_checkpoints must be >= 1")

    dense = resample_closed(centerline, samples)
    # Ensure counter-clockwise so travel direction (increasing index) is consistent.
    if _polygon_area(dense) < 0:
        dense = dense[::-1].copy()
    if clockwise:
        # Same shape driven the other way round: travel direction follows increasing index.
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

"""Headless (matplotlib) rendering of a track definition.

Unlike :mod:`neat_racer.visualization.renderer`, this module has no Pygame dependency,
so it works in a core install. It draws the corridor, numbered checkpoint gates, the
start/finish line, the centerline and the spawn pose, for quick visual inspection.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from neat_racer.simulation.track import Track  # noqa: E402

__all__ = ["render_track"]


def render_track(track: Track, path: str | Path) -> Path:
    """Render ``track`` to a PNG at ``path`` and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 7), dpi=130)

    outer = np.vstack([track.outer_boundary, track.outer_boundary[0]])
    inner = np.vstack([track.inner_boundary, track.inner_boundary[0]])
    ax.plot(outer[:, 0], outer[:, 1], color="black", linewidth=2)
    ax.plot(inner[:, 0], inner[:, 1], color="black", linewidth=2)

    centerline = track.metadata.get("centerline")
    if centerline:
        cl = np.asarray(centerline, dtype=float)
        ax.plot(cl[:, 0], cl[:, 1], color="#95a5a6", linewidth=0.8, linestyle="--")

    for i, cp in enumerate(track.checkpoints):
        ax.plot([cp[0][0], cp[1][0]], [cp[0][1], cp[1][1]], color="#3498db", linewidth=1.0)
        center = (np.asarray(cp[0]) + np.asarray(cp[1])) * 0.5
        ax.annotate(str(i), center, color="#2980b9", fontsize=7, ha="center", va="center")

    sf = track.start_finish
    ax.plot([sf[0][0], sf[1][0]], [sf[0][1], sf[1][1]], color="#e74c3c", linewidth=2.5, label="start/finish")

    spawn = track.spawn_pos
    heading = track.spawn_heading
    ax.plot(spawn[0], spawn[1], marker="o", color="#27ae60", markersize=6)
    ax.arrow(
        spawn[0], spawn[1], 35.0 * np.cos(heading), 35.0 * np.sin(heading),
        head_width=10, head_length=12, fc="#27ae60", ec="#27ae60", length_includes_head=True,
    )

    ax.set_aspect("equal")
    ax.set_title(f"{track.name}  ({len(track.checkpoints)} checkpoints)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path

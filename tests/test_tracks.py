from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from neat_racer.simulation.track import Track, load_track
from neat_racer.simulation.track_builder import build_annulus_track, resample_closed
from neat_racer.simulation.track_validator import has_errors, validate_track

BUNDLED_TRACKS = ["baseline_loop", "oval_speedway", "twisty_circuit", "serpentine_loop", "technical_coil"]


def _ellipse_centerline(n: int = 160, a: float = 300.0, b: float = 200.0) -> np.ndarray:
    t = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([450.0 + a * np.cos(t), 360.0 + b * np.sin(t)])


def test_resample_closed_even_spacing() -> None:
    pts = _ellipse_centerline(n=37)
    out = resample_closed(pts, 120)
    assert out.shape == (120, 2)
    loop = np.vstack([out, out[0]])
    seg = np.linalg.norm(np.diff(loop, axis=0), axis=1)
    # Even arc-length spacing: segment lengths cluster tightly around the mean.
    assert seg.std() / seg.mean() < 0.05


def test_build_annulus_track_is_valid(tmp_path: Path) -> None:
    payload = build_annulus_track(
        "unit_oval", "Unit Oval", _ellipse_centerline(), half_width=60.0, n_checkpoints=8
    )
    assert payload["track_id"] == "unit_oval"
    assert len(payload["checkpoints"]) == 8

    path = tmp_path / "unit_oval.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    track = load_track(path)
    issues = validate_track(track)
    assert not has_errors(issues), issues

    # Spawn sits near the corridor centre, well clear of the walls.
    assert track.is_inside_corridor(track.spawn_pos)
    assert not track.collision_at(track.spawn_pos, 7.5)


def test_build_annulus_track_rejects_bad_width() -> None:
    with pytest.raises(ValueError):
        build_annulus_track("x", "x", _ellipse_centerline(), half_width=0.0)


@pytest.mark.parametrize("track_id", BUNDLED_TRACKS)
def test_bundled_tracks_have_no_errors(track_id: str) -> None:
    path = Path("tracks") / f"{track_id}.json"
    assert path.exists(), f"missing bundled track {path}"
    issues = validate_track(load_track(path))
    assert not has_errors(issues), f"{track_id}: {[str(i) for i in issues]}"


def test_validator_flags_spawn_in_wall() -> None:
    outer = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]])
    inner = np.array([[30.0, 30.0], [70.0, 30.0], [70.0, 70.0], [30.0, 70.0]])
    bad = Track(
        track_id="bad",
        name="bad",
        scale=1.0,
        outer_boundary=outer,
        inner_boundary=inner,
        checkpoints=[(np.array([20.0, 20.0]), np.array([20.0, 80.0]))],
        start_finish=(np.array([20.0, 20.0]), np.array([20.0, 80.0])),
        spawn_pos=np.array([50.0, 50.0]),  # inside the inner hole -> not drivable
        spawn_heading=0.0,
        metadata={},
    )
    issues = validate_track(bad)
    assert has_errors(issues)
    assert any("spawn" in str(i) for i in issues)

"""Generate the bundled procedural tracks and validate them.

Run from the repo root:

    python tools/generate_tracks.py            # regenerate tracks/*.json
    python tools/generate_tracks.py --check     # validate only, don't overwrite

Tracks are built from a closed centerline via
:func:`neat_racer.simulation.track_builder.build_annulus_track`, then checked with
:func:`neat_racer.simulation.track_validator.validate_track`. The twisty tracks use a base ellipse
modulated by several radial harmonics: because the radius stays positive and the curve is
star-shaped about its center, the loop never self-intersects, so we get lots of twists and turns
in a thin corridor that still validates. The script fails loudly on any validation error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from neat_racer.simulation.track import load_track
from neat_racer.simulation.track_builder import build_annulus_track
from neat_racer.simulation.track_validator import has_errors, validate_track

TRACKS_DIR = Path("tracks")


def _harmonic_loop(cx, cy, rx, ry, harmonics, n=420):
    """A closed centerline: an ellipse whose radius is modulated by radial harmonics."""
    t = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    radial = np.ones_like(t)
    for k, amp, phase in harmonics:
        radial = radial + amp * np.cos(k * t + phase)
    x = cx + rx * radial * np.cos(t)
    y = cy + ry * radial * np.sin(t)
    return np.column_stack([x, y])


def _oval_speedway() -> dict:
    t = np.linspace(0.0, 2 * np.pi, 240, endpoint=False)
    cx, cy, a, b = 480.0, 360.0, 360.0, 230.0
    centerline = np.column_stack([cx + a * np.cos(t), cy + b * np.sin(t)])
    return build_annulus_track(
        "oval_speedway", "Oval Speedway", centerline, half_width=62.0,
        n_checkpoints=10, difficulty="easy",
    )


def _twisty_circuit() -> dict:
    centerline = _harmonic_loop(
        480, 360, 350, 240,
        harmonics=[(3, 0.11, 0.0), (5, 0.06, 1.1), (7, 0.035, 2.2)],
    )
    return build_annulus_track(
        "twisty_circuit", "Twisty Circuit", centerline, half_width=40.0,
        n_checkpoints=16, samples=360, difficulty="medium",
    )


def _serpentine_loop() -> dict:
    centerline = _harmonic_loop(
        480, 360, 350, 250,
        harmonics=[(4, 0.10, 0.6), (6, 0.035, 0.0)],
    )
    return build_annulus_track(
        "serpentine_loop", "Serpentine Loop", centerline, half_width=32.0,
        n_checkpoints=18, samples=400, difficulty="hard",
    )


def _technical_coil() -> dict:
    centerline = _harmonic_loop(
        480, 360, 345, 255,
        harmonics=[(5, 0.07, 0.0), (8, 0.03, 1.5)],
    )
    return build_annulus_track(
        "technical_coil", "Technical Coil", centerline, half_width=28.0,
        n_checkpoints=22, samples=440, difficulty="hard",
    )


BUILDERS = {
    "oval_speedway": _oval_speedway,
    "twisty_circuit": _twisty_circuit,
    "serpentine_loop": _serpentine_loop,
    "technical_coil": _technical_coil,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and validate procedural tracks")
    parser.add_argument("--check", action="store_true", help="Validate only; do not write files")
    args = parser.parse_args()

    TRACKS_DIR.mkdir(exist_ok=True)
    failures = 0

    for track_id, builder in BUILDERS.items():
        payload = builder()
        path = TRACKS_DIR / f"{track_id}.json"
        if not args.check:
            path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        if not path.exists():
            print(f"[skip] {track_id}: no file on disk to check")
            continue
        track = load_track(path)
        issues = validate_track(track)
        if has_errors(issues):
            status = "FAIL"
            failures += 1
        elif issues:
            status = "warn"
        else:
            status = "ok"
        verb = "checked" if args.check else "wrote"
        print(f"[{status}] {verb} {path}  ({len(track.checkpoints)} checkpoints)")
        for issue in issues:
            print(f"        - {issue}")

    if failures:
        print(f"\n{failures} track(s) failed validation", file=sys.stderr)
        return 1
    print("\nAll tracks valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

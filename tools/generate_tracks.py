"""Generate the bundled procedural tracks and validate them.

Run from the repo root:

    python tools/generate_tracks.py            # regenerate tracks/*.json
    python tools/generate_tracks.py --check     # validate only, don't overwrite

Each track is built from a closed centerline via
:func:`neat_racer.simulation.track_builder.build_annulus_track`, then checked with
:func:`neat_racer.simulation.track_validator.validate_track`. The script fails loudly
if any generated track does not validate.
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


def _oval_speedway() -> dict:
    t = np.linspace(0.0, 2 * np.pi, 220, endpoint=False)
    cx, cy, a, b = 460.0, 360.0, 330.0, 210.0
    centerline = np.column_stack([cx + a * np.cos(t), cy + b * np.sin(t)])
    return build_annulus_track(
        "oval_speedway", "Oval Speedway", centerline, half_width=78.0,
        n_checkpoints=8, difficulty="easy",
    )


def _chicane_loop() -> dict:
    t = np.linspace(0.0, 2 * np.pi, 260, endpoint=False)
    cx, cy, a, b = 460.0, 360.0, 320.0, 215.0
    # Lobed radius -> gentle S-curves along the loop, still star-convex.
    radial = 1.0 + 0.11 * np.cos(3.0 * t)
    centerline = np.column_stack([cx + a * radial * np.cos(t), cy + b * radial * np.sin(t)])
    return build_annulus_track(
        "chicane_loop", "Chicane Loop", centerline, half_width=52.0,
        n_checkpoints=12, difficulty="hard",
    )


def _hairpin_loop() -> dict:
    t = np.linspace(0.0, 2 * np.pi, 260, endpoint=False)
    cx, cy, a, b = 470.0, 360.0, 330.0, 200.0
    # Teardrop: the +x end is pinched tighter than the -x end -> a hairpin.
    squash = 1.0 - 0.33 * np.cos(t)
    centerline = np.column_stack([cx + a * np.cos(t), cy + b * squash * np.sin(t)])
    return build_annulus_track(
        "hairpin_loop", "Hairpin Loop", centerline, half_width=46.0,
        n_checkpoints=11, difficulty="hard",
    )


BUILDERS = {
    "oval_speedway": _oval_speedway,
    "chicane_loop": _chicane_loop,
    "hairpin_loop": _hairpin_loop,
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

        # Load through the real loader + validator so we test the on-disk shape.
        source = path if path.exists() else None
        if source is None:
            # --check with no file yet: write to a temp string and skip disk.
            print(f"[skip] {track_id}: no file on disk to check")
            continue
        track = load_track(source)
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

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
# Held-out tracks for the NEAT-vs-PPO study: never trained on, evaluation only.
HOLDOUT_DIR = TRACKS_DIR / "holdout"
HARD_HOLDOUT_DIR = HOLDOUT_DIR / "hard"


def _harmonic_loop(cx, cy, rx, ry, harmonics, n=420):
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
    # Several harmonics at different scales -> multiple turns in and out, still constant width.
    centerline = _harmonic_loop(
        480, 360, 340, 260,
        harmonics=[(3, 0.07, 0.0), (5, 0.05, 0.9), (7, 0.032, 2.1), (9, 0.018, 0.5)],
    )
    return build_annulus_track(
        "technical_coil", "Technical Coil", centerline, half_width=25.0,
        n_checkpoints=24, samples=480, difficulty="hard",
    )


def _holdout_ripple() -> dict:
    centerline = _harmonic_loop(
        480, 360, 330, 250,
        harmonics=[(4, 0.08, 1.3), (6, 0.05, 0.4), (8, 0.025, 2.7)],
    )
    return build_annulus_track(
        "holdout_ripple", "Holdout Ripple", centerline, half_width=34.0,
        n_checkpoints=20, samples=420, difficulty="medium",
    )


def _holdout_kidney() -> dict:
    centerline = _harmonic_loop(
        480, 360, 350, 230,
        harmonics=[(2, 0.12, 0.8), (3, 0.08, 2.0), (5, 0.04, 0.3)],
    )
    return build_annulus_track(
        "holdout_kidney", "Holdout Kidney", centerline, half_width=30.0,
        n_checkpoints=18, samples=420, difficulty="hard",
    )


# Out-of-family held-out candidates for study 2: each departs from the training family on at least
# one axis it never covers - clockwise travel, corridors narrower than the narrowest training track
# (half-width 25), or deep lobes that force hairpins. Which ones are held out is decided by a blinded
# difficulty calibration (tools/calibrate_holdouts.py); see HARD_HOLDOUT_SELECTED.
def _cand(track_id, name, harmonics, half_width, n_checkpoints, *, rx=340, ry=250, clockwise=False):
    def build() -> dict:
        centerline = _harmonic_loop(480, 360, rx, ry, harmonics=harmonics, n=480)
        return build_annulus_track(
            track_id, name, centerline, half_width=half_width, n_checkpoints=n_checkpoints,
            samples=480, difficulty="hard", clockwise=clockwise,
        )

    return build


def _chaikin_closed(points, iterations: int) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    for _ in range(iterations):
        nxt = np.roll(pts, -1, axis=0)
        pts = np.column_stack([0.75 * pts + 0.25 * nxt, 0.25 * pts + 0.75 * nxt]).reshape(-1, 2)
    return pts


def _waypoint_cand(track_id, name, waypoints, half_width, n_checkpoints, *, clockwise=False):
    # Polyline circuits rounded by corner cutting; used where harmonic loops can't make real hairpins.
    def build() -> dict:
        centerline = _chaikin_closed(waypoints, 5)
        return build_annulus_track(
            track_id, name, centerline, half_width=half_width, n_checkpoints=n_checkpoints,
            samples=520, difficulty="hard", clockwise=clockwise,
        )

    return build


HARD_CANDIDATES = {
    "hard_cw_twisty": _cand(
        "hard_cw_twisty", "Hard CW Twisty", [(3, 0.10, 0.4), (5, 0.06, 1.7), (7, 0.03, 0.9)], 30.0, 18,
        clockwise=True,
    ),
    "hard_cw_coil": _cand(
        "hard_cw_coil", "Hard CW Coil", [(4, 0.07, 2.2), (6, 0.05, 0.3), (9, 0.022, 1.4)], 25.0, 24,
        clockwise=True,
    ),
    "hard_narrow_loop": _cand(
        "hard_narrow_loop", "Hard Narrow Loop", [(3, 0.08, 1.0), (5, 0.04, 2.5)], 19.0, 20,
    ),
    "hard_narrow_wiggle": _cand(
        "hard_narrow_wiggle", "Hard Narrow Wiggle", [(4, 0.06, 0.2), (7, 0.035, 1.9), (10, 0.015, 0.6)], 21.0, 24,
    ),
    # One deep inward finger: two U-turns of roughly 80 px radius.
    "hard_hairpin_switchback": _waypoint_cand(
        "hard_hairpin_switchback", "Hard Hairpin Switchback",
        [(150, 140), (810, 140), (810, 580), (580, 580), (580, 300), (400, 300), (400, 580), (150, 580)],
        28.0, 22,
    ),
    # Two fingers from opposite sides, driven clockwise.
    "hard_cw_double_hairpin": _waypoint_cand(
        "hard_cw_double_hairpin", "Hard CW Double Hairpin",
        [(100, 120), (280, 120), (280, 420), (480, 420), (480, 120), (960, 120), (960, 620), (800, 620),
         (800, 320), (600, 320), (600, 620), (100, 620)],
        28.0, 24, clockwise=True,
    ),
    "hard_cw_hairpin": _cand(
        "hard_cw_hairpin", "Hard CW Hairpin", [(3, 0.24, 2.0), (6, 0.04, 0.7)], 28.0, 22, clockwise=True,
    ),
    "hard_narrow_cw_chicane": _cand(
        "hard_narrow_cw_chicane", "Hard Narrow CW Chicane", [(5, 0.07, 0.8), (8, 0.03, 2.4)], 22.0, 24,
        clockwise=True,
    ),
}

# Filled in from the calibration result; these are written to tracks/holdout/hard/.
HARD_HOLDOUT_SELECTED: list[str] = [
    "hard_narrow_cw_chicane", "hard_cw_coil", "hard_cw_hairpin", "hard_cw_twisty",
]


BUILDERS = {
    "oval_speedway": _oval_speedway,
    "twisty_circuit": _twisty_circuit,
    "serpentine_loop": _serpentine_loop,
    "technical_coil": _technical_coil,
}

HOLDOUT_BUILDERS = {
    "holdout_ripple": _holdout_ripple,
    "holdout_kidney": _holdout_kidney,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and validate procedural tracks")
    parser.add_argument("--check", action="store_true", help="Validate only; do not write files")
    parser.add_argument(
        "--candidates-dir", type=Path, default=None,
        help="Also write every hard held-out candidate into this directory (for calibration)",
    )
    args = parser.parse_args()

    TRACKS_DIR.mkdir(exist_ok=True)
    HOLDOUT_DIR.mkdir(exist_ok=True)
    HARD_HOLDOUT_DIR.mkdir(exist_ok=True)
    failures = 0

    targets = [
        (TRACKS_DIR, BUILDERS),
        (HOLDOUT_DIR, HOLDOUT_BUILDERS),
        (HARD_HOLDOUT_DIR, {k: HARD_CANDIDATES[k] for k in HARD_HOLDOUT_SELECTED}),
    ]
    if args.candidates_dir is not None:
        args.candidates_dir.mkdir(parents=True, exist_ok=True)
        targets.append((args.candidates_dir, HARD_CANDIDATES))
    for track_id, builder, out_dir in [(t, b, d) for d, builders in targets for t, b in builders.items()]:
        payload = builder()
        path = out_dir / f"{track_id}.json"
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

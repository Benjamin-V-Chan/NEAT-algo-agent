from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from neat_racer.simulation.track import load_track
from neat_racer.simulation.track_validator import corridor_span, has_errors, validate_track


def _snap(track, a, b) -> list[list[float]]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    span = corridor_span(track, a, b)
    if span is None:
        raise ValueError(f"gate {a.tolist()} -> {b.tolist()} does not lie across the corridor")
    t0, t1 = span
    return [np.round(a + (b - a) * t0, 2).tolist(), np.round(a + (b - a) * t1, 2).tolist()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stretch/trim every checkpoint and the start/finish gate so it spans the corridor "
        "wall to wall along its own line (keeps each gate's position and angle)."
    )
    parser.add_argument("track", type=Path)
    parser.add_argument("--check", action="store_true", help="Report only; do not rewrite the file")
    parser.add_argument(
        "--recenter-centerline", action="store_true",
        help="Replace metadata.centerline with the loop through the snapped gates' midpoints",
    )
    args = parser.parse_args()

    track = load_track(args.track)
    payload = json.loads(args.track.read_text(encoding="utf-8"))
    payload["checkpoints"] = [_snap(track, a, b) for a, b in track.checkpoints]
    payload["start_finish"] = _snap(track, *track.start_finish)
    if args.recenter_centerline:
        gates = [payload["start_finish"], *payload["checkpoints"], payload["start_finish"]]
        payload.setdefault("metadata", {})["centerline"] = [
            np.round((np.asarray(a) + np.asarray(b)) * 0.5, 2).tolist() for a, b in gates
        ]

    if not args.check:
        args.track.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        track = load_track(args.track)
    issues = validate_track(track)
    for issue in issues:
        print(f"  - {issue}")
    print(f"{'checked' if args.check else 'snapped'} {args.track}: {'FAIL' if has_errors(issues) else 'ok'}")
    return 1 if has_errors(issues) else 0


if __name__ == "__main__":
    sys.exit(main())

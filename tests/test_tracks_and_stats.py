from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neat_racer.config import dump_config_json, load_experiment_config, training_track_files
from neat_racer.simulation.track import load_track
from neat_racer.simulation.track_builder import _polygon_area, build_annulus_track
from neat_racer.simulation.track_validator import corridor_span, gate_wall_gaps, has_errors, validate_track
from neat_racer.study.evaluation import EVAL_COLUMNS, start_variants
from neat_racer.study.report import build_report, cliffs_delta, holm, mann_whitney_exact

ALL_TRACKS = sorted(Path("tracks").rglob("*.json"))


@pytest.mark.parametrize("path", ALL_TRACKS, ids=lambda p: p.stem)
def test_every_bundled_track_is_valid_and_gates_span_the_corridor(path) -> None:
    track = load_track(path)
    assert not has_errors(validate_track(track))
    assert all(max(a, b) <= 0.5 for _name, a, b in gate_wall_gaps(track))


def test_short_gate_is_rejected_and_snapping_restores_it() -> None:
    track = load_track("tracks/oval_speedway.json")
    a, b = track.checkpoints[3]
    mid = (a + b) * 0.5
    short = (mid + (a - mid) * 0.3, mid + (b - mid) * 0.3)  # gate covers only 30% of the corridor
    broken = dataclasses.replace(track, checkpoints=[*track.checkpoints[:3], short, *track.checkpoints[4:]])
    assert has_errors(validate_track(broken))

    t0, t1 = corridor_span(broken, *short)
    snapped = (short[0] + (short[1] - short[0]) * t0, short[0] + (short[1] - short[0]) * t1)
    np.testing.assert_allclose(snapped, (a, b), atol=1e-6)


def test_clockwise_build_reverses_travel_direction() -> None:
    t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    centerline = np.column_stack([400 + 250 * np.cos(t), 300 + 180 * np.sin(t)])
    ccw = build_annulus_track("ccw", "ccw", centerline, half_width=30.0)
    cw = build_annulus_track("cw", "cw", centerline, half_width=30.0, clockwise=True)
    assert _polygon_area(np.asarray(ccw["metadata"]["centerline"])[:-1]) > 0
    assert _polygon_area(np.asarray(cw["metadata"]["centerline"])[:-1]) < 0
    # Same corridor either way: the outer wall stays outside.
    assert abs(abs(_polygon_area(np.asarray(cw["outer_boundary"]))) -
               abs(_polygon_area(np.asarray(ccw["outer_boundary"])))) < 1e-6


def test_start_variants_are_deterministic_nominal_first_and_clear_of_walls() -> None:
    cfg = load_experiment_config("configs/neat_vs_ppo_v2.yaml")
    track = load_track("tracks/holdout/hard/hard_narrow_cw_chicane.json")
    first = start_variants(cfg, track, 5)
    second = start_variants(cfg, track, 5)
    assert first[0] is track
    assert len(first) == 5
    for x, y in zip(first, second, strict=True):
        assert np.array_equal(x.spawn_pos, y.spawn_pos) and x.spawn_heading == y.spawn_heading
    for variant in first[1:]:
        assert not variant.collision_at(variant.spawn_pos, cfg.car.collision_radius)
        assert abs(variant.spawn_heading - track.spawn_heading) <= np.radians(cfg.study.start_heading_jitter_deg)


def test_v2_config_split_is_disjoint_and_preregistered() -> None:
    cfg = load_experiment_config("configs/neat_vs_ppo_v2.yaml")
    train = set(training_track_files(cfg))
    assert len(train) == 5
    assert len(cfg.study.hard_holdout_tracks) >= 2
    assert not train & set(cfg.study.holdout_tracks) and not train & set(cfg.study.hard_holdout_tracks)
    assert len(cfg.study.seeds) == 10 and not set(cfg.study.seeds) & {1, 2, 3, 4, 5}
    assert cfg.study.primary_metrics == ["auc_train_completion", "final_holdout_hard_lap_completed"]


def test_exact_mann_whitney_known_values() -> None:
    # Complete separation: only 2 of C(2n, n) relabellings are as extreme.
    assert mann_whitney_exact([1, 2, 3], [4, 5, 6]) == pytest.approx(2 / 20)
    assert mann_whitney_exact(range(10), range(10, 20)) == pytest.approx(2 / 184756)
    assert mann_whitney_exact([1, 1, 1], [1, 1, 1]) == 1.0
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert cliffs_delta([2, 3], [1, 2]) == pytest.approx(0.75)


def test_generate_tracks_reproduces_bundled_files(tmp_path, monkeypatch) -> None:
    sys.path.insert(0, "tools")
    import generate_tracks

    for builders, folder in ((generate_tracks.BUILDERS, "tracks"),
                             (generate_tracks.HOLDOUT_BUILDERS, "tracks/holdout")):
        for track_id, builder in builders.items():
            payload = builder()
            on_disk = load_track(Path(folder) / f"{track_id}.json")
            np.testing.assert_allclose(np.asarray(payload["outer_boundary"]), on_disk.outer_boundary)
    for track_id in generate_tracks.HARD_HOLDOUT_SELECTED:
        payload = generate_tracks.HARD_CANDIDATES[track_id]()
        on_disk = load_track(Path("tracks/holdout/hard") / f"{track_id}.json")
        np.testing.assert_allclose(np.asarray(payload["inner_boundary"]), on_disk.inner_boundary)


def test_report_curves_and_auc_use_nominal_start_only(tmp_path) -> None:
    cfg = load_experiment_config("configs/neat_vs_ppo_v2.yaml")
    cfg.runtime.max_env_steps = 100
    cfg.study.eval_points = 2
    cfg.study.hard_holdout_tracks = []
    cfg.study.holdout_tracks = []
    cfg.simulation.train_tracks = ["tracks/oval_speedway.json"]
    dump_config_json(cfg, tmp_path / "study_config.json")
    for method in ("neat", "ppo"):
        rows = [{"method": method, "seed": 1, "env_steps": steps, "train_time_sec": steps, "mark": steps,
                 "final": 0, "split": "train", "track_id": "oval_speedway", "start": 0, "lap_completed": 1}
                for steps in (0, 50)]
        # Final policy: laps from the nominal start, fails from both perturbed starts.
        rows += [{"method": method, "seed": 1, "env_steps": 100, "train_time_sec": 100, "final": 1,
                  "split": "train", "track_id": "oval_speedway", "start": start, "lap_completed": int(start == 0)}
                 for start in range(3)]
        run_dir = tmp_path / "runs" / method / f"x_{method}-seed-1"
        run_dir.mkdir(parents=True)
        frame = pd.DataFrame(rows, columns=EVAL_COLUMNS).fillna({"crashed": 0, "max_progress": 0.0})
        frame.to_csv(run_dir / "eval_curve.csv", index=False)

    build_report(tmp_path)
    finals = pd.read_csv(tmp_path / "final_by_seed.csv")
    assert finals["auc_train_completion"].tolist() == [1.0, 1.0]
    assert finals["final_train_lap_completed"].tolist() == pytest.approx([1 / 3, 1 / 3])

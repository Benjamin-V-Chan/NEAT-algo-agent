from __future__ import annotations

import copy
import dataclasses
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neat_racer.config import ExperimentConfig, training_track_files
from neat_racer.evolution.fitness import CurriculumScheduler, FitnessAccumulator
from neat_racer.simulation.env import ControllerFn, PolicySnapshot, run_episode
from neat_racer.simulation.track import Track, load_track

EVAL_COLUMNS = [
    "method",
    "seed",
    "env_steps",
    "train_time_sec",
    "mark",
    "final",
    "split",
    "track_id",
    "start",
    "laps_completed",
    "lap_completed",
    "best_lap_time",
    "median_lap_time",
    "crashed",
    "termination",
    "max_progress",
    "eval_steps",
]


def eval_marks(cfg: ExperimentConfig) -> list[int]:
    budget = cfg.runtime.max_env_steps
    k = cfg.study.eval_points
    return [round(i * budget / k) for i in range(k + 1)]


def study_tracks(cfg: ExperimentConfig) -> list[tuple[str, Track]]:
    train = [("train", load_track(path)) for path in training_track_files(cfg)]
    holdout = [("holdout", load_track(path)) for path in cfg.study.holdout_tracks]
    hard = [("holdout_hard", load_track(path)) for path in cfg.study.hard_holdout_tracks]
    return train + holdout + hard


def start_variants(cfg: ExperimentConfig, track: Track, n: int) -> list[Track]:
    variants = [track]
    if n <= 1:
        return variants
    seed_material = [cfg.study.start_seed, *track.track_id.encode("utf-8")]
    rng = np.random.default_rng(seed_material)
    a, b = track.start_finish
    half_len = float(np.linalg.norm(b - a)) * 0.5
    along = (b - a) / max(2.0 * half_len, 1e-9)
    max_shift = cfg.study.start_lateral_jitter * half_len
    max_turn = np.radians(cfg.study.start_heading_jitter_deg)
    margin = 1.5 * cfg.car.collision_radius
    while len(variants) < n:
        pos = track.spawn_pos + along * rng.uniform(-max_shift, max_shift)
        heading = track.spawn_heading + rng.uniform(-max_turn, max_turn)
        if not track.collision_at(pos, margin):
            variants.append(dataclasses.replace(track, spawn_pos=pos, spawn_heading=float(heading)))
    return variants


def evaluate_controller(
    cfg: ExperimentConfig,
    tracks: list[tuple[str, Track]],
    make_controller: Callable[[], ControllerFn],
    starts: int = 1,
) -> list[dict[str, Any]]:
    eval_cfg = copy.deepcopy(cfg)
    eval_cfg.simulation.max_steps_per_episode = cfg.study.eval_max_steps
    eval_cfg.logging.save_per_step = False
    # Fitness is not an evaluation metric, but run_episode needs a step function.
    weights = CurriculumScheduler(cfg.fitness).current_weights()

    rows = []
    for split, base_track, start, track in [
        (split, base, i, variant)
        for split, base in tracks
        for i, variant in enumerate(start_variants(cfg, base, starts))
    ]:
        result = run_episode(eval_cfg, track, 0, 0, make_controller(), FitnessAccumulator(weights).step)
        lap_times = [float(e["event_value"]) for e in result.events if e["event_type"] == "lap_completed"]
        termination = next(
            (str(e["event_value"]) for e in result.events if e["event_type"] == "termination"), "time_limit"
        )
        rows.append(
            {
                "split": split,
                "track_id": base_track.track_id,
                "start": start,
                "laps_completed": result.laps_completed,
                "lap_completed": int(result.laps_completed > 0),
                "best_lap_time": result.best_lap_time,
                "median_lap_time": statistics.median(lap_times) if lap_times else None,
                "crashed": int(result.crash_flag),
                "termination": termination,
                "max_progress": result.max_progress,
                "eval_steps": result.env_steps,
            }
        )
    return rows


class StudyEvaluator:

    def __init__(self, cfg: ExperimentConfig, method: str, seed: int, out_dir: Path) -> None:
        self.cfg = cfg
        self.method = method
        self.seed = seed
        self.path = Path(out_dir) / "eval_curve.csv"
        self.budget = cfg.runtime.max_env_steps
        self.marks = eval_marks(cfg)
        self.next_mark = 0
        self.tracks = study_tracks(cfg)
        self.last_snapshot: PolicySnapshot | None = None

    def __call__(self, snapshot: PolicySnapshot) -> None:
        if snapshot.env_steps > self.budget:
            return
        self.last_snapshot = snapshot
        crossed = None
        while self.next_mark < len(self.marks) and self.marks[self.next_mark] <= snapshot.env_steps:
            crossed = self.marks[self.next_mark]
            self.next_mark += 1
        if crossed is not None:
            self._evaluate(snapshot, mark=crossed, final=False, starts=self.cfg.study.curve_eval_starts)

    def finalize(self) -> None:
        if self.last_snapshot is not None:
            self._evaluate(self.last_snapshot, mark=None, final=True, starts=self.cfg.study.final_eval_starts)

    def _evaluate(self, snapshot: PolicySnapshot, mark: int | None, final: bool, starts: int) -> None:
        rows = evaluate_controller(self.cfg, self.tracks, snapshot.make_controller, starts=starts)
        meta = {
            "method": self.method,
            "seed": self.seed,
            "env_steps": snapshot.env_steps,
            "train_time_sec": snapshot.train_time_sec,
            "mark": mark,
            "final": int(final),
        }
        frame = pd.DataFrame([{**meta, **row} for row in rows], columns=EVAL_COLUMNS)
        frame.to_csv(self.path, mode="a", header=not self.path.exists(), index=False)

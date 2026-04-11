"""Checkpoint/lap progression logic with anti-skip behavior."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neat_racer.config import ProgressSection
from neat_racer.simulation.geometry import segment_intersection
from neat_racer.simulation.track import Track
from neat_racer.utils.math_utils import clamp


@dataclass(slots=True)
class ProgressState:
    next_checkpoint_idx: int = 0
    checkpoints_passed_total: int = 0
    checkpoints_passed_current_cycle: int = 0
    ready_for_finish: bool = False
    laps_completed: int = 0
    lap_start_time: float = 0.0
    best_lap_time: float | None = None
    latest_lap_time: float | None = None


def update_progress(
    track: Track,
    progress_state: ProgressState,
    prev_pos: np.ndarray,
    curr_pos: np.ndarray,
    sim_time: float,
    cfg: ProgressSection,
) -> tuple[float, dict[str, float | int | bool]]:
    events: dict[str, float | int | bool] = {}
    num_checkpoints = len(track.checkpoints)

    next_gate = track.checkpoints[progress_state.next_checkpoint_idx]
    crossed_checkpoint, _ = segment_intersection(prev_pos, curr_pos, next_gate[0], next_gate[1])
    if crossed_checkpoint:
        progress_state.checkpoints_passed_total += 1
        progress_state.checkpoints_passed_current_cycle += 1
        progress_state.next_checkpoint_idx = (progress_state.next_checkpoint_idx + 1) % num_checkpoints
        events["checkpoint_crossed"] = True
        events["checkpoint_index"] = progress_state.next_checkpoint_idx

        if progress_state.checkpoints_passed_current_cycle >= num_checkpoints:
            progress_state.ready_for_finish = True

    lap_completed_now = False
    crossed_start, _ = segment_intersection(prev_pos, curr_pos, track.start_finish[0], track.start_finish[1])
    if crossed_start and (
        not cfg.require_start_finish_after_checkpoints or progress_state.ready_for_finish
    ):
        if progress_state.ready_for_finish:
            lap_completed_now = True
            progress_state.laps_completed += 1
            lap_time = sim_time - progress_state.lap_start_time
            progress_state.latest_lap_time = lap_time
            if progress_state.best_lap_time is None or lap_time < progress_state.best_lap_time:
                progress_state.best_lap_time = lap_time
            progress_state.lap_start_time = sim_time
            progress_state.ready_for_finish = False
            progress_state.checkpoints_passed_current_cycle = 0
            progress_state.next_checkpoint_idx = 0
            events["lap_completed"] = True
            events["lap_time"] = lap_time

    progress_scalar = _compute_progress_scalar(track, progress_state, curr_pos)
    events["lap_completed_now"] = lap_completed_now
    return progress_scalar, events


def _compute_progress_scalar(track: Track, state: ProgressState, curr_pos: np.ndarray) -> float:
    n = len(track.checkpoints)
    if n == 0:
        return float(state.laps_completed)

    passed = state.checkpoints_passed_current_cycle

    if passed == 0:
        start = track.start_center
        end = track.checkpoint_centers[0]
        frac = _segment_fraction(start, end, curr_pos)
        units = frac
    elif passed >= n:
        start = track.checkpoint_centers[-1]
        end = track.start_center
        frac = _segment_fraction(start, end, curr_pos)
        units = (n - 1) + frac
    else:
        start = track.checkpoint_centers[passed - 1]
        end = track.checkpoint_centers[passed]
        frac = _segment_fraction(start, end, curr_pos)
        units = passed + frac

    progress_in_lap = clamp(units / n, 0.0, 0.999999)
    return float(state.laps_completed + progress_in_lap)


def _segment_fraction(start: np.ndarray, end: np.ndarray, point: np.ndarray) -> float:
    vec = end - start
    denom = float(np.dot(vec, vec))
    if denom < 1e-9:
        return 0.0
    frac = float(np.dot(point - start, vec) / denom)
    return clamp(frac, 0.0, 1.0)

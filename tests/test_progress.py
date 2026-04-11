from __future__ import annotations

import numpy as np

from neat_racer.config import ProgressSection
from neat_racer.simulation.progress import ProgressState, update_progress


def test_checkpoint_progress_in_order(simple_track) -> None:
    state = ProgressState()
    cfg = ProgressSection()

    prev = np.array([10.0, 40.0])
    curr = np.array([25.0, 40.0])
    progress, events = update_progress(simple_track, state, prev, curr, sim_time=0.1, cfg=cfg)

    assert events.get("checkpoint_crossed")
    assert state.checkpoints_passed_total == 1
    assert progress > 0.0


def test_out_of_order_checkpoint_not_counted(simple_track) -> None:
    state = ProgressState()
    cfg = ProgressSection()

    prev = np.array([50.0, 90.0])
    curr = np.array([90.0, 90.0])
    _, events = update_progress(simple_track, state, prev, curr, sim_time=0.1, cfg=cfg)

    assert not events.get("checkpoint_crossed", False)
    assert state.checkpoints_passed_total == 0

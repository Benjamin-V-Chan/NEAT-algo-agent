from __future__ import annotations

import numpy as np
import pytest

from neat_racer.simulation.track import Track


@pytest.fixture()
def simple_track() -> Track:
    outer = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]])
    inner = np.array([[30.0, 30.0], [70.0, 30.0], [70.0, 70.0], [30.0, 70.0]])
    checkpoints = [
        (np.array([20.0, 20.0]), np.array([20.0, 80.0])),
        (np.array([20.0, 80.0]), np.array([80.0, 80.0])),
        (np.array([80.0, 80.0]), np.array([80.0, 20.0])),
        (np.array([80.0, 20.0]), np.array([20.0, 20.0])),
    ]
    start_finish = (np.array([20.0, 20.0]), np.array([20.0, 80.0]))
    return Track(
        track_id="test",
        name="test",
        scale=1.0,
        outer_boundary=outer,
        inner_boundary=inner,
        checkpoints=checkpoints,
        start_finish=start_finish,
        spawn_pos=np.array([25.0, 25.0]),
        spawn_heading=0.0,
        metadata={},
    )

"""Randomness controls for reproducible experiments."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SeedState:
    seed: int


def seed_everything(seed: int) -> SeedState:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return SeedState(seed=seed)

"""Math utility helpers used by simulation and fitness modules."""

from __future__ import annotations

import math


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def clamp01(value: float) -> float:
    return clamp(value, 0.0, 1.0)


def lerp(a: float, b: float, alpha: float) -> float:
    return a + (b - a) * alpha


def angle_wrap(theta: float) -> float:
    while theta > math.pi:
        theta -= 2 * math.pi
    while theta < -math.pi:
        theta += 2 * math.pi
    return theta


def safe_div(n: float, d: float) -> float:
    if abs(d) < 1e-12:
        return 0.0
    return n / d

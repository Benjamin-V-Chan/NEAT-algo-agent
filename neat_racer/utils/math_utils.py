from __future__ import annotations


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def clamp01(value: float) -> float:
    return clamp(value, 0.0, 1.0)

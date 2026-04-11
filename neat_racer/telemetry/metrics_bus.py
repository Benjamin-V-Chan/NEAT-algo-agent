"""In-memory telemetry bus for renderer/HUD overlays."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class MetricsBus:
    run_metrics: dict[str, Any] = field(default_factory=dict)
    generation_metrics: dict[str, Any] = field(default_factory=dict)
    car_metrics: dict[str, Any] = field(default_factory=dict)

    def update_run(self, **kwargs: Any) -> None:
        self.run_metrics.update(kwargs)

    def update_generation(self, **kwargs: Any) -> None:
        self.generation_metrics.update(kwargs)

    def update_car(self, **kwargs: Any) -> None:
        self.car_metrics.update(kwargs)

    def snapshot(self) -> dict[str, Any]:
        return {
            "run": dict(self.run_metrics),
            "generation": dict(self.generation_metrics),
            "car": dict(self.car_metrics),
        }

"""Structured logging of experiment artifacts and telemetry."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import pandas as pd

from neat_racer.config import ExperimentConfig, dump_config_json
from neat_racer.constants import (
    EVENT_COLUMNS,
    GENERATION_SUMMARY_COLUMNS,
    PER_CAR_SUMMARY_COLUMNS,
    PER_STEP_COLUMNS,
)


class ExperimentLogger:
    def __init__(self, config: ExperimentConfig, run_dir: Path) -> None:
        self.config = config
        self.run_dir = run_dir
        self.generation_rows: list[dict[str, Any]] = []
        self.car_rows: list[dict[str, Any]] = []
        self.step_rows: list[dict[str, Any]] = []
        self.event_rows: list[dict[str, Any]] = []
        self.run_summary: dict[str, Any] = {}

    def initialize(self, neat_config_text: str) -> None:
        dump_config_json(self.config, self.run_dir / "experiment_config.json")
        (self.run_dir / "neat_config_used.txt").write_text(neat_config_text, encoding="utf-8")

    def add_generation_row(self, row: dict[str, Any]) -> None:
        self.generation_rows.append(row)

    def add_car_row(self, row: dict[str, Any]) -> None:
        self.car_rows.append(row)

    def add_step_rows(self, rows: list[dict[str, Any]]) -> None:
        self.step_rows.extend(rows)

    def add_event_rows(self, rows: list[dict[str, Any]]) -> None:
        self.event_rows.extend(rows)

    def set_run_summary(self, summary: dict[str, Any]) -> None:
        self.run_summary = summary

    def finalize(self) -> None:
        _write_csv(self.run_dir / "generation_summary.csv", self.generation_rows, GENERATION_SUMMARY_COLUMNS)
        _write_csv(self.run_dir / "per_car_summary.csv", self.car_rows, PER_CAR_SUMMARY_COLUMNS)
        _write_csv(self.run_dir / "events.csv", self.event_rows, EVENT_COLUMNS)

        if self.config.logging.save_per_step and self.step_rows:
            self._write_steps()

        (self.run_dir / "run_summary.json").write_text(
            json.dumps(self.run_summary, indent=2),
            encoding="utf-8",
        )

    def _write_steps(self) -> None:
        frame = pd.DataFrame(self.step_rows)
        for c in PER_STEP_COLUMNS:
            if c not in frame.columns:
                frame[c] = None
        frame = frame[PER_STEP_COLUMNS]

        fmt = self.config.logging.per_step_format.lower()
        if fmt == "parquet":
            try:
                frame.to_parquet(
                    self.run_dir / "per_step.parquet",
                    index=False,
                    compression=self.config.logging.per_step_compression,
                )
                return
            except Exception:
                pass

        frame.to_csv(self.run_dir / "per_step.csv", index=False)



def row_from_episode(result: Any) -> dict[str, Any]:
    return {
        "generation": result.generation,
        "genome_id": result.genome_id,
        "final_fitness": result.fitness,
        "laps_completed": result.laps_completed,
        "best_lap_time": result.best_lap_time,
        "checkpoints_passed": result.checkpoints_passed,
        "max_progress": result.max_progress,
        "distance_traveled": result.distance_traveled,
        "progress_efficiency": (result.max_progress / result.distance_traveled) if result.distance_traveled > 1e-9 else 0.0,
        "survival_time": result.survival_time,
        "avg_speed": result.avg_speed,
        "max_speed": result.max_speed,
        "crash_flag": int(result.crash_flag),
        "crash_x": result.crash_x,
        "crash_y": result.crash_y,
        "wall_contacts": result.wall_contacts,
        "stagnation_events": result.stagnation_events,
        "steering_var": result.steering_var,
        "steering_jerk": result.steering_jerk,
        "throttle_var": result.throttle_var,
    }


def step_rows_from_episode(result: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in result.per_step:
        row = asdict(step)
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    frame = pd.DataFrame(rows)
    for c in columns:
        if c not in frame.columns:
            frame[c] = None
    frame = frame[columns]
    frame.to_csv(path, index=False)

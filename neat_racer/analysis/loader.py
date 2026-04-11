"""Data loading helpers for experiment analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError


class RunData:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.generation = _safe_read_csv(self.run_dir / "generation_summary.csv")
        self.per_car = _safe_read_csv(self.run_dir / "per_car_summary.csv")
        self.events = _safe_read_csv(self.run_dir / "events.csv")
        self.per_step = self._load_per_step()

    def _load_per_step(self) -> pd.DataFrame:
        parquet_path = self.run_dir / "per_step.parquet"
        csv_path = self.run_dir / "per_step.csv"
        if parquet_path.exists():
            try:
                return pd.read_parquet(parquet_path)
            except Exception:
                pass
        if csv_path.exists():
            return pd.read_csv(csv_path)
        return pd.DataFrame()


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except EmptyDataError:
            return pd.DataFrame()
    return pd.DataFrame()

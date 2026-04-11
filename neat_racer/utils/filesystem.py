"""Filesystem helpers for experiment output management."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re


def _slugify(name: str) -> str:
    lowered = name.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    return slug or "experiment"


def create_experiment_dir(output_root: str | Path, experiment_name: str) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _slugify(experiment_name)

    candidate = root / f"{stamp}_{slug}"
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = root / f"{stamp}_{slug}_{suffix:02d}"

    candidate.mkdir(parents=True)
    for child in ("plots", "replays", "champions"):
        (candidate / child).mkdir(parents=True, exist_ok=True)
    return candidate


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

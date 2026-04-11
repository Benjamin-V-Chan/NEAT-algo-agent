"""Cross-run comparison utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def compare_runs(run_dirs: list[str], output_path: str | None = None) -> pd.DataFrame:
    rows = []
    series = []

    for rd in run_dirs:
        run_dir = Path(rd)
        gen_path = run_dir / "generation_summary.csv"
        if not gen_path.exists():
            continue
        gen = pd.read_csv(gen_path)
        if gen.empty:
            continue

        best_row = gen.iloc[gen["best_fitness"].idxmax()]
        rows.append(
            {
                "run_dir": str(run_dir),
                "max_best_fitness": float(gen["best_fitness"].max()),
                "max_mean_fitness": float(gen["mean_fitness"].max()),
                "best_lap_time": float(gen["best_lap_time"].dropna().min()) if gen["best_lap_time"].notna().any() else None,
                "first_completion_gen": int(gen[gen["completion_rate"] > 0]["generation"].min())
                if (gen["completion_rate"] > 0).any()
                else None,
            }
        )
        series.append((run_dir.name, gen))

    summary = pd.DataFrame(rows)
    if output_path:
        out = Path(output_path)
        out.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out / "run_comparison.csv", index=False)
        _plot_best_fitness_series(series, out / "run_comparison_best_fitness.png")

    return summary


def _plot_best_fitness_series(series: list[tuple[str, pd.DataFrame]], path: Path) -> None:
    if not series:
        return
    fig, ax = plt.subplots(figsize=(10, 5), dpi=130)
    for name, frame in series:
        ax.plot(frame["generation"], frame["best_fitness"], label=name)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness")
    ax.set_title("Best Fitness Across Runs")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

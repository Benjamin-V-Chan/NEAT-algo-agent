"""Analysis and plotting routines for NEAT Racer run artifacts."""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib
import numpy as np
import pandas as pd

from neat_racer.analysis.loader import RunData

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class AnalysisReport:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.plots_dir = self.run_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def generate(self) -> None:
        data = RunData(self.run_dir)
        self._plot_training_curves(data.generation)
        self._plot_lap_progress(data.generation)
        self._plot_fitness_distribution(data.per_car)
        self._plot_crash_heatmap(data.per_car)
        self._plot_control_distributions(data)
        self._write_correlation_summary(data.per_car)
        self._write_analysis_summary(data)

    def _plot_training_curves(self, generation: pd.DataFrame) -> None:
        if generation.empty:
            return
        fig, ax = plt.subplots(figsize=(10, 5), dpi=130)
        ax.plot(generation["generation"], generation["best_fitness"], label="best")
        ax.plot(generation["generation"], generation["mean_fitness"], label="mean")
        ax.plot(generation["generation"], generation["median_fitness"], label="median")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.set_title("Training Curves")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.plots_dir / "training_curves.png")
        plt.close(fig)

    def _plot_lap_progress(self, generation: pd.DataFrame) -> None:
        if generation.empty:
            return
        fig, ax = plt.subplots(figsize=(10, 5), dpi=130)
        ax.plot(generation["generation"], generation["best_lap_time"], marker="o")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Best Lap Time")
        ax.set_title("Lap Time Progression")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.plots_dir / "lap_time_progression.png")
        plt.close(fig)

    def _plot_fitness_distribution(self, per_car: pd.DataFrame) -> None:
        if per_car.empty:
            return
        fig, ax = plt.subplots(figsize=(12, 6), dpi=130)
        grouped = [group["final_fitness"].to_numpy() for _, group in per_car.groupby("generation")]
        positions = sorted(per_car["generation"].unique())
        ax.boxplot(grouped, positions=positions, widths=0.7, showfliers=False)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.set_title("Fitness Distribution by Generation")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(self.plots_dir / "fitness_distribution_by_generation.png")
        plt.close(fig)

    def _plot_crash_heatmap(self, per_car: pd.DataFrame) -> None:
        if per_car.empty:
            return
        crashes = per_car[per_car["crash_flag"] == 1].dropna(subset=["crash_x", "crash_y"])
        if crashes.empty:
            return

        fig, ax = plt.subplots(figsize=(8, 6), dpi=130)
        heat = ax.hist2d(crashes["crash_x"], crashes["crash_y"], bins=36, cmap="inferno")
        fig.colorbar(heat[3], ax=ax, label="Crash density")
        ax.set_title("Crash Heatmap")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(self.plots_dir / "crash_heatmap.png")
        plt.close(fig)

    def _plot_control_distributions(self, data: RunData) -> None:
        if data.per_step.empty:
            return
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=130)
        axes[0].hist(data.per_step["steering_cmd"], bins=30, color="#e67e22")
        axes[0].set_title("Steering")
        axes[1].hist(data.per_step["throttle_cmd"], bins=30, color="#2ecc71")
        axes[1].set_title("Throttle")
        axes[2].hist(data.per_step["brake_cmd"], bins=30, color="#3498db")
        axes[2].set_title("Brake")
        for ax in axes:
            ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(self.plots_dir / "controls_diagnostics.png")
        plt.close(fig)

    def _write_correlation_summary(self, per_car: pd.DataFrame) -> None:
        if per_car.empty:
            return
        cols = [
            "final_fitness",
            "laps_completed",
            "max_progress",
            "distance_traveled",
            "avg_speed",
            "max_speed",
            "wall_contacts",
            "steering_var",
            "steering_jerk",
            "throttle_var",
        ]
        existing = [c for c in cols if c in per_car.columns]
        corr = per_car[existing].corr(numeric_only=True)
        corr.to_csv(self.plots_dir / "correlation_summary.csv")

    def _write_analysis_summary(self, data: RunData) -> None:
        if data.generation.empty:
            return

        summary: dict[str, object] = {}
        gen = data.generation
        summary["best_fitness"] = float(gen["best_fitness"].max())
        summary["best_fitness_generation"] = int(gen.loc[gen["best_fitness"].idxmax(), "generation"])

        lap_rows = data.per_car[data.per_car["laps_completed"] > 0] if not data.per_car.empty else pd.DataFrame()
        summary["first_generation_lap_completion"] = (
            int(lap_rows["generation"].min()) if not lap_rows.empty else None
        )
        completion_rows = gen[gen["completion_rate"] > 0]
        summary["first_generation_nonzero_completion_rate"] = (
            int(completion_rows["generation"].min()) if not completion_rows.empty else None
        )

        total_checkpoints = self._infer_total_checkpoints()
        milestone_values = [1]
        if total_checkpoints is not None and total_checkpoints > 1:
            milestone_values = sorted(
                {
                    1,
                    max(1, int(round(total_checkpoints * 0.25))),
                    max(1, int(round(total_checkpoints * 0.50))),
                    max(1, int(round(total_checkpoints * 0.75))),
                    total_checkpoints,
                }
            )

        checkpoint_milestones: dict[str, int | None] = {}
        if not data.per_car.empty:
            per_gen_max = data.per_car.groupby("generation")["checkpoints_passed"].max()
            for value in milestone_values:
                reached = per_gen_max[per_gen_max >= value]
                checkpoint_milestones[str(value)] = int(reached.index.min()) if not reached.empty else None
        summary["checkpoint_milestones"] = checkpoint_milestones

        (self.plots_dir / "analysis_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

    def _infer_total_checkpoints(self) -> int | None:
        cfg_path = self.run_dir / "experiment_config.json"
        if not cfg_path.exists():
            return None
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            track_path = cfg["simulation"]["track_file"]
        except Exception:
            return None

        try:
            track_file = Path(track_path)
            if not track_file.is_absolute() and not track_file.exists():
                candidate = self.run_dir.parent.parent / track_file
                if candidate.exists():
                    track_file = candidate
            track_json = json.loads(track_file.read_text(encoding="utf-8"))
            checkpoints = track_json.get("checkpoints", [])
            return len(checkpoints)
        except Exception:
            return None


def analyze_run(run_dir: str | Path) -> None:
    AnalysisReport(run_dir).generate()

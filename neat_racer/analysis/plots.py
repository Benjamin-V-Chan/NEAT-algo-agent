"""Analysis and plotting routines for NEAT Racer run artifacts."""

from __future__ import annotations

import json
from pathlib import Path

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
        self._config: dict | None = None

    def _experiment_config(self) -> dict:
        if self._config is None:
            cfg_path = self.run_dir / "experiment_config.json"
            try:
                self._config = json.loads(cfg_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self._config = {}
        return self._config

    def _config_value(self, path: tuple[str, ...], default):
        node = self._experiment_config()
        for key in path:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def generate(self) -> None:
        data = RunData(self.run_dir)
        self._plot_training_curves(data.generation)
        self._plot_lap_progress(data.generation)
        self._plot_fitness_distribution(data.per_car)
        self._plot_crash_heatmap(data.per_car)
        self._plot_control_distributions(data)
        self._plot_evolution_dynamics(data.generation)
        self._plot_curriculum_progression(data.generation)
        self._plot_correlation_heatmap(data.per_car)
        self._plot_champion_trajectory(data)
        self._plot_speed_heatmap(data)
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

        bins = self._config_value(("analysis", "crash_heatmap_bins"), default=36)
        fig, ax = plt.subplots(figsize=(8, 6), dpi=130)
        heat = ax.hist2d(crashes["crash_x"], crashes["crash_y"], bins=bins, cmap="inferno")
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

    def _plot_evolution_dynamics(self, generation: pd.DataFrame) -> None:
        if generation.empty or "num_species" not in generation.columns:
            return
        if generation["num_species"].isna().all():
            return
        fig, (ax_species, ax_complexity) = plt.subplots(2, 1, figsize=(10, 7), dpi=130, sharex=True)

        ax_species.plot(generation["generation"], generation["num_species"], color="#8e44ad", marker="o")
        ax_species.set_ylabel("Species")
        ax_species.set_title("Speciation & Network Complexity")
        ax_species.grid(alpha=0.3)

        for col, label, color in [
            ("best_nodes", "best nodes", "#2980b9"),
            ("mean_nodes", "mean nodes", "#7fb3d5"),
            ("best_connections", "best connections", "#c0392b"),
            ("mean_connections", "mean connections", "#e59866"),
        ]:
            if col in generation.columns:
                ax_complexity.plot(generation["generation"], generation[col], label=label, color=color)
        ax_complexity.set_xlabel("Generation")
        ax_complexity.set_ylabel("Count")
        ax_complexity.grid(alpha=0.3)
        ax_complexity.legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(self.plots_dir / "evolution_dynamics.png")
        plt.close(fig)

    def _plot_curriculum_progression(self, generation: pd.DataFrame) -> None:
        if generation.empty:
            return
        fig, ax_left = plt.subplots(figsize=(10, 5), dpi=130)
        ax_left.plot(
            generation["generation"], generation["completion_rate"],
            color="#27ae60", marker="o", label="completion rate",
        )
        ax_left.set_xlabel("Generation")
        ax_left.set_ylabel("Lap completion rate", color="#27ae60")
        ax_left.set_ylim(-0.02, 1.02)
        ax_left.tick_params(axis="y", labelcolor="#27ae60")
        ax_left.grid(alpha=0.3)

        ax_right = ax_left.twinx()
        ax_right.plot(generation["generation"], generation["crash_count"], color="#c0392b", label="crashes")
        if "farthest_progress" in generation.columns:
            ax_right.plot(
                generation["generation"], generation["farthest_progress"], color="#2980b9",
                linestyle="--", label="farthest progress (laps)",
            )
        ax_right.set_ylabel("Crashes / farthest progress")

        lines_left, labels_left = ax_left.get_legend_handles_labels()
        lines_right, labels_right = ax_right.get_legend_handles_labels()
        ax_left.legend(lines_left + lines_right, labels_left + labels_right, loc="upper left", fontsize=8)
        ax_left.set_title("Curriculum Progression")
        fig.tight_layout()
        fig.savefig(self.plots_dir / "curriculum_progression.png")
        plt.close(fig)

    def _plot_correlation_heatmap(self, per_car: pd.DataFrame) -> None:
        if per_car.empty:
            return
        cols = [
            "final_fitness", "laps_completed", "max_progress", "distance_traveled",
            "avg_speed", "max_speed", "wall_contacts", "steering_var", "steering_jerk", "throttle_var",
        ]
        existing = [c for c in cols if c in per_car.columns]
        corr = per_car[existing].corr(numeric_only=True)
        if corr.empty:
            return
        fig, ax = plt.subplots(figsize=(9, 8), dpi=130)
        im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1.0, vmax=1.0)
        ax.set_xticks(range(len(existing)))
        ax.set_xticklabels(existing, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(existing)))
        ax.set_yticklabels(existing, fontsize=8)
        for i in range(len(existing)):
            for j in range(len(existing)):
                ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="black" if abs(corr.iat[i, j]) < 0.6 else "white")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
        ax.set_title("Behavioral Correlations")
        fig.tight_layout()
        fig.savefig(self.plots_dir / "correlation_heatmap.png")
        plt.close(fig)

    def _plot_champion_trajectory(self, data: RunData) -> None:
        if data.per_step.empty or data.per_car.empty:
            return
        track = self._load_track_boundaries()
        if track is None:
            return

        best_row = data.per_car.loc[data.per_car["final_fitness"].idxmax()]
        traj = data.per_step[
            (data.per_step["genome_id"] == best_row["genome_id"])
            & (data.per_step["generation"] == best_row["generation"])
        ].sort_values("tick")
        if traj.empty:
            return

        outer, inner = track
        fig, ax = plt.subplots(figsize=(9, 7), dpi=130)
        ax.plot(outer[:, 0], outer[:, 1], color="black", linewidth=2)
        ax.plot(inner[:, 0], inner[:, 1], color="black", linewidth=2)
        sc = ax.scatter(traj["x"], traj["y"], c=traj["speed"], cmap="viridis", s=6)
        fig.colorbar(sc, ax=ax, label="Speed")
        ax.set_aspect("equal")
        ax.set_title(
            f"Champion Trajectory (gen {int(best_row['generation'])}, genome {int(best_row['genome_id'])})"
        )
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(self.plots_dir / "champion_trajectory.png")
        plt.close(fig)

    def _plot_speed_heatmap(self, data: RunData) -> None:
        if data.per_step.empty:
            return
        track = self._load_track_boundaries()
        step = data.per_step.dropna(subset=["x", "y", "speed"])
        if step.empty:
            return
        fig, ax = plt.subplots(figsize=(9, 7), dpi=130)
        hb = ax.hexbin(step["x"], step["y"], C=step["speed"], gridsize=40, cmap="magma", reduce_C_function=np.mean)
        fig.colorbar(hb, ax=ax, label="Mean speed")
        if track is not None:
            outer, inner = track
            ax.plot(outer[:, 0], outer[:, 1], color="white", linewidth=1.5)
            ax.plot(inner[:, 0], inner[:, 1], color="white", linewidth=1.5)
        ax.set_aspect("equal")
        ax.set_title("Mean Speed Across the Track")
        fig.tight_layout()
        fig.savefig(self.plots_dir / "speed_heatmap.png")
        plt.close(fig)

    def _load_track_boundaries(self):
        """Return (outer, inner) closed boundary arrays for the run's track, or None."""
        track_path = self._config_value(("simulation", "track_file"), default=None)
        if not track_path:
            return None
        try:
            track_file = Path(track_path)
            if not track_file.is_absolute() and not track_file.exists():
                candidate = self.run_dir.parent.parent / track_file
                if candidate.exists():
                    track_file = candidate
            track_json = json.loads(track_file.read_text(encoding="utf-8"))
            outer = np.asarray(track_json["outer_boundary"], dtype=float)
            inner = np.asarray(track_json["inner_boundary"], dtype=float)
            outer = np.vstack([outer, outer[0]])
            inner = np.vstack([inner, inner[0]])
            return outer, inner
        except Exception:
            return None

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
        track_path = self._config_value(("simulation", "track_file"), default=None)
        if not track_path:
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

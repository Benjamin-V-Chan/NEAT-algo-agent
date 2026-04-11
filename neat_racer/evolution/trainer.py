"""NEAT training orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import json
import pickle
from pathlib import Path
import time
from typing import Any

import neat
import numpy as np

from neat_racer.config import ExperimentConfig
from neat_racer.evolution.fitness import CurriculumScheduler, FitnessAccumulator
from neat_racer.evolution.neat_config_builder import render_neat_config
from neat_racer.simulation.env import network_controller, run_episode
from neat_racer.simulation.track import load_track
from neat_racer.telemetry.logger import ExperimentLogger, row_from_episode, step_rows_from_episode
from neat_racer.telemetry.metrics_bus import MetricsBus
from neat_racer.utils.filesystem import create_experiment_dir
from neat_racer.visualization.renderer import LiveRenderer


@dataclass(slots=True)
class TrainingResult:
    run_dir: Path
    best_genome_id: int
    best_fitness: float
    best_lap_time: float | None


class NeatTrainer:
    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        self.track = load_track(cfg.simulation.track_file)
        self.metrics_bus = MetricsBus()
        self.scheduler = CurriculumScheduler(cfg.fitness)

        self.run_dir = create_experiment_dir(cfg.logging.output_root, cfg.experiment.name)
        self.logger = ExperimentLogger(cfg, self.run_dir)

        self.renderer: LiveRenderer | None = None
        if cfg.rendering.enabled and cfg.runtime.mode == "live":
            self.renderer = LiveRenderer(cfg.rendering, self.metrics_bus)

        self.best_genome = None
        self.best_genome_id = -1
        self.best_fitness = float("-inf")
        self.best_lap_time: float | None = None
        self.champion_history: list[dict[str, Any]] = []
        self.first_generation_lap_completion: int | None = None

    def run(self) -> TrainingResult:
        neat_text = render_neat_config(self.cfg)
        neat_cfg_path = self.run_dir / "neat_config_used.txt"
        neat_cfg_path.write_text(neat_text, encoding="utf-8")
        self.logger.initialize(neat_text)

        if self.renderer:
            self.renderer.initialize(self.track)
            self.renderer.save_track_snapshot(self.track, self.run_dir / "track_snapshot.png")
        else:
            self._save_track_snapshot_without_renderer()

        neat_config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(neat_cfg_path),
        )

        population = neat.Population(neat_config)
        population.add_reporter(neat.StdOutReporter(True))

        generation_index = 0

        def eval_genomes(genomes: list[tuple[int, neat.DefaultGenome]], _: neat.Config) -> None:
            nonlocal generation_index
            self._evaluate_generation(generation_index, genomes, neat_config)
            generation_index += 1

        winner = population.run(eval_genomes, self.cfg.runtime.max_generations)

        if winner is not None and self.best_genome is None:
            self.best_genome = winner

        if self.renderer:
            self.renderer.shutdown()

        self._finalize_artifacts()

        return TrainingResult(
            run_dir=self.run_dir,
            best_genome_id=self.best_genome_id,
            best_fitness=self.best_fitness,
            best_lap_time=self.best_lap_time,
        )

    def _evaluate_generation(
        self,
        generation: int,
        genomes: list[tuple[int, neat.DefaultGenome]],
        neat_config: neat.Config,
    ) -> None:
        t0 = time.perf_counter()
        weights = self.scheduler.current_weights()

        results = []
        crash_count = 0
        lap_finish_count = 0

        self.metrics_bus.update_generation(generation=generation, alive_count=len(genomes))

        for i, (genome_id, genome) in enumerate(genomes):
            net = neat.nn.FeedForwardNetwork.create(genome, neat_config)
            accumulator = FitnessAccumulator(weights)

            def _fitness_step(features: dict) -> float:
                return accumulator.step(features)

            render_step = None
            if self.renderer:
                render_step = self._build_render_step(generation, genome_id)

            episode = run_episode(
                self.cfg,
                self.track,
                generation=generation,
                genome_id=genome_id,
                controller=network_controller(net.activate),
                fitness_step_fn=_fitness_step,
                render_step=render_step,
            )

            genome.fitness = episode.fitness
            results.append(episode)

            if episode.crash_flag:
                crash_count += 1
            if episode.laps_completed > 0:
                lap_finish_count += 1
                if self.first_generation_lap_completion is None:
                    self.first_generation_lap_completion = generation

            self.logger.add_car_row(row_from_episode(episode))
            self.logger.add_event_rows(episode.events)
            if self.cfg.logging.save_per_step:
                self.logger.add_step_rows(step_rows_from_episode(episode))

            if episode.fitness > self.best_fitness:
                self.best_fitness = episode.fitness
                self.best_genome = genome
                self.best_genome_id = genome_id

            if episode.best_lap_time is not None:
                if self.best_lap_time is None or episode.best_lap_time < self.best_lap_time:
                    self.best_lap_time = episode.best_lap_time

            self.metrics_bus.update_car(
                genome_id=genome_id,
                speed=episode.avg_speed,
                steer=episode.steering_var,
                throttle=episode.throttle_var,
                brake=0.0,
                checkpoint=episode.checkpoints_passed,
                lap=episode.laps_completed,
                sensor_angles_deg=self.cfg.sensors.angles_deg,
                sensor_max_range=self.cfg.sensors.max_range,
            )

            if self.renderer:
                controls = self.renderer.poll_events()
                if controls.quit_requested:
                    raise KeyboardInterrupt("Quit requested by user")

            self.metrics_bus.update_generation(alive_count=max(0, len(genomes) - i - 1))

        fitnesses = np.asarray([r.fitness for r in results], dtype=float)
        progresses = np.asarray([r.max_progress for r in results], dtype=float)

        best_lap_candidates = [r.best_lap_time for r in results if r.best_lap_time is not None]
        gen_best_lap = min(best_lap_candidates) if best_lap_candidates else None
        completion_rate = lap_finish_count / max(1, len(results))
        self.scheduler.update(completion_rate)

        gen_best_idx = int(np.argmax(fitnesses))
        gen_best = results[gen_best_idx]

        row = {
            "generation": generation,
            "population_size": len(results),
            "alive_peak": len(results),
            "best_fitness": float(np.max(fitnesses)),
            "mean_fitness": float(np.mean(fitnesses)),
            "median_fitness": float(np.median(fitnesses)),
            "farthest_progress": float(np.max(progresses)),
            "laps_completed_count": int(sum(r.laps_completed for r in results)),
            "best_lap_time": gen_best_lap,
            "completion_rate": completion_rate,
            "crash_count": crash_count,
            "eval_time_sec": time.perf_counter() - t0,
        }
        self.logger.add_generation_row(row)

        self.metrics_bus.update_generation(**row)
        self.metrics_bus.update_run(
            best_fitness=self.best_fitness,
            best_lap_time=self.best_lap_time,
            completion_rate_ema=self.scheduler.smoothed_completion_rate,
            curriculum_alpha=self.scheduler.alpha,
        )

        self.champion_history.append(
            {
                "generation": generation,
                "genome_id": gen_best.genome_id,
                "fitness": gen_best.fitness,
                "lap_count": gen_best.laps_completed,
                "best_lap_time": gen_best.best_lap_time,
            }
        )

        self._save_generation_champion(generation, genomes, int(gen_best.genome_id))

    def _build_render_step(self, generation: int, genome_id: int):
        def _render(track, car, sensor_vec, overlay):
            if not self.renderer:
                return
            controls = self.renderer.poll_events()
            while controls.paused and not controls.quit_requested:
                self.renderer.draw_frame(track, [car], 0, sensor_vec, overlay)
                controls = self.renderer.poll_events()
            if controls.quit_requested:
                raise KeyboardInterrupt("Quit requested by user")
            self.metrics_bus.update_car(
                genome_id=genome_id,
                speed=car.speed,
                steer=overlay.get("steer_cmd", car.steering_cmd),
                throttle=overlay.get("throttle_cmd", 0.0),
                brake=overlay.get("brake_cmd", 0.0),
                checkpoint=car.checkpoint_index,
                lap=car.lap_count,
                sensor_angles_deg=self.cfg.sensors.angles_deg,
                sensor_max_range=self.cfg.sensors.max_range,
            )
            self.metrics_bus.update_generation(generation=generation)
            self.renderer.draw_frame(track, [car], 0, sensor_vec, overlay)

        return _render

    def _save_generation_champion(self, generation: int, genomes: list[tuple[int, neat.DefaultGenome]], genome_id: int) -> None:
        candidate = None
        for gid, genome in genomes:
            if gid == genome_id:
                candidate = genome
                break
        if candidate is None:
            return

        out = self.run_dir / "champions" / f"generation_{generation:04d}_genome_{genome_id}.pkl"
        with out.open("wb") as fh:
            pickle.dump(candidate, fh)

    def _finalize_artifacts(self) -> None:
        if self.best_genome is not None:
            with (self.run_dir / "best_genome.pkl").open("wb") as fh:
                pickle.dump(self.best_genome, fh)
            (self.run_dir / "best_genome_metadata.json").write_text(
                json.dumps(
                    {
                        "best_genome_id": self.best_genome_id,
                        "best_fitness": self.best_fitness,
                        "best_lap_time": self.best_lap_time,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        (self.run_dir / "champion_history.json").write_text(
            json.dumps(self.champion_history, indent=2),
            encoding="utf-8",
        )

        summary = {
            "experiment_name": self.cfg.experiment.name,
            "seed": self.cfg.experiment.seed,
            "best_genome_id": self.best_genome_id,
            "best_fitness": self.best_fitness,
            "best_lap_time": self.best_lap_time,
            "first_generation_lap_completion": self.first_generation_lap_completion,
            "curriculum_completion_rate_ema": self.scheduler.smoothed_completion_rate,
            "curriculum_alpha": self.scheduler.alpha,
        }
        self.logger.set_run_summary(summary)
        self.logger.finalize()

    def _save_track_snapshot_without_renderer(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6), dpi=130)
        outer = np.vstack([self.track.outer_boundary, self.track.outer_boundary[0]])
        inner = np.vstack([self.track.inner_boundary, self.track.inner_boundary[0]])
        ax.plot(outer[:, 0], outer[:, 1], color="black", linewidth=2)
        ax.plot(inner[:, 0], inner[:, 1], color="black", linewidth=2)
        for cp in self.track.checkpoints:
            ax.plot([cp[0][0], cp[1][0]], [cp[0][1], cp[1][1]], color="#3498db", linewidth=0.8)
        sf = self.track.start_finish
        ax.plot([sf[0][0], sf[1][0]], [sf[0][1], sf[1][1]], color="#e74c3c", linewidth=2)
        ax.set_aspect("equal")
        ax.set_title(self.track.name)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(self.run_dir / "track_snapshot.png")
        plt.close(fig)

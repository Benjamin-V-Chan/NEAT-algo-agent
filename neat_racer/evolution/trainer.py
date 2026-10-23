from __future__ import annotations

import copy
import json
import pickle
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import neat
import numpy as np

from neat_racer.config import ExperimentConfig, training_track_files
from neat_racer.evolution.fitness import CurriculumScheduler, FitnessAccumulator
from neat_racer.evolution.neat_config_builder import render_neat_config
from neat_racer.simulation.env import (
    CarAgent,
    PolicySnapshot,
    network_controller,
    run_episode,
    run_population,
)
from neat_racer.simulation.track import load_track
from neat_racer.telemetry.logger import ExperimentLogger, row_from_episode, step_rows_from_episode
from neat_racer.utils.filesystem import create_experiment_dir
from neat_racer.visualization.renderer import LiveRenderer
from neat_racer.visualization.track_plot import render_track


@dataclass(slots=True)
class TrainingResult:
    run_dir: Path
    best_genome_id: int
    best_fitness: float
    best_lap_time: float | None
    env_steps: int = 0


class _BudgetExhausted(Exception):
    pass


# --- Parallel evaluation workers (module-level so they pickle under 'spawn') ---
_WORKER_STATE: dict[str, Any] = {}


def _parallel_worker_init(cfg: ExperimentConfig, tracks: list[Any], neat_cfg_path: str) -> None:
    neat_config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        neat_cfg_path,
    )
    _WORKER_STATE["cfg"] = cfg
    _WORKER_STATE["tracks"] = tracks
    _WORKER_STATE["neat_config"] = neat_config


def _parallel_worker_eval(task: tuple[int, int, Any, Any, int]) -> Any:
    generation, genome_id, genome, weights, track_idx = task
    cfg = _WORKER_STATE["cfg"]
    track = _WORKER_STATE["tracks"][track_idx]
    neat_config = _WORKER_STATE["neat_config"]
    net = neat.nn.FeedForwardNetwork.create(genome, neat_config)
    accumulator = FitnessAccumulator(weights)
    return run_episode(
        cfg,
        track,
        generation=generation,
        genome_id=genome_id,
        controller=network_controller(net.activate),
        fitness_step_fn=accumulator.step,
    )


class NeatTrainer:
    def __init__(
        self, cfg: ExperimentConfig, policy_hook: Callable[[PolicySnapshot], None] | None = None
    ) -> None:
        self.cfg = cfg
        # Every genome drives every training track; the first one is used for snapshots/rendering.
        self.tracks = [load_track(path) for path in training_track_files(cfg)]
        self.track = self.tracks[0]
        self.policy_hook = policy_hook
        self.scheduler = CurriculumScheduler(cfg.fitness)

        self.run_dir = create_experiment_dir(cfg.logging.output_root, cfg.experiment.name)
        self.logger = ExperimentLogger(cfg, self.run_dir)

        self.renderer: LiveRenderer | None = None
        if cfg.rendering.enabled and cfg.runtime.mode == "live":
            self.renderer = LiveRenderer(cfg.rendering)

        self.best_genome = None
        self.best_genome_id = -1
        self.best_fitness = float("-inf")
        self.best_lap_time: float | None = None
        self.champion_history: list[dict[str, Any]] = []
        self.first_generation_lap_completion: int | None = None
        self.env_steps = 0
        self._t_start = time.perf_counter()
        self._hook_time = 0.0

    def _train_time(self) -> float:
        return time.perf_counter() - self._t_start - self._hook_time

    def run(self) -> TrainingResult:
        neat_text = render_neat_config(self.cfg)
        neat_cfg_path = self.run_dir / "neat_config_used.txt"
        neat_cfg_path.write_text(neat_text, encoding="utf-8")
        self.logger.initialize(neat_text)

        render_track(self.track, self.run_dir / "track_snapshot.png")
        if self.renderer:
            self.renderer.initialize(self.track)

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
        budget = self.cfg.runtime.max_env_steps
        self._t_start = time.perf_counter()

        def eval_genomes(genomes: list[tuple[int, neat.DefaultGenome]], _: neat.Config) -> None:
            nonlocal generation_index
            # A generation can't be split, so the budget is checked before starting one.
            if budget and self.env_steps >= budget:
                raise _BudgetExhausted
            self._evaluate_generation(generation_index, genomes, neat_config, population)
            generation_index += 1

        winner = None
        try:
            winner = population.run(eval_genomes, self.cfg.runtime.max_generations)
        except _BudgetExhausted:
            print(f"Env-step budget reached ({self.env_steps} >= {budget}) — stopping.")
        except KeyboardInterrupt:
            # Live window closed (or Ctrl-C): stop early but keep what we have.
            print("Training interrupted — finalizing artifacts from completed generations.")

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
            env_steps=self.env_steps,
        )

    @staticmethod
    def _genome_complexity(genome: neat.DefaultGenome) -> tuple[int, int]:
        nodes = len(genome.nodes)
        connections = sum(1 for c in genome.connections.values() if c.enabled)
        return nodes, connections

    def _evaluate_generation(
        self,
        generation: int,
        genomes: list[tuple[int, neat.DefaultGenome]],
        neat_config: neat.Config,
        population: neat.Population,
    ) -> None:
        t0 = time.perf_counter()
        weights = self.scheduler.current_weights()
        crash_count = 0
        lap_finish_count = 0

        use_parallel = self.cfg.runtime.num_workers > 1 and self.renderer is None
        if use_parallel:
            per_track = self._run_episodes_parallel(generation, genomes, weights)
        else:
            per_track = [
                self._run_episodes_serial(generation, genomes, neat_config, weights, track, render=idx == 0)
                for idx, track in enumerate(self.tracks)
            ]
        # per_genome[i] holds genome i's episode on each training track.
        per_genome = [list(episodes) for episodes in zip(*per_track, strict=True)]
        results = [episode for episodes in per_genome for episode in episodes]

        # Aggregation runs in genome order, so best-fitness tie-breaking and all
        # logging match the serial path regardless of how results were produced.
        genome_fitness: list[float] = []
        for (genome_id, genome), episodes in zip(genomes, per_genome, strict=True):
            # Mean over training tracks; with a single track this is just the episode fitness.
            fitness = float(np.mean([episode.fitness for episode in episodes]))
            genome.fitness = fitness
            genome_fitness.append(fitness)

            for episode in episodes:
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

                if episode.best_lap_time is not None and (
                    self.best_lap_time is None or episode.best_lap_time < self.best_lap_time
                ):
                    self.best_lap_time = episode.best_lap_time

            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_genome = genome
                self.best_genome_id = genome_id

        self.env_steps += sum(r.env_steps for r in results)

        fitnesses = np.asarray(genome_fitness, dtype=float)
        progresses = np.asarray([r.max_progress for r in results], dtype=float)

        best_lap_candidates = [r.best_lap_time for r in results if r.best_lap_time is not None]
        gen_best_lap = min(best_lap_candidates) if best_lap_candidates else None
        completion_rate = lap_finish_count / max(1, len(results))
        self.scheduler.update(completion_rate)

        gen_best_idx = int(np.argmax(fitnesses))
        gen_best_id, gen_best_genome = genomes[gen_best_idx]
        gen_best_laps = [r.best_lap_time for r in per_genome[gen_best_idx] if r.best_lap_time is not None]

        complexities = [self._genome_complexity(genome) for _, genome in genomes]
        node_counts = np.asarray([c[0] for c in complexities], dtype=float)
        conn_counts = np.asarray([c[1] for c in complexities], dtype=float)
        best_nodes, best_conns = complexities[gen_best_idx]
        num_species = len(getattr(population.species, "species", {}))

        row = {
            "generation": generation,
            "population_size": len(genomes),
            "alive_peak": len(genomes),
            "best_fitness": float(np.max(fitnesses)),
            "mean_fitness": float(np.mean(fitnesses)),
            "median_fitness": float(np.median(fitnesses)),
            "farthest_progress": float(np.max(progresses)),
            "laps_completed_count": int(sum(r.laps_completed for r in results)),
            "best_lap_time": gen_best_lap,
            "completion_rate": completion_rate,
            "crash_count": crash_count,
            "eval_time_sec": time.perf_counter() - t0,
            "num_species": int(num_species),
            "best_nodes": int(best_nodes),
            "best_connections": int(best_conns),
            "mean_nodes": float(np.mean(node_counts)),
            "mean_connections": float(np.mean(conn_counts)),
            "env_steps": self.env_steps,
            "wall_time_sec": self._train_time(),
        }
        self.logger.add_generation_row(row)

        self.champion_history.append(
            {
                "generation": generation,
                "genome_id": gen_best_id,
                "fitness": float(fitnesses[gen_best_idx]),
                "lap_count": sum(r.laps_completed for r in per_genome[gen_best_idx]),
                "best_lap_time": min(gen_best_laps) if gen_best_laps else None,
            }
        )

        self._save_generation_champion(generation, genomes, int(gen_best_id))

        if self.policy_hook is not None:
            champion = copy.deepcopy(gen_best_genome)
            hook_t0 = time.perf_counter()
            self.policy_hook(
                PolicySnapshot(
                    env_steps=self.env_steps,
                    train_time_sec=self._train_time(),
                    make_controller=lambda: network_controller(
                        neat.nn.FeedForwardNetwork.create(champion, neat_config).activate
                    ),
                )
            )
            self._hook_time += time.perf_counter() - hook_t0

    def _build_agents(
        self,
        generation: int,
        genomes: list[tuple[int, neat.DefaultGenome]],
        neat_config: neat.Config,
        weights: Any,
        track: Any,
    ) -> list[CarAgent]:
        agents: list[CarAgent] = []
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, neat_config)
            accumulator = FitnessAccumulator(weights)
            agents.append(
                CarAgent(
                    self.cfg,
                    track,
                    generation=generation,
                    genome_id=genome_id,
                    controller=network_controller(net.activate),
                    fitness_step_fn=accumulator.step,
                )
            )
        return agents

    def _run_episodes_serial(
        self,
        generation: int,
        genomes: list[tuple[int, neat.DefaultGenome]],
        neat_config: neat.Config,
        weights: Any,
        track: Any,
        render: bool = True,
    ) -> list[Any]:
        # The whole generation drives at once (lockstep). With a live renderer the viewer sees the
        # entire field racing and crashing out; headless, it is identical to running them serially.
        # With several training tracks only the first one is rendered.
        agents = self._build_agents(generation, genomes, neat_config, weights, track)
        render_all = self._build_render_all(generation, len(agents)) if self.renderer and render else None
        return run_population(self.cfg, track, agents, render_all=render_all)

    def _run_episodes_parallel(
        self,
        generation: int,
        genomes: list[tuple[int, neat.DefaultGenome]],
        weights: Any,
    ) -> list[list[Any]]:
        tasks = [
            (generation, genome_id, genome, weights, track_idx)
            for track_idx in range(len(self.tracks))
            for genome_id, genome in genomes
        ]
        neat_cfg_path = str(self.run_dir / "neat_config_used.txt")
        with ProcessPoolExecutor(
            max_workers=self.cfg.runtime.num_workers,
            initializer=_parallel_worker_init,
            initargs=(self.cfg, self.tracks, neat_cfg_path),
        ) as pool:
            # pool.map preserves task order, so results stay aligned with `genomes` per track.
            results = list(pool.map(_parallel_worker_eval, tasks))
        n = len(genomes)
        return [results[i * n : (i + 1) * n] for i in range(len(self.tracks))]

    def _build_render_all(self, generation: int, total: int):
        def _render(track, agents, tick):
            r = self.renderer
            if not r:
                return
            overlay = self._live_overlay(generation, agents, tick, total)
            controls = r.frame(track, agents, overlay, tick)
            if controls.quit_requested:
                raise KeyboardInterrupt("Quit requested by user")

        return _render

    def _live_overlay(self, generation: int, agents: list[CarAgent], tick: int, total: int) -> dict:
        alive = sum(1 for a in agents if a.alive)
        gen_best = max((a.fitness_total for a in agents), default=0.0)
        lead = max(agents, key=lambda a: a.fitness_total, default=None)
        return {
            "generation": generation,
            "tick": tick,
            "alive": alive,
            "total": total,
            "gen_best_fitness": gen_best,
            "run_best_fitness": self.best_fitness if self.best_fitness > float("-inf") else 0.0,
            "run_best_lap": self.best_lap_time,
            "lead_speed": lead.car.speed if lead else 0.0,
            "lead_lap": lead.car.lap_count if lead else 0,
            "lead_checkpoint": lead.car.checkpoint_index if lead else 0,
        }

    def _save_generation_champion(
        self, generation: int, genomes: list[tuple[int, neat.DefaultGenome]], genome_id: int
    ) -> None:
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
            "method": "neat",
            "env_steps": self.env_steps,
            "train_time_sec": self._train_time(),
            "train_tracks": [t.track_id for t in self.tracks],
        }
        self.logger.set_run_summary(summary)
        self.logger.finalize()

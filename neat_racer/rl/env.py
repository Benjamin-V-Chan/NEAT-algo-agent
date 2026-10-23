from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from neat_racer.config import ExperimentConfig
from neat_racer.constants import NEAT_OUTPUT_COUNT
from neat_racer.evolution.fitness import ActiveFitnessWeights, CurriculumScheduler, FitnessAccumulator
from neat_racer.evolution.neat_config_builder import observation_size
from neat_racer.simulation.car import CarControl, decode_network_outputs
from neat_racer.simulation.env import CarAgent, EpisodeResult, sense_population
from neat_racer.simulation.track import Track


def _no_controller(_obs: np.ndarray) -> CarControl:
    raise RuntimeError("RacerEnv drives CarAgent through step(); the agent's controller is unused")


class RacerEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: ExperimentConfig,
        tracks: Track | list[Track],
        weights: ActiveFitnessWeights | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.tracks = [tracks] if isinstance(tracks, Track) else list(tracks)
        if not self.tracks:
            raise ValueError("RacerEnv needs at least one track")
        self.weights = weights or CurriculumScheduler(cfg.fitness).current_weights()
        self.max_steps = cfg.simulation.max_steps_per_episode

        n_obs = observation_size(cfg)
        n_rays = len(cfg.sensors.angles_deg)
        low = np.zeros(n_obs, dtype=np.float32)
        low[n_rays + 2] = -1.0  # progress_delta is clipped to [-1, 1]; every other feature is in [0, 1]
        self.observation_space = spaces.Box(low=low, high=np.ones(n_obs, dtype=np.float32), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(NEAT_OUTPUT_COUNT,), dtype=np.float32)

        self.episode_index = 0
        self.agent: CarAgent | None = None
        self.track: Track = self.tracks[0]
        self._tick = 0

    def set_fitness_weights(self, weights: ActiveFitnessWeights) -> None:
        # Takes effect from the next episode, like a NEAT generation picking up new curriculum weights.
        self.weights = weights

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            # A seeded reset restarts the track cycle (and per-episode noise streams), so the same
            # seed always replays the same episode sequence.
            self.episode_index = 0
        options = options or {}
        if "track_index" in options:
            self.track = self.tracks[int(options["track_index"]) % len(self.tracks)]
        else:
            self.track = self.tracks[self.episode_index % len(self.tracks)]

        accumulator = FitnessAccumulator(self.weights)
        # genome_id doubles as the per-episode sensor-noise stream id, as it does for NEAT agents.
        self.agent = CarAgent(
            self.cfg, self.track, generation=0, genome_id=self.episode_index,
            controller=_no_controller, fitness_step_fn=accumulator.step,
        )
        self.episode_index += 1
        self._tick = 0
        return self._observe(), {"track_id": self.track.track_id}

    def _observe(self) -> np.ndarray:
        agent = self.agent
        ray_norm, alignments = sense_population(self.cfg, self.track, [agent])
        alignment = None if alignments is None else float(alignments[0])
        obs = agent._observe(self._tick, ray_norm[0], alignment)
        return obs.astype(np.float32)

    def step(self, action: np.ndarray):
        agent = self.agent
        if agent is None or not agent.alive or self._tick >= self.max_steps:
            raise RuntimeError("step() called on a finished episode; call reset()")

        agent._apply(decode_network_outputs(np.asarray(action, dtype=float).tolist()))
        collision = bool(self.track.collision_mask(agent.car.pos[None, :], self.cfg.car.collision_radius)[0])
        alive = agent._resolve(self._tick, collision)
        reward = float(agent.last_fitness_delta)
        self._tick += 1

        terminated = not alive
        truncated = alive and self._tick >= self.max_steps
        info: dict[str, Any] = {"track_id": self.track.track_id}
        if terminated or truncated:
            info["episode_result"] = self.episode_result()
        return self._observe(), reward, terminated, truncated, info

    def episode_result(self) -> EpisodeResult:
        if self.agent is None:
            raise RuntimeError("No episode has been started")
        return self.agent.finish()

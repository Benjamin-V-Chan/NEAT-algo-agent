"""Episode simulation environment.

The core is :class:`CarAgent` — one car + its network + its per-episode bookkeeping, advanced
**one tick at a time** via :meth:`CarAgent.step`. :func:`run_population` steps a whole field of
agents in lockstep (so a live viewer can show the entire population racing and thinning out at
once), while :func:`run_episode` is a thin single-car wrapper kept for replay and tests.

Because cars never interact (no car-to-car collisions), stepping N cars together is numerically
identical to running them one after another — only the scheduling changes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from neat_racer.config import ExperimentConfig
from neat_racer.simulation.car import CarControl, CarState, decode_network_outputs, integrate_car_step
from neat_racer.simulation.progress import ProgressState, update_progress
from neat_racer.simulation.sensors import build_observation, raycast_distances
from neat_racer.simulation.termination import TerminationManager
from neat_racer.simulation.track import Track


@dataclass(slots=True)
class EpisodeStepTelemetry:
    tick: int
    sim_time: float
    generation: int
    genome_id: int
    car_id: int
    x: float
    y: float
    heading: float
    speed: float
    accel: float
    steering_cmd: float
    throttle_cmd: float
    brake_cmd: float
    sensor_vector: list[float]
    checkpoint_idx: int
    lap_count: int
    progress_scalar: float
    alive: bool
    collision_flag: bool
    event_code: str
    fitness_delta: float


@dataclass(slots=True)
class EpisodeResult:
    generation: int
    genome_id: int
    fitness: float
    crash_flag: bool
    crash_x: float | None
    crash_y: float | None
    max_progress: float
    checkpoints_passed: int
    laps_completed: int
    best_lap_time: float | None
    survival_time: float
    distance_traveled: float
    avg_speed: float
    max_speed: float
    wall_contacts: int
    stagnation_events: int
    steering_var: float
    steering_jerk: float
    throttle_var: float
    completion_bonus_awarded: bool
    useful_speed_reward: float
    time_penalty: float
    instability_penalty: float
    wall_penalty: float
    per_step: list[EpisodeStepTelemetry] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)


ControllerFn = Callable[[np.ndarray], CarControl]


def _agent_rng(seed: int, genome_id: int) -> np.random.Generator:
    """A deterministic per-car RNG so sensor noise is reproducible regardless of scheduling."""
    mix = (int(seed) * 1_000_003) ^ (int(genome_id) & 0xFFFFFFFF)
    return np.random.default_rng(mix & 0x7FFFFFFFFFFFFFFF)


class CarAgent:
    """One car driven by one controller, advanced a tick at a time."""

    __slots__ = (
        "cfg", "track", "generation", "genome_id", "controller", "fitness_step_fn", "rng",
        "car", "progress_state", "termination", "max_steps", "dt", "n_rays",
        "max_progress", "last_progress", "fitness_total", "useful_speed_reward",
        "time_penalty", "wall_penalty", "completion_bonus_awarded", "crash_x", "crash_y",
        "per_step", "events", "alive", "ticks", "last_fitness_delta",
        "render_rays", "last_control",
    )

    def __init__(
        self,
        cfg: ExperimentConfig,
        track: Track,
        generation: int,
        genome_id: int,
        controller: ControllerFn,
        fitness_step_fn: Callable[[dict], float],
    ) -> None:
        self.cfg = cfg
        self.track = track
        self.generation = generation
        self.genome_id = genome_id
        self.controller = controller
        self.fitness_step_fn = fitness_step_fn
        self.rng = _agent_rng(cfg.experiment.seed, genome_id)

        self.car = CarState(pos=track.spawn_pos.copy(), heading=track.spawn_heading)
        self.progress_state = ProgressState(lap_start_time=0.0)
        self.termination = TerminationManager(cfg.termination)
        self.max_steps = cfg.simulation.max_steps_per_episode
        self.dt = cfg.simulation.dt
        self.n_rays = len(cfg.sensors.angles_deg)

        self.max_progress = 0.0
        self.last_progress = 0.0
        self.fitness_total = 0.0
        self.useful_speed_reward = 0.0
        self.time_penalty = 0.0
        self.wall_penalty = 0.0
        self.completion_bonus_awarded = False
        self.crash_x: float | None = None
        self.crash_y: float | None = None
        self.per_step: list[EpisodeStepTelemetry] = []
        self.events: list[dict] = []
        self.alive = True
        self.ticks = 0
        self.last_fitness_delta = 0.0
        self.render_rays: np.ndarray | None = None
        self.last_control: CarControl | None = None

    def step(self, tick: int, capture_rays: bool = False) -> bool:
        """Advance one tick. Returns True if the car is still alive afterward."""
        if not self.alive:
            return False

        cfg = self.cfg
        track = self.track
        car = self.car
        sim_time = tick * self.dt

        speed_norm = 0.0 if cfg.car.max_speed <= 1e-9 else car.speed / cfg.car.max_speed
        progress_delta = car.progress_scalar - self.last_progress

        obs = build_observation(track, car, cfg.sensors, speed_norm, progress_delta)
        if cfg.dynamics.sensor_noise_enabled and cfg.dynamics.sensor_noise_std > 0:
            noise = self.rng.normal(0.0, cfg.dynamics.sensor_noise_std, self.n_rays)
            obs[: self.n_rays] = np.clip(obs[: self.n_rays] + noise, 0.0, 1.0)

        control = self.controller(obs)
        self.last_control = control
        step = integrate_car_step(car, control, cfg.car, self.dt, cfg.dynamics)

        # Wall contact. In fatal mode (default) the first touch ends the car; otherwise contacts
        # bleed speed, bounce the car back, and count against a budget (see TerminationManager).
        collision_flag = track.collision_at(car.pos, cfg.car.collision_radius)
        fatal = cfg.termination.wall_contact_fatal and collision_flag
        if collision_flag:
            if not car.in_contact:
                car.wall_contacts += 1
            car.in_contact = True
            car.pos = step.prev_pos.copy()
            car.speed *= cfg.car.collision_speed_retain
        else:
            car.in_contact = False

        current_progress, progress_events = update_progress(
            track, self.progress_state, step.prev_pos, step.current_pos, sim_time, cfg.progress
        )
        car.progress_scalar = current_progress
        car.checkpoint_index = self.progress_state.next_checkpoint_idx
        car.lap_count = self.progress_state.laps_completed
        self.max_progress = max(self.max_progress, current_progress)

        step_features = {
            "progress_scalar": current_progress,
            "progress_delta": current_progress - self.last_progress,
            "checkpoints_passed": self.progress_state.checkpoints_passed_total,
            "laps_completed": self.progress_state.laps_completed,
            "speed_norm": speed_norm,
            "time_norm": tick / self.max_steps,
            "collision_flag": collision_flag,
            "stagnation_events": car.stall_events,
            "wall_contacts": car.wall_contacts,
            "steering_cmd": car.steering_cmd,
            "throttle_cmd": control.throttle,
            "brake_cmd": control.brake,
            "best_lap_time": self.progress_state.best_lap_time,
            "lap_completed_now": bool(progress_events.get("lap_completed_now", False)),
        }
        fitness_delta = self.fitness_step_fn(step_features)
        self.fitness_total += fitness_delta
        self.last_fitness_delta = fitness_delta

        self.useful_speed_reward += max(0.0, current_progress - self.last_progress) * car.speed
        self.time_penalty += tick / self.max_steps
        self.wall_penalty += 1.0 if collision_flag else 0.0

        event_code = "none"
        if progress_events.get("checkpoint_crossed"):
            event_code = "checkpoint"
            self._add_event(tick, sim_time, "checkpoint_crossed", progress_events.get("checkpoint_index"))
        if progress_events.get("lap_completed"):
            self.completion_bonus_awarded = True
            event_code = "lap"
            self._add_event(tick, sim_time, "lap_completed", progress_events.get("lap_time"))

        if capture_rays or cfg.logging.save_per_step:
            self.render_rays = raycast_distances(track, car.pos, car.heading, cfg.sensors)
        if cfg.logging.save_per_step:
            self._append_step(tick, sim_time, control, event_code, fitness_delta)

        self.last_progress = current_progress
        self.ticks += 1

        # Termination: fatal wall, or the usual stagnation / spin / contact-budget checks.
        reason: str | None = None
        if fatal:
            reason = "wall_crash"
        else:
            term = self.termination.update(car, current_progress)
            if term.done:
                reason = term.reason

        if reason is not None:
            car.alive = False
            self.alive = False
            if reason in ("wall_crash", "too_many_wall_contacts"):
                car.crashed = True
                self.crash_x, self.crash_y = float(car.pos[0]), float(car.pos[1])
            self._add_event(tick, sim_time, "termination", reason)

        return self.alive

    def _add_event(self, tick: int, sim_time: float, event_type: str, value) -> None:
        self.events.append(
            {
                "generation": self.generation,
                "genome_id": self.genome_id,
                "tick": tick,
                "sim_time": sim_time,
                "event_type": event_type,
                "event_value": value,
                "x": float(self.car.pos[0]),
                "y": float(self.car.pos[1]),
                "details": "",
            }
        )

    def _append_step(
        self, tick: int, sim_time: float, control: CarControl, event_code: str, fitness_delta: float
    ) -> None:
        car = self.car
        rays = self.render_rays if self.render_rays is not None else np.zeros(self.n_rays)
        self.per_step.append(
            EpisodeStepTelemetry(
                tick=tick,
                sim_time=sim_time,
                generation=self.generation,
                genome_id=self.genome_id,
                car_id=self.genome_id,
                x=float(car.pos[0]),
                y=float(car.pos[1]),
                heading=float(car.heading),
                speed=float(car.speed),
                accel=float(car.acceleration),
                steering_cmd=float(car.steering_cmd),
                throttle_cmd=float(control.throttle),
                brake_cmd=float(control.brake),
                sensor_vector=[float(v) for v in rays],
                checkpoint_idx=car.checkpoint_index,
                lap_count=car.lap_count,
                progress_scalar=float(car.progress_scalar),
                alive=self.alive,
                collision_flag=bool(self.car.in_contact),
                event_code=event_code,
                fitness_delta=float(fitness_delta),
            )
        )

    def finish(self) -> EpisodeResult:
        car = self.car
        steer_arr = np.asarray(car.steering_history, dtype=float) if car.steering_history else np.asarray([0.0])
        throttle_arr = np.asarray(car.throttle_history, dtype=float) if car.throttle_history else np.asarray([0.0])
        avg_speed = car.speed_integral / max(1, car.steps)
        instability_penalty = float(
            np.var(steer_arr) + car.steering_jerk_accum / max(1, car.steps) + np.var(throttle_arr)
        )
        return EpisodeResult(
            generation=self.generation,
            genome_id=self.genome_id,
            fitness=float(self.fitness_total),
            crash_flag=bool(car.crashed),
            crash_x=self.crash_x,
            crash_y=self.crash_y,
            max_progress=float(self.max_progress),
            checkpoints_passed=self.progress_state.checkpoints_passed_total,
            laps_completed=self.progress_state.laps_completed,
            best_lap_time=self.progress_state.best_lap_time,
            survival_time=float(car.steps * self.dt),
            distance_traveled=float(car.total_distance),
            avg_speed=float(avg_speed),
            max_speed=float(car.max_speed),
            wall_contacts=car.wall_contacts,
            stagnation_events=car.stall_events,
            steering_var=float(np.var(steer_arr)),
            steering_jerk=float(car.steering_jerk_accum / max(1, car.steps)),
            throttle_var=float(np.var(throttle_arr)),
            completion_bonus_awarded=self.completion_bonus_awarded,
            useful_speed_reward=float(self.useful_speed_reward),
            time_penalty=float(self.time_penalty),
            instability_penalty=instability_penalty,
            wall_penalty=float(self.wall_penalty),
            per_step=self.per_step,
            events=self.events,
        )


RenderAllFn = Callable[[Track, list[CarAgent], int], None]


def run_population(
    cfg: ExperimentConfig,
    track: Track,
    agents: list[CarAgent],
    render_all: RenderAllFn | None = None,
) -> list[EpisodeResult]:
    """Step every agent in lockstep until all are done or the step budget is hit."""
    max_steps = cfg.simulation.max_steps_per_episode
    capture = render_all is not None
    for tick in range(max_steps):
        any_alive = False
        for agent in agents:
            # step() early-returns False for already-dead agents, so no extra guard is needed.
            if agent.step(tick, capture_rays=capture):
                any_alive = True
        if render_all is not None:
            render_all(track, agents, tick)
        if not any_alive:
            break
    return [agent.finish() for agent in agents]


def run_episode(
    cfg: ExperimentConfig,
    track: Track,
    generation: int,
    genome_id: int,
    controller: ControllerFn,
    fitness_step_fn: Callable[[dict], float],
    render_step: Callable[..., None] | None = None,  # retained for signature compatibility
) -> EpisodeResult:
    """Run a single car to completion (used by replay and tests)."""
    agent = CarAgent(cfg, track, generation, genome_id, controller, fitness_step_fn)
    return run_population(cfg, track, [agent])[0]


def network_controller(net_activate: Callable[[list[float]], list[float]]) -> ControllerFn:
    def _controller(observation: np.ndarray) -> CarControl:
        outputs = net_activate(observation.tolist())
        return decode_network_outputs(outputs)

    return _controller

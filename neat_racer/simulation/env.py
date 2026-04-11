"""Episode simulation environment for one car on one track."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

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


def run_episode(
    cfg: ExperimentConfig,
    track: Track,
    generation: int,
    genome_id: int,
    controller: ControllerFn,
    fitness_step_fn: Callable[[dict], float],
    render_step: Callable[[Track, CarState, np.ndarray, dict], None] | None = None,
) -> EpisodeResult:
    car = CarState(pos=track.spawn_pos.copy(), heading=track.spawn_heading)
    progress_state = ProgressState(lap_start_time=0.0)
    termination = TerminationManager(cfg.termination)

    max_steps = cfg.simulation.max_steps_per_episode
    dt = cfg.simulation.dt

    max_progress = 0.0
    last_progress = 0.0
    fitness_total = 0.0
    useful_speed_reward = 0.0
    time_penalty = 0.0
    wall_penalty = 0.0
    completion_bonus_awarded = False
    crash_x = None
    crash_y = None

    per_step: list[EpisodeStepTelemetry] = []
    events: list[dict] = []

    for tick in range(max_steps):
        sim_time = tick * dt
        speed_norm = 0.0 if cfg.car.max_speed <= 1e-9 else car.speed / cfg.car.max_speed
        progress_delta = car.progress_scalar - last_progress

        obs = build_observation(track, car, cfg.sensors, speed_norm, progress_delta)
        control = controller(obs)

        step = integrate_car_step(car, control, cfg.car, dt)

        collision_flag = track.collision_at(car.pos, cfg.car.collision_radius)
        if collision_flag:
            car.wall_contacts += 1

        current_progress, progress_events = update_progress(
            track,
            progress_state,
            step.prev_pos,
            step.current_pos,
            sim_time,
            cfg.progress,
        )
        car.progress_scalar = current_progress
        car.checkpoint_index = progress_state.next_checkpoint_idx
        car.lap_count = progress_state.laps_completed
        max_progress = max(max_progress, current_progress)

        step_features = {
            "progress_scalar": current_progress,
            "progress_delta": current_progress - last_progress,
            "checkpoints_passed": progress_state.checkpoints_passed_total,
            "laps_completed": progress_state.laps_completed,
            "speed_norm": speed_norm,
            "time_norm": tick / max_steps,
            "collision_flag": collision_flag,
            "stagnation_events": car.stall_events,
            "wall_contacts": car.wall_contacts,
            "steering_cmd": car.steering_cmd,
            "throttle_cmd": control.throttle,
            "brake_cmd": control.brake,
            "best_lap_time": progress_state.best_lap_time,
            "lap_completed_now": bool(progress_events.get("lap_completed_now", False)),
        }
        fitness_delta = fitness_step_fn(step_features)
        fitness_total += fitness_delta

        useful_speed_reward += max(0.0, current_progress - last_progress) * car.speed
        time_penalty += tick / max_steps
        wall_penalty += 1.0 if collision_flag else 0.0

        event_code = "none"
        if progress_events.get("checkpoint_crossed"):
            event_code = "checkpoint"
            events.append(
                {
                    "generation": generation,
                    "genome_id": genome_id,
                    "tick": tick,
                    "sim_time": sim_time,
                    "event_type": "checkpoint_crossed",
                    "event_value": progress_events.get("checkpoint_index"),
                    "x": float(car.pos[0]),
                    "y": float(car.pos[1]),
                    "details": "",
                }
            )

        if progress_events.get("lap_completed"):
            completion_bonus_awarded = True
            event_code = "lap"
            events.append(
                {
                    "generation": generation,
                    "genome_id": genome_id,
                    "tick": tick,
                    "sim_time": sim_time,
                    "event_type": "lap_completed",
                    "event_value": progress_events.get("lap_time"),
                    "x": float(car.pos[0]),
                    "y": float(car.pos[1]),
                    "details": "",
                }
            )

        if render_step:
            sensor_vec = raycast_distances(track, car.pos, car.heading, cfg.sensors)
            render_step(
                track,
                car,
                sensor_vec,
                {
                    "generation": generation,
                    "genome_id": genome_id,
                    "tick": tick,
                    "fitness": fitness_total,
                    "max_progress": max_progress,
                    "steer_cmd": float(car.steering_cmd),
                    "throttle_cmd": float(control.throttle),
                    "brake_cmd": float(control.brake),
                },
            )

        if cfg.logging.save_per_step:
            sensor_vec = raycast_distances(track, car.pos, car.heading, cfg.sensors)
            per_step.append(
                EpisodeStepTelemetry(
                    tick=tick,
                    sim_time=sim_time,
                    generation=generation,
                    genome_id=genome_id,
                    car_id=genome_id,
                    x=float(car.pos[0]),
                    y=float(car.pos[1]),
                    heading=float(car.heading),
                    speed=float(car.speed),
                    accel=float(car.acceleration),
                    steering_cmd=float(car.steering_cmd),
                    throttle_cmd=float(control.throttle),
                    brake_cmd=float(control.brake),
                    sensor_vector=[float(v) for v in sensor_vec],
                    checkpoint_idx=car.checkpoint_index,
                    lap_count=car.lap_count,
                    progress_scalar=float(car.progress_scalar),
                    alive=True,
                    collision_flag=collision_flag,
                    event_code=event_code,
                    fitness_delta=float(fitness_delta),
                )
            )

        last_progress = current_progress

        if collision_flag:
            car.crashed = True
            car.alive = False
            crash_x, crash_y = float(car.pos[0]), float(car.pos[1])

        term_result = termination.update(car, current_progress)
        if term_result.done:
            car.alive = False
            events.append(
                {
                    "generation": generation,
                    "genome_id": genome_id,
                    "tick": tick,
                    "sim_time": sim_time,
                    "event_type": "termination",
                    "event_value": term_result.reason,
                    "x": float(car.pos[0]),
                    "y": float(car.pos[1]),
                    "details": "",
                }
            )

        if not car.alive:
            break

    steer_arr = np.asarray(car.steering_history, dtype=float) if car.steering_history else np.asarray([0.0])
    throttle_arr = np.asarray(car.throttle_history, dtype=float) if car.throttle_history else np.asarray([0.0])

    avg_speed = car.speed_integral / max(1, car.steps)

    instability_penalty = float(np.var(steer_arr) + car.steering_jerk_accum / max(1, car.steps) + np.var(throttle_arr))

    return EpisodeResult(
        generation=generation,
        genome_id=genome_id,
        fitness=float(fitness_total),
        crash_flag=bool(car.crashed),
        crash_x=crash_x,
        crash_y=crash_y,
        max_progress=float(max_progress),
        checkpoints_passed=progress_state.checkpoints_passed_total,
        laps_completed=progress_state.laps_completed,
        best_lap_time=progress_state.best_lap_time,
        survival_time=float(car.steps * dt),
        distance_traveled=float(car.total_distance),
        avg_speed=float(avg_speed),
        max_speed=float(car.max_speed),
        wall_contacts=car.wall_contacts,
        stagnation_events=car.stall_events,
        steering_var=float(np.var(steer_arr)),
        steering_jerk=float(car.steering_jerk_accum / max(1, car.steps)),
        throttle_var=float(np.var(throttle_arr)),
        completion_bonus_awarded=completion_bonus_awarded,
        useful_speed_reward=float(useful_speed_reward),
        time_penalty=float(time_penalty),
        instability_penalty=float(instability_penalty),
        wall_penalty=float(wall_penalty),
        per_step=per_step,
        events=events,
    )


def network_controller(net_activate: Callable[[list[float]], list[float]]) -> ControllerFn:
    def _controller(observation: np.ndarray) -> CarControl:
        outputs = net_activate(observation.tolist())
        return decode_network_outputs(outputs)

    return _controller

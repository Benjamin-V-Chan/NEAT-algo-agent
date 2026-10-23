from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from gymnasium.utils.env_checker import check_env  # noqa: E402

from neat_racer.config import ExperimentConfig  # noqa: E402
from neat_racer.evolution.fitness import CurriculumScheduler, FitnessAccumulator  # noqa: E402
from neat_racer.rl.env import RacerEnv  # noqa: E402
from neat_racer.simulation.car import decode_network_outputs  # noqa: E402
from neat_racer.simulation.env import run_episode  # noqa: E402
from neat_racer.simulation.track import load_track  # noqa: E402


def _ray_follower(obs: np.ndarray) -> np.ndarray:
    # Steer toward the more open side; the raw [-1, 1] outputs a NEAT network would emit.
    left = obs[7] + obs[8]
    right = obs[0] + obs[1]
    return np.array([np.clip(2.5 * (left - right), -1.0, 1.0), -0.1, -1.0])


def _open_loop(n: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [np.array([rng.uniform(-0.3, 0.3), rng.uniform(-0.6, 0.2), -1.0]) for _ in range(n)]


def _neat_path(cfg, track, policy):
    """Run the existing NEAT episode loop, recording what the controller saw and did."""
    seen_obs, actions = [], []

    def controller(obs):
        raw = policy(len(seen_obs), obs)
        seen_obs.append(obs.copy())
        actions.append(raw)
        return decode_network_outputs(raw.tolist())

    weights = CurriculumScheduler(cfg.fitness).current_weights()
    result = run_episode(cfg, track, 0, 0, controller, FitnessAccumulator(weights).step)
    return result, seen_obs, actions


def _env_path(cfg, track, actions):
    env = RacerEnv(cfg, track)
    obs, _ = env.reset(seed=0)
    seen_obs, rewards, positions = [obs], [], []
    terminated = truncated = False
    info = {}
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        seen_obs.append(obs)
        rewards.append(reward)
        positions.append(env.agent.car.pos.copy())
        if terminated or truncated:
            break
    return seen_obs, rewards, positions, terminated, truncated, info


@pytest.mark.parametrize(
    ("track_file", "policy_kind", "noise"),
    [
        ("tracks/twisty_circuit.json", "feedback", False),
        ("tracks/oval_speedway.json", "feedback", True),
        ("tracks/technical_coil.json", "open_loop", False),
    ],
)
def test_env_matches_neat_path_step_for_step(track_file, policy_kind, noise) -> None:
    cfg = ExperimentConfig()
    cfg.simulation.max_steps_per_episode = 400
    cfg.logging.save_per_step = True
    cfg.dynamics.sensor_noise_enabled = noise
    track = load_track(track_file)

    if policy_kind == "feedback":
        def policy(_t, obs):
            return _ray_follower(obs)
    else:
        script = _open_loop(cfg.simulation.max_steps_per_episode)

        def policy(t, _obs):
            return script[t]

    neat_result, neat_obs, neat_actions = _neat_path(cfg, track, policy)
    env_obs, rewards, positions, terminated, truncated, info = _env_path(cfg, track, neat_actions)

    steps = len(neat_actions)
    assert steps > 20  # a non-trivial trajectory
    assert len(rewards) == steps
    # Observations: identical inputs at every tick (the env exposes them as float32).
    for t in range(steps):
        np.testing.assert_array_equal(env_obs[t], neat_obs[t].astype(np.float32))
    # Dynamics + rewards: same positions and same per-step fitness increments.
    for t, row in enumerate(neat_result.per_step):
        assert positions[t][0] == row.x and positions[t][1] == row.y
        assert rewards[t] == row.fitness_delta
    assert sum(rewards) == pytest.approx(neat_result.fitness, rel=0, abs=1e-9)
    # Same episode end, same bookkeeping.
    env_result = info["episode_result"]
    neat_terminated = any(e["event_type"] == "termination" for e in neat_result.events)
    assert terminated == neat_terminated
    assert truncated == (not neat_terminated)
    for field in ("crash_flag", "max_progress", "checkpoints_passed", "laps_completed", "wall_contacts",
                  "distance_traveled", "survival_time", "env_steps"):
        assert getattr(env_result, field) == getattr(neat_result, field), field


def test_truncation_at_max_steps() -> None:
    cfg = ExperimentConfig()
    cfg.simulation.max_steps_per_episode = 30
    env = RacerEnv(cfg, load_track("tracks/oval_speedway.json"))
    obs, _ = env.reset(seed=0)
    for _ in range(30):
        obs, _r, terminated, truncated, info = env.step(_ray_follower(obs))
        assert not terminated
    assert truncated
    assert info["episode_result"].env_steps == 30
    with pytest.raises(RuntimeError):
        env.step(np.zeros(3))


def test_crash_terminates(simple_track) -> None:
    env = RacerEnv(ExperimentConfig(), simple_track)
    env.reset(seed=0)
    terminated = False
    for _ in range(300):
        _obs, reward, terminated, truncated, info = env.step(np.array([-1.0, 1.0, -1.0]))
        if terminated or truncated:
            break
    assert terminated and not truncated
    assert info["episode_result"].crash_flag
    assert reward < 0  # the collision penalty lands on the crash step


def test_env_cycles_tracks_and_passes_checker() -> None:
    tracks = [load_track("tracks/oval_speedway.json"), load_track("tracks/twisty_circuit.json")]
    env = RacerEnv(ExperimentConfig(), tracks)
    seen = [env.reset(seed=0)[1]["track_id"], env.reset()[1]["track_id"], env.reset()[1]["track_id"]]
    assert seen == ["oval_speedway", "twisty_circuit", "oval_speedway"]
    assert env.reset(options={"track_index": 1})[1]["track_id"] == "twisty_circuit"
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)
    check_env(RacerEnv(ExperimentConfig(), tracks), skip_render_check=True)

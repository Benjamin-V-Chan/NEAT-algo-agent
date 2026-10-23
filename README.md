# NEAT Racer Research Platform

A modular, data-heavy [NEAT](https://neat-python.readthedocs.io/)-driven autonomous racing
simulator for evolving control policies on closed-loop tracks. Cars perceive the world only
through local ray sensors and a few internal-state features — there is no privileged optimal
path — and evolution shapes a neural controller from "barely moves" to "drives clean laps."

## Background
This project was built for the **Northrop Grumman High School Involvement Partnership (HIP)
Program**, where I was advised by Northrop Grumman engineers. The original platform was developed
from 2025 to mid-2026 over the course of the program. I came back to it a couple of months after the program
ended and revised it substantially — reworking the simulation core, vectorizing the geometry, and
rebuilding the live visualization — which is why the more recent commits exist.

## Highlights
- 2D top-down racing simulation with deterministic, fixed-step kinematic physics
- Local perception only (ray sensors + internal state), no global track knowledge
- `neat-python` evolution loop with champion persistence and a replay mode
- Curriculum-aware fitness shaping that shifts from *make progress* to *lap fast*
- Structured experiment artifacts (CSV / JSON / Parquet, plots, champion pickles)
- Headless-first training with **optional** live Pygame rendering of the **whole population
  racing at once**, with the sim rate decoupled from the display rate for smooth fast-forward
- Vectorized geometry: every car's ray sensors and wall collisions are resolved for the entire
  generation in a couple of batched numpy calls per tick
- Post-run analysis and multi-run comparison tooling
- A Gymnasium wrapper + stable-baselines3 PPO baseline on the identical environment, and a
  NEAT-vs-PPO comparison study runnable from one command (see [NEAT vs PPO](#neat-vs-ppo))

## Project Layout
```text
neat_racer/
  analysis/       # run loaders, plots (incl. evolution/curriculum/trajectory), comparison
  evolution/      # NEAT config builder, curriculum fitness, trainer (serial + parallel eval)
  replay/         # saved-champion replay
  rl/             # Gymnasium env wrapper (env.py) + stable-baselines3 PPO trainer (train_ppo.py)
  simulation/     # geometry, track, car dynamics, sensors, progress, termination,
                  #   track_builder (procedural tracks) + track_validator
  study/          # NEAT-vs-PPO protocol: shared evaluation harness, runner, report
  telemetry/      # structured logging (CSV / Parquet) + schemas
  utils/          # seeds, filesystem, math helpers
  visualization/  # pygame renderer + HUD (optional) and a matplotlib track_plot
configs/          # YAML experiment configs
tracks/           # JSON tracks (baseline_loop, oval_speedway, twisty_circuit, serpentine_loop, technical_coil)
  holdout/        # held-out evaluation-only tracks for the NEAT-vs-PPO study
    hard/         # out-of-family held-out tier (clockwise / narrow), chosen by blinded calibration
tools/            # generate_tracks.py, snap_track_gates.py, calibrate_holdouts.py
outputs/          # timestamped run artifacts (auto-created, git-ignored)
saved_artifacts/  # optional static artifacts storage
tests/            # unit + smoke tests
main.py           # CLI entrypoint
```

## Installation
Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .                   # core: headless training, analysis, replay
```

Optional extras:
```bash
pip install -e '.[viz]'            # live Pygame rendering (pygame-ce)
pip install -e '.[parquet]'        # columnar per-step telemetry (pyarrow)
pip install -e '.[rl]'             # PPO baseline + NEAT-vs-PPO study (gymnasium, stable-baselines3, torch)
pip install -e '.[dev]'            # pytest + ruff (+ viz + parquet + rl) for development
```

> Rendering uses `pygame-ce`, the maintained community fork, which ships prebuilt wheels for
> current Python versions. All non-rendering functionality works without it.

## CLI Usage
```bash
# Train headless
python main.py train --config configs/default_experiment.yaml

# Train with live rendering (requires the 'viz' extra)
python main.py train --config configs/default_experiment.yaml --live

# Replay the saved champion of a run
python main.py replay --run-dir outputs/<timestamped_run>            # live window
python main.py replay --run-dir outputs/<timestamped_run> --headless # export trajectory only

# (Re)generate analysis plots for an existing run
python main.py analyze --run-dir outputs/<timestamped_run>

# Compare multiple runs
python main.py compare --runs outputs/run_a outputs/run_b --output-dir outputs/comparisons/manual

# Batch across seeds (auto-compares at the end)
python main.py batch --config configs/default_experiment.yaml --seeds 1 2 3 4

# Train PPO on the same environment (single run; budget = runtime.max_env_steps)
python main.py train-ppo --config configs/neat_vs_ppo.yaml --seed 1

# Run the full NEAT-vs-PPO study, or rebuild its report
python main.py study --config configs/neat_vs_ppo.yaml
python main.py study-report --study-dir outputs/study/<timestamped_study>

# Validate / visualize track files
python main.py track --validate-all tracks
python main.py track --render tracks/technical_coil.json --out coil.png
```

The installed console script `neat-racer` is equivalent to `python main.py`.

### Parallel training
Set `runtime.num_workers` above 1 to evaluate each generation's episodes across worker processes
(headless only). Because the default model is noise-free, per-episode results are deterministic
given their inputs, so a parallel run reproduces the serial run's outcome exactly — only faster.
```yaml
runtime:
  num_workers: 4
```

## Controls (Live Mode)
The display runs at a steady `target_fps` while the simulation advances
`sim_ticks_per_frame` ticks per drawn frame, so a whole generation is watchable in
seconds — the "watch speed" is adjustable live without dropping the frame rate.

- `SPACE` — pause / unpause
- `[` / `]` — slower / faster watch speed (sim ticks per frame)
- `S` — toggle sensor rays
- `T` — toggle the leader trail
- `ESC` or window close — stop the run (artifacts from completed generations are still saved)

## Configuration
Primary config: `configs/default_experiment.yaml` (fast variant: `configs/smoke_experiment.yaml`).
YAML keys map 1:1 onto the typed dataclasses in `neat_racer/config.py`; **unknown keys raise an
error**, so the config is self-documenting and typo-safe. Main sections:

| Section       | Purpose |
|---------------|---------|
| `experiment`  | name, description, seed |
| `runtime`     | mode, generation count, determinism, `num_workers` (parallel eval), `max_env_steps` budget |
| `simulation`  | timestep, max episode steps, track file, optional multi-track `train_tracks` |
| `car`         | kinematics, drag/steer limits, wall-contact speed retention |
| `sensors`     | ray angles/range and the optional alignment feature |
| `dynamics`    | optional realism (off by default): understeer, sensor noise, surface grip |
| `termination` | stagnation / spin / wall-contact-budget kill switches |
| `progress`    | checkpoint radius and lap sequencing rules |
| `fitness`     | early/late weight sets and the curriculum blend target |
| `neat`        | population size and genome hyperparameters |
| `ppo`         | stable-baselines3 PPO hyperparameters |
| `study`       | NEAT-vs-PPO protocol: seeds, held-out tracks, evaluation schedule |
| `logging`     | output root and per-step telemetry toggles |
| `rendering`   | window and overlay settings |
| `analysis`    | post-train analysis behavior |
| `replay`      | replay defaults |

## Mathematical Framework
### Vehicle motion (per fixed step `dt`)
- `a_long = a_max·throttle − b_max·brake − drag·v − rolling·sign(v)`
- `v ← clip(v + a_long·dt, 0, v_max)`  *(speed is non-negative — the car cannot reverse)*
- `δ = δ_max·steer_cmd`
- `yaw_rate = (v / L)·tan(δ)·steer_gain(v)`, with `steer_gain = max(min_gain, 1 − decay·(v/v_max))`
- `θ ← θ + yaw_rate·dt`,  `x ← x + v·cos(θ)·dt`,  `y ← y + v·sin(θ)·dt`

### Sensing
- Raycasts at fixed relative angles, intersected against all track wall segments
- Normalized: `s_i = min(distance_i, max_range) / max_range`
- Observation vector = sensor rays + `[speed_norm, steer_norm, progress_delta]` (+ optional
  centerline-alignment cosine). Its length auto-drives the NEAT network's input count.

### Progress / laps
- Checkpoints must be crossed **in strict order** (gate segment intersection)
- A lap requires a full checkpoint cycle followed by crossing the start/finish line
- Scalar progress: `laps + (checkpoint_index + intra-sector_fraction) / num_checkpoints`

### Optional dynamics (realism)
All **off by default** (defaults reproduce the base model exactly), enabled per-run under
`dynamics`:
- **Understeer** — grip-limited cornering: when lateral acceleration `v·yaw_rate` exceeds the grip
  budget `max_lateral_accel`, the yaw rate is scaled down so the car washes wide.
- **Sensor noise** — Gaussian noise (`sensor_noise_std`) on the normalized rays the controller
  sees (telemetry/rendering rays stay clean), reproducible under the run seed.
- **Surface grip** — a global `surface_grip` multiplier on tractive/braking force and the lateral
  grip budget (`< 1` = slippery).

### Wall contacts
By default (`termination.wall_contact_fatal: true`) the **first** wall contact ends the car
immediately as a crash — so a car is only as good as the cleanest line it can hold, and the live
field thins out visibly as cars make mistakes.

A gentler budget model is available with `wall_contact_fatal: false`: each contact bleeds speed
(`v ← v·collision_speed_retain`), bounces the car back to its last valid position, and counts
against a budget, and the episode ends (flagged as a crash) only when `wall_contacts` exceeds
`termination.max_wall_contacts`.

### Fitness (per-step accumulation)
Rewards progress, checkpoints, lap completion, a lap-quality bonus, and useful speed;
penalizes elapsed time, collisions, stalling, control instability, and wall contacts.

**Curriculum schedule:** an exponential moving average of the population completion rate drives
`α = clamp(EMA / completion_rate_target, 0, 1)`, and the active weights blend the two weight sets
as `W = (1−α)·W_early + α·W_late`. Early on, weights reward *any* progress; as the population
starts finishing laps, weight shifts toward speed and lap time.

## Output Artifacts
Each run creates `outputs/<timestamp>_<experiment>/` containing:
- `experiment_config.json`, `neat_config_used.txt` — exact resolved configuration
- `run_summary.json`, `generation_summary.csv`, `per_car_summary.csv`, `events.csv`
- `per_step.parquet` (or CSV fallback) when `logging.save_per_step` is enabled
- `best_genome.pkl` + `best_genome_metadata.json`, `champion_history.json`
- `champions/` per-generation champion pickles
- `plots/` — training curves, lap-time progression, fitness distributions, crash heatmap,
  control diagnostics, evolution dynamics (speciation + network complexity), curriculum
  progression, a behavioral-correlation heatmap, the champion's speed-colored trajectory, a
  track speed heatmap, and correlation + analysis summaries
- `replays/` — champion trajectory exports
- `track_snapshot.png`

## NEAT vs PPO
A controlled comparison of NEAT against PPO (reinforcement learning, stable-baselines3) on this
simulator. It asks two questions: which method learns faster per environment step, and which one
generalizes better to tracks it never trained on.

### Protocol
The protocol was fixed before any run.
- **Same environment.**
  - `neat_racer/rl/env.py` is a single-car Gymnasium env that drives the same `CarAgent` the NEAT
    loop uses. Physics, ray sensors, the 13-float observation, the action decoding, checkpoint/lap
    rules and termination are all shared code.
  - `tests/test_rl_env.py` asserts that observations, positions, rewards and the termination tick
    match the NEAT path step for step when both are given the same actions.
- **Same reward.**
  - PPO's per-step reward is the NEAT per-step fitness increment, so an episode's undiscounted
    return equals its NEAT fitness. The curriculum is the same scheduler, stepped per PPO rollout
    instead of per generation.
  - Deliberate differences: PPO discounts (γ = 0.99) and normalizes rewards, and the 1800-tick cap
    is treated as a truncation.
- **Same budget.**
  - Each run gets 3,000,000 env steps, where 1 step is one simulated tick of one car (0.05 car-s).
    NEAT is charged for every tick of every genome on every track.
  - The final policy is the latest one trained on ≤ 3M steps: for NEAT, the champion of the last
    generation that finished within budget; for PPO, the policy after the last whole rollout.
  - Wall-clock time is logged too.
- **Train/test split.**
  - Training uses the 5 bundled tracks.
  - Two held-out tiers are used only for evaluation:
    - *in-family* (`tracks/holdout/`): 2 tracks from the same generator as the training tracks;
    - *hard* (`tracks/holdout/hard/`): 4 out-of-family tracks. They are clockwise (every training
      track runs counter-clockwise), and one is also narrower than any training track.
  - The hard tracks were picked by a **blinded calibration** (`tools/calibrate_holdouts.py`).
    Earlier final policies drove 8 candidates, only pooled completion was kept, and the 4
    candidates closest to 50% were selected. This places them where the methods *can* differ,
    without looking at which method does better.
- **Seeds and settings.**
  - Study 2 uses 10 seeds per method (Study 1 used 5).
  - NEAT uses the repo's settings (population 120).
  - PPO uses SB3 defaults with 5 envs (one per training track), 1024 steps per env per rollout, and
    batch size 256.
  - Neither side was tuned for the study.
- **Pre-registered primary tests** (Study 2):
  - P1 is train lap-completion AUC (sample efficiency).
  - P2 is hard held-out lap completion (generalization).
  - Each uses an exact Mann-Whitney test, Holm-corrected. A method "wins" only if the adjusted
    p < 0.05.
- **Metrics.**
  - Lap completion rate, best and median lap time, crash rate, and progress, on training and on
    held-out tracks.
  - Learning curves of those metrics against env steps, with 95% t-bands over seeds.
  - Each evaluation runs the policy deterministically for 1800-tick episodes, through the same
    evaluation function for both methods. Curve points use one nominal start per track. Final
    policies are evaluated from 5 fixed start conditions per track.

### Results — Study 2 (`outputs/study/20261023_071745_neat_vs_ppo_v2/`)
| final policies, mean over 10 seeds | NEAT | PPO | PPO − NEAT (95% CI) | Holm p |
|---|---|---|---|---|
| **P1** Train lap-completion AUC (sample efficiency) | 0.912 | 0.933 | 0.021 ± 0.021 | **0.020** |
| **P2** Hard held-out lap completion | 0.390 | 0.595 | 0.205 ± 0.337 | 0.126 |
| In-family held-out lap completion | 1.00 | 1.00 | — | — |
| Hard held-out crash rate | 0.66 | 0.46 | — | — |
| Median env steps until all 5 train tracks lapped | 401,657 | 102,400 | — | — |

**Conclusion.**
- *Learning speed: PPO wins.* The effect is small: it is pre-registered and significant after Holm
  correction. PPO laps every training track after a median of about 100k env steps, against about
  400k for NEAT. Both sit at the ceiling after about 1M steps.
- *Generalization: no detectable difference.* PPO's point estimate is higher on the hard
  clockwise tracks (0.60 vs 0.39). However, outcomes vary hugely between seeds, and the 95% CI on
  the difference runs from −0.13 to +0.54.
  - The test did discriminate: pooled hard-tier completion is about 0.49, not at ceiling or floor.
  - On in-family held-out tracks both methods are perfect.
- *A descriptive trend.* PPO's hard-tier performance peaks around 500k steps (0.98 from the nominal
  start) and decays to 0.65 by 3M while its training score stays at 1.00. It looks like
  specialization to counter-clockwise travel.
- *Wall-clock.* NEAT is about 2× cheaper per env step in this implementation. Study 2 ran 14 jobs
  at once on a 15-core machine, so its absolute timings are contention-bound and are not quoted.

Figures are `figures/learning_curves.png` and `figures/learning_curves_wallclock.png` in the study
directory, next to `results.md` with the per-track tables.

### Study 1 (superseded)
Study 1 (`outputs/study/20261023_063532_neat_vs_ppo/`, 5 seeds) found a tie:
- AUC 0.75 vs 0.79, p = 0.85.
- Both methods at 1.00 on the in-family held-out tracks.

It could not discriminate, for two reasons:
- **`baseline_loop.json` was defective.** Its checkpoint gates did not span the corridor, so a car
  on the outer line lapped at full speed without crossing them and was killed by the stagnation
  rule. That defect explained every training-side difference. The gates are now snapped wall to
  wall (`tools/snap_track_gates.py`), and the track validator rejects such layouts.
- **The held-out tracks were too easy.** Study 2 fixes both problems.

### Reproduce
```bash
pip install -e '.[rl]'
python main.py study --config configs/neat_vs_ppo_v2.yaml   # Study 2: 20 runs, ~2 h with 14 in parallel
python main.py study --config configs/neat_vs_ppo.yaml      # Study 1: 10 runs, ~17 min
python main.py study-report --study-dir outputs/study/<timestamped_study>   # re-render tables and figures
```
`study` trains every (method, seed) run, evaluates them on the step schedule, and writes the
following into `outputs/study/<timestamp>_<name>/`:
- `results.md`, including the pre-registered primary-test table
- the CSV tables
- the figures
- per-run directories in the usual telemetry format, so `analyze` and `compare` work on PPO runs
  too

Rerunning `configs/neat_vs_ppo.yaml` now uses the fixed `baseline_loop`, so it will not reproduce
Study 1's numbers exactly.

## Extending
- **Tracks** — drop a JSON file in `tracks/` (see `tracks/baseline_loop.json` for the schema), or
  generate one procedurally from a centerline with `neat_racer.simulation.track_builder` (see
  `tools/generate_tracks.py`; `build_annulus_track(..., clockwise=True)` builds the mirror-direction
  circuit). Validate/visualize any track with `python main.py track`. Bundled
  tracks: `baseline_loop`, `oval_speedway`, `twisty_circuit`, `serpentine_loop`, `technical_coil`.
  - The validator rejects any checkpoint or start/finish gate that stops more than one collision
    radius short of a wall. A car could drive around such a gate, never earn progress, and be killed
    by the stagnation rule.
  - For a hand-drawn track, `python tools/snap_track_gates.py tracks/<file>.json` stretches every
    gate wall to wall along its own line. `--recenter-centerline` additionally rebuilds the alignment
    centerline from the gate midpoints.
- **Observations** — add features in `simulation/sensors.py`
- **Dynamics** — adjust the model/constraints in `simulation/car.py`
- **Reward shaping** — edit `evolution/fitness.py`
- **Analytics** — add plots in `analysis/plots.py`

## Testing
```bash
pytest -q                # unit + smoke tests
ruff check neat_racer tests
```

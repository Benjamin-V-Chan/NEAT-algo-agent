# NEAT Racer Research Platform

A modular, data-heavy [NEAT](https://neat-python.readthedocs.io/)-driven autonomous racing
simulator for evolving control policies on closed-loop tracks. Cars perceive the world only
through local ray sensors and a few internal-state features — there is no privileged optimal
path — and evolution shapes a neural controller from "barely moves" to "drives clean laps."

## Highlights
- 2D top-down racing simulation with deterministic, fixed-step kinematic physics
- Local perception only (ray sensors + internal state), no global track knowledge
- `neat-python` evolution loop with champion persistence and a replay mode
- Curriculum-aware fitness shaping that shifts from *make progress* to *lap fast*
- Structured experiment artifacts (CSV / JSON / Parquet, plots, champion pickles)
- Headless-first training with **optional** live Pygame rendering + telemetry HUD
- Post-run analysis and multi-run comparison tooling

## Project Layout
```text
neat_racer/
  analysis/       # run loaders, plots (incl. evolution/curriculum/trajectory), comparison
  evolution/      # NEAT config builder, curriculum fitness, trainer (serial + parallel eval)
  replay/         # saved-champion replay
  simulation/     # geometry, track, car dynamics, sensors, progress, termination,
                  #   track_builder (procedural tracks) + track_validator
  telemetry/      # metrics bus + structured logging
  utils/          # seeds, filesystem, math helpers
  visualization/  # pygame renderer + HUD (optional) and a matplotlib track_plot
configs/          # YAML experiment configs
tracks/           # JSON track definitions (baseline_loop, oval_speedway, chicane_loop, hairpin_loop)
tools/            # helper scripts (generate_tracks.py)
outputs/          # timestamped run artifacts (auto-created, git-ignored)
saved_artifacts/  # optional static artifacts storage
tests/            # unit + smoke + reproducibility tests
main.py           # CLI entrypoint
CHANGELOG.md      # release notes
```

## Installation
Requires Python 3.11+ (tested through 3.14).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .                   # core: headless training, analysis, replay
```

Optional extras:
```bash
pip install -e '.[viz]'            # live Pygame rendering (pygame-ce)
pip install -e '.[parquet]'        # columnar per-step telemetry (pyarrow)
pip install -e '.[dev]'            # pytest + ruff (+ viz + parquet) for development
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

# Validate / visualize track files
python main.py track --validate-all tracks
python main.py track --render tracks/chicane_loop.json --out chicane.png
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
- `SPACE` — pause / unpause
- `+` / `-` — speed up / slow down the render loop
- `S` — toggle sensor rays
- Window close — stop the run

## Configuration
Primary config: `configs/default_experiment.yaml` (fast variant: `configs/smoke_experiment.yaml`).
YAML keys map 1:1 onto the typed dataclasses in `neat_racer/config.py`; **unknown keys raise an
error**, so the config is self-documenting and typo-safe. Main sections:

| Section       | Purpose |
|---------------|---------|
| `experiment`  | name, description, seed |
| `runtime`     | mode, generation count, determinism, `num_workers` (parallel eval) |
| `simulation`  | timestep, max episode steps, track file |
| `car`         | kinematics, drag/steer limits, wall-contact speed retention |
| `sensors`     | ray angles/range and the optional alignment feature |
| `dynamics`    | optional realism (off by default): understeer, sensor noise, surface grip |
| `termination` | stagnation / spin / wall-contact-budget kill switches |
| `progress`    | checkpoint radius and lap sequencing rules |
| `fitness`     | early/late weight sets and the curriculum blend target |
| `neat`        | population size and genome hyperparameters |
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
Touching a wall is **not** instantly fatal. Each new contact bleeds speed
(`v ← v·collision_speed_retain`), bounces the car back to its last valid position, and counts
against a budget. The episode ends (flagged as a crash) only when `wall_contacts` exceeds
`termination.max_wall_contacts` — giving evolution a smoother gradient than sudden death.

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

## Extending
- **Tracks** — drop a JSON file in `tracks/` (see `tracks/baseline_loop.json` for the schema), or
  generate one procedurally from a centerline with `neat_racer.simulation.track_builder` (see
  `tools/generate_tracks.py`). Validate/visualize any track with `python main.py track`. Bundled
  tracks: `baseline_loop`, `oval_speedway`, `chicane_loop`, `hairpin_loop`.
- **Observations** — add features in `simulation/sensors.py`
- **Dynamics** — adjust the model/constraints in `simulation/car.py`
- **Reward shaping** — edit `evolution/fitness.py`
- **Analytics** — add plots in `analysis/plots.py`

## Testing
```bash
pytest -q                # unit + smoke tests
ruff check neat_racer tests
```

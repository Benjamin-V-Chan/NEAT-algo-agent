# NEAT Racer Research Platform

A modular, data-heavy NEAT-driven autonomous racing simulator for evolving control policies on unknown closed-loop tracks.

## Highlights
- 2D top-down racing simulation with deterministic fixed-step physics
- Local perception only (ray sensors + internal state), no privileged optimal path input
- `neat-python` evolution loop with champion persistence and replay mode
- Curriculum-aware fitness shaping from progress to lap-time optimization
- Structured experiment artifacts (`csv/json/parquet`, plots, champion files)
- Headless-first training with optional live Pygame rendering and telemetry HUD
- Analysis and run comparison tooling for post-run diagnostics

## Project Layout
```text
neat_racer/
  analysis/       # run loaders, plots, multi-run comparison
  evolution/      # NEAT config builder, fitness scheduler, trainer
  replay/         # saved-champion replay
  simulation/     # geometry, track, car dynamics, sensors, progress, termination
  telemetry/      # metrics bus and structured logging
  utils/          # seeds, filesystem, math helpers
  visualization/  # pygame renderer and HUD
configs/          # YAML experiment configs + neat config template
tracks/           # JSON track definitions
outputs/          # timestamped run artifacts (auto-created)
saved_artifacts/  # optional static artifacts storage
tests/            # unit + smoke tests
main.py           # CLI entrypoint
```

## Installation
```bash
python -m venv .venv
. .venv/Scripts/activate   # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -e .
```

Optional Parquet support:
```bash
pip install -e .[parquet]
```

## CLI Usage
```bash
# Train headless
python main.py train --config configs/default_experiment.yaml

# Train with live rendering
python main.py train --config configs/default_experiment.yaml --live

# Replay saved champion
python main.py replay --run-dir outputs/<timestamped_run>

# Analyze existing run
python main.py analyze --run-dir outputs/<timestamped_run>

# Compare multiple runs
python main.py compare --runs outputs/run_a outputs/run_b --output-dir outputs/comparisons/manual

# Batch seeds
python main.py batch --config configs/default_experiment.yaml --seeds 1 2 3 4
```

## Controls (Live Mode)
- `SPACE`: pause/unpause
- `+` / `-`: speed up / slow down rendering loop
- `S`: toggle sensor rays
- Window close: stop run

## Configuration
Primary config: `configs/default_experiment.yaml`.

Main sections:
- `experiment`: name, description, seed
- `runtime`: mode, deterministic policy, generation count
- `simulation`: timestep, max episode steps, track path
- `car`: kinematic + drag/steer constraints
- `sensors`: ray angles/range and optional alignment feature
- `termination`: stagnation/spin/wall-retry kill switches
- `progress`: checkpoint and lap sequencing rules
- `fitness`: early/late weights and curriculum blending target
- `neat`: population and genome hyperparameters
- `logging`: output root and per-step telemetry toggles
- `rendering`: window and overlay settings
- `analysis`: post-train analysis behavior
- `replay`: replay defaults

## Mathematical Framework
### Vehicle motion
At each fixed step `dt`:
- `a_long = a_max*throttle - b_max*brake - drag_coeff*v - rolling_coeff*sign(v)`
- `v_{t+1} = clip(v_t + a_long*dt, 0, v_max)`
- `delta = delta_max * steer_cmd`
- `yaw_rate = (v/L) * tan(delta) * steer_gain(v)` with speed-dependent steering attenuation
- `theta_{t+1} = theta_t + yaw_rate*dt`
- `x_{t+1} = x_t + v_{t+1}*cos(theta_{t+1})*dt`
- `y_{t+1} = y_t + v_{t+1}*sin(theta_{t+1})*dt`

### Sensing
- Configurable raycasts at fixed relative angles
- Ray-segment intersection against track walls
- Sensor values normalized: `s_i = min(distance_i, sensor_max)/sensor_max`

### Progress / laps
- Checkpoints must be crossed in strict order
- Full cycle + start/finish crossing required for lap completion
- Scalar progress: `laps_completed + (checkpoint_index + sector_fraction)/num_checkpoints`

### Fitness
Per-step accumulation implements:
- progress and checkpoint rewards
- lap completion and lap-quality bonus
- useful-speed reward weighted by progress delta
- penalties for time, collision, stagnation, wall scraping, instability

Curriculum schedule:
- Exponential moving average of completion rate
- `alpha = clamp(smoothed_completion_rate / completion_rate_target, 0, 1)`
- Active weights: `W = (1-alpha)*W_early + alpha*W_late`

## Output Artifacts
Each run creates `outputs/<timestamp>_<experiment>/` with:
- `experiment_config.json`
- `neat_config_used.txt`
- `run_summary.json`
- `generation_summary.csv`
- `per_car_summary.csv`
- `events.csv`
- optional `per_step.parquet` (or csv fallback)
- `best_genome.pkl`
- `best_genome_metadata.json`
- `champions/` per-generation champions
- `plots/` (training curves, lap progression, distributions, crash heatmap, correlations)
- `replays/` champion trajectory exports
- `track_snapshot.png`

## Extending
- Add tracks by dropping JSON files in `tracks/`
- Add observation features in `simulation/sensors.py`
- Adjust dynamics and constraints in `simulation/car.py`
- Modify reward shaping in `evolution/fitness.py`
- Add analytics in `analysis/plots.py`

## Testing
```bash
pytest -q
```

Smoke config for fast checks: `configs/smoke_experiment.yaml`.

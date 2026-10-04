# Changelog

All notable changes to NEAT Racer are documented here. This project loosely follows
[Keep a Changelog](https://keepachangelog.com/) and [Semantic Versioning](https://semver.org/).

## [Unreleased]
### Added
- `CHANGELOG.md` (this file).
- GitHub Actions CI workflow (`.github/workflows/ci.yml`): ruff + pytest matrix across
  Python 3.11 / 3.12 / 3.13, triggered on pushes to `main` and on pull requests.

## [0.2.0]
Focus: make the project install-and-run on modern Python, and make the code honest about its
own configuration (no dead config keys, no documented-but-unimplemented behavior).

### Added
- Optional `viz` extra backed by `pygame-ce` (maintained fork with wheels for Python 3.13/3.14);
  the renderer lazily imports it and raises a clear "install `.[viz]`" message when absent.
- Full wall-contact model in `simulation/env.py`: contacts bleed speed
  (`car.collision_speed_retain`), bounce the car to its last valid position, and count against a
  budget; the episode ends (flagged as a crash) only when `termination.max_wall_contacts` is
  exceeded — a smoother evolutionary gradient than instant death.
- `tests/test_env.py` and `tests/test_sensors.py`; a `ruff` configuration with the full lint
  surface clean.

### Changed
- Rendering is now opt-in via the `viz` extra. Headless training / analysis / replay need no
  native graphics dependency, so `pip install -e .` works on Python 3.11–3.14.
- `analysis.crash_heatmap_bins` is now honored (previously hardcoded to 36).
- `progress.require_start_finish_after_checkpoints` now actually controls lap logic (a nested
  conditional made it a no-op before).
- Rewrote the README to be cross-platform and code-accurate; aligned `requirements.txt` with the
  extras model; bumped version 0.1.0 → 0.2.0; replaced stray `.vscode` settings with useful ones.

### Removed
- Dead config/code: `neat.config_file` and the unused static `configs/neat_config.ini` (the
  trainer renders its INI from `render_neat_config`); `termination.max_reverse_steps`,
  `car.reverse_steps`, and the reverse-limit termination branch (speed is clamped ≥ 0, so the car
  cannot reverse and the branch never fired); and the redundant `scripts/run.py` (an exact
  duplicate of `main.py`).

### Fixed
- Loop-variable closure bug in the trainer (now passes `accumulator.step` directly).
- Dead local in `analysis/compare.py`; set the matplotlib `Agg` backend there for headless safety.

## [0.1.0]
- Initial framework: 2D kinematic racing simulation, local ray-sensor perception, `neat-python`
  evolution loop, curriculum fitness, structured run artifacts, analysis/comparison tooling, and
  an optional live Pygame renderer.

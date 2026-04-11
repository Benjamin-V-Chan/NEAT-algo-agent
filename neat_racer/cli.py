"""Command line interface for NEAT Racer."""

from __future__ import annotations

import argparse
from pathlib import Path

from neat_racer.analysis.compare import compare_runs
from neat_racer.analysis.plots import analyze_run
from neat_racer.config import load_experiment_config
from neat_racer.evolution.trainer import NeatTrainer
from neat_racer.replay.runner import replay_best_genome
from neat_racer.utils.random_utils import seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NEAT Racer research platform")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train a NEAT population")
    train.add_argument("--config", type=Path, default=Path("configs/default_experiment.yaml"))
    train.add_argument("--live", action="store_true", help="Enable live rendering")
    train.add_argument("--seed", type=int, default=None)
    train.add_argument("--no-analysis", action="store_true")

    replay = sub.add_parser("replay", help="Replay best genome from a run dir")
    replay.add_argument("--run-dir", type=Path, required=True)
    replay.add_argument("--headless", action="store_true")

    analyze = sub.add_parser("analyze", help="Generate plots for a run")
    analyze.add_argument("--run-dir", type=Path, required=True)

    compare = sub.add_parser("compare", help="Compare multiple runs")
    compare.add_argument("--runs", nargs="+", required=True, help="Run directories")
    compare.add_argument("--output-dir", type=Path, default=Path("outputs/comparisons"))

    batch = sub.add_parser("batch", help="Run repeated experiments across seeds")
    batch.add_argument("--config", type=Path, default=Path("configs/default_experiment.yaml"))
    batch.add_argument("--seeds", nargs="+", type=int, required=True)
    batch.add_argument("--live", action="store_true")

    return parser


def cmd_train(args: argparse.Namespace) -> Path:
    cfg = load_experiment_config(args.config)
    if args.seed is not None:
        cfg.experiment.seed = args.seed

    cfg.rendering.enabled = bool(args.live)
    cfg.runtime.mode = "live" if args.live else "headless"

    seed_everything(cfg.experiment.seed)

    trainer = NeatTrainer(cfg)
    result = trainer.run()

    if cfg.analysis.enabled_post_train and not args.no_analysis:
        analyze_run(result.run_dir)

    return result.run_dir


def cmd_replay(args: argparse.Namespace) -> Path:
    return replay_best_genome(args.run_dir, live=not args.headless)


def cmd_analyze(args: argparse.Namespace) -> None:
    analyze_run(args.run_dir)


def cmd_compare(args: argparse.Namespace) -> Path:
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    compare_runs(args.runs, output_path=str(out_dir))
    return out_dir


def cmd_batch(args: argparse.Namespace) -> list[Path]:
    run_dirs: list[Path] = []
    for seed in args.seeds:
        cfg = load_experiment_config(args.config)
        cfg.experiment.seed = seed
        cfg.experiment.name = f"{cfg.experiment.name}_seed_{seed}"
        cfg.rendering.enabled = bool(args.live)
        cfg.runtime.mode = "live" if args.live else "headless"
        seed_everything(seed)

        trainer = NeatTrainer(cfg)
        result = trainer.run()
        if cfg.analysis.enabled_post_train:
            analyze_run(result.run_dir)
        run_dirs.append(result.run_dir)

    compare_out = Path("outputs/comparisons") / run_dirs[0].name if run_dirs else Path("outputs/comparisons/empty")
    compare_out.mkdir(parents=True, exist_ok=True)
    compare_runs([str(r) for r in run_dirs], output_path=str(compare_out))
    return run_dirs


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        run_dir = cmd_train(args)
        print(f"Training complete. Run dir: {run_dir}")
        return 0

    if args.command == "replay":
        replay_path = cmd_replay(args)
        print(f"Replay trajectory written to: {replay_path}")
        return 0

    if args.command == "analyze":
        cmd_analyze(args)
        print("Analysis complete.")
        return 0

    if args.command == "compare":
        out_dir = cmd_compare(args)
        print(f"Comparison artifacts written to: {out_dir}")
        return 0

    if args.command == "batch":
        runs = cmd_batch(args)
        print(f"Batch complete. Runs: {[str(r) for r in runs]}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

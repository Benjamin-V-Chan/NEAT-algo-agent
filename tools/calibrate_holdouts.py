from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

from neat_racer.config import load_experiment_config, load_experiment_config_json
from neat_racer.simulation.track import load_track
from neat_racer.study.evaluation import evaluate_controller


def _final_controllers(study_dir: Path) -> list:
    import neat
    from stable_baselines3 import PPO

    from neat_racer.rl.train_ppo import policy_controller
    from neat_racer.simulation.env import network_controller

    finals = pd.read_csv(study_dir / "final_by_seed.csv")
    factories = []
    for _, row in finals.iterrows():
        run_dir = next((study_dir / "runs" / row["method"]).glob(f"*{row['method']}-seed-{int(row['seed'])}"))
        if row["method"] == "neat":
            gen = pd.read_csv(run_dir / "generation_summary.csv")
            generation = int(gen.loc[gen["env_steps"] == row["final_policy_env_steps"], "generation"].iloc[0])
            champion_path = next((run_dir / "champions").glob(f"generation_{generation:04d}_genome_*.pkl"))
            genome = pickle.loads(champion_path.read_bytes())
            neat_cfg = neat.Config(
                neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation,
                str(run_dir / "neat_config_used.txt"),
            )

            def make(genome=genome, neat_cfg=neat_cfg):
                return network_controller(neat.nn.FeedForwardNetwork.create(genome, neat_cfg).activate)
        else:
            policy = PPO.load(run_dir / "ppo_model.zip", device="cpu").policy

            def make(policy=policy):
                return policy_controller(policy)

        factories.append(make)
    return factories


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Blinded difficulty calibration of held-out candidates: every final policy of a "
        "finished study drives every candidate; only POOLED results (all policies together) are kept."
    )
    parser.add_argument("--study-dir", type=Path, required=True, help="Finished study whose final policies probe")
    parser.add_argument("--candidates-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/neat_vs_ppo_v2.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--band", type=float, nargs=2, default=(0.15, 0.85))
    parser.add_argument("--max-selected", type=int, default=4)
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    probe_cfg = load_experiment_config_json(args.study_dir / "study_config.json")
    # Candidates are driven under the study-2 evaluation settings (episode length, start conditions).
    probe_cfg.study = cfg.study
    candidates = [("candidate", load_track(p)) for p in sorted(args.candidates_dir.glob("*.json"))]
    factories = _final_controllers(args.study_dir)

    rows = []
    for make in factories:
        rows.extend(evaluate_controller(probe_cfg, candidates, make, starts=cfg.study.final_eval_starts))
    episodes = pd.DataFrame(rows)  # no method/seed column exists: results are pooled by construction
    pooled = (
        episodes.groupby("track_id")
        .agg(pooled_completion=("lap_completed", "mean"), pooled_progress=("max_progress", "mean"),
             pooled_crash=("crashed", "mean"), episodes=("lap_completed", "count"))
        .reset_index()
    )
    lo, hi = args.band
    pooled["in_band"] = pooled["pooled_completion"].between(lo, hi)
    pooled["distance_to_half"] = (pooled["pooled_completion"] - 0.5).abs()
    eligible = pooled[pooled["in_band"]].sort_values(["distance_to_half", "track_id"])
    selected = eligible["track_id"].head(args.max_selected).tolist()

    args.out.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(args.out / "pooled_difficulty.csv", index=False)
    (args.out / "selection.json").write_text(
        json.dumps(
            {"probe_study": str(args.study_dir), "policies": len(factories),
             "starts_per_track": cfg.study.final_eval_starts, "band": [lo, hi],
             "max_selected": args.max_selected, "selected": selected},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(pooled.round(3).to_string(index=False))
    print(f"Selected ({len(selected)}): {selected}")
    return 0 if len(selected) >= 2 else 1


if __name__ == "__main__":
    sys.exit(main())

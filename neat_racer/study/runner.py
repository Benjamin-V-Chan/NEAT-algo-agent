from __future__ import annotations

import contextlib
import json
import multiprocessing as mp
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from neat_racer.config import ExperimentConfig, dump_config_json
from neat_racer.study.evaluation import StudyEvaluator
from neat_racer.utils.random_utils import seed_everything

METHODS = ("neat", "ppo")


def run_job(cfg: ExperimentConfig, method: str, seed: int, study_dir: str) -> dict[str, Any]:
    study_path = Path(study_dir)
    cfg.experiment.seed = seed
    cfg.experiment.name = f"{method}_seed_{seed}"
    cfg.logging.output_root = str(study_path / "runs" / method)
    cfg.runtime.num_workers = 1
    cfg.runtime.mode = "headless"
    cfg.rendering.enabled = False

    log_path = study_path / "logs" / f"{method}_seed_{seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        seed_everything(seed)
        if method == "neat":
            from neat_racer.evolution.trainer import NeatTrainer

            trainer: Any = NeatTrainer(cfg)
        elif method == "ppo":
            from neat_racer.rl.train_ppo import PpoTrainer

            trainer = PpoTrainer(cfg)
        else:
            raise ValueError(f"Unknown method: {method}")

        evaluator = StudyEvaluator(cfg, method, seed, trainer.run_dir)
        trainer.policy_hook = evaluator
        result = trainer.run()
        train_time = float(trainer._train_time())
        evaluator.finalize()

        if cfg.analysis.enabled_post_train:
            from neat_racer.analysis.plots import analyze_run

            try:
                analyze_run(result.run_dir)
            except Exception:  # noqa: BLE001 - per-run plots are a convenience, not part of the study
                traceback.print_exc(file=log)

    return {
        "method": method,
        "seed": seed,
        "run_dir": str(result.run_dir),
        "env_steps": int(trainer.env_steps),
        "final_policy_env_steps": evaluator.last_snapshot.env_steps if evaluator.last_snapshot else None,
        "train_time_sec": train_time,
        "job_wall_time_sec": time.perf_counter() - t0,
    }


def create_study_dir(output_root: str | Path, name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(output_root) / f"{stamp}_{name}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def run_study(cfg: ExperimentConfig, methods: tuple[str, ...] = METHODS) -> Path:
    if cfg.runtime.max_env_steps <= 0:
        raise ValueError("The study needs runtime.max_env_steps > 0 (the shared env-step budget)")
    study_dir = create_study_dir(cfg.study.output_root, cfg.experiment.name)
    dump_config_json(cfg, study_dir / "study_config.json")

    jobs = [(method, seed) for seed in cfg.study.seeds for method in methods]
    workers = min(cfg.study.max_parallel_runs, len(jobs))
    print(f"Study dir: {study_dir}")
    print(f"Running {len(jobs)} jobs ({', '.join(methods)} x seeds {cfg.study.seeds}) on {workers} processes")

    t0 = time.perf_counter()
    records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        futures = {pool.submit(run_job, cfg, method, seed, str(study_dir)): (method, seed) for method, seed in jobs}
        for future in as_completed(futures):
            method, seed = futures[future]
            try:
                record = future.result()
                print(
                    f"  done {method} seed {seed}: {record['env_steps']} env steps, "
                    f"{record['train_time_sec']:.0f}s training"
                )
            except Exception as exc:  # noqa: BLE001 - keep the other runs; the failure is recorded
                record = {"method": method, "seed": seed, "error": repr(exc)}
                print(f"  FAILED {method} seed {seed}: {exc!r}")
            records.append(record)

    total = time.perf_counter() - t0
    (study_dir / "jobs.json").write_text(
        json.dumps({"study_wall_time_sec": total, "parallel_processes": workers, "jobs": records}, indent=2),
        encoding="utf-8",
    )
    print(f"All jobs finished in {total / 60:.1f} min")

    from neat_racer.study.report import build_report

    build_report(study_dir)
    return study_dir

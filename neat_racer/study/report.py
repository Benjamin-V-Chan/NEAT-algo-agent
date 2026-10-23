from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from neat_racer.analysis.compare import compare_runs
from neat_racer.config import load_experiment_config_json
from neat_racer.study.evaluation import eval_marks

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHOD_COLORS = {"neat": "#2a78d6", "ppo": "#eb6834"}
METHOD_LABELS = {"neat": "NEAT", "ppo": "PPO"}
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SPLIT_ORDER = ("train", "holdout", "holdout_hard")
SPLIT_LABELS = {
    "train": "Training tracks",
    "holdout": "Held-out (in-family)",
    "holdout_hard": "Held-out (hard, out-of-family)",
}
METRICS = {
    "lap_completed": "Lap completion rate",
    "max_progress": "Mean progress (laps)",
    "crashed": "Crash rate",
}
# Two-sided 97.5% Student-t quantiles; df > 30 uses the normal value.
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
         10: 2.228, 15: 2.131, 20: 2.086, 30: 2.042}


def t975(df: int) -> float:
    if df <= 0:
        return float("nan")
    key = max(k for k in _T975 if k <= df) if df <= 30 else None
    return _T975[key] if key is not None else 1.96


def mean_ci(values) -> tuple[float, float, int]:
    arr = np.asarray([v for v in values if v is not None and not pd.isna(v)], dtype=float)
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan"), 0
    if n == 1:
        return float(arr[0]), float("nan"), 1
    return float(arr.mean()), t975(n - 1) * float(arr.std(ddof=1)) / math.sqrt(n), n


def mann_whitney_exact(a, b) -> float:
    a = list(a)
    b = list(b)
    pooled = a + b
    n_a = len(a)
    if n_a == 0 or not b:
        return float("nan")
    ranks = pd.Series(pooled).rank().to_numpy()
    observed = ranks[:n_a].sum()
    expected = n_a * (len(pooled) + 1) / 2
    combos = np.fromiter(
        itertools.chain.from_iterable(itertools.combinations(range(len(pooled)), n_a)), dtype=np.int16
    ).reshape(-1, n_a)
    sums = ranks[combos].sum(axis=1)
    return float(np.mean(np.abs(sums - expected) >= abs(observed - expected) - 1e-9))


def cliffs_delta(x, y) -> float:
    x = np.asarray(list(x), dtype=float)[:, None]
    y = np.asarray(list(y), dtype=float)[None, :]
    if x.size == 0 or y.size == 0:
        return float("nan")
    return float(np.mean(x > y) - np.mean(x < y))


def welch_diff_ci(x, y) -> tuple[float, float]:
    x = np.asarray(list(x), dtype=float)
    y = np.asarray(list(y), dtype=float)
    if len(x) < 2 or len(y) < 2:
        return float("nan"), float("nan")
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    diff = float(x.mean() - y.mean())
    se = math.sqrt(vx + vy)
    if se == 0:
        return diff, 0.0
    df = (vx + vy) ** 2 / (vx**2 / (len(x) - 1) + vy**2 / (len(y) - 1))
    return diff, t975(int(df)) * se


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    m = len(pvalues)
    adjusted = [float("nan")] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[idx]))
        adjusted[idx] = running
    return adjusted


def splits_in(evals: pd.DataFrame) -> list[str]:
    present = set(evals["split"])
    return [s for s in SPLIT_ORDER if s in present]


def load_evals(study_dir: Path) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted((study_dir / "runs").glob("*/*/eval_curve.csv"))]
    if not frames:
        raise FileNotFoundError(f"No eval_curve.csv files under {study_dir / 'runs'}")
    evals = pd.concat(frames, ignore_index=True)
    if "start" not in evals:
        evals["start"] = 0
    return evals


def per_policy_scores(evals: pd.DataFrame) -> pd.DataFrame:
    keys = ["method", "seed", "env_steps", "train_time_sec", "final", "split"]
    scores = evals.groupby(keys, as_index=False).agg(
        lap_completed=("lap_completed", "mean"),
        max_progress=("max_progress", "mean"),
        crashed=("crashed", "mean"),
        tracks=("track_id", "count"),
    )
    return scores


def curves_on_grid(
    scores: pd.DataFrame, grid: list[float], x: str = "env_steps", stop_at_end: bool = False
) -> pd.DataFrame:
    rows = []
    for (method, seed, split), run in scores.groupby(["method", "seed", "split"]):
        run = run.sort_values(x).drop_duplicates(subset=[x], keep="last")
        xs = run[x].to_numpy()
        for g in grid:
            idx = np.searchsorted(xs, g, side="right") - 1
            if stop_at_end and g > xs[-1]:
                idx = -1
            row = {"method": method, "seed": seed, "split": split, "grid": g}
            for metric in METRICS:
                row[metric] = float(run[metric].iloc[idx]) if idx >= 0 else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_curves(
    grid_df: pd.DataFrame, path: Path, x_label: str, x_scale: float, x_unit: str, splits: list[str]
) -> None:
    metrics = ["lap_completed", "max_progress"]
    fig, axes = plt.subplots(
        len(metrics), len(splits), figsize=(5.5 * len(splits), 7.2), dpi=140, sharex=True, facecolor=SURFACE,
        squeeze=False,
    )
    for r, metric in enumerate(metrics):
        for c, split in enumerate(splits):
            ax = axes[r, c]
            _style(ax)
            end_labels = []
            for method in ("neat", "ppo"):
                sub = grid_df[(grid_df["method"] == method) & (grid_df["split"] == split)]
                if sub.empty:
                    continue
                stats = sub.groupby("grid")[metric].apply(lambda s: mean_ci(s.tolist()))
                gx = np.asarray(stats.index, dtype=float) / x_scale
                mean = np.asarray([v[0] for v in stats])
                half = np.nan_to_num(np.asarray([v[1] for v in stats]))
                counts = np.asarray([v[2] for v in stats])
                n = int(counts.max())
                # Only plot points where every seed contributes (runs end at different wall times).
                mean = np.where(counts == n, mean, np.nan)
                half = np.where(counts == n, half, 0.0)
                color = METHOD_COLORS[method]
                # Each run holds its latest evaluation until the next, so draw steps; clip the band to
                # the metric's valid range (rates in [0, 1], progress >= 0).
                upper_cap = 1.0 if metric != "max_progress" else np.inf
                ax.fill_between(
                    gx, np.clip(mean - half, 0.0, upper_cap), np.clip(mean + half, 0.0, upper_cap),
                    color=color, alpha=0.15, linewidth=0, step="post",
                )
                ax.plot(
                    gx, mean, color=color, linewidth=2, drawstyle="steps-post",
                    label=f"{METHOD_LABELS[method]} (n={n} seeds)",
                )
                finite = np.where(np.isfinite(mean))[0]
                if len(finite):
                    i = finite[-1]
                    end_labels.append((mean[i], gx[i], f"{METHOD_LABELS[method]} {mean[i]:.2f}"))
            if metric == "lap_completed":
                ax.set_ylim(-0.03, 1.03)
            # Direct end labels, nudged apart when the two series finish at (nearly) the same value.
            lo, hi = ax.get_ylim()
            end_labels.sort()
            for j, (y, x_end, text) in enumerate(end_labels):
                close = len(end_labels) == 2 and abs(end_labels[1][0] - end_labels[0][0]) < 0.06 * (hi - lo)
                dy = (-5 if j == 0 else 5) if close else 0
                ax.annotate(text, (x_end, y), xytext=(4, dy), textcoords="offset points",
                            fontsize=8, color=INK_2, va="center")
            ax.set_title(f"{METRICS[metric]} — {SPLIT_LABELS[split]}", fontsize=10, color=INK, loc="left")
            if r == len(metrics) - 1:
                ax.set_xlabel(f"{x_label} ({x_unit})", fontsize=9, color=INK_2)
    axes[0, 0].legend(fontsize=8, frameon=False, labelcolor=INK_2, loc="lower right")
    fig.suptitle(
        "NEAT vs PPO — mean over seeds (step = latest evaluated policy), shaded = 95% t-interval clipped to "
        "the metric's range; deterministic episodes (curve points use the nominal start)",
        fontsize=10, color=INK_2, x=0.01, ha="left",
    )
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def _fmt(v, digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.{digits}f}"


def _fmt_ci(mean: float, half: float, digits: int = 2) -> str:
    if math.isnan(mean):
        return "—"
    return f"{mean:.{digits}f} ± {half:.{digits}f}" if not math.isnan(half) else f"{mean:.{digits}f}"


def _md_table(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def build_report(study_dir: str | Path, exclude_tracks: tuple[str, ...] = ()) -> Path:
    study_dir = Path(study_dir)
    cfg = load_experiment_config_json(study_dir / "study_config.json")
    budget = cfg.runtime.max_env_steps
    out_dir = study_dir / f"posthoc_excluding_{'_'.join(exclude_tracks)}" if exclude_tracks else study_dir
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    evals = load_evals(study_dir)
    evals = evals[~evals["track_id"].isin(exclude_tracks)]
    evals.to_csv(out_dir / "eval_all.csv", index=False)
    scores = per_policy_scores(evals)
    # Curves (and the AUC built from them) use the nominal start only, including at the final policy,
    # so every curve point is measured the same way; final-policy metrics use every start.
    curve_scores = per_policy_scores(evals[evals["start"] == 0])
    splits = splits_in(evals)
    train_track_count = int(evals[evals["split"] == "train"]["track_id"].nunique())

    grid = eval_marks(cfg)
    grid_df = curves_on_grid(curve_scores, grid)
    grid_df.to_csv(out_dir / "learning_curves_grid.csv", index=False)
    plot_curves(grid_df, fig_dir / "learning_curves.png", "Training env steps", 1e6, "millions", splits)
    max_wall = float(curve_scores["train_time_sec"].max())
    wall_grid = list(np.linspace(0, max_wall, 41))
    wall_df = curves_on_grid(curve_scores, wall_grid, x="train_time_sec", stop_at_end=True)
    plot_curves(wall_df, fig_dir / "learning_curves_wallclock.png", "Training wall time", 60.0, "minutes", splits)

    jobs_path = study_dir / "jobs.json"
    jobs = json.loads(jobs_path.read_text(encoding="utf-8")) if jobs_path.exists() else {}
    job_by_key = {(j["method"], j["seed"]): j for j in jobs.get("jobs", []) if "error" not in j}
    run_rows = []
    for (method, seed), run in scores.groupby(["method", "seed"]):
        final = run[run["final"] == 1]
        curve = run[run["final"] == 0]
        per_eval = evals[(evals["method"] == method) & (evals["seed"] == seed) & (evals["split"] == "train")]
        # A policy "laps all training tracks" when every training episode it was given completed a lap.
        all_lapped = per_eval.groupby(["env_steps", "final"]).agg(
            done=("lap_completed", "sum"), episodes=("lap_completed", "count"), tracks=("track_id", "nunique")
        ).reset_index()
        hit = all_lapped[(all_lapped["done"] == all_lapped["episodes"]) & (all_lapped["tracks"] == train_track_count)]
        hit = hit.set_index("env_steps")
        train_grid = grid_df[(grid_df["method"] == method) & (grid_df["seed"] == seed) & (grid_df["split"] == "train")]

        def final_metric(split: str, metric: str, frame=final) -> float:
            sel = frame[frame["split"] == split]
            return float(sel[metric].iloc[0]) if not sel.empty else float("nan")

        job = job_by_key.get((method, seed), {})
        run_rows.append(
            {
                "method": method,
                "seed": int(seed),
                "final_policy_env_steps": int(final["env_steps"].iloc[0]) if not final.empty else None,
                "total_env_steps_consumed": job.get("env_steps"),
                "train_time_sec": job.get("train_time_sec"),
                "evals": int(curve["env_steps"].nunique()),
                "steps_to_all_train_laps": int(hit.index.min()) if not hit.empty else None,
                "auc_train_completion": float(np.nan_to_num(train_grid["lap_completed"].to_numpy()).mean()),
                **{f"final_{split}_{metric}": final_metric(split, metric) for split in splits for metric in METRICS},
            }
        )
        for split in splits[1:]:
            run_rows[-1][f"gap_train_minus_{split}"] = (
                run_rows[-1]["final_train_lap_completed"] - run_rows[-1][f"final_{split}_lap_completed"]
            )
    runs = pd.DataFrame(run_rows).sort_values(["method", "seed"])
    runs.to_csv(out_dir / "final_by_seed.csv", index=False)

    summary_rows = []
    headline = [("auc_train_completion", "Train lap-completion AUC (0–1)")]
    for metric, label in (("lap_completed", "lap completion"), ("max_progress", "progress (laps)"),
                          ("crashed", "crash rate")):
        headline += [(f"final_{split}_{metric}", f"Final {label} — {SPLIT_LABELS[split]}") for split in splits]
    headline += [(f"gap_train_minus_{split}", f"Completion gap train − {SPLIT_LABELS[split]}")
                 for split in splits[1:]]
    for key, label in headline:
        row = {"metric": label}
        for method in ("neat", "ppo"):
            m, h, n = mean_ci(runs[runs["method"] == method][key].tolist())
            row[METHOD_LABELS[method]] = _fmt_ci(m, h)
        a = runs[runs["method"] == "neat"][key].dropna().tolist()
        b = runs[runs["method"] == "ppo"][key].dropna().tolist()
        row["Mann-Whitney p (exact, 2-sided)"] = _fmt(mann_whitney_exact(a, b), 3)
        summary_rows.append(row)
    steps_row = {"metric": f"Env steps until a policy laps all {train_track_count} train tracks"}
    for method in ("neat", "ppo"):
        sub = runs[runs["method"] == method]
        reached = sub["steps_to_all_train_laps"].dropna()
        steps_row[METHOD_LABELS[method]] = (
            f"{len(reached)}/{len(sub)} runs; median of those {int(reached.median()):,}"
            if len(reached) else f"0/{len(sub)} runs (censored at {budget:,})"
        )
    # Censored runs count as B (ties at the budget), which the rank test handles.
    censored = {
        m: runs[runs["method"] == m]["steps_to_all_train_laps"].fillna(budget).tolist() for m in ("neat", "ppo")
    }
    steps_row["Mann-Whitney p (exact, 2-sided)"] = _fmt(mann_whitney_exact(censored["neat"], censored["ppo"]), 3)
    summary_rows.append(steps_row)
    summary = pd.DataFrame(summary_rows).fillna("")
    summary.to_csv(out_dir / "summary.csv", index=False)

    primary_rows = []
    primary = [m for m in cfg.study.primary_metrics if m in runs.columns]
    raw_p = []
    for key in primary:
        neat_v = runs[runs["method"] == "neat"][key].dropna().tolist()
        ppo_v = runs[runs["method"] == "ppo"][key].dropna().tolist()
        raw_p.append(mann_whitney_exact(neat_v, ppo_v))
        diff, half = welch_diff_ci(ppo_v, neat_v)
        primary_rows.append(
            {
                "primary metric": key,
                "NEAT mean": _fmt(float(np.mean(neat_v)) if neat_v else float("nan"), 3),
                "PPO mean": _fmt(float(np.mean(ppo_v)) if ppo_v else float("nan"), 3),
                "PPO − NEAT (95% Welch CI)": _fmt_ci(diff, half, 3),
                "Cliff's δ (PPO vs NEAT)": _fmt(cliffs_delta(ppo_v, neat_v), 2),
                "n (NEAT/PPO)": f"{len(neat_v)}/{len(ppo_v)}",
                "exact MW p": _fmt(raw_p[-1], 4),
            }
        )
    for row, adj in zip(primary_rows, holm(raw_p) if raw_p else [], strict=True):
        row["Holm-adjusted p"] = _fmt(adj, 4)
    primary_table = pd.DataFrame(primary_rows)
    primary_table.to_csv(out_dir / "primary_tests.csv", index=False)

    final_eval = evals[evals["final"] == 1]
    track_rows = []
    for (split, track_id), sub_track in final_eval.groupby(["split", "track_id"]):
        for method in ("neat", "ppo"):
            sub = sub_track[sub_track["method"] == method]
            if sub.empty:
                continue
            m, h, _ = mean_ci(sub["max_progress"].tolist())
            completers = sub[sub["lap_completed"] == 1]
            track_rows.append(
                {
                    "split": split,
                    "track": track_id,
                    "method": METHOD_LABELS[method],
                    "lapped": f"{len(completers)}/{len(sub)}",
                    "crashed": f"{int(sub['crashed'].sum())}/{len(sub)}",
                    "progress (laps)": _fmt_ci(m, h),
                    "best lap (s)": _fmt(completers["best_lap_time"].min() if len(completers) else None),
                    "median best lap (s)": _fmt(completers["best_lap_time"].median() if len(completers) else None),
                }
            )
    tracks = pd.DataFrame(track_rows).sort_values(["split", "track", "method"], ascending=[False, True, True])
    tracks.to_csv(out_dir / "generalization_table.csv", index=False)

    wall_rows = []
    for method in ("neat", "ppo"):
        sub = runs[runs["method"] == method].dropna(subset=["train_time_sec", "total_env_steps_consumed"])
        if sub.empty:
            continue
        m, h, _ = mean_ci((sub["train_time_sec"] / 60).tolist())
        rate = (sub["total_env_steps_consumed"] / sub["train_time_sec"]).tolist()
        rm, rh, _ = mean_ci(rate)
        wall_rows.append(
            {"method": METHOD_LABELS[method], "train time (min)": _fmt_ci(m, h, 1),
             "env steps / s": _fmt_ci(rm, rh, 0)}
        )
    wall = pd.DataFrame(wall_rows)

    # Existing comparison tooling works on both methods' run dirs (same telemetry format).
    if not exclude_tracks:
        run_dirs = sorted(str(p.parent) for p in (study_dir / "runs").glob("*/*/generation_summary.csv"))
        compare_runs(run_dirs, output_path=str(study_dir / "compare"))

    numeric_cols = ["auc_train_completion"] + [f"final_{s}_lap_completed" for s in splits] + [
        f"final_{s}_{m}" for s in splits[1:] for m in ("max_progress", "crashed")
    ]
    per_seed_cols = ["method", "seed", "final_policy_env_steps", "train_time_sec", "steps_to_all_train_laps",
                     *numeric_cols]
    per_seed = runs[per_seed_cols].copy()
    per_seed["train_time_sec"] = per_seed["train_time_sec"].map(lambda v: _fmt(v, 0))
    for c in numeric_cols:
        per_seed[c] = per_seed[c].map(_fmt)
    per_seed["steps_to_all_train_laps"] = per_seed["steps_to_all_train_laps"].map(
        lambda v: "never" if pd.isna(v) else f"{int(v):,}"
    )

    lines = [
        f"# NEAT vs PPO — study results (`{study_dir.name}`)",
        "",
        f"Budget: {budget:,} env steps per run · seeds {list(cfg.study.seeds)} · "
        + " · ".join(
            f"{SPLIT_LABELS[s]}: {sorted(evals[evals['split'] == s]['track_id'].unique())}" for s in splits
        ),
        f"Final evaluation: {cfg.study.final_eval_starts} start(s) per track; curve evaluations: "
        f"{cfg.study.curve_eval_starts}.",
        f"Study wall time: {_fmt(jobs.get('study_wall_time_sec', float('nan')) / 60, 1)} min on "
        f"{jobs.get('parallel_processes', '?')} parallel processes.",
        "",
        "Generated by `neat_racer/study/report.py` from the eval_curve.csv files under `runs/`.",
        *(
            [f"**Post-hoc sensitivity analysis:** tracks {list(exclude_tracks)} excluded from scoring "
             "(not part of the pre-registered protocol; same saved evaluations, nothing re-run)."]
            if exclude_tracks else []
        ),
        "Means ± 95% t-interval over seeds.",
        "",
        *(
            ["## Pre-registered primary tests (Holm-corrected)", _md_table(primary_table), ""]
            if not primary_table.empty else []
        ),
        "## Summary",
        _md_table(summary),
        "",
        "## Final policies per track",
        _md_table(tracks),
        "",
        "## Per seed",
        _md_table(per_seed),
        "",
        "## Wall-clock",
        _md_table(wall) if not wall.empty else "(no completed jobs)",
        "",
        "## Figures",
        "- `figures/learning_curves.png` — lap completion and progress vs env steps, per split",
        "- `figures/learning_curves_wallclock.png` — the same vs training wall time",
        "",
    ]
    out = out_dir / "results.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Study report written to: {out}")
    return out

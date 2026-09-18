"""Fetch COLE per-seed training return + final XP matrix from wandb artifacts.

Curves and matrices (per `cole.py:1132-1175`):
  - Partner training return: `metrics["returned_episode_returns"]` shape
    (NUM_SEEDS, POP_SIZE, NUM_PARTNER_UPDATES). Per-seed mean over the pop axis.
  - Final XP matrix: `out["final_xp_matrix"]` (top-level, NOT under metrics)
    shape (NUM_SEEDS, POP_SIZE, POP_SIZE). One matrix per seed; we plot the
    mean-over-seeds version (the same one wandb logs as `Eval/LastXPMatrix`).

COLE runs in this project are not configured with `train_ego: true`, so there
is no ego curve to plot. Some COLE neurips:benchmark runs (e.g. yn8h1tua) were
logged with `log_train_out=False` and have no train_run artifact at all — we
skip those with a warning rather than failing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.training_curves.common import (
    DEFAULT_CACHE_DIR,
    CurveData,
    fetch_train_run_metrics_cached,
    find_benchmark_runs,
    get_config_value,
    make_curve,
)


@dataclass
class COLERunCurves:
    run_id: str
    task: str
    partner: CurveData
    xp_matrix: np.ndarray  # shape (POP_SIZE, POP_SIZE), seed-averaged


def _partner_curve(metrics: dict, timesteps_per_iter: int) -> CurveData:
    arr = np.asarray(metrics["returned_episode_returns"])
    if arr.ndim != 3:
        raise ValueError(
            f"partner returned_episode_returns expected 3D (seeds, iters, updates), got {arr.shape}"
        )
    n_seeds, n_iters, n_updates_per_iter = arr.shape
    per_seed = arr.reshape(n_seeds, n_iters * n_updates_per_iter)
    total_env_steps = timesteps_per_iter * n_iters
    return make_curve(per_seed, total_env_steps, n_segments=n_iters)


def _has_saved_train_run(run) -> bool:
    return any("saved_train_run" in a.name for a in run.logged_artifacts())


def fetch_cole_curves_for_task(
    task: str,
    entity: str = "aht-project",
    project: str = "aht-benchmark",
    cache_dir=DEFAULT_CACHE_DIR,
    force_recompute: bool = False,
) -> list[COLERunCurves]:
    runs = find_benchmark_runs(
        algorithm="cole",
        task=task,
        entity=entity,
        project=project,
    )
    if not runs:
        raise ValueError(
            f"No finished COLE neurips:benchmark runs for task={task!r} in {entity}/{project}."
        )

    out: list[COLERunCurves] = []
    for run in runs:
        if not _has_saved_train_run(run):
            print(
                f"[cole] skipping run {run.id} (task={task}): no saved_train_run artifact "
                f"(likely log_train_out=False — needs rerun)."
            )
            continue

        print(f"\n[cole] processing run {run.id}  task={task}  state={run.state}")
        timesteps_per_iter = get_config_value(
            run.config, "algorithm.TOTAL_TIMESTEPS_PER_ITERATION"
        )
        if timesteps_per_iter is None:
            raise ValueError(
                f"Run {run.id} missing algorithm.TOTAL_TIMESTEPS_PER_ITERATION."
            )

        result = fetch_train_run_metrics_cached(
            run,
            artifact_kind="saved_train_run",
            entity=entity,
            project=project,
            cache_dir=cache_dir,
            force_recompute=force_recompute,
            reduce_per_update=True,
            extra_top_level_keys=("final_xp_matrix",),
        )
        partner_metrics = result["metrics"]
        xp_per_seed = np.asarray(result["final_xp_matrix"])
        if xp_per_seed.ndim != 3:
            raise ValueError(
                f"final_xp_matrix expected 3D (seeds, pop, pop), got {xp_per_seed.shape}"
            )
        xp_mean = xp_per_seed.mean(axis=0)  # match Eval/LastXPMatrix log

        out.append(
            COLERunCurves(
                run_id=run.id,
                task=task,
                partner=_partner_curve(partner_metrics, timesteps_per_iter),
                xp_matrix=xp_mean,
            )
        )

    if not out:
        raise ValueError(
            f"All COLE neurips:benchmark runs for {task} were missing saved_train_run "
            "artifacts. Need a rerun with logger.log_train_out=true."
        )
    return out

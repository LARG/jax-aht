"""Fetch CoMeDi per-seed training curves from wandb train_run artifacts.

Curves (per user request 2026-05-03):
  - Partner training return: `metrics["returned_episode_returns"]` from
    `saved_train_run`, shape (NUM_SEEDS, NUM_ITERATIONS, NUM_UPDATES_PER_ITER).
    CoMeDi adds one new partner per iteration, each iteration is its own
    IPPO training pass against the existing pool. We concatenate iterations
    along the time axis to produce a single (NUM_SEEDS, total_updates) curve
    showing the whole sequential training timeline.
  - Ego training return: `metrics["returned_episode_returns"]` from
    `ego_train_run`, shape (NUM_SEEDS, NUM_EGO_TRAIN_SEEDS, NUM_EGO_UPDATES).
    Same as FCP — mirrors `Train/Ego_returned_episode_returns`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.training_curves.common import (
    DEFAULT_CACHE_DIR,
    CurveData,
    extract_ego_curve,
    fetch_train_run_metrics_cached,
    find_benchmark_runs,
    get_config_value,
    make_curve,
)


@dataclass
class CoMeDiRunCurves:
    run_id: str
    task: str
    partner: CurveData
    ego: CurveData


def _partner_curve(metrics: dict, timesteps_per_iter: int) -> CurveData:
    arr = np.asarray(metrics["returned_episode_returns"])
    if arr.ndim != 3:
        raise ValueError(
            f"partner returned_episode_returns expected 3D (seeds, iters, updates), got {arr.shape}"
        )
    n_seeds, n_iters, n_updates_per_iter = arr.shape
    # Concatenate iterations along the time axis: each iteration is a separate
    # IPPO pass for a new partner, so the natural "training timeline" runs
    # iteration 0 → ... → iteration N-1.
    per_seed = arr.reshape(n_seeds, n_iters * n_updates_per_iter)
    total_env_steps = timesteps_per_iter * n_iters
    return make_curve(per_seed, total_env_steps, n_segments=n_iters)


def fetch_comedi_curves_for_task(
    task: str,
    entity: str = "aht-project",
    project: str = "aht-benchmark",
    cache_dir=DEFAULT_CACHE_DIR,
    force_recompute: bool = False,
) -> list[CoMeDiRunCurves]:
    runs = find_benchmark_runs(
        algorithm="comedi",
        task=task,
        entity=entity,
        project=project,
    )
    if not runs:
        raise ValueError(
            f"No finished CoMeDi neurips:benchmark runs for task={task!r} in {entity}/{project}."
        )

    out: list[CoMeDiRunCurves] = []
    for run in runs:
        print(f"\n[comedi] processing run {run.id}  task={task}  state={run.state}")
        timesteps_per_iter = get_config_value(
            run.config, "algorithm.TOTAL_TIMESTEPS_PER_ITERATION"
        )
        ego_total = get_config_value(
            run.config, "algorithm.ego_train_algorithm.TOTAL_TIMESTEPS"
        )
        if timesteps_per_iter is None or ego_total is None:
            raise ValueError(
                f"Run {run.id} missing config (TOTAL_TIMESTEPS_PER_ITERATION="
                f"{timesteps_per_iter}, ego_total={ego_total})."
            )

        partner_metrics = fetch_train_run_metrics_cached(
            run,
            artifact_kind="saved_train_run",
            entity=entity,
            project=project,
            cache_dir=cache_dir,
            force_recompute=force_recompute,
            reduce_per_update=True,
        )
        ego_metrics = fetch_train_run_metrics_cached(
            run,
            artifact_kind="ego_train_run",
            entity=entity,
            project=project,
            cache_dir=cache_dir,
            force_recompute=force_recompute,
            reduce_per_update=True,
        )

        out.append(
            CoMeDiRunCurves(
                run_id=run.id,
                task=task,
                partner=_partner_curve(partner_metrics, timesteps_per_iter),
                ego=extract_ego_curve(ego_metrics, ego_total),
            )
        )
    return out

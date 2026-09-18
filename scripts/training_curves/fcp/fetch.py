"""Fetch FCP per-seed training and ego-eval curves from wandb train_run artifacts.

Curve definitions (from inspection of run ikrlj1qe, FCP/coord_ring):
  - Partner curve: `metrics["returned_episode_returns"]` from `saved_train_run`,
    shape (NUM_SEEDS, PARTNER_POP_SIZE, NUM_PARTNER_UPDATES). Mean over the
    pop axis to get one curve per seed. This mirrors what `fcp.py:96-105` logs
    as `Train/Partner_returned_episode_returns` to wandb (but preserves seeds.

  - Ego curve: `metrics["returned_episode_returns"]` from `ego_train_run`,
    shape (NUM_SEEDS, NUM_EGO_TRAIN_SEEDS, NUM_EGO_UPDATES). Mirrors
    `Train/Ego_returned_episode_returns`. We deliberately do NOT use
    `eval_ep_last_info["returned_episode_returns"]` because that's the
    held-out eval return — sparse (NUM_CHECKPOINTS-limited).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.training_curves.common import (
    CurveData,
    DEFAULT_CACHE_DIR,
    extract_ego_curve,
    fetch_train_run_metrics_cached,
    find_benchmark_runs,
    get_config_value,
    make_curve,
)


@dataclass
class FCPRunCurves:
    run_id: str
    task: str
    partner: CurveData
    ego: CurveData


def _partner_curve(metrics: dict, total_timesteps: int) -> CurveData:
    arr = np.asarray(metrics["returned_episode_returns"])
    if arr.ndim != 3:
        raise ValueError(
            f"partner returned_episode_returns expected 3D (seeds, pop, updates), got {arr.shape}"
        )
    return make_curve(arr.mean(axis=1), total_timesteps)


def fetch_fcp_curves_for_task(
    task: str,
    entity: str = "aht-project",
    project: str = "aht-benchmark",
    cache_dir=DEFAULT_CACHE_DIR,
    force_recompute: bool = False,
) -> list[FCPRunCurves]:
    """Pull every FCP `neurips:benchmark` run on `task` and return per-seed curves."""
    runs = find_benchmark_runs(
        algorithm="fcp", task=task, entity=entity, project=project,
    )
    if not runs:
        raise ValueError(
            f"No finished FCP neurips:benchmark runs for task={task!r} in {entity}/{project}."
        )

    out: list[FCPRunCurves] = []
    for run in runs:
        print(f"\n[fcp] processing run {run.id}  task={task}  state={run.state}")
        partner_total = get_config_value(run.config, "algorithm.TOTAL_TIMESTEPS")
        ego_total = get_config_value(run.config, "algorithm.ego_train_algorithm.TOTAL_TIMESTEPS")
        if partner_total is None or ego_total is None:
            raise ValueError(
                f"Run {run.id} missing TOTAL_TIMESTEPS config "
                f"(partner={partner_total}, ego={ego_total})."
            )

        partner_metrics = fetch_train_run_metrics_cached(
            run, artifact_kind="saved_train_run",
            entity=entity, project=project,
            cache_dir=cache_dir, force_recompute=force_recompute,
            reduce_per_update=True,
        )
        ego_metrics = fetch_train_run_metrics_cached(
            run, artifact_kind="ego_train_run",
            entity=entity, project=project,
            cache_dir=cache_dir, force_recompute=force_recompute,
            reduce_per_update=True,
        )

        out.append(FCPRunCurves(
            run_id=run.id,
            task=task,
            partner=_partner_curve(partner_metrics, partner_total),
            ego=extract_ego_curve(ego_metrics, ego_total),
        ))

    return out

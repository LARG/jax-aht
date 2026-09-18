"""Fetch TrajeDi per-seed training-return curves from wandb train_run artifacts.

TrajeDi's `outs` is a single dict with `metrics["returned_episode_returns"]`
shape (NUM_SEEDS, NUM_UPDATES). Unlike FCP/CoMeDi, there is no inner pop axis
in the metric tree — the partner population is folded into the update axis
because the algorithm interleaves SP/XP/MP rollouts inside each update step.
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
class TrajeDiRunCurves:
    run_id: str
    task: str
    train: CurveData


def _train_curve(metrics: dict, total_timesteps: int) -> CurveData:
    arr = np.asarray(metrics["returned_episode_returns"])
    if arr.ndim != 2:
        raise ValueError(
            f"TrajeDi returned_episode_returns expected 2D (seeds, updates), got {arr.shape}"
        )
    return make_curve(arr, total_timesteps)


def fetch_trajedi_curves_for_task(
    task: str,
    entity: str = "aht-project",
    project: str = "aht-benchmark",
    cache_dir=DEFAULT_CACHE_DIR,
    force_recompute: bool = False,
) -> list[TrajeDiRunCurves]:
    runs = find_benchmark_runs(
        algorithm="trajedi",
        task=task,
        entity=entity,
        project=project,
    )
    if not runs:
        raise ValueError(
            f"No finished TrajeDi neurips:benchmark runs for task={task!r} in {entity}/{project}."
        )

    out: list[TrajeDiRunCurves] = []
    for run in runs:
        print(f"\n[trajedi] processing run {run.id}  task={task}  state={run.state}")
        total = get_config_value(run.config, "algorithm.TOTAL_TIMESTEPS")
        if total is None:
            raise ValueError(f"Run {run.id} missing algorithm.TOTAL_TIMESTEPS.")

        metrics = fetch_train_run_metrics_cached(
            run,
            artifact_kind="saved_train_run",
            entity=entity,
            project=project,
            cache_dir=cache_dir,
            force_recompute=force_recompute,
            reduce_per_update=True,
        )
        out.append(
            TrajeDiRunCurves(
                run_id=run.id,
                task=task,
                train=_train_curve(metrics, total),
            )
        )
    return out

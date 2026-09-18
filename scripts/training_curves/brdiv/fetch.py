"""Fetch BRDIV per-seed training and ego curves from wandb train_run artifacts.

BRDIV's `metrics["eval_ep_last_info"]["returned_episode_returns"]` has shape
(NUM_SEEDS, NUM_PARTNER_UPDATES, POP_SIZE^2, NUM_EVAL_EPS, NUM_AGENTS) — see
`BRDiv.py:769-797`. The SP/XP split via `_get_all_ids` is shared with LBRDIV
and lives in `common.sp_xp_masks`.
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
    sp_xp_masks,
)


@dataclass
class BRDivRunCurves:
    run_id: str
    task: str
    sp_partner: CurveData
    xp_partner: CurveData
    ego: CurveData


def _partner_curves(metrics: dict, total_timesteps: int) -> tuple[CurveData, CurveData]:
    arr = np.asarray(metrics["eval_ep_last_info"]["returned_episode_returns"])
    if arr.ndim != 5:
        raise ValueError(f"partner eval_ep_last_info expected 5D, got {arr.shape}")
    _, _, n_pairs, _, _ = arr.shape
    pop_size = int(round(n_pairs ** 0.5))
    if pop_size * pop_size != n_pairs:
        raise ValueError(f"expected pop^2 pairs, got {n_pairs}")

    sp_mask, xp_mask = sp_xp_masks(pop_size)
    sp = arr[:, :, sp_mask].mean(axis=(-3, -2, -1))
    xp = arr[:, :, xp_mask].mean(axis=(-3, -2, -1))
    return make_curve(sp, total_timesteps), make_curve(xp, total_timesteps)


def fetch_brdiv_curves_for_task(
    task: str,
    entity: str = "aht-project",
    project: str = "aht-benchmark",
    cache_dir=DEFAULT_CACHE_DIR,
    force_recompute: bool = False,
) -> list[BRDivRunCurves]:
    runs = find_benchmark_runs(
        algorithm="brdiv", task=task, entity=entity, project=project,
    )
    if not runs:
        raise ValueError(
            f"No finished BRDIV neurips:benchmark runs for task={task!r} in {entity}/{project}."
        )

    out: list[BRDivRunCurves] = []
    for run in runs:
        print(f"\n[brdiv] processing run {run.id}  task={task}  state={run.state}")
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

        sp_curve, xp_curve = _partner_curves(partner_metrics, partner_total)
        out.append(BRDivRunCurves(
            run_id=run.id, task=task,
            sp_partner=sp_curve, xp_partner=xp_curve,
            ego=extract_ego_curve(ego_metrics, ego_total),
        ))
    return out

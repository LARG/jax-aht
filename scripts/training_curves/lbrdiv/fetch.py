"""Fetch LBRDIV per-seed training and ego curves from wandb train_run artifacts.

Curve definitions (from `LBRDiv.py:1017-1037` and inspection):
  - Partner SP curve: `metrics["eval_ep_last_info"]["returned_episode_returns"]`
    from `saved_train_run`, shape
    (NUM_SEEDS, NUM_PARTNER_UPDATES, POP_SIZE^2, NUM_EVAL_EPISODES, NUM_AGENTS).
    Take the SP-mask slice (where conf_id == br_id), then mean over
    (pair, eval_eps, agents) → (NUM_SEEDS, NUM_PARTNER_UPDATES).
  - Partner XP curve: same array, ~SP-mask slice, same averaging.
  - Ego curve: identical to FCP — `metrics["returned_episode_returns"]` from
    `ego_train_run`, shape (NUM_SEEDS, NUM_EGO_TRAIN_SEEDS, NUM_EGO_UPDATES).
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
    make_env_steps_axis,
    sp_xp_masks,
)


@dataclass
class LMCurves:
    """Lagrange-multiplier curves: one line per (i,j) pair, mean over seeds.

    LBRDiv stores `lms_horizontal` / `lms_vertical` shape
    (NUM_SEEDS, NUM_PARTNER_UPDATES, POP_SIZE, POP_SIZE). To match the wandb
    visualization we mean over seeds and flatten the last two axes into a
    single "pair" axis, giving (POP_SIZE^2, NUM_PARTNER_UPDATES).
    """

    values: np.ndarray  # (pop_size^2, num_updates)
    env_steps: np.ndarray
    pair_labels: list[str]


@dataclass
class LBRDivRunCurves:
    run_id: str
    task: str
    sp_partner: CurveData
    xp_partner: CurveData
    ego: CurveData
    lm_horizontal: LMCurves
    lm_vertical: LMCurves


def _partner_curves(metrics: dict, total_timesteps: int) -> tuple[CurveData, CurveData]:
    arr = np.asarray(metrics["eval_ep_last_info"]["returned_episode_returns"])
    if arr.ndim != 5:
        raise ValueError(
            f"partner eval_ep_last_info expected 5D "
            f"(seeds, updates, pop^2, eval_eps, agents), got {arr.shape}"
        )
    _, _, n_pairs, _, _ = arr.shape
    pop_size = round(n_pairs**0.5)
    if pop_size * pop_size != n_pairs:
        raise ValueError(f"expected pop^2 pairs, got {n_pairs} (sqrt={pop_size})")

    sp_mask, xp_mask = sp_xp_masks(pop_size)

    # Mean over (pair, eval_eps, agents) → (seeds, updates)
    sp_per_seed = arr[:, :, sp_mask].mean(axis=(-3, -2, -1))
    xp_per_seed = arr[:, :, xp_mask].mean(axis=(-3, -2, -1))

    return make_curve(sp_per_seed, total_timesteps), make_curve(
        xp_per_seed, total_timesteps
    )


def _lm_curves(arr: np.ndarray, total_timesteps: int) -> LMCurves:
    if arr.ndim != 4:
        raise ValueError(
            f"LM array expected 4D (seeds, updates, pop, pop), got {arr.shape}"
        )
    _, n_updates, pop_size, _ = arr.shape
    seed_mean = arr.mean(axis=0)  # (updates, pop, pop)
    flat = seed_mean.reshape(n_updates, pop_size * pop_size).T  # (pop^2, updates)
    pair_labels = [f"({i},{j})" for i in range(pop_size) for j in range(pop_size)]
    env_steps = make_env_steps_axis(n_updates, total_timesteps)
    return LMCurves(values=flat, env_steps=env_steps, pair_labels=pair_labels)


def fetch_lbrdiv_curves_for_task(
    task: str,
    entity: str = "aht-project",
    project: str = "aht-benchmark",
    cache_dir=DEFAULT_CACHE_DIR,
    force_recompute: bool = False,
) -> list[LBRDivRunCurves]:
    runs = find_benchmark_runs(
        algorithm="lbrdiv",
        task=task,
        entity=entity,
        project=project,
    )
    if not runs:
        raise ValueError(
            f"No finished LBRDIV neurips:benchmark runs for task={task!r} in {entity}/{project}."
        )

    out: list[LBRDivRunCurves] = []
    for run in runs:
        print(f"\n[lbrdiv] processing run {run.id}  task={task}  state={run.state}")
        partner_total = get_config_value(run.config, "algorithm.TOTAL_TIMESTEPS")
        ego_total = get_config_value(
            run.config, "algorithm.ego_train_algorithm.TOTAL_TIMESTEPS"
        )
        if partner_total is None or ego_total is None:
            raise ValueError(
                f"Run {run.id} missing TOTAL_TIMESTEPS config "
                f"(partner={partner_total}, ego={ego_total})."
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

        sp_curve, xp_curve = _partner_curves(partner_metrics, partner_total)
        out.append(
            LBRDivRunCurves(
                run_id=run.id,
                task=task,
                sp_partner=sp_curve,
                xp_partner=xp_curve,
                ego=extract_ego_curve(ego_metrics, ego_total),
                lm_horizontal=_lm_curves(
                    np.asarray(partner_metrics["lms_horizontal"]), partner_total
                ),
                lm_vertical=_lm_curves(
                    np.asarray(partner_metrics["lms_vertical"]), partner_total
                ),
            )
        )

    return out

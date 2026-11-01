"""Fetch ROTATE per-seed training/eval curves from wandb train_run + heldout artifacts.

ROTATE is open-ended (OEL) — its `saved_train_run` artifact root is a tuple
`(teammate_outs, ego_outs)`. Each side has its own `metrics` subtree.

We use the **dense per-update training metrics** (not the eval metrics) so the
within-iter resolution isn't bottlenecked by `NUM_CHECKPOINTS` (which is 1 in
the benchmark runs and only writes 2 eval points per OEL iter):

  - ego_outs["metrics"]["returned_episode_returns"]
    shape (NUM_SEEDS, NUM_OEL_ITERS, NUM_EGO_SEEDS, NUM_EGO_UPDATES) — ego's
    masked-mean episode return during its training rollouts (paired with the
    current confederate). Logged every update step → dense.

  - teammate_outs["metrics"]["average_returns_br"] (older runs:
    "average_rewards_br")
    shape (NUM_SEEDS, NUM_OEL_ITERS, NUM_PARTNER_SEEDS, NUM_PARTNER_UPDATES) —
    confederate's average return when paired with the best-response during
    conf-vs-BR training rollouts. Logged every update step → dense. The older
    `average_rewards_br` spelling is per-step reward, so it sits on a lower
    scale than the newer return-based metric.

  - teammate_outs["metrics"]["train_regret"]
    same shape — `average_returns_br - average_returns_ego`, the regret the
    confederate objective maximizes. Absent from runs predating 2026-08-18.

  - heldout_eval_metrics artifact (separate, OEL): 5D
    (NUM_SEEDS, NUM_OEL_ITERS, NUM_HELDOUT_AGENTS, NUM_EVAL_EPS, NUM_AGENTS).
    Mean over (heldout_agents, eval_eps, agents) → one return per (seed,
    oel_iter). Intrinsically sparse — one measurement per iter.

We concatenate the OEL-iter axis with the within-iter step axis to produce a
single timeline for plotting (analogous to CoMeDi/COLE's iteration handling).
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
from scripts.wandb_utils.wandb_cache import fetch_run_eval_metrics_cached


@dataclass
class RotateRunCurves:
    run_id: str
    task: str
    ego_vs_conf: CurveData  # ego eval against confederate, per-seed
    conf_vs_confbr: CurveData  # confederate eval against best-response, per-seed
    ego_vs_heldout: CurveData | None  # None if heldout_eval_metrics artifact is missing
    train_regret: CurveData | None  # None for runs predating regret logging


# The 2026-08-18 ROTATE sync from continual-aht renamed the confederate-vs-BR
# metric (and switched it from per-step reward to episode return). Accept both
# so older and newer artifacts both load.
_CONF_VS_BR_KEYS = ("average_returns_br", "average_rewards_br")


def _first_present(metrics: dict, keys: tuple[str, ...]) -> str | None:
    for k in keys:
        if k in metrics:
            return k
    return None


def _reduce_partner_metric(
    arr: np.ndarray,
    name: str,
    partner_total_per_iter: int,
) -> CurveData:
    """Collapse a 4D partner-side metric to a per-seed curve on a single timeline.

    Input is (seeds, oel_iters, partner_seeds, partner_updates). We average over
    partner_seeds and concatenate the OEL-iter axis with the within-iter update
    axis, matching how CoMeDi/COLE flatten their iteration structure.
    """
    if arr.ndim != 4:
        raise ValueError(
            f"{name} expected 4D (seeds, oel_iters, partner_seeds, partner_updates), "
            f"got {arr.shape}"
        )
    per_seed = arr.mean(axis=2)  # mean over partner_seeds → (seeds, oel_iters, updates)
    n_seeds, n_iters, n_updates = per_seed.shape
    flat = per_seed.reshape(n_seeds, n_iters * n_updates)
    return make_curve(flat, partner_total_per_iter * n_iters, n_segments=n_iters)


def _conf_vs_confbr(teammate_metrics: dict, partner_total_per_iter: int) -> CurveData:
    """Per-seed dense conf-vs-BR signal during conf-vs-BR training rollouts.

    Uses the dense per-update metric rather than the sparse eval metric so we get
    one point per partner update (rather than NUM_CHECKPOINTS points per OEL
    iter). Newer runs log this as episode return (`average_returns_br`); older
    ones logged per-step reward (`average_rewards_br`), which sits on a lower
    scale — check which key was used before comparing across runs.
    """
    key = _first_present(teammate_metrics, _CONF_VS_BR_KEYS)
    if key is None:
        raise ValueError(
            f"no conf-vs-BR metric found; tried {_CONF_VS_BR_KEYS}. "
            f"Available: {sorted(teammate_metrics)}"
        )
    return _reduce_partner_metric(
        np.asarray(teammate_metrics[key]),
        key,
        partner_total_per_iter,
    )


def _train_regret(
    teammate_metrics: dict, partner_total_per_iter: int
) -> CurveData | None:
    """Per-seed dense regret signal logged by ROTATE as `Losses/TrainRegret`.

    `train_regret` is `average_returns_br - average_returns_ego` computed per
    partner update (`rotate.py`), i.e. how much better the best-response does
    against the confederate than the ego agent does — the quantity ROTATE's
    confederate objective maximizes. Shape matches `average_rewards_br`:
    (seeds, oel_iters, partner_seeds, partner_updates).

    Returns None for runs that predate regret logging (added 2026-08-18), so
    older artifacts degrade to an empty panel rather than failing collection.
    """
    if "train_regret" not in teammate_metrics:
        return None
    return _reduce_partner_metric(
        np.asarray(teammate_metrics["train_regret"]),
        "train_regret",
        partner_total_per_iter,
    )


def _ego_vs_conf(ego_metrics: dict, ego_total_per_iter: int) -> CurveData:
    """Per-seed dense ego-vs-conf signal: ego's training-rollout return.

    Uses `returned_episode_returns` (the dense per-update masked-mean episode
    return) rather than the sparse `eval_ep_last_info` metric. Ego rolls out
    against the current confederate during its training, so this is "ego paired
    with conf" measured every update step.
    """
    arr = np.asarray(ego_metrics["returned_episode_returns"])
    if arr.ndim != 4:
        raise ValueError(
            f"ego training return expected 4D (seeds, oel_iters, ego_seeds, ego_updates), "
            f"got {arr.shape}"
        )
    per_seed = arr.mean(axis=2)  # mean over ego_seeds → (seeds, oel_iters, updates)
    n_seeds, n_iters, n_updates = per_seed.shape
    flat = per_seed.reshape(n_seeds, n_iters * n_updates)
    return make_curve(flat, ego_total_per_iter * n_iters, n_segments=n_iters)


def _ego_vs_heldout(
    heldout_metrics: dict, ego_total_per_iter: int, n_iters: int
) -> CurveData:
    """Per-seed ego-vs-heldout return, one point per OEL iter.

    `heldout_metrics["returned_episode_returns"]` is shape
    (NUM_SEEDS, NUM_OEL_ITERS, NUM_HELDOUT_AGENTS, NUM_EVAL_EPS, NUM_AGENTS).
    Mean over the last three axes to get one return per (seed, oel_iter).
    The x-axis is cumulative env-steps spent on ego training up to that iter.
    """
    arr = np.asarray(heldout_metrics["returned_episode_returns"])
    if arr.ndim != 5:
        raise ValueError(
            f"heldout eval expected 5D (seeds, oel_iters, heldout, eval_eps, agents), got {arr.shape}"
        )
    per_seed = arr.mean(axis=(2, 3, 4))  # (seeds, oel_iters)
    # X-axis: cumulative env steps at the end of each OEL iter.
    # Iter k (0-indexed) corresponds to ego_total_per_iter * (k+1) env-steps.
    env_steps = np.arange(1, per_seed.shape[1] + 1) * ego_total_per_iter
    return CurveData(values=per_seed, env_steps=env_steps)


def fetch_rotate_curves_for_task(
    task: str,
    entity: str = "aht-project",
    project: str = "aht-benchmark",
    cache_dir=DEFAULT_CACHE_DIR,
    force_recompute: bool = False,
) -> list[RotateRunCurves]:
    runs = find_benchmark_runs(
        algorithm="rotate",
        task=task,
        entity=entity,
        project=project,
    )
    if not runs:
        raise ValueError(
            f"No finished ROTATE neurips:benchmark runs for task={task!r} in {entity}/{project}."
        )

    out: list[RotateRunCurves] = []
    for run in runs:
        # Some neurips:benchmark ROTATE runs were logged with log_train_out=False
        # and have no saved_train_run artifact. Skip rather than fail — they need
        # a rerun to be plottable.
        if not any("saved_train_run" in a.name for a in run.logged_artifacts()):
            print(
                f"[rotate] skipping run {run.id} (task={task}): no saved_train_run "
                "artifact (log_train_out=False — needs rerun)."
            )
            continue
        print(f"\n[rotate] processing run {run.id}  task={task}  state={run.state}")
        partner_total = get_config_value(
            run.config, "algorithm.TIMESTEPS_PER_ITER_PARTNER"
        )
        ego_total = get_config_value(run.config, "algorithm.TIMESTEPS_PER_ITER_EGO")
        if partner_total is None or ego_total is None:
            raise ValueError(
                f"Run {run.id} missing TIMESTEPS_PER_ITER_PARTNER / TIMESTEPS_PER_ITER_EGO "
                f"(partner={partner_total}, ego={ego_total})."
            )

        teammate_metrics = fetch_train_run_metrics_cached(
            run,
            artifact_kind="saved_train_run",
            entity=entity,
            project=project,
            cache_dir=cache_dir,
            force_recompute=force_recompute,
            tuple_index=0,
        )
        ego_metrics = fetch_train_run_metrics_cached(
            run,
            artifact_kind="saved_train_run",
            entity=entity,
            project=project,
            cache_dir=cache_dir,
            force_recompute=force_recompute,
            tuple_index=1,
        )

        conf_curve = _conf_vs_confbr(teammate_metrics, partner_total)
        ego_curve = _ego_vs_conf(ego_metrics, ego_total)
        regret_curve = _train_regret(teammate_metrics, partner_total)
        if regret_curve is None:
            print(
                f"  [warn] run {run.id} has no `train_regret` "
                "(predates 2026-08-18 regret logging)."
            )

        # Heldout eval — separate artifact, already a flat dict of arrays.
        # Some neurips:benchmark runs are missing this artifact; in that case
        # we render the heldout panel as N/A rather than failing the whole run.
        try:
            heldout = fetch_run_eval_metrics_cached(
                run.id,
                entity,
                project,
                cache_dir=cache_dir,
                force_recompute=force_recompute,
            )
            n_iters = np.asarray(
                teammate_metrics["eval_ep_last_info_br"]["returned_episode_returns"]
            ).shape[1]
            heldout_curve = _ego_vs_heldout(heldout, ego_total, n_iters)
        except ValueError as e:
            print(f"  [warn] heldout eval not available for {run.id}: {e}")
            heldout_curve = None

        out.append(
            RotateRunCurves(
                run_id=run.id,
                task=task,
                ego_vs_conf=ego_curve,
                conf_vs_confbr=conf_curve,
                ego_vs_heldout=heldout_curve,
                train_regret=regret_curve,
            )
        )

    if not out:
        raise ValueError(
            f"All ROTATE neurips:benchmark runs for {task} were missing saved_train_run "
            "artifacts. Need a rerun with logger.log_train_out=true."
        )
    return out

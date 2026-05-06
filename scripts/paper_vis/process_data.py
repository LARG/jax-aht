"""Load and summarize heldout eval metrics pulled from wandb."""
import pickle
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np

from common.plot_utils import get_metric_names
from common.stat_utils import compute_aggregate_stat_and_ci_per_task, compute_aggregate_stat_and_ci
from scripts.paper_vis.plot_globals import (
    GLOBAL_HELDOUT_CONFIG, BENCHMARK_PROJECT, ENTITY, TASK_TO_ENV_NAME,
)
from scripts.wandb_utils.wandb_cache import fetch_run_eval_metrics_cached, DEFAULT_CACHE_DIR


def detect_failed_seeds(
    eval_metrics: dict,
    metric_name: str,
    relative_threshold: float = 0.10,
    breadth_threshold: float = 0.80,
    oel_method: bool = False,
) -> np.ndarray:
    """Return a boolean mask (num_seeds,) where True indicates a collapsed seed.

    A seed is flagged when its mean return is below `relative_threshold` * the
    best seed's mean return for more than `breadth_threshold` fraction of
    heldout agents.

    Handles both standard 4D arrays (num_seeds, num_heldout_agents, num_eval_eps,
    num_agents_per_game) and OEL 5D arrays (adds num_oel_iter as axis 1; uses
    the last iteration).
    """
    data = eval_metrics[metric_name]
    if oel_method and data.ndim == 5:
        data = data[:, -1]  # (num_seeds, num_heldout_agents, num_eval_eps, num_agents_per_game)

    # Mean over eval_eps and agents_per_game -> (num_seeds, num_heldout_agents)
    seed_teammate_means = data.mean(axis=(-1, -2))

    best_per_teammate = seed_teammate_means.max(axis=0)  # (num_heldout_agents,)
    valid = best_per_teammate > 0
    if not valid.any():
        return np.zeros(seed_teammate_means.shape[0], dtype=bool)

    below = seed_teammate_means[:, valid] < relative_threshold * best_per_teammate[valid]
    return below.mean(axis=-1) > breadth_threshold


def load_results_for_task(
    task_name: str,
    run_specs: List[Tuple[str, str, bool]],
    force_recompute: bool = False,
    renormalize_metrics: bool = False,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    filter_failed_seeds: bool = False,
    failed_seed_relative_threshold: float = 0.10,
    failed_seed_breadth_threshold: float = 0.80,
) -> dict:
    """Fetch wandb eval-metric artifacts and compute summary stats for each method.

    Args:
        task_name: e.g. "overcooked-v1/coord_ring"
        run_specs: list of (display_name, run_id, is_oel) tuples
        force_recompute: skip summary-stats cache and recompute
        renormalize_metrics: if True, apply best-return normalization
        cache_dir: root of the local cache tree
        filter_failed_seeds: if True, remove seeds that collapsed (near-zero
            return across most heldout agents) before computing summary stats
        failed_seed_relative_threshold: seed-vs-teammate mean must be below
            this fraction of the best seed's mean to count as failed
        failed_seed_breadth_threshold: fraction of heldout agents that must
            be below the relative threshold to flag a seed as failed

    Returns:
        dict mapping display_name -> summary_data dict
    """
    cache_dir = Path(cache_dir)
    env_name = TASK_TO_ENV_NAME[task_name]
    metric_names = get_metric_names(env_name)

    best_returns = None
    if renormalize_metrics:
        from scripts.paper_vis.compute_best_returns import load_best_returns
        best_returns = load_best_returns(task_name, run_specs, cache_dir=cache_dir,
                                         force_recompute=force_recompute)
        print(f"Loaded best returns for {task_name}: {best_returns}")

    results = {}
    for display_name, run_id, is_oel in run_specs:
        run_ids = run_id if isinstance(run_id, list) else [run_id]
        run_id_str = "+".join(run_ids)

        safe_name = display_name.replace("/", "_").replace(" ", "_")
        suffix = "_renorm" if renormalize_metrics else ""
        if filter_failed_seeds:
            suffix += f"_filtered{failed_seed_relative_threshold}x{failed_seed_breadth_threshold}"
        summary_cache = (
            cache_dir / "summary_stats" / task_name / f"{safe_name}__{run_id_str}{suffix}.pkl"
        )

        if not force_recompute and summary_cache.exists():
            with open(summary_cache, "rb") as f:
                results[display_name] = pickle.load(f)
            print(f"Loaded cached summary for {display_name}")
            continue

        parts = [
            fetch_run_eval_metrics_cached(rid, ENTITY, BENCHMARK_PROJECT, cache_dir, force_recompute)
            for rid in run_ids
        ]
        if len(parts) == 1:
            eval_metrics = parts[0]
        else:
            eval_metrics = {
                k: np.concatenate([p[k] for p in parts], axis=0)
                for k in parts[0]
            }

        if renormalize_metrics and best_returns is not None:
            from scripts.paper_vis.compute_best_returns import (
                renormalize_eval_metrics,
                get_performance_bounds_from_run_config,
            )
            from scripts.wandb_utils.wandb_cache import fetch_run_config_cached
            run_config = fetch_run_config_cached(run_ids[0], ENTITY, BENCHMARK_PROJECT, cache_dir)
            perf_bounds = get_performance_bounds_from_run_config(run_config, task_name)
            eval_metrics = renormalize_eval_metrics(eval_metrics, perf_bounds, best_returns)

        had_filtered_seeds = False
        if filter_failed_seeds:
            failed_mask = detect_failed_seeds(
                eval_metrics, metric_names[0],
                relative_threshold=failed_seed_relative_threshold,
                breadth_threshold=failed_seed_breadth_threshold,
                oel_method=is_oel,
            )
            if failed_mask.any():
                had_filtered_seeds = True
                seed_means = eval_metrics[metric_names[0]].mean(
                    axis=tuple(range(1, eval_metrics[metric_names[0]].ndim))
                )
                warnings.warn(
                    f"[{display_name}] Filtering {failed_mask.sum()} failed seed(s) "
                    f"at index {np.where(failed_mask)[0].tolist()} "
                    f"(per-seed means: {seed_means.round(4).tolist()})",
                    stacklevel=2,
                )
                eval_metrics = {k: v[~failed_mask] for k, v in eval_metrics.items()}

        summary_data = heldout_metrics_per_agent(
            GLOBAL_HELDOUT_CONFIG, eval_metrics, metric_names, oel_method=is_oel
        )
        summary_data["_filtered_seeds"] = had_filtered_seeds

        summary_cache.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_cache, "wb") as f:
            pickle.dump(summary_data, f)
        print(f"Saved summary for {display_name} to {summary_cache}")

        results[display_name] = summary_data

    return results


def load_results_for_task_merged(
    task_name: str,
    run_specs: List[Tuple[str, str, bool]],
    bc_run_specs: List[Tuple[str, str, bool]],
    force_recompute: bool = False,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    filter_failed_seeds: bool = False,
    failed_seed_relative_threshold: float = 0.10,
    failed_seed_breadth_threshold: float = 0.80,
) -> dict:
    """Like `load_results_for_task` but merges each ego's training-time heldout
    artifact with its BC heldout-eval artifact along the partner axis BEFORE
    running the reducer. Both source artifacts are already per-partner
    normalized by their `performance_bounds`, so the concat is comparable.

    `bc_run_specs` indexes by display_name; cells without a matching BC entry
    fall back to the standard pool only. Failed-seed filtering uses the same
    `detect_failed_seeds` heuristic and threshold defaults as `load_results_for_task`.
    """
    cache_dir = Path(cache_dir)
    env_name = TASK_TO_ENV_NAME[task_name]
    metric_names = get_metric_names(env_name)

    bc_by_name = {dn: rid for dn, rid, _ in bc_run_specs}

    results = {}
    for display_name, run_id, is_oel in run_specs:
        run_ids = run_id if isinstance(run_id, list) else [run_id]
        run_id_str = "+".join(run_ids)

        bc_id = bc_by_name.get(display_name)
        bc_id_str = bc_id if bc_id else "none"

        safe_name = display_name.replace("/", "_").replace(" ", "_")
        suffix = ""
        if filter_failed_seeds:
            suffix += f"_filtered{failed_seed_relative_threshold}x{failed_seed_breadth_threshold}"
        summary_cache = (
            cache_dir / "summary_stats_merged" / task_name
            / f"{safe_name}__{run_id_str}__bc_{bc_id_str}{suffix}.pkl"
        )

        if not force_recompute and summary_cache.exists():
            with open(summary_cache, "rb") as f:
                results[display_name] = pickle.load(f)
            print(f"Loaded cached merged summary for {display_name}")
            continue

        # Pull the standard heldout-eval artifact(s) (handles list-pooling).
        old_parts = [
            fetch_run_eval_metrics_cached(rid, ENTITY, BENCHMARK_PROJECT, cache_dir, force_recompute)
            for rid in run_ids
        ]
        old_metrics = old_parts[0] if len(old_parts) == 1 else {
            k: np.concatenate([p[k] for p in old_parts], axis=0) for k in old_parts[0]
        }

        if bc_id is None:
            print(f"  no BC run registered for {display_name}; using old artifact only.")
            eval_metrics = old_metrics
        else:
            bc_metrics = fetch_run_eval_metrics_cached(
                bc_id, ENTITY, BENCHMARK_PROJECT, cache_dir, force_recompute
            )
            partner_axis = 2 if is_oel else 1   # OEL is 5D with seeds×iter prefix; partner axis shifts by 1
            common = sorted(set(old_metrics) & set(bc_metrics))
            eval_metrics = {
                k: np.concatenate([old_metrics[k], bc_metrics[k]], axis=partner_axis)
                for k in common
            }
            print(f"  merged {display_name}: old shape "
                  f"{old_metrics[common[0]].shape} + BC shape "
                  f"{bc_metrics[common[0]].shape} → {eval_metrics[common[0]].shape}")

        had_filtered_seeds = False
        if filter_failed_seeds:
            failed_mask = detect_failed_seeds(
                eval_metrics, metric_names[0],
                relative_threshold=failed_seed_relative_threshold,
                breadth_threshold=failed_seed_breadth_threshold,
                oel_method=is_oel,
            )
            if failed_mask.any():
                had_filtered_seeds = True
                seed_means = eval_metrics[metric_names[0]].mean(
                    axis=tuple(range(1, eval_metrics[metric_names[0]].ndim))
                )
                warnings.warn(
                    f"[{display_name}] Filtering {failed_mask.sum()} failed seed(s) "
                    f"at index {np.where(failed_mask)[0].tolist()} "
                    f"(per-seed means: {seed_means.round(4).tolist()})",
                    stacklevel=2,
                )
                eval_metrics = {k: v[~failed_mask] for k, v in eval_metrics.items()}

        summary_data = heldout_metrics_per_agent(
            GLOBAL_HELDOUT_CONFIG, eval_metrics, metric_names, oel_method=is_oel
        )
        summary_data["_filtered_seeds"] = had_filtered_seeds

        summary_cache.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_cache, "wb") as f:
            pickle.dump(summary_data, f)
        print(f"Saved merged summary for {display_name} → {summary_cache}")

        results[display_name] = summary_data

    return results


def heldout_metrics_per_agent(config, eval_metrics, metric_names: tuple, oel_method: bool):
    """Compute aggregate stat and CI over heldout agents.

    For OEL methods (5D arrays), uses the last iteration.
    For standard methods (4D arrays), uses all seeds.

    Shape conventions:
        OEL:      (num_seeds, num_oel_iter, num_heldout_agents, num_eval_eps, num_agents_per_game)
        Standard: (num_seeds, num_heldout_agents, num_eval_eps, num_agents_per_game)
    """
    num_heldout_agents = eval_metrics[metric_names[0]].shape[-3]

    summary_data = {}
    aggregate_stat = config["global_heldout_settings"]["AGGREGATE_STAT"]

    for metric_name in metric_names:
        if oel_method:
            data = (
                eval_metrics[metric_name][:, -1]
                .mean(axis=-1)
                .transpose(0, 2, 1)
                .reshape(-1, num_heldout_agents)
            )
        else:
            data = (
                eval_metrics[metric_name]
                .mean(axis=-1)
                .transpose(0, 2, 1)
                .reshape(-1, num_heldout_agents)
            )

        data = np.array(data)

        point_est_per_task, interval_ests_per_task = compute_aggregate_stat_and_ci_per_task(
            data, aggregate_stat, return_interval_est=True
        )
        lower_ci_per_task = interval_ests_per_task[:, 0]
        upper_ci_per_task = interval_ests_per_task[:, 1]

        point_est_all, interval_ests_all = compute_aggregate_stat_and_ci(
            data, aggregate_stat, return_interval_est=True
        )
        lower_ci_all = interval_ests_all[0]
        upper_ci_all = interval_ests_all[1]

        summary_data[metric_name] = {
            f"overall_{aggregate_stat}": point_est_all,
            "overall_lower_ci": lower_ci_all,
            "overall_upper_ci": upper_ci_all,
            f"{aggregate_stat}_per_agent": point_est_per_task,
            "per_agent_lower_ci": lower_ci_per_task,
            "per_agent_upper_ci": upper_ci_per_task,
        }

    return summary_data

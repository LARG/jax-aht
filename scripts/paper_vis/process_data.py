"""Load and summarize heldout eval metrics pulled from wandb."""
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np

from common.plot_utils import get_metric_names
from common.stat_utils import compute_aggregate_stat_and_ci_per_task, compute_aggregate_stat_and_ci
from scripts.paper_vis.plot_globals import (
    GLOBAL_HELDOUT_CONFIG, BENCHMARK_PROJECT, ENTITY, TASK_TO_ENV_NAME,
)
from scripts.wandb_utils.wandb_cache import fetch_run_eval_metrics_cached, DEFAULT_CACHE_DIR


def load_results_for_task(
    task_name: str,
    run_specs: List[Tuple[str, str, bool]],
    force_recompute: bool = False,
    renormalize_metrics: bool = False,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> dict:
    """Fetch wandb eval-metric artifacts and compute summary stats for each method.

    Args:
        task_name: e.g. "overcooked-v1/coord_ring"
        run_specs: list of (display_name, run_id, is_oel) tuples
        force_recompute: skip summary-stats cache and recompute
        renormalize_metrics: if True, apply best-return normalization
        cache_dir: root of the local cache tree

    Returns:
        dict mapping display_name -> summary_data dict
    """
    cache_dir = Path(cache_dir)
    env_name = TASK_TO_ENV_NAME[task_name]
    metric_names = get_metric_names(env_name)

    best_returns = None
    if renormalize_metrics:
        from scripts.paper_vis.compute_best_returns import load_best_returns
        best_returns = load_best_returns(task_name, run_specs, cache_dir=cache_dir)
        print(f"Loaded best returns for {task_name}: {best_returns}")

    results = {}
    for display_name, run_id, is_oel in run_specs:
        run_ids = run_id if isinstance(run_id, list) else [run_id]
        run_id_str = "+".join(run_ids)

        safe_name = display_name.replace("/", "_").replace(" ", "_")
        suffix = "_renorm" if renormalize_metrics else ""
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

        summary_data = heldout_metrics_per_agent(
            GLOBAL_HELDOUT_CONFIG, eval_metrics, metric_names, oel_method=is_oel
        )

        summary_cache.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_cache, "wb") as f:
            pickle.dump(summary_data, f)
        print(f"Saved summary for {display_name} to {summary_cache}")

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

"""Compute and cache the best observed returns for heldout agents.

Normalization pipeline:
  1. The stored eval metrics are already min-max normalized using per-agent
     performance bounds that were active when each run was evaluated.
  2. We unnormalize them:  raw = normalized * (upper - lower) + lower
  3. We find the best raw return across all benchmark runs for each heldout agent.
  4. These best returns are used to re-normalize:  new = raw / best_return
     so that 1.0 corresponds to the best observed performance.

Performance bounds are read from each run's wandb config (not from the live
codebase) to ensure reproducibility across config changes.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from scripts.paper_vis.plot_globals import BENCHMARK_PROJECT, ENTITY
from scripts.wandb_utils.wandb_cache import (
    DEFAULT_CACHE_DIR,
    fetch_run_config_cached,
    fetch_run_eval_metrics_cached,
)


# ---------------------------------------------------------------------------
# Performance-bounds helpers
# ---------------------------------------------------------------------------

def get_performance_bounds_from_run_config(run_config: dict, task_name: str) -> List[Optional[dict]]:
    """Extract per-heldout-agent performance bounds from a wandb run config.

    Iterates over the heldout set config in the same order that
    ``load_heldout_set`` would, producing one bounds-dict per individual agent
    (i.e. one entry per model checkpoint for RL agents, one entry for each
    heuristic agent).

    Returns:
        List of dicts ``{metric_name: [lower, upper]}``, one per heldout agent,
        in the same order as the heldout-agent dimension of the eval metrics.
        Entries are ``None`` when no bounds are defined.
    """
    heldout_set_config = run_config.get("heldout_set", {}).get(task_name, {})
    if not heldout_set_config:
        raise ValueError(
            f"No heldout_set config found for task '{task_name}' in run config. "
            f"Available tasks: {list(run_config.get('heldout_set', {}).keys())}"
        )

    bounds_list = []
    for agent_config in heldout_set_config.values():
        performance_bounds = agent_config.get("performance_bounds", None)

        if "path" in agent_config:
            # RL agent — one entry per model checkpoint
            idx_list = agent_config.get("idx_list", [])
            n_models = len(idx_list)

            if performance_bounds is None:
                bounds_list.extend([None] * n_models)
                continue

            # Detect whether bounds are per-model (list-of-lists) or shared
            first_val = next(iter(performance_bounds.values()))
            per_model = isinstance(first_val[0], (list, tuple))

            for i in range(n_models):
                if per_model:
                    bounds_list.append({k: v[i] for k, v in performance_bounds.items()})
                else:
                    bounds_list.append({k: v for k, v in performance_bounds.items()})
        else:
            # Heuristic agent — single entry
            bounds_list.append(
                {k: v for k, v in performance_bounds.items()} if performance_bounds else None
            )

    return bounds_list


# ---------------------------------------------------------------------------
# Per-run returns extraction
# ---------------------------------------------------------------------------

def extract_returns_for_run(
    eval_metrics: dict,
    perf_bounds: List[Optional[dict]],
    is_oel: bool,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Unnormalize eval metrics and compute the best mean return per heldout agent.

    Returns:
        Dict mapping metric_name ->
            (mean_returns, best_seed_indices, best_iter_indices)
        Arrays have shape (num_heldout_agents,).
        ``best_iter_indices`` is all-None for non-OEL methods.
    """
    results = {}

    for metric_name, data in eval_metrics.items():
        data = np.array(data)
        unnorm = np.copy(data)

        if is_oel:
            if data.ndim != 5:
                print(f"Warning: expected 5-D OEL data for {metric_name}, got {data.ndim}-D. Skipping.")
                continue
            num_heldout = data.shape[2]

            for h in range(min(num_heldout, len(perf_bounds))):
                bounds = perf_bounds[h]
                if bounds and metric_name in bounds:
                    lo, hi = bounds[metric_name]
                    unnorm[:, :, h, :, :] = data[:, :, h, :, :] * (hi - lo) + lo

            # Mean over agents-per-game and eval episodes
            # shape → (num_seeds, num_oel_iter, num_heldout)
            mean_over_eps = unnorm.mean(axis=-1).mean(axis=-1)

            mean_returns = np.zeros(num_heldout)
            best_seed_idx = np.zeros(num_heldout, dtype=int)
            best_iter_idx = np.zeros(num_heldout, dtype=int)
            for h in range(num_heldout):
                s, it = np.unravel_index(
                    np.argmax(mean_over_eps[:, :, h]),
                    mean_over_eps[:, :, h].shape,
                )
                mean_returns[h] = mean_over_eps[s, it, h]
                best_seed_idx[h] = s
                best_iter_idx[h] = it

        elif data.ndim == 4:
            num_heldout = data.shape[1]

            for h in range(min(num_heldout, len(perf_bounds))):
                bounds = perf_bounds[h]
                if bounds and metric_name in bounds:
                    lo, hi = bounds[metric_name]
                    unnorm[:, h, :, :] = data[:, h, :, :] * (hi - lo) + lo

            # Mean over agents-per-game and eval episodes
            # shape → (num_seeds, num_heldout)
            mean_over_eps = unnorm.mean(axis=-1).mean(axis=-1)

            mean_returns = np.zeros(num_heldout)
            best_seed_idx = np.zeros(num_heldout, dtype=int)
            best_iter_idx = np.full(num_heldout, None, dtype=object)
            for h in range(num_heldout):
                s = int(np.argmax(mean_over_eps[:, h]))
                mean_returns[h] = mean_over_eps[s, h]
                best_seed_idx[h] = s
        else:
            print(f"Warning: unexpected data shape {data.shape} for {metric_name}. Skipping.")
            continue

        results[metric_name] = (mean_returns, best_seed_idx, best_iter_idx)

    return results


# ---------------------------------------------------------------------------
# Best-returns computation and caching
# ---------------------------------------------------------------------------

def compute_best_returns(
    task_name: str,
    all_run_specs: List[Tuple[str, str, bool]],
    entity: str = ENTITY,
    project: str = BENCHMARK_PROJECT,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> dict:
    """Compute the best unnormalized return per heldout agent across all runs.

    The best returns are guaranteed to be at least as high as each agent's
    original upper bound (so that re-normalization never makes results look
    worse than the original normalization).

    Returns:
        Dict mapping metric_name -> list of best returns (one per heldout agent)
    """
    best_returns: Dict[str, np.ndarray] = {}
    original_upper_bounds: Dict[str, List[float]] = {}

    for display_name, run_id, is_oel in all_run_specs:
        if not run_id:
            continue

        run_ids = run_id if isinstance(run_id, list) else [run_id]
        print(f"\nProcessing {display_name} (run {'+'.join(run_ids)}) ...")
        parts = [fetch_run_eval_metrics_cached(rid, entity, project, cache_dir) for rid in run_ids]
        if len(parts) == 1:
            eval_metrics = parts[0]
        else:
            eval_metrics = {
                k: np.concatenate([p[k] for p in parts], axis=0)
                for k in parts[0]
            }
        run_config = fetch_run_config_cached(run_ids[0], entity, project, cache_dir)
        perf_bounds = get_performance_bounds_from_run_config(run_config, task_name)

        # Record original upper bounds (from first run that has them)
        for h, bounds in enumerate(perf_bounds):
            if bounds:
                for metric_name, (lo, hi) in bounds.items():
                    if metric_name not in original_upper_bounds:
                        original_upper_bounds[metric_name] = []
                    if h >= len(original_upper_bounds[metric_name]):
                        original_upper_bounds[metric_name].append(hi)

        returns_data = extract_returns_for_run(eval_metrics, perf_bounds, is_oel)

        # Determine reference heldout count from the first metric of this run
        if returns_data:
            sample_metric = next(iter(returns_data))
            run_n_heldout = len(returns_data[sample_metric][0])
            if best_returns:
                ref_n_heldout = len(next(iter(best_returns.values())))
                if run_n_heldout != ref_n_heldout:
                    print(
                        f"  WARNING: skipping {display_name} — heldout agent count "
                        f"{run_n_heldout} != expected {ref_n_heldout} "
                        f"(run was evaluated against a different heldout set)"
                    )
                    continue

        for metric_name, (cur_returns, _, _) in returns_data.items():
            if metric_name not in best_returns:
                best_returns[metric_name] = cur_returns.copy()
            else:
                best_returns[metric_name] = np.maximum(best_returns[metric_name], cur_returns)

    if not best_returns:
        raise ValueError(f"No valid returns found for task '{task_name}'.")

    # Ensure best returns are at least as high as the original upper bounds
    for metric_name, values in best_returns.items():
        uppers = original_upper_bounds.get(metric_name, [])
        for i in range(min(len(values), len(uppers))):
            if values[i] < uppers[i]:
                print(
                    f"Clamping best return for heldout agent {i}, {metric_name}: "
                    f"{values[i]:.4f} → {uppers[i]:.4f}"
                )
                values[i] = uppers[i]

    return {k: v.tolist() for k, v in best_returns.items()}


def load_best_returns(
    task_name: str,
    all_run_specs: List[Tuple[str, str, bool]],
    entity: str = ENTITY,
    project: str = BENCHMARK_PROJECT,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force_recompute: bool = False,
) -> dict:
    """Return cached best returns, computing and caching them if necessary."""
    safe_task = task_name.replace("/", "__")
    cache_path = Path(cache_dir) / "best_returns" / f"{safe_task}.json"

    if not force_recompute and cache_path.exists():
        print(f"Loading best returns from cache: {cache_path}")
        with open(cache_path, "r") as f:
            return json.load(f)

    best_returns = compute_best_returns(task_name, all_run_specs, entity, project, cache_dir)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(best_returns, f, indent=2)
    print(f"Saved best returns to {cache_path}")
    return best_returns


# ---------------------------------------------------------------------------
# Renormalization
# ---------------------------------------------------------------------------

def renormalize_eval_metrics(
    eval_metrics: dict,
    perf_bounds: List[Optional[dict]],
    best_returns: dict,
) -> dict:
    """Unnormalize eval metrics then renormalize by best observed returns.

    Args:
        eval_metrics: raw (already normalized) arrays from wandb artifact
        perf_bounds: per-heldout-agent bounds from ``get_performance_bounds_from_run_config``
        best_returns: dict metric_name -> list of best returns per agent

    Returns:
        dict with the same keys as eval_metrics, values rescaled so that 1.0
        corresponds to the best observed return for each heldout agent.
    """
    renorm = {}

    for metric_name, data in eval_metrics.items():
        data = np.array(data)
        out = np.copy(data)

        if data.ndim == 5:
            num_heldout = data.shape[2]
            heldout_dim = 2
        elif data.ndim == 4:
            num_heldout = data.shape[1]
            heldout_dim = 1
        else:
            print(f"Warning: unexpected shape {data.shape} for {metric_name}. Keeping original.")
            renorm[metric_name] = data
            continue

        if metric_name not in best_returns:
            renorm[metric_name] = data
            continue

        br = best_returns[metric_name]
        agents_to_process = min(num_heldout, len(perf_bounds), len(br))

        for h in range(agents_to_process):
            bounds = perf_bounds[h]
            if not bounds or metric_name not in bounds:
                continue
            lo, hi = bounds[metric_name]
            best = br[h]
            if best <= 0:
                print(f"Warning: best_return={best} for {metric_name} agent {h}. Keeping original.")
                continue

            if heldout_dim == 2:
                raw = data[:, :, h, :, :] * (hi - lo) + lo
                out[:, :, h, :, :] = raw / best
            else:
                raw = data[:, h, :, :] * (hi - lo) + lo
                out[:, h, :, :] = raw / best

        renorm[metric_name] = out

    return renorm

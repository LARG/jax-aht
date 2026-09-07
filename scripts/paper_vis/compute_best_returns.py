"""Compute and cache the best observed returns for heldout agents.

Normalization pipeline:
  1. The stored eval metrics are already min-max normalized using per-agent
     performance bounds that were active when each run was evaluated.
  2. We unnormalize them:  raw = normalized * (upper - lower) + lower
  3. We find the best raw return across all benchmark runs for each heldout agent.
  4. These best returns are used to re-normalize:  new = raw / best_return
     so that 1.0 corresponds to the best observed performance.

Performance bounds are read from each run's wandb config (not from the live
codebase) to ensure reproducibility across config changes. Partners are matched
across runs *by name* (see ``heldout_partners``), so runs evaluated against
different-sized heldout sets (e.g. with/without the human proxy) contribute to
the same per-partner maxima. The cached JSON lists best returns in the live
yaml's partner order (``_labels`` key) so that older consumers indexing by
position keep working.
"""

import json
from pathlib import Path

import numpy as np

from scripts.paper_vis.heldout_partners import (
    canonical_task_labels,
    load_run_eval_metrics,
)
from scripts.paper_vis.plot_globals import BENCHMARK_PROJECT, ENTITY
from scripts.wandb_utils.wandb_cache import DEFAULT_CACHE_DIR

# ---------------------------------------------------------------------------
# Per-run returns extraction
# ---------------------------------------------------------------------------


def extract_returns_for_run(
    eval_metrics: dict,
    perf_bounds: list[dict | None],
    is_oel: bool,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
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
                print(
                    f"Warning: expected 5-D OEL data for {metric_name}, got {data.ndim}-D. Skipping."
                )
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
            print(
                f"Warning: unexpected data shape {data.shape} for {metric_name}. Skipping."
            )
            continue

        results[metric_name] = (mean_returns, best_seed_idx, best_iter_idx)

    return results


# ---------------------------------------------------------------------------
# Best-returns computation and caching
# ---------------------------------------------------------------------------


def compute_best_returns(
    task_name: str,
    all_run_specs: list[tuple[str, str, bool]],
    entity: str = ENTITY,
    project: str = BENCHMARK_PROJECT,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    bc_run_specs: list[tuple[str, str, bool]] | None = None,
) -> dict:
    """Compute the best unnormalized return per heldout partner across all runs.

    Partners are matched by name across runs, so a run that lacks the human
    proxy simply contributes nothing to that partner's maximum. ``bc_run_specs``
    (display_name -> BC eval run id) lets runs without a built-in human proxy
    contribute their separate BC evaluation.

    The best returns are guaranteed to be at least as high as each partner's
    original upper bound (so that re-normalization never makes results look
    worse than the original normalization).

    Returns:
        Dict mapping metric_name -> list of best returns, in the order given by
        the ``_labels`` entry (the live yaml's canonical partner order).
    """
    bc_by_name = {dn: rid for dn, rid, _ in (bc_run_specs or [])}
    best: dict[str, dict[str, float]] = {}
    original_upper: dict[str, dict[str, float]] = {}

    for display_name, run_id, is_oel in all_run_specs:
        if not run_id:
            continue
        run_ids = run_id if isinstance(run_id, list) else [run_id]
        print(f"\nProcessing {display_name} (run {'+'.join(run_ids)}) ...")
        eval_metrics, labels, perf_bounds, _ = load_run_eval_metrics(
            task_name,
            run_id,
            is_oel,
            cache_dir=cache_dir,
            bc_run_id=bc_by_name.get(display_name),
        )

        for label, bounds in zip(labels, perf_bounds):
            if not bounds:
                continue
            for metric_name, (_lo, hi) in bounds.items():
                cur = original_upper.setdefault(metric_name, {})
                cur[label] = max(cur.get(label, hi), hi)

        returns_data = extract_returns_for_run(eval_metrics, perf_bounds, is_oel)
        for metric_name, (cur_returns, _, _) in returns_data.items():
            cur = best.setdefault(metric_name, {})
            for label, value in zip(labels, cur_returns):
                cur[label] = max(cur.get(label, -np.inf), float(value))

    if not best:
        raise ValueError(f"No valid returns found for task '{task_name}'.")

    # Ensure best returns are at least as high as the original upper bounds
    for metric_name, values in best.items():
        for label, hi in original_upper.get(metric_name, {}).items():
            if label in values and values[label] < hi:
                print(
                    f"Clamping best return for heldout agent {label}, {metric_name}: "
                    f"{values[label]:.4f} -> {hi:.4f}"
                )
                values[label] = hi

    canonical = canonical_task_labels(task_name)
    extra = sorted({lbl for v in best.values() for lbl in v} - set(canonical))
    if extra:
        print(f"WARNING: partners not in live yaml for {task_name}: {extra}")
    labels_out = canonical + extra
    out = {
        metric_name: [values.get(lbl) for lbl in labels_out]
        for metric_name, values in best.items()
    }
    out["_labels"] = labels_out
    return out


def _run_specs_fingerprint(
    all_run_specs: list[tuple[str, str, bool]],
    bc_run_specs: list[tuple[str, str, bool]] | None = None,
) -> str:
    """Stable short hash over the run IDs in all_run_specs (and BC run IDs).

    Ego and unified plots pass different run_specs for the same task, so the
    best-returns cache must be keyed on run IDs as well as task name to avoid
    cross-contamination between plot types.
    """
    import hashlib

    run_ids_str = "|".join(
        ("+".join(rid) if isinstance(rid, list) else rid)
        for _, rid, _ in all_run_specs
        if rid
    )
    if bc_run_specs:
        run_ids_str += "|bc:" + "|".join(
            ("+".join(rid) if isinstance(rid, list) else rid)
            for _, rid, _ in bc_run_specs
            if rid
        )
    return hashlib.md5(run_ids_str.encode()).hexdigest()[:8]


def load_best_returns(
    task_name: str,
    all_run_specs: list[tuple[str, str, bool]],
    entity: str = ENTITY,
    project: str = BENCHMARK_PROJECT,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force_recompute: bool = False,
    cache_filename: str | None = None,
    bc_run_specs: list[tuple[str, str, bool]] | None = None,
) -> dict:
    """Return cached best returns, computing and caching them if necessary.

    If `cache_filename` is provided, it overrides the default fingerprint-based
    name (relative to `cache_dir/best_returns/`). Used by the BC-only flow so
    the plotter can find the JSON without knowing the run-IDs.
    """
    safe_task = task_name.replace("/", "__")
    if cache_filename:
        cache_path = Path(cache_dir) / "best_returns" / cache_filename
    else:
        fingerprint = _run_specs_fingerprint(all_run_specs, bc_run_specs)
        cache_path = (
            Path(cache_dir) / "best_returns" / f"{safe_task}__{fingerprint}.json"
        )

    if not force_recompute and cache_path.exists():
        with open(cache_path, "r") as f:
            cached = json.load(f)
        if "_labels" in cached:
            print(f"Loading best returns from cache: {cache_path}")
            return cached
        print(f"Best-returns cache {cache_path} predates named partners; recomputing.")

    best_returns = compute_best_returns(
        task_name, all_run_specs, entity, project, cache_dir, bc_run_specs=bc_run_specs
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(best_returns, f, indent=2)
    print(f"Saved best returns to {cache_path}")

    # Invalidate renorm summary stats that were computed against old best_returns
    stale_dir = Path(cache_dir) / "summary_stats" / task_name
    if stale_dir.exists():
        stale_files = list(stale_dir.glob("*_renorm*.pkl"))
        for f in stale_files:
            f.unlink()
        if stale_files:
            print(
                f"Deleted {len(stale_files)} stale renorm cache file(s) in {stale_dir}"
            )

    return best_returns


# ---------------------------------------------------------------------------
# Renormalization
# ---------------------------------------------------------------------------


def renormalize_eval_metrics(
    eval_metrics: dict,
    perf_bounds: list[dict | None],
    best_returns: dict,
    labels: list[str],
) -> dict:
    """Unnormalize eval metrics then renormalize by best observed returns.

    Args:
        eval_metrics: raw (already normalized) arrays from wandb artifact
        perf_bounds: per-partner bounds aligned with ``labels``
        best_returns: output of ``compute_best_returns`` (has a ``_labels`` key)
        labels: partner labels along the artifact's partner axis

    Returns:
        dict with the same keys as eval_metrics, values rescaled so that 1.0
        corresponds to the best observed return for each heldout partner.
    """
    br_labels = best_returns["_labels"]
    renorm = {}

    for metric_name, data in eval_metrics.items():
        data = np.array(data)
        out = np.copy(data)

        if data.ndim == 5:
            heldout_dim = 2
        elif data.ndim == 4:
            heldout_dim = 1
        else:
            print(
                f"Warning: unexpected shape {data.shape} for {metric_name}. Keeping original."
            )
            renorm[metric_name] = data
            continue

        if metric_name not in best_returns:
            renorm[metric_name] = data
            continue

        br = dict(zip(br_labels, best_returns[metric_name]))
        num_heldout = data.shape[heldout_dim]
        if num_heldout != len(labels):
            raise ValueError(
                f"{metric_name}: {num_heldout} partners in data but {len(labels)} labels"
            )

        for h, (label, bounds) in enumerate(zip(labels, perf_bounds)):
            if not bounds or metric_name not in bounds:
                continue
            best = br.get(label)
            if best is None or best <= 0:
                print(
                    f"Warning: no usable best_return for {metric_name} partner '{label}'. Keeping original."
                )
                continue
            lo, hi = bounds[metric_name]

            if heldout_dim == 2:
                raw = data[:, :, h, :, :] * (hi - lo) + lo
                out[:, :, h, :, :] = raw / best
            else:
                raw = data[:, h, :, :] * (hi - lo) + lo
                out[:, h, :, :] = raw / best

        renorm[metric_name] = out

    return renorm

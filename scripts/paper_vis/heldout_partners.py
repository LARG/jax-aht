"""Per-run heldout partner bookkeeping for the paper plots.

Heldout partners are identified by name rather than artifact index: wandb
reorders config dict keys, so the partner order is read from the logged
``HeldoutEval/FinalEgoVsHeldout`` table and bounds are looked up by name.
For runs whose heldout set has no ``human_proxy``, ``load_run_eval_metrics``
appends the separate BC eval run (``BC_BENCHMARK_RUNS``) along the partner axis.
"""

import re
import warnings
from pathlib import Path

import numpy as np
import omegaconf

from scripts.paper_vis.plot_globals import (
    BENCHMARK_PROJECT,
    ENTITY,
    GLOBAL_HELDOUT_CONFIG,
    HUMAN_PROXY_AGENTS,
)
from scripts.wandb_utils.wandb_cache import (
    DEFAULT_CACHE_DIR,
    fetch_run_config_cached,
    fetch_run_eval_metrics_cached,
    fetch_run_heldout_names_cached,
)

HUMAN_PROXY_LABEL = "human_proxy"
# Partner keys used by the separate BC heldout-eval runs; the first one found is
# the canonical human proxy (bc_run_0 for overcooked, bc for LBF).
BC_RUN_CANONICAL_KEYS = ("bc_run_0", "bc")

HumanProxySource = str | None  # "builtin" | "bc_merge" | None


def is_human_proxy(label: str) -> bool:
    return any(p in label for p in HUMAN_PROXY_AGENTS)


def _n_models(agent_cfg: dict) -> int:
    """Number of partner-axis entries an agent config expands to."""
    if "path" in agent_cfg and "idx_list" in agent_cfg:
        return len(agent_cfg["idx_list"])
    bounds = agent_cfg.get("performance_bounds") or {}
    if bounds:
        first = next(iter(bounds.values()))
        if isinstance(first[0], (list, tuple)):
            return len(first)
    return 1


def _label(base: str, i: int, n: int) -> str:
    return base if n == 1 else f"{base}[{i}]"


def expand_labels(heldout_cfg: dict, key_order: list[str]) -> list[str]:
    """Expand agent keys into one canonical label per partner-axis entry."""
    labels: list[str] = []
    for key in key_order:
        n = _n_models(heldout_cfg[key])
        labels.extend(_label(key, i, n) for i in range(n))
    return labels


def canonical_task_labels(task_name: str) -> list[str]:
    """Canonical partner labels for a task, in live-yaml order (human proxy last)."""
    cfg = omegaconf.OmegaConf.to_container(
        GLOBAL_HELDOUT_CONFIG["heldout_set"][task_name], resolve=True
    )
    cfg = {k: v for k, v in cfg.items() if v is not None}
    return expand_labels(cfg, list(cfg.keys()))


def labels_from_table(columns: list[str], heldout_cfg: dict) -> list[str]:
    """Map logged table columns (``"comedi (1, 0)"``) to canonical labels."""
    counts: dict[str, int] = {}
    labels = []
    for col in columns:
        base = re.sub(r"\s*\(.*\)$", "", col)
        i = counts.get(base, 0)
        counts[base] = i + 1
        n = _n_models(heldout_cfg[base]) if base in heldout_cfg else 1
        labels.append(_label(base, i, n))
    return labels


def bounds_for_label(heldout_cfg: dict, label: str) -> dict | None:
    """``{metric: [lo, hi]}`` for one partner label, or None if undefined."""
    m = re.match(r"^(.*)\[(\d+)\]$", label)
    base, i = (m.group(1), int(m.group(2))) if m else (label, 0)
    bounds = (heldout_cfg.get(base) or {}).get("performance_bounds")
    if not bounds:
        return None
    return {
        metric: list(v[i]) if isinstance(v[0], (list, tuple)) else list(v)
        for metric, v in bounds.items()
    }


def get_run_partners(
    run_id: str,
    task_name: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> tuple[list[str], list[dict | None]]:
    """Return (labels, bounds) along the partner axis of a run's heldout artifact.

    Bounds are read from the run's own wandb config (the bounds active at eval
    time), looked up by partner name.
    """
    run_config = fetch_run_config_cached(run_id, ENTITY, BENCHMARK_PROJECT, cache_dir)
    heldout_cfg = run_config.get("heldout_set", {}).get(task_name)
    if not heldout_cfg:
        raise ValueError(
            f"No heldout_set config for task '{task_name}' in run {run_id}. "
            f"Available: {list(run_config.get('heldout_set', {}).keys())}"
        )
    heldout_cfg = {k: v for k, v in heldout_cfg.items() if v is not None}

    columns = fetch_run_heldout_names_cached(
        run_id, ENTITY, BENCHMARK_PROJECT, cache_dir
    )
    if columns is not None:
        labels = labels_from_table(columns, heldout_cfg)
    else:
        warnings.warn(
            f"Run {run_id} has no logged heldout table; assuming live-yaml partner "
            f"order for {task_name}.",
            stacklevel=2,
        )
        yaml_keys = [
            k
            for k in GLOBAL_HELDOUT_CONFIG["heldout_set"][task_name]
            if k in heldout_cfg
        ]
        yaml_keys += [k for k in heldout_cfg if k not in yaml_keys]
        labels = expand_labels(heldout_cfg, yaml_keys)

    bounds = [bounds_for_label(heldout_cfg, lbl) for lbl in labels]
    return labels, bounds


def _partner_axis(is_oel: bool) -> int:
    # OEL artifacts are (seeds, oel_iter, partners, eps, agents); others drop oel_iter.
    return 2 if is_oel else 1


def load_run_eval_metrics(
    task_name: str,
    run_id,
    is_oel: bool,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force_recompute: bool = False,
    bc_run_id: str | list[str] | None = None,
) -> tuple[dict, list[str], list[dict | None], HumanProxySource]:
    """Load a run's heldout eval metrics with named, bounds-aligned partners.

    ``run_id`` may be a list of run ids evaluated against the same heldout set;
    their seeds are pooled along axis 0.

    If the run's own heldout set contains no human proxy and ``bc_run_id`` is
    given, the canonical partner of that separate BC evaluation is appended to
    the partner axis (labelled ``human_proxy``) with its own bounds. For pooled
    ``run_id`` lists, ``bc_run_id`` must be a list of the same length (one BC
    eval per source run, in the same order) so seeds line up.

    Returns:
        (eval_metrics, labels, bounds, human_proxy_source) where
        human_proxy_source is "builtin", "bc_merge", or None.
    """
    run_ids = run_id if isinstance(run_id, list) else [run_id]
    parts = [
        fetch_run_eval_metrics_cached(
            rid, ENTITY, BENCHMARK_PROJECT, cache_dir, force_recompute
        )
        for rid in run_ids
    ]
    eval_metrics = (
        parts[0]
        if len(parts) == 1
        else {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}
    )

    labels, bounds = get_run_partners(run_ids[0], task_name, cache_dir)
    for rid in run_ids[1:]:
        other_labels, _ = get_run_partners(rid, task_name, cache_dir)
        if other_labels != labels:
            raise ValueError(
                f"Pooled runs {run_ids[0]} and {rid} have different heldout partners "
                f"for {task_name}:\n  {labels}\n  {other_labels}"
            )

    axis = _partner_axis(is_oel)
    sample = next(iter(eval_metrics.values()))
    if sample.shape[axis] != len(labels):
        raise ValueError(
            f"Run {run_ids[0]} ({task_name}): artifact has {sample.shape[axis]} partners "
            f"but {len(labels)} labels were resolved: {labels}"
        )

    if any(is_human_proxy(lbl) for lbl in labels):
        return eval_metrics, labels, bounds, "builtin"

    if bc_run_id is None:
        return eval_metrics, labels, bounds, None

    bc_run_ids = bc_run_id if isinstance(bc_run_id, list) else [bc_run_id]
    if len(bc_run_ids) != len(run_ids):
        raise ValueError(
            f"{'+'.join(run_ids)} ({task_name}): {len(run_ids)} pooled run(s) but "
            f"{len(bc_run_ids)} BC eval run(s) given: {bc_run_ids}"
        )
    bc_parts = [
        fetch_run_eval_metrics_cached(
            rid, ENTITY, BENCHMARK_PROJECT, cache_dir, force_recompute
        )
        for rid in bc_run_ids
    ]
    bc_metrics = (
        bc_parts[0]
        if len(bc_parts) == 1
        else {k: np.concatenate([p[k] for p in bc_parts], axis=0) for k in bc_parts[0]}
    )
    bc_labels, bc_bounds = get_run_partners(bc_run_ids[0], task_name, cache_dir)
    for rid in bc_run_ids[1:]:
        other_labels, _ = get_run_partners(rid, task_name, cache_dir)
        if other_labels != bc_labels:
            raise ValueError(
                f"Pooled BC evals {bc_run_ids[0]} and {rid} have different partners "
                f"for {task_name}:\n  {bc_labels}\n  {other_labels}"
            )
    bc_run_id = "+".join(bc_run_ids)
    bc_idx = next(
        (i for i, lbl in enumerate(bc_labels) if lbl in BC_RUN_CANONICAL_KEYS), 0
    )
    bc_axis = _partner_axis(False)
    common = [k for k in eval_metrics if k in bc_metrics]
    merged = {}
    for k in common:
        bc_slice = np.take(bc_metrics[k], indices=[bc_idx], axis=bc_axis)
        if is_oel:
            # Old-style OEL run without built-in human proxy: BC eval is only for the
            # final iterate, so replicate it across the oel_iter axis.
            bc_slice = np.broadcast_to(
                bc_slice[:, None],
                (bc_slice.shape[0], eval_metrics[k].shape[1]) + bc_slice.shape[1:],
            )
        merged[k] = np.concatenate([eval_metrics[k], bc_slice], axis=axis)
    labels = labels + [HUMAN_PROXY_LABEL]
    bounds = bounds + [bc_bounds[bc_idx]]
    print(
        f"  merged BC eval {bc_run_id} (partner '{bc_labels[bc_idx]}') into "
        f"{'+'.join(run_ids)}: {sample.shape} -> {merged[common[0]].shape}"
    )
    return merged, labels, bounds, "bc_merge"

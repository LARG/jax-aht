"""Cached wandb sweep data fetcher.

Stores fetched run data as pickled DataFrames under a local cache directory
so repeated calls skip the network round-trip.

Config keys are stored as top-level columns.
Summary metric values are stored as '_summary.<name>' columns.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import wandb


DEFAULT_CACHE_DIR = Path("results/figures/cache")


def _normalize(s: str) -> str:
    """Lowercase and strip all separator characters for fuzzy name matching."""
    return re.sub(r"[/_\-]", "", s.lower())


def _check_sweep_name(sweep_name: str, expected_parts: list[str]) -> None:
    norm = _normalize(sweep_name)
    for part in expected_parts:
        if _normalize(part) not in norm:
            raise ValueError(
                f"Sweep name '{sweep_name}' does not contain expected part '{part}'. "
                f"Check that the sweep ID matches the intended algorithm/task."
            )


def _json_safe(v) -> bool:
    try:
        json.dumps(v)
        return True
    except (TypeError, ValueError):
        return False


def _cache_path(sweep_id: str, entity: str, project: str, cache_dir: Path) -> Path:
    return cache_dir / f"{entity}__{project}__{sweep_id}.pkl"


def fetch_sweep_cached(
    sweep_id: str,
    entity: str,
    project: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force_recompute: bool = False,
    expected_name_parts: list[str] | None = None,
) -> pd.DataFrame:
    """Return a DataFrame of all finished runs in a sweep.

    Config keys are stored as top-level columns.
    Summary keys are stored as '_summary.<name>' columns.

    Args:
        force_recompute: Re-fetch from wandb and overwrite the cache even if it exists.
        expected_name_parts: If provided, assert that the sweep name contains each part
            (case- and separator-insensitive). Raises ValueError on mismatch.
    """
    path = _cache_path(sweep_id, entity, project, Path(cache_dir))

    # Always resolve the sweep object — it's a lightweight metadata call and lets us
    # validate the sweep name even when serving results from the local cache.
    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")

    if expected_name_parts:
        _check_sweep_name(sweep.name, expected_name_parts)

    if not force_recompute and path.exists():
        print(f"Loading from cache: {path}")
        return pd.read_pickle(path)

    print(f"Fetching from wandb: {entity}/{project}/{sweep_id} ...")
    rows = []
    for run in sweep.runs:
        if run.state not in ("finished", "crashed"):
            continue
        row: dict = {"_run_id": run.id}
        row.update(run.config)
        for k, v in run.summary.items():
            if not k.startswith("_") and _json_safe(v):
                row[f"_summary.{k}"] = v
        rows.append(row)

    if not rows:
        raise ValueError(f"No finished runs found in sweep {sweep_id}.")

    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)
    print(f"Cached: {path}")
    return df


def filter_hparams(df: pd.DataFrame, algorithm: str, filtered_kv: dict) -> pd.DataFrame:
    """Drop rows where a hyperparameter matches an excluded value.

    Handles both bare keys (e.g. 'TRAJEDI_COEF') and wandb-prefixed keys
    (e.g. 'algorithm.TRAJEDI_COEF') so it works on DataFrames from either
    apply_best_hparams (bare) or print_sweep_summary (prefixed).
    """
    for bare_key, excluded_vals in filtered_kv.get(algorithm, {}).items():
        for col in (bare_key, f"algorithm.{bare_key}"):
            if col in df.columns:
                df = df[~df[col].isin(excluded_vals)]
                break
    return df


def fetch_sweep_bare_keys(sweep_id: str, entity: str, project: str) -> list[str]:
    """Return swept parameter names from the wandb sweep config, with 'algorithm.' prefix stripped."""
    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    return [k.removeprefix("algorithm.") for k in sweep.config.get("parameters", {}).keys()]


def load_sweep_df(
    task: str,
    algorithm: str,
    force_recompute: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Validate, fetch, and score a sweep DataFrame.

    Looks up the sweep ID from plot_globals, fetches from wandb (with caching),
    and returns (DataFrame with '_score' column, swept bare parameter keys).
    Swept keys come from the wandb sweep config, not the local param_sweep YAML.
    """
    from vis.plot_globals import (
        ENTITY, HYPERPARAM_PROJECT, HYPERPARAM_DEFAULT_METRIC,
        HYPERPARAM_SWEEPS, TASK_LEGACY_NAMES,
    )
    if task not in HYPERPARAM_SWEEPS:
        raise ValueError(f"Task '{task}' not found. Available: {list(HYPERPARAM_SWEEPS)}")
    if algorithm not in HYPERPARAM_SWEEPS[task]:
        raise ValueError(
            f"Algorithm '{algorithm}' not found for task '{task}'. "
            f"Available: {list(HYPERPARAM_SWEEPS[task])}"
        )
    sweep_id = HYPERPARAM_SWEEPS[task][algorithm]
    task_name_for_check = TASK_LEGACY_NAMES.get(task, task)
    bare_keys = fetch_sweep_bare_keys(sweep_id, ENTITY, HYPERPARAM_PROJECT)
    df = fetch_sweep_cached(
        sweep_id, ENTITY, HYPERPARAM_PROJECT,
        force_recompute=force_recompute,
        expected_name_parts=[algorithm, task_name_for_check],
    )
    return extract_metric(df, HYPERPARAM_DEFAULT_METRIC), bare_keys


def build_hparam_df(
    raw_df: pd.DataFrame,
    algorithm: str,
    bare_keys: list[str],
) -> pd.DataFrame:
    """Build a clean DataFrame of swept hparams and scores.

    Extracts bare_keys from raw_df (checking both flat 'algorithm.KEY' columns
    and the nested 'algorithm' dict as a fallback), drops rows with missing values,
    and applies FILTERED_HYPERPARAMETER_KV.
    """
    from vis.plot_globals import FILTERED_HYPERPARAMETER_KV

    cols: dict[str, pd.Series] = {}
    for key in bare_keys:
        flat_col = f"algorithm.{key}"
        if flat_col in raw_df.columns:
            cols[key] = raw_df[flat_col]
        elif "algorithm" in raw_df.columns:
            cols[key] = raw_df["algorithm"].apply(
                lambda d, k=key: d.get(k) if isinstance(d, dict) else None
            )
        else:
            raise ValueError(f"Swept key '{key}' not found in sweep data.")

    sweep_df = pd.DataFrame(cols)
    sweep_df["_score"] = raw_df["_score"].values
    sweep_df = sweep_df.dropna()
    return filter_hparams(sweep_df, algorithm, FILTERED_HYPERPARAMETER_KV)


def extract_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Add a '_score' column from a summary metric, dropping rows where it is missing."""
    col = f"_summary.{metric}"
    if col not in df.columns:
        available = [c.removeprefix("_summary.") for c in df.columns if c.startswith("_summary.")]
        raise ValueError(f"Metric '{metric}' not found. Available summary metrics: {available}")
    result = df.dropna(subset=[col]).copy()
    result["_score"] = result[col].astype(float)
    return result

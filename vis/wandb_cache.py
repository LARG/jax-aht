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


def extract_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Add a '_score' column from a summary metric, dropping rows where it is missing."""
    col = f"_summary.{metric}"
    if col not in df.columns:
        available = [c.removeprefix("_summary.") for c in df.columns if c.startswith("_summary.")]
        raise ValueError(f"Metric '{metric}' not found. Available summary metrics: {available}")
    result = df.dropna(subset=[col]).copy()
    result["_score"] = result[col].astype(float)
    return result

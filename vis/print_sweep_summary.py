"""Print a summary of hyperparameter sweep results for a given algorithm and task.

Reports:
  - Total finished runs
  - Number of unique hyperparameter combinations (some combos may have been run multiple times)
  - Top-N combinations ranked by mean score across duplicate runs

Usage:
    python vis/print_sweep_summary.py --algo-type ego --task lbf --algorithm ppo_ego
    python vis/print_sweep_summary.py --algo-type unified --task lbf --algorithm brdiv --top-n 10
"""

from __future__ import annotations

import argparse

import pandas as pd

from vis.plot_globals import (
    ENTITY,
    HYPERPARAM_PROJECT,
    HYPERPARAM_DEFAULT_METRIC,
    EGO_HYPERPARAM_SWEEPS,
    UNIFIED_HYPERPARAM_SWEEPS,
)
from vis.wandb_cache import fetch_sweep_cached, extract_metric


def get_hyperparam_cols(df: pd.DataFrame) -> list[str]:
    """Return config columns that actually vary across runs (the swept dimensions)."""
    cols = []
    for c in df.columns:
        if c.startswith("_"):
            continue
        try:
            if df[c].nunique() > 1:
                cols.append(c)
        except TypeError:
            pass
    return cols


def summarize_sweep(df: pd.DataFrame, top_n: int) -> None:
    param_cols = get_hyperparam_cols(df)

    # Drop rows where any hyperparam is unhashable (nested dicts etc.)
    hashable_mask = df[param_cols].apply(
        lambda col: col.map(lambda v: not isinstance(v, (dict, list)))
    ).all(axis=1)
    df = df[hashable_mask].copy()

    print(f"\nTotal finished runs:              {len(df)}")

    grouped = (
        df.groupby(param_cols, dropna=False)["_score"]
        .agg(runs="count", mean_score="mean", max_score="max", std_score="std")
        .reset_index()
        .sort_values("mean_score", ascending=False)
    )

    n_unique = len(grouped)
    n_duplicated = int((grouped["runs"] > 1).sum())
    print(f"Unique hyperparam combinations:   {n_unique}")
    print(f"Combinations run more than once:  {n_duplicated}")

    print(f"\nTop {top_n} combinations by mean score:")
    top = grouped.head(top_n).reset_index(drop=True)
    top.index += 1
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(top.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print unique hyperparam count and top combinations for a sweep."
    )
    parser.add_argument("--algo-type", choices=["ego", "unified"], required=True)
    parser.add_argument("--task", required=True, help="Task name (e.g. lbf).")
    parser.add_argument("--algorithm", required=True,
                        help="Algorithm name (e.g. ppo_ego, brdiv).")
    parser.add_argument("--top-n", type=int, default=5,
                        help="Number of top combinations to display (default: 5).")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Re-fetch from wandb, ignoring the local cache.")
    args = parser.parse_args()

    sweep_map = EGO_HYPERPARAM_SWEEPS if args.algo_type == "ego" else UNIFIED_HYPERPARAM_SWEEPS

    if args.task not in sweep_map:
        raise ValueError(f"Task '{args.task}' not found. Available: {list(sweep_map)}")
    if args.algorithm not in sweep_map[args.task]:
        raise ValueError(
            f"Algorithm '{args.algorithm}' not found for task '{args.task}'. "
            f"Available: {list(sweep_map[args.task])}"
        )

    sweep_id = sweep_map[args.task][args.algorithm]
    print(f"Sweep: {args.algo_type}/{args.task}/{args.algorithm} ({sweep_id})")

    df = fetch_sweep_cached(
        sweep_id, ENTITY, HYPERPARAM_PROJECT,
        force_recompute=args.force_recompute,
        expected_name_parts=[args.algorithm, args.task],
    )
    df = extract_metric(df, HYPERPARAM_DEFAULT_METRIC)
    summarize_sweep(df, args.top_n)


if __name__ == "__main__":
    main()

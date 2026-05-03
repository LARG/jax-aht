"""Print a summary of hyperparameter sweep results for a given algorithm and task.

Reports:
  - Total finished runs
  - Number of unique hyperparameter combinations
  - Best mean score across all combinations
  - Top-N combinations ranked by mean score

Usage:
    python scripts/paper_vis/print_sweep_summary.py --task lbf/lbf_7x7_nolevels --algorithm ppo_ego
    python scripts/paper_vis/print_sweep_summary.py --task lbf/lbf_7x7_nolevels --algorithm brdiv --top-n 10
    python scripts/paper_vis/print_sweep_summary.py --task lbf/lbf_7x7_nolevels  # all algorithms
"""

from __future__ import annotations

import argparse

import pandas as pd

from scripts.paper_vis.plot_globals import HYPERPARAM_SWEEPS
from scripts.wandb_utils.wandb_cache import load_sweep_df, build_hparam_df


def summarize_sweep(sweep_df: pd.DataFrame, bare_keys: list[str], top_n: int) -> None:
    print(f"\nTotal finished runs:              {len(sweep_df)}")

    grouped = (
        sweep_df.groupby(bare_keys, dropna=False)["_score"]
        .agg(runs="count", mean_score="mean", max_score="max", std_score="std")
        .reset_index()
        .sort_values("mean_score", ascending=False)
    )

    n_unique = len(grouped)
    n_duplicated = int((grouped["runs"] > 1).sum())
    print(f"Unique hyperparam combinations:   {n_unique}")
    print(f"Combinations run more than once:  {n_duplicated}")
    print(f"Best mean score:                  {grouped['mean_score'].iloc[0]:.4f}")

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
    parser.add_argument("--task", required=True, help="Task name (e.g. lbf/lbf_7x7_nolevels).")
    parser.add_argument("--algorithm", default=None,
                        help="Algorithm name (e.g. ppo_ego, brdiv). Omit to run all algorithms for the task.")
    parser.add_argument("--top-n", type=int, default=5,
                        help="Number of top combinations to display (default: 5).")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Re-fetch from wandb, ignoring the local cache.")
    args = parser.parse_args()

    if args.task not in HYPERPARAM_SWEEPS:
        raise ValueError(f"Task '{args.task}' not found. Available: {list(HYPERPARAM_SWEEPS)}")

    algorithms = [args.algorithm] if args.algorithm else list(HYPERPARAM_SWEEPS[args.task])

    for algorithm in algorithms:
        if len(algorithms) > 1:
            print(f"\n{'='*60}")
        print(f"Sweep: {args.task}/{algorithm}")
        raw_df, bare_keys = load_sweep_df(args.task, algorithm, args.force_recompute)
        sweep_df = build_hparam_df(raw_df, algorithm, bare_keys)
        summarize_sweep(sweep_df, bare_keys, args.top_n)


if __name__ == "__main__":
    main()

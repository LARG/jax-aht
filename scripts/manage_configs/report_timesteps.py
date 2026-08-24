#!/usr/bin/env python3
"""
Report the total number of training timesteps for every algorithm × task.

For each entry point directory provided, the script resolves each algorithm
config, computes the total timestep budget using the formula appropriate to
that algorithm family, and prints a table in a human-readable K / M / B scale.

Timestep formulas by entry point / algorithm:

    ego_agent_training   (all algos):
        TOTAL_TIMESTEPS

    teammate_generation  (brdiv, fcp, lbrdiv):
        TOTAL_TIMESTEPS + ego_train_algorithm.TOTAL_TIMESTEPS

    teammate_generation  (comedi):
        TOTAL_TIMESTEPS_PER_ITERATION * PARTNER_POP_SIZE
            + ego_train_algorithm.TOTAL_TIMESTEPS

    open_ended_training  (cole):
        PARTNER_POP_SIZE * TOTAL_TIMESTEPS_PER_ITERATION

    open_ended_training  (rotate, open_ended_minimax):
        (TIMESTEPS_PER_ITER_PARTNER + TIMESTEPS_PER_ITER_EGO)
            * NUM_OPEN_ENDED_ITERS

    open_ended_training  (paired):
        TOTAL_TIMESTEPS

Usage:
    python scripts/manage_configs/report_timesteps.py <entry_point_dir> [<entry_point_dir> ...]

Examples:
    python scripts/manage_configs/report_timesteps.py teammate_generation/
    python scripts/manage_configs/report_timesteps.py ego_agent_training/ teammate_generation/ open_ended_training/
"""

import sys
from pathlib import Path

from scripts.manage_configs.helpers import ALL_TASKS, resolve_algo_config, compute_total_timesteps, format_human



def config_path_for_task(
    algo_configs_root: Path, algo: str, task: str
) -> Path:
    parts = task.split("/")
    if len(parts) == 1:
        return algo_configs_root / algo / f"{task}.yaml"
    return algo_configs_root / algo / "/".join(parts[:-1]) / f"{parts[-1]}.yaml"


def report_entry_point(root: Path, skip_algos: set[str] | None = None) -> None:
    algo_configs_root = root / "configs" / "algorithm"
    if not algo_configs_root.exists():
        print(f"  WARNING: algorithm configs not found: {algo_configs_root}", file=sys.stderr)
        return

    algos = sorted(
        d.name for d in algo_configs_root.iterdir()
        if d.is_dir() and (not skip_algos or d.name not in skip_algos)
    )

    # Collect data: rows are tasks, columns are algorithms.
    # data[task][algo] = formatted string (or "-" if not applicable)
    data: dict[str, dict[str, str]] = {}
    present_tasks: list[str] = []

    for task in ALL_TASKS:
        row: dict[str, str] = {}
        task_present = False
        for algo in algos:
            cfg_path = config_path_for_task(algo_configs_root, algo, task)
            if not cfg_path.exists():
                row[algo] = "-"
                continue
            try:
                resolved = resolve_algo_config(cfg_path, algo_configs_root)
                total = compute_total_timesteps(resolved, algo=algo)
                row[algo] = format_human(total) if total is not None else "?"
            except Exception as e:
                row[algo] = f"ERR({e})"
            task_present = True
        if task_present:
            data[task] = row
            present_tasks.append(task)

    if not data:
        print("  No data found.")
        return

    # Print a table.
    task_col_w = max(len(t) for t in present_tasks) + 2
    algo_col_w = {a: max(len(a), max(len(data[t].get(a, "-")) for t in present_tasks)) + 2
                  for a in algos}

    header = f"{'Task':<{task_col_w}}" + "".join(f"{a:<{algo_col_w[a]}}" for a in algos)
    sep = "-" * len(header)
    print(header)
    print(sep)
    for task in present_tasks:
        row_str = f"{task:<{task_col_w}}"
        for a in algos:
            row_str += f"{data[task].get(a, '-'):<{algo_col_w[a]}}"
        print(row_str)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "entry_point_dirs",
        nargs="+",
        metavar="entry_point_dir",
        help="One or more entry point directories (e.g. teammate_generation/).",
    )
    parser.add_argument(
        "--skip-algos",
        nargs="+",
        metavar="ALGO",
        default=[],
        help="Algorithm names to exclude from the report (e.g. --skip-algos comedi rotate).",
    )
    args = parser.parse_args()
    skip_algos = set(args.skip_algos) if args.skip_algos else None

    for ep_dir in args.entry_point_dirs:
        root = Path(ep_dir)
        print(f"\n{'=' * 60}")
        print(f"Entry point: {root}")
        print(f"{'=' * 60}")
        report_entry_point(root, skip_algos=skip_algos)

    print()


if __name__ == "__main__":
    main()

"""Recompute and overwrite the cached best-returns files for all tasks.

Uses locally cached wandb artifact pickles (eval metrics + run configs) when
available; only downloads from wandb for runs that have not been cached yet.

Run from repo root: python scripts/paper_vis/recompute_best_returns.py [--tasks ...]
"""

import argparse

from scripts.paper_vis.compute_best_returns import load_best_returns
from scripts.paper_vis.plot_globals import (
    BC_BENCHMARK_RUNS,
    BENCHMARK_PROJECT,
    EGO_BENCHMARK_RUNS,
    ENTITY,
    METHOD_TO_DISPLAY_NAME,
    OEL_METHODS,
    UNIFIED_BENCHMARK_RUNS,
    paper_tasks,
)
from scripts.wandb_utils.wandb_cache import DEFAULT_CACHE_DIR


def _display_name(method_name: str, teammate_type: str | None = None) -> str:
    name = METHOD_TO_DISPLAY_NAME.get(method_name, method_name)
    return f"{name} ({teammate_type})" if teammate_type else name


def build_run_specs(task_name: str) -> list[tuple[str, str | list[str], bool]]:
    """Collect all (display_name, run_id, is_oel) tuples for a task.

    Combines unified-benchmark and ego-benchmark runs so that best returns
    are computed across every available method.
    """
    specs: list[tuple] = []
    seen_run_ids: set[str] = set()

    def _add(method_name: str, run_id, teammate_type: str | None = None):
        if not run_id:
            return
        key = "+".join(run_id if isinstance(run_id, list) else [run_id])
        if key in seen_run_ids:
            return
        seen_run_ids.add(key)
        specs.append(
            (
                _display_name(method_name, teammate_type),
                run_id,
                method_name in OEL_METHODS,
            )
        )

    for method_name, run_id in UNIFIED_BENCHMARK_RUNS.get(task_name, {}).items():
        _add(method_name, run_id)

    for method_name, teammate_runs in EGO_BENCHMARK_RUNS.get(task_name, {}).items():
        for teammate_type, run_id in teammate_runs.items():
            _add(method_name, run_id, teammate_type)

    return specs


def build_bc_run_specs(task_name: str) -> list[tuple[str, str, bool]]:
    """Collect (display_name, bc_run_id, is_oel) tuples from BC_BENCHMARK_RUNS.

    Display names must match those produced by ``build_run_specs`` so the
    loader can pair each benchmark run with its separate BC evaluation. Only
    runs whose own heldout set lacks the human proxy actually use them.
    """
    specs: list[tuple] = []
    for method_name, entry in BC_BENCHMARK_RUNS.get(task_name, {}).items():
        items = entry.items() if isinstance(entry, dict) else [(None, entry)]
        for teammate_type, run_id in items:
            if not run_id:
                continue
            specs.append(
                (
                    _display_name(method_name, teammate_type),
                    run_id,
                    method_name in OEL_METHODS,
                )
            )
    return specs


def main():
    parser = argparse.ArgumentParser(
        description="Recompute best-returns cache from local wandb artifact pickles"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Tasks to recompute (default: PAPER_TASKS)",
    )
    parser.add_argument(
        "--include_bc",
        action="store_true",
        help="Also merge the separate BC heldout evals (BC_BENCHMARK_RUNS) into runs "
        "whose own heldout set lacks the human proxy, so the human_proxy entry "
        "of the best returns covers every method.",
    )
    args = parser.parse_args()

    task_list = (
        args.tasks
        if args.tasks
        else paper_tasks(UNIFIED_BENCHMARK_RUNS, EGO_BENCHMARK_RUNS)
    )

    for task_name in task_list:
        run_specs = build_run_specs(task_name)
        if not run_specs:
            print(f"No benchmark runs configured for '{task_name}', skipping.")
            continue
        bc_specs = build_bc_run_specs(task_name) if args.include_bc else None
        print(f"\n{'=' * 60}")
        print(f"Recomputing best returns for: {task_name}")
        print(f"  {len(run_specs)} run spec(s): {[s[0] for s in run_specs]}")
        if bc_specs:
            print(f"  {len(bc_specs)} BC eval run(s) available for merging")
        safe = task_name.replace("/", "__")
        best_returns = load_best_returns(
            task_name,
            run_specs,
            entity=ENTITY,
            project=BENCHMARK_PROJECT,
            cache_dir=DEFAULT_CACHE_DIR,
            force_recompute=True,
            cache_filename=f"{safe}.json",
            bc_run_specs=bc_specs,
        )
        labels = best_returns["_labels"]
        for metric, values in best_returns.items():
            if metric.startswith("_"):
                continue
            missing = [lbl for lbl, v in zip(labels, values) if v is None]
            if missing:
                print(f"  WARNING: no best return for {metric} partners {missing}")
        print(f"  {len(labels)} partners: {labels}")


if __name__ == "__main__":
    main()

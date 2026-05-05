"""Recompute and overwrite the cached best-returns files for all tasks.

Uses locally cached wandb artifact pickles (eval metrics + run configs) when
available; only downloads from wandb for runs that have not been cached yet.

Run from repo root: python scripts/paper_vis/recompute_best_returns.py [--tasks ...]
"""
import argparse

from scripts.paper_vis.compute_best_returns import load_best_returns
from scripts.paper_vis.plot_globals import (
    ENTITY, BENCHMARK_PROJECT,
    EGO_BENCHMARK_RUNS, UNIFIED_BENCHMARK_RUNS,
    METHOD_TO_DISPLAY_NAME, OEL_METHODS,
)
from scripts.wandb_utils.wandb_cache import DEFAULT_CACHE_DIR


def build_run_specs(task_name: str) -> list[tuple[str, str | list[str], bool]]:
    """Collect all (display_name, run_id, is_oel) tuples for a task.

    Combines unified-benchmark and ego-benchmark runs so that best returns
    are computed across every available method.  Runs evaluated against a
    different-sized heldout set are skipped with a warning inside
    compute_best_returns.
    """
    specs: list[tuple] = []
    seen_run_ids: set[str] = set()

    def _add(method_name: str, run_id):
        if not run_id:
            return
        key = "+".join(run_id if isinstance(run_id, list) else [run_id])
        if key in seen_run_ids:
            return
        seen_run_ids.add(key)
        specs.append((
            METHOD_TO_DISPLAY_NAME.get(method_name, method_name),
            run_id,
            method_name in OEL_METHODS,
        ))

    for method_name, run_id in UNIFIED_BENCHMARK_RUNS.get(task_name, {}).items():
        _add(method_name, run_id)

    for method_name, teammate_runs in EGO_BENCHMARK_RUNS.get(task_name, {}).items():
        for run_id in teammate_runs.values():
            _add(method_name, run_id)

    return specs


def main():
    parser = argparse.ArgumentParser(
        description="Recompute best-returns cache from local wandb artifact pickles"
    )
    parser.add_argument(
        "--tasks", nargs="+",
        help="Tasks to recompute (default: all tasks with benchmark runs)",
    )
    args = parser.parse_args()

    all_tasks = sorted(set(UNIFIED_BENCHMARK_RUNS) | set(EGO_BENCHMARK_RUNS))
    task_list = args.tasks if args.tasks else all_tasks

    for task_name in task_list:
        run_specs = build_run_specs(task_name)
        if not run_specs:
            print(f"No benchmark runs configured for '{task_name}', skipping.")
            continue
        print(f"\n{'='*60}")
        print(f"Recomputing best returns for: {task_name}")
        print(f"  {len(run_specs)} run spec(s): {[s[0] for s in run_specs]}")
        load_best_returns(
            task_name,
            run_specs,
            entity=ENTITY,
            project=BENCHMARK_PROJECT,
            cache_dir=DEFAULT_CACHE_DIR,
            force_recompute=True,
        )


if __name__ == "__main__":
    main()

"""Recompute and overwrite the cached best-returns files for all tasks.

Uses locally cached wandb artifact pickles (eval metrics + run configs) when
available; only downloads from wandb for runs that have not been cached yet.

Run from repo root: python scripts/paper_vis/recompute_best_returns.py [--tasks ...]
"""
import argparse
import json
from pathlib import Path

from scripts.paper_vis.compute_best_returns import compute_best_returns, load_best_returns
from scripts.paper_vis.plot_globals import (
    ENTITY, BENCHMARK_PROJECT,
    EGO_BENCHMARK_RUNS, UNIFIED_BENCHMARK_RUNS, BC_BENCHMARK_RUNS,
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


def build_bc_run_specs(task_name: str) -> list[tuple[str, str, bool]]:
    """Collect (display_name, run_id, is_oel) tuples from BC_BENCHMARK_RUNS for a task.

    Each cell evaluates one ego against the BC partner(s) for that task. All
    cells share the same heldout-set (the bc_proxy entry), so their eval-metric
    artifacts have the same partner axis and can be aggregated by the standard
    compute_best_returns pipeline.
    """
    specs: list[tuple] = []
    seen: set[str] = set()
    for method_name, teammate_runs in BC_BENCHMARK_RUNS.get(task_name, {}).items():
        for run_id in teammate_runs.values():
            if not run_id or run_id in seen:
                continue
            seen.add(run_id)
            specs.append((
                METHOD_TO_DISPLAY_NAME.get(method_name, method_name),
                run_id,
                method_name in OEL_METHODS,
            ))
    return specs


def main():
    parser = argparse.ArgumentParser(
        description="Recompute best-returns cache from local wandb artifact pickles"
    )
    parser.add_argument(
        "--tasks", nargs="+",
        help="Tasks to recompute (default: all tasks with benchmark runs)",
    )
    parser.add_argument(
        "--include_bc", action="store_true",
        help="In default mode, also compute best_returns from BC_BENCHMARK_RUNS "
             "and append them to the per-metric arrays so the bounds-gap plot "
             "shows BC partners at the end of each task panel (matching the "
             "live yaml's bc_proxy entry order).",
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
        safe = task_name.replace("/", "__")
        old_br = load_best_returns(
            task_name,
            run_specs,
            entity=ENTITY,
            project=BENCHMARK_PROJECT,
            cache_dir=DEFAULT_CACHE_DIR,
            force_recompute=True,
            cache_filename=f"{safe}.json",
        )

        if args.include_bc:
            bc_specs = build_bc_run_specs(task_name)
            if not bc_specs:
                print(f"  --include_bc: no BC runs registered for '{task_name}', leaving as-is.")
                continue
            print(f"  --include_bc: appending BC best_returns from {len(bc_specs)} run(s)")
            bc_br = compute_best_returns(
                task_name, bc_specs, entity=ENTITY,
                project=BENCHMARK_PROJECT, cache_dir=DEFAULT_CACHE_DIR,
            )
            # Restrict to bc_run_0 only for overcooked (5 BC partners → 1).
            if "overcooked" in task_name:
                bc_br = {k: v[:1] for k, v in bc_br.items()}
            merged = {}
            for metric in set(old_br) | set(bc_br):
                old_vals = list(old_br.get(metric, []))
                bc_vals = list(bc_br.get(metric, []))
                merged[metric] = old_vals + bc_vals
            merged_path = Path(DEFAULT_CACHE_DIR) / "best_returns" / f"{safe}.json"
            with open(merged_path, "w") as f:
                json.dump(merged, f, indent=2)
            lens = {m: len(v) for m, v in merged.items()}
            print(f"  wrote merged (old+BC) best returns -> {merged_path}  "
                  f"(per-metric lengths: {lens})")


if __name__ == "__main__":
    main()

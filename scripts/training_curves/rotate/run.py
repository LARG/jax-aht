"""CLI: fetch + plot ROTATE training/eval curves for a task."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.training_curves.common import (
    DEFAULT_CACHE_DIR,
    task_to_safe_filename,
    update_wandb_run_index,
)
from scripts.training_curves.rotate.fetch import fetch_rotate_curves_for_task
from scripts.training_curves.rotate.plot import plot_rotate_run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--entity", default="aht-project")
    p.add_argument("--project", default="aht-benchmark")
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--out-dir", default="results/figures/training_curves/rotate")
    p.add_argument("--force-recompute", action="store_true")
    args = p.parse_args()

    runs = fetch_rotate_curves_for_task(
        task=args.task,
        entity=args.entity,
        project=args.project,
        cache_dir=Path(args.cache_dir),
        force_recompute=args.force_recompute,
    )
    out_dir = Path(args.out_dir)
    entries: dict[str, dict] = {}
    for r in runs:
        # ROTATE has multiple runs for some tasks (e.g. forced_coord). Disambiguate
        # by run id when more than one run is returned for the same task.
        if len(runs) > 1:
            fname = f"{task_to_safe_filename(args.task)}__{r.run_id}.png"
            label = f"{r.task} (run {r.run_id})"
        else:
            fname = f"{task_to_safe_filename(args.task)}.png"
            label = r.task
        plot_rotate_run(r, out_dir / fname)
        entries[fname] = {"label": label, "run_id": r.run_id}
    update_wandb_run_index(out_dir, entries, args.entity, args.project)


if __name__ == "__main__":
    main()

"""CLI: fetch + plot TrajeDi training-return curve for a task."""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.training_curves.common import (
    DEFAULT_CACHE_DIR,
    task_to_safe_filename,
    update_wandb_run_index,
)
from scripts.training_curves.trajedi.fetch import fetch_trajedi_curves_for_task
from scripts.training_curves.trajedi.plot import plot_trajedi_run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--entity", default="aht-project")
    p.add_argument("--project", default="aht-benchmark")
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--out-dir", default="results/figures/training_curves/trajedi")
    p.add_argument("--force-recompute", action="store_true")
    args = p.parse_args()

    runs = fetch_trajedi_curves_for_task(
        task=args.task, entity=args.entity, project=args.project,
        cache_dir=Path(args.cache_dir), force_recompute=args.force_recompute,
    )
    out_dir = Path(args.out_dir)
    entries: dict[str, dict] = {}
    for r in runs:
        fname = f"{task_to_safe_filename(args.task)}.png"
        plot_trajedi_run(r, out_dir / fname)
        entries[fname] = {"label": r.task, "run_id": r.run_id}
    update_wandb_run_index(out_dir, entries, args.entity, args.project)


if __name__ == "__main__":
    main()

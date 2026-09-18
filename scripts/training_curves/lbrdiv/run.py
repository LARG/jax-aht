"""CLI: fetch + plot LBRDIV training curves for a task.

Usage:
    PYTHONPATH=. python scripts/training_curves/lbrdiv/run.py --task overcooked-v1/coord_ring
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.training_curves.common import (
    DEFAULT_CACHE_DIR,
    task_to_safe_filename,
    update_wandb_run_index,
)
from scripts.training_curves.lbrdiv.fetch import fetch_lbrdiv_curves_for_task
from scripts.training_curves.lbrdiv.plot import plot_lbrdiv_run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--entity", default="aht-project")
    p.add_argument("--project", default="aht-benchmark")
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--out-dir", default="results/figures/training_curves/lbrdiv")
    p.add_argument("--force-recompute", action="store_true")
    args = p.parse_args()

    runs = fetch_lbrdiv_curves_for_task(
        task=args.task,
        entity=args.entity,
        project=args.project,
        cache_dir=Path(args.cache_dir),
        force_recompute=args.force_recompute,
    )
    out_dir = Path(args.out_dir)
    entries: dict[str, dict] = {}
    for r in runs:
        fname = f"{task_to_safe_filename(args.task)}.png"
        plot_lbrdiv_run(r, out_dir / fname)
        entries[fname] = {"label": r.task, "run_id": r.run_id}
    update_wandb_run_index(out_dir, entries, args.entity, args.project)


if __name__ == "__main__":
    main()

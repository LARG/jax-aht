"""CLI: fetch + plot FCP training curves for a task.

Usage:
    PYTHONPATH=. python scripts/training_curves/fcp/run.py --task overcooked-v1/coord_ring
    PYTHONPATH=. python scripts/training_curves/fcp/run.py --task overcooked-v1/coord_ring --force-recompute
"""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.training_curves.common import (
    DEFAULT_CACHE_DIR,
    task_to_safe_filename,
    update_wandb_run_index,
)
from scripts.training_curves.fcp.fetch import fetch_fcp_curves_for_task
from scripts.training_curves.fcp.plot import plot_fcp_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True,
                        help="e.g. overcooked-v1/coord_ring or lbf/lbf_7x7_nolevels")
    parser.add_argument("--entity", default="aht-project")
    parser.add_argument("--project", default="aht-benchmark")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--out-dir", default="results/figures/training_curves/fcp",
                        help="Output directory for the plot.")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Bypass the local artifact-metrics cache.")
    args = parser.parse_args()

    runs = fetch_fcp_curves_for_task(
        task=args.task,
        entity=args.entity,
        project=args.project,
        cache_dir=Path(args.cache_dir),
        force_recompute=args.force_recompute,
    )

    out_dir = Path(args.out_dir)
    entries: dict[str, dict] = {}
    for run_curves in runs:
        fname = f"{task_to_safe_filename(args.task)}.png"
        plot_fcp_run(run_curves, out_dir / fname)
        entries[fname] = {"label": run_curves.task, "run_id": run_curves.run_id}
    update_wandb_run_index(out_dir, entries, args.entity, args.project)


if __name__ == "__main__":
    main()

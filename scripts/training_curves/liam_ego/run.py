"""CLI: fetch + plot LIAM Ego training-return curve for a task."""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.training_curves.common import (
    DEFAULT_CACHE_DIR,
    task_to_safe_filename,
    update_wandb_run_index,
)
from scripts.training_curves.ego_common import (
    fetch_ego_curves_for_task,
    plot_ego_run,
)

ALG = "liam_ego"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--entity", default="aht-project")
    p.add_argument("--project", default="aht-benchmark")
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--out-dir", default=f"results/figures/training_curves/{ALG}")
    p.add_argument("--force-recompute", action="store_true")
    args = p.parse_args()

    runs = fetch_ego_curves_for_task(
        algorithm=ALG, task=args.task,
        entity=args.entity, project=args.project,
        cache_dir=Path(args.cache_dir), force_recompute=args.force_recompute,
    )
    out_dir = Path(args.out_dir)
    entries: dict[str, dict] = {}
    for r in runs:
        suffix = r.teammate_type or r.run_id
        fname = f"{task_to_safe_filename(args.task)}__{suffix}.png"
        plot_ego_run(r, out_dir / fname)
        label = f"{r.task} ({suffix})" if r.teammate_type else f"{r.task} (run {r.run_id})"
        entries[fname] = {"label": label, "run_id": r.run_id}
    update_wandb_run_index(out_dir, entries, args.entity, args.project)


if __name__ == "__main__":
    main()

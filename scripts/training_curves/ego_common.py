"""Shared fetcher + plotter for the standalone ego algorithms.

`ppo_ego`, `liam_ego`, `meliba_ego` all save an `ego_train_run` artifact with
`metrics["returned_episode_returns"]` shape (NUM_SEEDS, NUM_EGO_TRAIN_SEEDS,
NUM_EGO_UPDATES). They share `algorithm.TOTAL_TIMESTEPS` for the env-step axis.
The per-algo folders are thin wrappers that only set the algorithm name.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from scripts.paper_vis.plot_globals import (
    AXIS_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    TASK_TO_PLOT_TITLE,
    TITLE_FONTSIZE,
)
from scripts.training_curves.common import (
    DEFAULT_CACHE_DIR,
    EGO_DISPLAY_NAMES,
    CurveData,
    extract_ego_curve,
    fetch_train_run_metrics_cached,
    find_benchmark_runs,
    get_config_value,
)


@dataclass
class EgoRunCurves:
    run_id: str
    task: str
    algorithm: str
    teammate_type: str  # e.g. "fcp_teammates", "rotate_teammates", or "" if unknown
    train: CurveData


def _build_run_id_to_teammate_type(algorithm: str) -> dict[str, str]:
    """Reverse-lookup of EGO_BENCHMARK_RUNS: run_id -> teammate_type ("fcp_teammates" etc).

    The EGO_BENCHMARK_RUNS mapping in plot_globals tags each (task, ego_algo,
    teammate_type) tuple with a wandb run id. We invert it so that during
    fetch we can recover the experimental condition for any run id.
    Algorithm-name normalization: plot_globals inconsistently uses "ppo_ego"
    (with suffix) but "liam" / "meliba" (without). Match against both.
    """
    from scripts.paper_vis.plot_globals import EGO_BENCHMARK_RUNS

    candidates = {algorithm, algorithm.removesuffix("_ego")}
    rid_to_type: dict[str, str] = {}
    for by_alg in EGO_BENCHMARK_RUNS.values():
        for alg, by_teammate in by_alg.items():
            if alg not in candidates:
                continue
            for teammate_type, rid in by_teammate.items():
                if rid:
                    rid_to_type[rid] = teammate_type
    return rid_to_type


def fetch_ego_curves_for_task(
    algorithm: str,
    task: str,
    entity: str = "aht-project",
    project: str = "aht-benchmark",
    cache_dir=DEFAULT_CACHE_DIR,
    force_recompute: bool = False,
) -> list[EgoRunCurves]:
    runs = find_benchmark_runs(
        algorithm=algorithm,
        task=task,
        entity=entity,
        project=project,
    )
    if not runs:
        raise ValueError(
            f"No finished {algorithm} neurips:benchmark runs for task={task!r} "
            f"in {entity}/{project}."
        )

    rid_to_type = _build_run_id_to_teammate_type(algorithm)

    out: list[EgoRunCurves] = []
    for run in runs:
        teammate_type = rid_to_type.get(run.id, "")
        print(
            f"\n[{algorithm}] processing run {run.id}  task={task}  "
            f"state={run.state}  teammate_type={teammate_type or '?'}"
        )
        # Standalone ego algos use TOTAL_TIMESTEPS at top level (not under
        # ego_train_algorithm — that nesting is the partner+ego pipeline only).
        total = get_config_value(run.config, "algorithm.TOTAL_TIMESTEPS")
        if total is None:
            raise ValueError(f"Run {run.id} missing algorithm.TOTAL_TIMESTEPS.")

        metrics = fetch_train_run_metrics_cached(
            run,
            artifact_kind="ego_train_run",
            entity=entity,
            project=project,
            cache_dir=cache_dir,
            force_recompute=force_recompute,
            reduce_per_update=True,
        )
        out.append(
            EgoRunCurves(
                run_id=run.id,
                task=task,
                algorithm=algorithm,
                teammate_type=teammate_type,
                train=extract_ego_curve(metrics, total),
            )
        )
    return out


def plot_ego_run(curves: EgoRunCurves, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    n_seeds = curves.train.values.shape[0]
    cmap = plt.get_cmap("tab10")
    for s in range(n_seeds):
        ax.plot(
            curves.train.env_steps,
            curves.train.values[s],
            color=cmap(s % 10),
            label=f"seed {s}",
            linewidth=1.5,
        )

    task_title = TASK_TO_PLOT_TITLE.get(curves.task, curves.task)
    method = EGO_DISPLAY_NAMES.get(curves.algorithm, curves.algorithm)
    suffix = f" — {curves.teammate_type}" if curves.teammate_type else ""
    ax.set_title(f"{method}{suffix} — {task_title}", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Environment Steps", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Training return", fontsize=AXIS_LABEL_FONTSIZE)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=LEGEND_FONTSIZE, loc="best", ncol=max(1, n_seeds // 5))

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")

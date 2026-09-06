"""Visualize the distribution of sweep performance across hyperparameter settings.

For a given task and algorithm type (ego, teammate_gen or unified), generates a grid of
subplots — one per algorithm — where each point is one unique hyperparameter setting
(mean score over its seeds). The y-axis shows performance; the x-axis is meaningless
(points are spread with jitter for readability).

The settings shown are exactly those considered when the benchmark configs were chosen:
``scripts/manage_configs/apply_best_hparams.py --max-hparams 140 --seed 0`` (see
``select_hparam_settings`` there). Pass ``--max-hparams 0`` to show every setting.

To run:
    python scripts/paper_vis/plot_sweep_distribution.py --algo-type ego --task lbf/lbf_12x12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gaussian_kde

from scripts.manage_configs.apply_best_hparams import select_hparam_settings
from scripts.paper_vis.plot_globals import (
    HYPERPARAM_SWEEPS,
    METHOD_TO_DISPLAY_NAME,
    SAVE_DIR,
    TASK_TO_DISPLAY_NAME,
)
from scripts.utils import ALGO_TO_ENTRY_POINT


def plot_distribution(
    scores_by_algo: dict[str, np.ndarray],
    title: str,
    out_path: Path,
) -> None:
    algos = list(scores_by_algo.keys())
    n = len(algos)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 3, nrows * 3.5),
        squeeze=False,
    )

    rng = np.random.default_rng(0)
    y_max = max(scores.max() for scores in scores_by_algo.values())

    for idx, algo in enumerate(algos):
        ax = axes[idx // ncols][idx % ncols]
        scores = scores_by_algo[algo]
        x = rng.uniform(-0.3, 0.3, size=len(scores))

        ax.scatter(x, scores, alpha=0.5, s=18, color="steelblue")
        median = np.median(scores)
        ax.axhline(
            median,
            color="crimson",
            linewidth=1.2,
            linestyle="--",
            label=f"median={median:.3f}",
        )

        ax.set_title(METHOD_TO_DISPLAY_NAME[algo], fontsize=10)
        ax.set_ylabel("Mean Normalized Return", fontsize=8)
        ax.set_xticks([])
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(0, y_max + 0.05)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)

        # KDE curve on the right
        divider = make_axes_locatable(ax)
        ax_kde = divider.append_axes("right", size="40%", pad=0.05)
        y_vals = np.linspace(0, y_max + 0.05, 300)
        kde = gaussian_kde(scores, bw_method="scott")
        density = kde(y_vals)
        ax_kde.plot(density, y_vals, color="steelblue", linewidth=1.2)
        ax_kde.fill_betweenx(y_vals, density, alpha=0.25, color="steelblue")
        ax_kde.axhline(median, color="crimson", linewidth=1.2, linestyle="--")
        peak_y = y_vals[np.argmax(density)]
        peak_d = density.max()
        ax_kde.annotate(
            f"{peak_y:.3f}",
            xy=(peak_d, peak_y),
            xytext=(-4, -10),
            textcoords="offset points",
            fontsize=7,
            va="center",
            color="steelblue",
        )
        ax_kde.set_ylim(0, y_max + 0.05)
        ax_kde.set_xticks([])
        ax_kde.set_yticks([])
        ax_kde.spines[["top", "right", "bottom"]].set_visible(False)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(title, fontsize=12, y=0.97)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot performance distribution across all hyperparameter settings."
    )
    parser.add_argument(
        "--algo-type",
        choices=["ego", "teammate_gen", "unified"],
        required=True,
        help="Which sweep family to visualize.",
    )
    parser.add_argument(
        "--task", required=True, help="Task name (e.g. lbf/lbf_7x7_nolevels)."
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Re-fetch from wandb, ignoring the local cache.",
    )
    parser.add_argument(
        "--max-hparams",
        type=int,
        default=140,
        help="Max unique hyperparam settings to visualize per algorithm. If the "
        "sweep ran more, a seeded random subset is drawn — the same subset "
        "apply_best_hparams.py used to pick the benchmark configs "
        "(default: 140). Pass 0 to show every setting.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for the --max-hparams subsample (default: 0, matching "
        "apply_best_hparams.py).",
    )
    args = parser.parse_args()

    ALGO_TYPE = args.algo_type
    TASK = args.task
    FORCE_RECOMPUTE = args.force_recompute
    MAX_NUM_TO_VISUALIZE = args.max_hparams or None
    SEED = args.seed

    ALGO_TYPE_TO_ENTRY_POINT = {
        "ego": "ego_agent_training",
        "teammate_gen": "teammate_generation",
        "unified": "open_ended_training",
    }
    all_task_sweeps = HYPERPARAM_SWEEPS[TASK]
    task_sweeps = {
        algo: sid
        for algo, sid in all_task_sweeps.items()
        if ALGO_TO_ENTRY_POINT.get(algo) == ALGO_TYPE_TO_ENTRY_POINT[ALGO_TYPE]
    }

    scores_by_algo: dict[str, np.ndarray] = {}
    for algo in task_sweeps:
        df, _ = select_hparam_settings(
            TASK, algo, FORCE_RECOMPUTE, max_hparams=MAX_NUM_TO_VISUALIZE, seed=SEED
        )
        scores_by_algo[algo] = df["_score"].values
        print(f"  {algo}: {len(df)} hparam settings")

    out_path = (
        Path(SAVE_DIR) / f"sweep_distribution_{ALGO_TYPE}_{TASK.replace('/', '_')}.pdf"
    )
    title = f"{TASK_TO_DISPLAY_NAME[TASK]}"
    plot_distribution(scores_by_algo, title, out_path)

"""Visualize the distribution of sweep performance across all hyperparameter settings.

For a given task and algorithm type (ego or unified), generates a grid of subplots —
one per algorithm — where each point is one sweep run. The y-axis shows performance;
the x-axis is meaningless (points are spread with jitter for readability).

To run: edit the config block at the bottom of this file, then:
    python vis/plot_sweep_distribution.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gaussian_kde

from vis.plot_globals import (
    SAVE_DIR,
    ENTITY,
    METHOD_TO_DISPLAY_NAME, TASK_TO_DISPLAY_NAME,
    HYPERPARAM_PROJECT,
    HYPERPARAM_DEFAULT_METRIC,
    EGO_HYPERPARAM_SWEEPS,
    UNIFIED_HYPERPARAM_SWEEPS,
)
from vis.wandb_cache import fetch_sweep_cached, extract_metric


def _get_hparam_cols(df: pd.DataFrame) -> list[str]:
    """Return config columns that vary across runs and have hashable values.

    Mirrors the logic in print_sweep_summary.py: skip columns where nunique()
    raises TypeError (entirely unhashable, e.g. always-dict columns).
    """
    cols = []
    for c in df.columns:
        if c.startswith("_"):
            continue
        try:
            if df[c].nunique() > 1:
                cols.append(c)
        except TypeError:
            pass
    return cols


def deduplicate_and_sample(
    df: pd.DataFrame,
    max_num: int | None,
    seed: int = 0,
) -> pd.DataFrame:
    """Aggregate duplicate hyperparam combinations and optionally subsample.

    Rows with identical hyperparameter settings are collapsed by taking the
    mean _score.  If more than *max_num* unique combinations remain, randomly
    sample down to that limit.
    """
    hparam_cols = _get_hparam_cols(df)
    if hparam_cols:
        # Drop individual rows where a hyperparam value is unhashable (e.g. a
        # dict nested inside an otherwise scalar column).
        hashable_mask = df[hparam_cols].apply(
            lambda col: col.map(lambda v: not isinstance(v, (dict, list)))
        ).all(axis=1)
        df = df[hashable_mask].copy()
        df = df.groupby(hparam_cols, as_index=False, dropna=False).agg({"_score": "mean"})
    if max_num is not None and len(df) > max_num:
        df = df.sample(n=max_num, random_state=seed)
    return df


def plot_distribution(
    scores_by_algo: dict[str, np.ndarray],
    title: str,
    out_path: Path,
) -> None:
    algos = list(scores_by_algo.keys())
    n = len(algos)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
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
        ax.axhline(median, color="crimson", linewidth=1.2, linestyle="--",
                   label=f"median={median:.3f}")

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
        ax_kde.annotate(f"{peak_y:.3f}", xy=(peak_d, peak_y),
                        xytext=(-4, -10), textcoords="offset points",
                        fontsize=7, va="center", color="steelblue")
        ax_kde.set_ylim(0, y_max + 0.05)
        ax_kde.set_xticks([])
        ax_kde.set_yticks([])
        ax_kde.spines[["top", "right", "bottom"]].set_visible(False)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(title, fontsize=12, y=0.95)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot performance distribution across all hyperparameter settings."
    )
    parser.add_argument("--algo-type", choices=["ego", "unified"], required=True,
                        help="Which sweep family to visualize.")
    parser.add_argument("--task", required=True,
                        help="Task name (e.g. lbf).")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Re-fetch from wandb, ignoring the local cache.")
    parser.add_argument("--max-hparams", type=int, default=400,
                        help="Max unique hyperparam combinations to visualize per algorithm. "
                             "If exceeded, randomly sample down to this limit.")
    args = parser.parse_args()

    ALGO_TYPE = args.algo_type
    TASK = args.task
    FORCE_RECOMPUTE = args.force_recompute
    MAX_NUM_TO_VISUALIZE = args.max_hparams

    sweep_map = EGO_HYPERPARAM_SWEEPS if ALGO_TYPE == "ego" else UNIFIED_HYPERPARAM_SWEEPS
    task_sweeps = sweep_map[TASK]

    scores_by_algo: dict[str, np.ndarray] = {}
    for algo, sweep_id in task_sweeps.items():
        df = fetch_sweep_cached(sweep_id, ENTITY, HYPERPARAM_PROJECT,
                                force_recompute=FORCE_RECOMPUTE,
                                expected_name_parts=[algo, TASK])
        df = extract_metric(df, HYPERPARAM_DEFAULT_METRIC)
        df = deduplicate_and_sample(df, MAX_NUM_TO_VISUALIZE)
        scores_by_algo[algo] = df["_score"].values
        print(f"  {algo}: {len(df)} unique hparam combos")

    out_path = Path(SAVE_DIR) / f"sweep_distribution_{ALGO_TYPE}_{TASK.replace('/', '_')}.pdf"
    title = f"{TASK_TO_DISPLAY_NAME[TASK]}: Hyperparameter Performance Distribution ({ALGO_TYPE.capitalize()} Algorithms)"
    plot_distribution(scores_by_algo, title, out_path)

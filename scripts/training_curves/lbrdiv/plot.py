"""Plot LBRDIV per-seed training curves: SP partner, XP partner, ego."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.paper_vis.plot_globals import (
    AXIS_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    METHOD_TO_DISPLAY_NAME,
    TASK_TO_PLOT_TITLE,
    TITLE_FONTSIZE,
)
from scripts.training_curves.lbrdiv.fetch import LBRDivRunCurves


def _eval_indices(values: np.ndarray) -> np.ndarray:
    """Indices where any seed's value changes — these are the true eval points.

    LBRDIV writes the eval metric only at NUM_CHECKPOINTS update steps
    (see `LBRDiv.py:1017-1037` + IPPO checkpoint logic in `ippo.py:373`).
    Between evals, the metric carries the previous value, producing long
    plateaus. We filter to the change-points so the plot reflects only
    real evaluations.
    """
    diffs = np.any(np.diff(values, axis=1) != 0, axis=0)
    idxs = [0] + (np.nonzero(diffs)[0] + 1).tolist()
    return np.asarray(idxs)


def _plot_panel(ax, env_steps: np.ndarray, values: np.ndarray, title: str, ylabel: str,
                sparse_eval: bool = False):
    n_seeds = values.shape[0]
    cmap = plt.get_cmap("tab10")
    if sparse_eval:
        idxs = _eval_indices(values)
        x = env_steps[idxs]
        for s in range(n_seeds):
            ax.plot(x, values[s, idxs], color=cmap(s % 10), label=f"seed {s}",
                    linewidth=1.5, marker="o", markersize=5)
        if len(idxs) <= 5:
            ax.text(0.02, 0.98, f"{len(idxs)} eval points (NUM_CHECKPOINTS limited)",
                    transform=ax.transAxes, va="top", fontsize=10, color="gray")
    else:
        for s in range(n_seeds):
            ax.plot(env_steps, values[s], color=cmap(s % 10), label=f"seed {s}", linewidth=1.5)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Environment Steps", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=LEGEND_FONTSIZE, loc="best", ncol=max(1, n_seeds // 5))


def _plot_lm_panel(ax, lm_curves, title: str, ylabel: str):
    """Plot pop_size^2 Lagrange-multiplier curves, one line per (i,j) pair."""
    n_pairs, _ = lm_curves.values.shape
    cmap = plt.get_cmap("viridis")
    for k in range(n_pairs):
        ax.plot(lm_curves.env_steps, lm_curves.values[k],
                color=cmap(k / max(1, n_pairs - 1)),
                label=lm_curves.pair_labels[k], linewidth=1.0, alpha=0.8)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Environment Steps", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.grid(True, alpha=0.3)
    # Legend with all pop_size^2 entries — keep it small and outside if it gets large.
    ncol = max(1, n_pairs // 8)
    ax.legend(fontsize=8, loc="best", ncol=ncol, title="(conf, br)")


def plot_lbrdiv_run(curves: LBRDivRunCurves, out_path: Path):
    """5-panel figure: SP partner | XP partner | ego | LM horizontal | LM vertical.

    Top row: partner-return panels + ego-return panel (per-seed).
    Bottom row: LM_H + LM_V (mean over seeds, one line per pair).
    """
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))

    task_title = TASK_TO_PLOT_TITLE.get(curves.task, curves.task)
    method = METHOD_TO_DISPLAY_NAME.get("lbrdiv", "LBRDiv")
    fig.suptitle(f"{method} — {task_title}", fontsize=TITLE_FONTSIZE + 2)

    _plot_panel(
        axes[0, 0], curves.sp_partner.env_steps, curves.sp_partner.values,
        title="Partner SP return",
        ylabel="Self-play return (mean over pop)",
        sparse_eval=True,
    )
    _plot_panel(
        axes[0, 1], curves.xp_partner.env_steps, curves.xp_partner.values,
        title="Partner XP return",
        ylabel="Cross-play return (mean over off-diag pairs)",
        sparse_eval=True,
    )
    _plot_panel(
        axes[0, 2], curves.ego.env_steps, curves.ego.values,
        title="Ego training return",
        ylabel="Return (vs partner pop)",
    )
    _plot_lm_panel(
        axes[1, 0], curves.lm_horizontal,
        title="Losses/LMs_Horizontal (mean over seeds)",
        ylabel="LM value",
    )
    _plot_lm_panel(
        axes[1, 1], curves.lm_vertical,
        title="Losses/LMs_Vertical (mean over seeds)",
        ylabel="LM value",
    )
    axes[1, 2].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")

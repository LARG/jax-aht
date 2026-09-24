"""Plot BRDIV per-seed training curves: SP partner, XP partner, ego."""

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
from scripts.training_curves.brdiv.fetch import BRDivRunCurves


def _eval_indices(values: np.ndarray) -> np.ndarray:
    diffs = np.any(np.diff(values, axis=1) != 0, axis=0)
    return np.asarray([0] + (np.nonzero(diffs)[0] + 1).tolist())


def _plot_panel(
    ax,
    env_steps: np.ndarray,
    values: np.ndarray,
    title: str,
    ylabel: str,
    sparse_eval: bool = False,
):
    n_seeds = values.shape[0]
    cmap = plt.get_cmap("tab10")
    if sparse_eval:
        idxs = _eval_indices(values)
        x = env_steps[idxs]
        for s in range(n_seeds):
            ax.plot(
                x,
                values[s, idxs],
                color=cmap(s % 10),
                label=f"seed {s}",
                linewidth=1.5,
                marker="o",
                markersize=5,
            )
        if len(idxs) <= 5:
            ax.text(
                0.02,
                0.98,
                f"{len(idxs)} eval points (NUM_CHECKPOINTS limited)",
                transform=ax.transAxes,
                va="top",
                fontsize=10,
                color="gray",
            )
    else:
        for s in range(n_seeds):
            ax.plot(
                env_steps,
                values[s],
                color=cmap(s % 10),
                label=f"seed {s}",
                linewidth=1.5,
            )
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Environment Steps", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=LEGEND_FONTSIZE, loc="best", ncol=max(1, n_seeds // 5))


def plot_brdiv_run(curves: BRDivRunCurves, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    task_title = TASK_TO_PLOT_TITLE.get(curves.task, curves.task)
    method = METHOD_TO_DISPLAY_NAME.get("brdiv", "BRDiv")
    fig.suptitle(f"{method} — {task_title}", fontsize=TITLE_FONTSIZE + 2)

    _plot_panel(
        axes[0],
        curves.sp_partner.env_steps,
        curves.sp_partner.values,
        title="Partner SP return",
        ylabel="Self-play return (mean over pop)",
        sparse_eval=True,
    )
    _plot_panel(
        axes[1],
        curves.xp_partner.env_steps,
        curves.xp_partner.values,
        title="Partner XP return",
        ylabel="Cross-play return (mean over off-diag pairs)",
        sparse_eval=True,
    )
    _plot_panel(
        axes[2],
        curves.ego.env_steps,
        curves.ego.values,
        title="Ego training return",
        ylabel="Return (vs partner pop)",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")

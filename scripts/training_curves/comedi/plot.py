"""Plot CoMeDi per-seed training curves: partner train + ego train."""

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
from scripts.training_curves.comedi.fetch import CoMeDiRunCurves


def _plot_panel(ax, env_steps: np.ndarray, values: np.ndarray, title: str, ylabel: str):
    n_seeds = values.shape[0]
    cmap = plt.get_cmap("tab10")
    for s in range(n_seeds):
        ax.plot(
            env_steps, values[s], color=cmap(s % 10), label=f"seed {s}", linewidth=1.5
        )
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Environment Steps", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=LEGEND_FONTSIZE, loc="best", ncol=max(1, n_seeds // 5))


def plot_comedi_run(curves: CoMeDiRunCurves, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    task_title = TASK_TO_PLOT_TITLE.get(curves.task, curves.task)
    method = METHOD_TO_DISPLAY_NAME.get("comedi", "CoMeDi")
    fig.suptitle(f"{method} — {task_title}", fontsize=TITLE_FONTSIZE + 2)

    _plot_panel(
        axes[0],
        curves.partner.env_steps,
        curves.partner.values,
        title="Partner training return",
        ylabel="Return (mean over partner pop)",
    )
    _plot_panel(
        axes[1],
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

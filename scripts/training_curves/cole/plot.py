"""Plot COLE per-seed partner training return + final XP-matrix heatmap."""

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
from scripts.training_curves.cole.fetch import COLERunCurves


def _plot_curve_panel(
    ax, env_steps: np.ndarray, values: np.ndarray, title: str, ylabel: str
):
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


def _plot_xp_matrix_panel(ax, mat: np.ndarray):
    pop_size = mat.shape[0]
    im = ax.imshow(mat, cmap="viridis", origin="upper", aspect="auto")
    ax.set_title("Final XP matrix (mean over seeds)", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Partner index (column)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Partner index (row)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xticks(range(pop_size))
    ax.set_yticks(range(pop_size))
    plt.colorbar(im, ax=ax, label="Return")

    # Annotate cell values when the matrix is small enough to read.
    # Auto-pick decimal places: integers for Overcooked-scale returns (0-250),
    # 2 decimals for LBF-scale returns (0-1).
    if pop_size <= 12:
        decimals = 0 if mat.max() >= 10 else 2
        fmt = f"{{:.{decimals}f}}"
        for i in range(pop_size):
            for j in range(pop_size):
                ax.text(
                    j,
                    i,
                    fmt.format(mat[i, j]),
                    ha="center",
                    va="center",
                    color="white" if mat[i, j] < mat.max() * 0.5 else "black",
                    fontsize=8,
                )


def plot_cole_run(curves: COLERunCurves, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    task_title = TASK_TO_PLOT_TITLE.get(curves.task, curves.task)
    method = METHOD_TO_DISPLAY_NAME.get("cole", "COLE")
    fig.suptitle(f"{method} — {task_title}", fontsize=TITLE_FONTSIZE + 2)

    _plot_curve_panel(
        axes[0],
        curves.partner.env_steps,
        curves.partner.values,
        title="Partner training return",
        ylabel="Return (mean over partner pop)",
    )
    _plot_xp_matrix_panel(axes[1], curves.xp_matrix)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")

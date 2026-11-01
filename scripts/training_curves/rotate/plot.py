"""Plot ROTATE per-seed curves: ego vs conf, conf vs confBR, ego vs heldout."""

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
from scripts.training_curves.rotate.fetch import RotateRunCurves


def _eval_indices(values: np.ndarray) -> np.ndarray:
    """Indices where any seed's value changes — the true eval points.

    ROTATE writes the eval metric only at NUM_CHECKPOINTS update steps within
    each OEL iter; between evals the metric repeats. Filtering to change-points
    gives a clean per-iter trajectory instead of plateau staircases.
    """
    if values.shape[1] <= 1:
        return np.arange(values.shape[1])
    diffs = np.any(np.diff(values, axis=1) != 0, axis=0)
    return np.asarray([0] + (np.nonzero(diffs)[0] + 1).tolist())


def _plot_panel(ax, env_steps, values, title, ylabel, marker=None, sparse_eval=False):
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
                markersize=4,
            )
    else:
        for s in range(n_seeds):
            ax.plot(
                env_steps,
                values[s],
                color=cmap(s % 10),
                label=f"seed {s}",
                linewidth=1.5,
                marker=marker,
                markersize=4,
            )
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Environment Steps", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=LEGEND_FONTSIZE, loc="best", ncol=max(1, n_seeds // 5))


def plot_rotate_run(curves: RotateRunCurves, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    task_title = TASK_TO_PLOT_TITLE.get(curves.task, curves.task)
    method = METHOD_TO_DISPLAY_NAME.get("rotate", "ROTATE")
    fig.suptitle(f"{method} — {task_title}", fontsize=TITLE_FONTSIZE + 2)

    _plot_panel(
        axes[0],
        curves.ego_vs_conf.env_steps,
        curves.ego_vs_conf.values,
        title="Ego vs Conf (training return)",
        ylabel="Return",
    )
    _plot_panel(
        axes[1],
        curves.conf_vs_confbr.env_steps,
        curves.conf_vs_confbr.values,
        title="Conf vs Conf-BR (avg per-step reward)",
        ylabel="Avg per-step reward",
    )
    # Heldout has only one point per OEL iter — render with markers.
    if curves.ego_vs_heldout is not None:
        _plot_panel(
            axes[2],
            curves.ego_vs_heldout.env_steps,
            curves.ego_vs_heldout.values,
            title="Ego vs Heldout (per-iter eval)",
            ylabel="Return (heldout-mean)",
            marker="o",
        )
    else:
        axes[2].axis("off")
        axes[2].text(
            0.5,
            0.5,
            "Ego vs Heldout\n(heldout_eval_metrics artifact missing)",
            ha="center",
            va="center",
            fontsize=14,
            color="gray",
            transform=axes[2].transAxes,
        )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")

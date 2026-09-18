"""Plot TrajeDi per-seed training-return curve."""
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
from scripts.training_curves.trajedi.fetch import TrajeDiRunCurves


def plot_trajedi_run(curves: TrajeDiRunCurves, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    n_seeds = curves.train.values.shape[0]
    cmap = plt.get_cmap("tab10")
    for s in range(n_seeds):
        ax.plot(curves.train.env_steps, curves.train.values[s],
                color=cmap(s % 10), label=f"seed {s}", linewidth=1.5)

    task_title = TASK_TO_PLOT_TITLE.get(curves.task, curves.task)
    method = METHOD_TO_DISPLAY_NAME.get("trajedi", "TrajeDi")
    ax.set_title(f"{method} — {task_title}", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Environment Steps", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Training return", fontsize=AXIS_LABEL_FONTSIZE)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=LEGEND_FONTSIZE, loc="best", ncol=max(1, n_seeds // 5))

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")

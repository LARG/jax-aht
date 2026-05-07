"""Plot original performance bounds vs best-seen BR bounds per teammate and task.

For each task with a cached best-returns file, shows a stacked bar chart where:
  - Bottom segment (blue):  original max bound from global_heldout_settings.yaml
  - Top segment (orange):   additional amount the best-seen BR exceeds that bound

Run from repo root: python scripts/paper_vis/plot_bounds_comparison.py
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import omegaconf

from scripts.paper_vis.plot_globals import (
    GLOBAL_HELDOUT_CONFIG,
    HUMAN_PROXY_AGENTS,
    RL_AGENTS,
    SAVE_DIR,
    TASK_TO_METRIC_NAME,
    TASK_TO_DISPLAY_NAME,
    TITLE_FONTSIZE,
    AXIS_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
)
from scripts.wandb_utils.wandb_cache import DEFAULT_CACHE_DIR

BEST_RETURNS_CACHE_DIR = Path(DEFAULT_CACHE_DIR) / "best_returns"

COLOR_ORIGINAL = "#4C72B0"
COLOR_DELTA = "#DD8452"

HATCH_PATTERN = "///"
HATCH_PREFIXES = ("seq_agent", "entitled_agent", "greedy_")


def _agent_sort_key(name: str) -> tuple:
    """Return (group, sub_order) for sorting: RL(0), heuristic(1), human_proxy(2).
    Within heuristics, independent_agent is placed third (sub_order=2)."""
    def _matches(name, patterns):
        return any(p.strip("*") in name for p in patterns)

    if _matches(name, HUMAN_PROXY_AGENTS):
        return (2, 0)
    if _matches(name, RL_AGENTS):
        return (0, 0)
    # Within heuristics: onion=0, plate=1, independent=2, others keep original order
    if "independent" in name:
        return (1, 2)
    if "onion" in name:
        return (1, 0)
    if "plate" in name:
        return (1, 1)
    return (1, 3)


def _should_hatch(label: str) -> bool:
    return any(label.startswith(p) for p in HATCH_PREFIXES)


def get_flat_bounds(task_config: dict, metric_name: str):
    """Return (labels, original_maxes) aligned with the best_returns agent order.

    Follows the same expansion logic as get_performance_bounds_from_run_config:
    RL agents (with 'path') expand one entry per idx_list element; heuristic
    agents produce a single entry.
    """
    labels: list[str] = []
    original_maxes: list[float | None] = []

    for teammate_name, agent_config in task_config.items():
        bounds = agent_config.get("performance_bounds", None)

        if "path" in agent_config:
            idx_list = agent_config.get("idx_list", [])
            n_models = len(idx_list)

            if bounds is None or metric_name not in bounds:
                for i in range(n_models):
                    lbl = teammate_name if n_models == 1 else f"{teammate_name}[{i}]"
                    labels.append(lbl)
                    original_maxes.append(None)
                continue

            metric_bounds = bounds[metric_name]
            per_model = isinstance(metric_bounds[0], (list, tuple))

            for i in range(n_models):
                lbl = teammate_name if n_models == 1 else f"{teammate_name}[{i}]"
                labels.append(lbl)
                if per_model:
                    original_maxes.append(float(metric_bounds[i][1]))
                else:
                    original_maxes.append(float(metric_bounds[1]))
        else:
            # Heuristic agent — single entry, OR a single block with per-instance
            # bounds expressed as a list-of-lists (e.g. bc_proxy with N partners
            # in one human_proxy entry).
            if bounds is None or metric_name not in bounds:
                labels.append(teammate_name)
                original_maxes.append(None)
                continue
            metric_bounds = bounds[metric_name]
            if isinstance(metric_bounds[0], (list, tuple)):
                n = len(metric_bounds)
                for i in range(n):
                    lbl = teammate_name if n == 1 else f"{teammate_name}[{i}]"
                    labels.append(lbl)
                    original_maxes.append(float(metric_bounds[i][1]))
            else:
                labels.append(teammate_name)
                original_maxes.append(float(metric_bounds[1]))

    return labels, original_maxes


def load_best_returns(task_name: str) -> dict | None:
    safe = task_name.replace("/", "__")
    path = BEST_RETURNS_CACHE_DIR / f"{safe}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def plot_bounds_comparison(save_dir: str, show_plots: bool = False):
    heldout_cfg = omegaconf.OmegaConf.to_container(
        GLOBAL_HELDOUT_CONFIG["heldout_set"], resolve=True
    )

    tasks_with_cache = [t for t in heldout_cfg if load_best_returns(t) is not None]
    if not tasks_with_cache:
        print("No best_returns cache files found under", BEST_RETURNS_CACHE_DIR)
        return

    ncols = min(2, len(tasks_with_cache))
    nrows = (len(tasks_with_cache) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 8, nrows * 4))

    # Normalise axes to a 2-D array for uniform indexing
    axes = np.array(axes).reshape(nrows, ncols)

    for idx, task_name in enumerate(tasks_with_cache):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        metric_name = TASK_TO_METRIC_NAME.get(task_name, "returned_episode_returns")
        labels, original_maxes = get_flat_bounds(heldout_cfg[task_name], metric_name)
        best_returns = load_best_returns(task_name)
        best_vals = best_returns.get(metric_name, [])

        n = min(len(labels), len(best_vals))
        if len(labels) != len(best_vals):
            print(
                f"Warning [{task_name}]: {len(labels)} yaml entries vs "
                f"{len(best_vals)} best_returns entries — using first {n}."
            )
        labels = labels[:n]
        orig_arr = np.array(
            [v if v is not None else 0.0 for v in original_maxes[:n]], dtype=float
        )
        best_arr = np.array(best_vals[:n], dtype=float)

        # Reorder: heuristic first, then RL, then human_proxy
        order = sorted(range(n), key=lambda i: _agent_sort_key(labels[i]))
        labels = [labels[i] for i in order]
        orig_arr = orig_arr[order]
        best_arr = best_arr[order]

        delta_arr = np.maximum(0.0, best_arr - orig_arr)

        x = np.arange(n)
        bar_w = 0.6

        bars_orig = ax.bar(x, orig_arr, width=bar_w, color=COLOR_ORIGINAL, alpha=0.85,
                           label="Original bound", zorder=3)
        bars_delta = ax.bar(x, delta_arr, width=bar_w, bottom=orig_arr, color=COLOR_DELTA,
                            alpha=0.85, label="Best-seen BR delta", zorder=3)

        for i, label in enumerate(labels):
            if _should_hatch(label):
                bars_orig[i].set_hatch(HATCH_PATTERN)
                bars_delta[i].set_hatch(HATCH_PATTERN)

        task_display = TASK_TO_DISPLAY_NAME.get(task_name, task_name)
        ax.set_title(task_display, fontsize=TITLE_FONTSIZE)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=AXIS_LABEL_FONTSIZE - 8)
        ax.tick_params(axis="y", labelsize=AXIS_LABEL_FONTSIZE - 8)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        if idx == 0:
            ax.legend(fontsize=LEGEND_FONTSIZE, loc="lower right")

    # Hide any unused subplot panels
    for idx in range(len(tasks_with_cache), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    # Single unified y-axis label
    first_task = tasks_with_cache[0]
    first_metric = TASK_TO_METRIC_NAME.get(first_task, "returned_episode_returns")
    unified_ylabel = first_metric.replace("_", " ").title()
    fig.text(0.0, 0.5, unified_ylabel, va="center", rotation="vertical",
             fontsize=AXIS_LABEL_FONTSIZE)

    plt.suptitle("Original vs Best-Seen BR Performance Bounds",
                 fontsize=TITLE_FONTSIZE + 2, y=1.01)
    plt.tight_layout(rect=[0.02, 0.05, 1.0, 0.98])

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "bounds_comparison.pdf")
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Saved figure to {save_path}")

    if show_plots:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot original vs best-seen BR performance bounds"
    )
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR,
                        help="Directory to save figures (default: %(default)s)")
    parser.add_argument("--show_plots", action="store_true",
                        help="Display plots interactively in addition to saving")
    args = parser.parse_args()

    plot_bounds_comparison(save_dir=args.save_dir, show_plots=args.show_plots)

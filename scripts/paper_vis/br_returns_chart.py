"""Visualize the raw BR returns (upper performance bounds) for each teammate
in each task from global_heldout_settings.yaml.

Each subfigure corresponds to one task. Within each subfigure, every heldout
teammate is a bar on the x-axis; the bar height is the upper bound of
`returned_episode_returns` from `performance_bounds`. For multi-checkpoint
agents the bar is split into individual entries named `<agent>_0`, `<agent>_1`,
etc.
"""
import argparse
import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path when running as a script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np

from scripts.paper_vis.plot_globals import (
    AXIS_LABEL_FONTSIZE,
    GLOBAL_HELDOUT_CONFIG,
    HEURISTIC_AGENTS,
    HUMAN_PROXY_AGENTS,
    LEGEND_FONTSIZE,
    RL_AGENTS,
    SAVE_DIR,
    TASK_TO_PLOT_TITLE,
    TITLE_FONTSIZE,
)
from scripts.paper_vis.plot_bounds_comparison import COLOR_ORIGINAL, COLOR_DELTA

COLOR_HUMAN_PROXY = "#9467BD"


def _agent_color(name: str) -> str:
    def _matches(name, patterns):
        return any(p.strip("*") in name for p in patterns)

    if _matches(name, RL_AGENTS):
        return COLOR_ORIGINAL
    if _matches(name, HUMAN_PROXY_AGENTS):
        return COLOR_HUMAN_PROXY
    return COLOR_DELTA

plt.rcParams["xtick.labelsize"] = AXIS_LABEL_FONTSIZE
plt.rcParams["ytick.labelsize"] = AXIS_LABEL_FONTSIZE

METRIC_KEY = "returned_episode_returns"


def _extract_br_returns(agent_config):
    """Return a list of (label_suffix, upper_bound) pairs for one agent config.

    Handles both flat  [lower, upper]  and nested  [[l0,u0],[l1,u1],...]
    formats stored under performance_bounds -> returned_episode_returns.
    """
    perf = agent_config.get("performance_bounds", None)
    if perf is None or METRIC_KEY not in perf:
        return []

    bounds = perf[METRIC_KEY]

    # Flat single pair: [lower, upper]
    if isinstance(bounds[0], (int, float)):
        return [("", float(bounds[1]))]

    # List of pairs: [[l0, u0], [l1, u1], ...]
    return [(f"_{i}", float(pair[1])) for i, pair in enumerate(bounds)]


def _agent_sort_key(name: str) -> tuple:
    """Return (group, sub_order) for sorting: heuristic(0), RL(1), human_proxy(2).
    Within heuristics, independent_agent is placed third (sub_order=2)."""
    def _matches(name, patterns):
        return any(p.strip("*") in name for p in patterns)

    if _matches(name, HUMAN_PROXY_AGENTS):
        return (2, 0)
    if _matches(name, RL_AGENTS):
        return (1, 0)
    # Within heuristics: onion=0, plate=1, independent=2, others keep original order
    if "independent" in name:
        return (0, 2)
    if "onion" in name:
        return (0, 0)
    if "plate" in name:
        return (0, 1)
    return (0, 3)


def extract_task_data(task_cfg):
    """Return (names, values) lists for all agents in a task config, ordered
    by agent type: heuristic first, then RL, then human_proxy."""
    names = []
    values = []
    for agent_name, agent_cfg in task_cfg.items():
        entries = _extract_br_returns(agent_cfg)
        if not entries:
            continue
        if len(entries) == 1:
            suffix, val = entries[0]
            names.append(agent_name)
            values.append(val)
        else:
            for suffix, val in entries:
                names.append(f"{agent_name}{suffix}")
                values.append(val)
    # Sort by agent type: heuristic (0), RL (1), human_proxy (2)
    paired = sorted(zip(names, values), key=lambda nv: _agent_sort_key(nv[0]))
    names = [n for n, _ in paired]
    values = [v for _, v in paired]
    return names, values


def plot_br_returns(save: bool, savedir: str, show_plot: bool, savename: str):
    heldout_set = GLOBAL_HELDOUT_CONFIG["heldout_set"]
    task_names = list(heldout_set.keys())

    n_tasks = len(task_names)
    n_cols = 3
    n_rows = (n_tasks + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 8, n_rows * 5),
    )
    axes = np.array(axes).reshape(-1)

    for ax_idx, task_name in enumerate(task_names):
        ax = axes[ax_idx]
        task_cfg = heldout_set[task_name]
        names, values = extract_task_data(task_cfg)

        if not names:
            ax.set_visible(False)
            continue

        x = np.arange(len(names))

        colors = [_agent_color(n) for n in names]
        ax.bar(x, values, color=colors, alpha=0.85, zorder=10)

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=LEGEND_FONTSIZE)
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)
        ax.set_axisbelow(True)

        display_name = TASK_TO_PLOT_TITLE.get(task_name, task_name)
        ax.set_title(display_name, fontsize=TITLE_FONTSIZE)

    # Hide any unused subplots
    for ax_idx in range(n_tasks, len(axes)):
        axes[ax_idx].set_visible(False)

    # Single shared y-axis label on the left side of the figure
    fig.text(0.04, 0.5, "Max Returned Episode Return", va="center", rotation="vertical",
             fontsize=AXIS_LABEL_FONTSIZE)

    # Legend in the bottom-right subplot, lower-right corner, on top of bars
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_DELTA, alpha=0.85, label="Heuristic"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_ORIGINAL, alpha=0.85, label="RL-Based"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_HUMAN_PROXY, alpha=0.85, label="Human Data"),
    ]
    # Place on the last visible subplot
    last_visible = axes[n_tasks - 1]
    leg = last_visible.legend(handles=legend_handles, loc="lower right", fontsize=LEGEND_FONTSIZE,
                              title="Agent Type", title_fontsize=LEGEND_FONTSIZE,
                              framealpha=0.9)
    leg.set_zorder(20)

    plt.tight_layout(rect=[0.02, 0.05, 1.0, 1.0])

    if save:
        os.makedirs(savedir, exist_ok=True)
        out_path = os.path.join(savedir, f"{savename}.pdf")
        plt.savefig(out_path, bbox_inches="tight")
        print(f"Saved figure to {out_path}")

    if show_plot:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot raw BR returns for all teammates across all tasks."
    )
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR,
                        help="Directory to save the figure.")
    parser.add_argument("--savename", type=str, default="br_returns",
                        help="Filename stem for the saved PDF.")
    parser.add_argument("--show_plots", action="store_true",
                        help="Display the figure interactively.")
    args = parser.parse_args()

    plot_br_returns(
        save=True,
        savedir=args.save_dir,
        show_plot=args.show_plots,
        savename=args.savename,
    )

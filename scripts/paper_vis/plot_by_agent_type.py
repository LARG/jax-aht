"""Plot unified-method performance broken down by heldout agent type.

For each task, produces a bar chart where the x-axis is the RL agent type
(comedi, lbrdiv, ippo, brdiv, obl) plus a heuristic bucket for non-RL agents.
Each group of bars shows per-method performance aggregated over all heldout
agents of that type.

Run from repo root:
    conda activate bench311
    PYTHONPATH=. python scripts/paper_vis/plot_by_agent_type.py
"""
import argparse
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import omegaconf

from scripts.paper_vis.process_data import load_results_for_task
from scripts.paper_vis.plot_globals import (
    GLOBAL_HELDOUT_CONFIG,
    SAVE_DIR,
    UNIFIED_BENCHMARK_RUNS,
    METHOD_TO_DISPLAY_NAME,
    OEL_METHODS,
    RL_AGENTS,
    TASK_TO_METRIC_NAME,
    TASK_TO_DISPLAY_NAME,
    TASK_TO_AXIS_DISPLAY_NAME,
    TITLE_FONTSIZE,
    AXIS_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
)

HEURISTIC_KEY = "heuristic"

# Display names for x-axis tick labels
AGENT_TYPE_DISPLAY = {
    "comedi": "CoMeDi",
    "lbrdiv": "LBRDiv",
    "ippo":   "IPPO",
    "brdiv":  "BRDiv",
    "obl":    "OBL",
    HEURISTIC_KEY: "Heuristic",
}


def get_heldout_agent_labels(task_config: dict) -> list[str]:
    """Return agent labels in heldout-set order (matches per-agent stat arrays)."""
    labels = []
    for teammate_name, agent_config in task_config.items():
        if "path" in agent_config:
            n = len(agent_config.get("idx_list", []))
            for i in range(n):
                labels.append(teammate_name if n == 1 else f"{teammate_name}[{i}]")
        else:
            labels.append(teammate_name)
    return labels


def classify_agent(label: str) -> str:
    """Return the first matching RL agent type, or 'heuristic'.

    RL_AGENTS is checked in order (lbrdiv before brdiv) so that 'lbrdiv-conf'
    is correctly assigned to lbrdiv rather than brdiv.
    """
    for rl_type in RL_AGENTS:
        if rl_type in label:
            return rl_type
    return HEURISTIC_KEY


def group_indices_by_type(labels: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        groups[classify_agent(label)].append(i)
    return dict(groups)


def plot_tasks(
    task_list: list[str],
    save_dir: str,
    use_best_returns_normalization: bool,
    force_recompute: bool,
    show_plots: bool,
):
    agg_stat = str(GLOBAL_HELDOUT_CONFIG["global_heldout_settings"]["AGGREGATE_STAT"])

    ncols = min(2, len(task_list))
    nrows = (len(task_list) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 10, nrows * 5))
    axes = np.array(axes).reshape(nrows, ncols)

    for idx, task_name in enumerate(task_list):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        # Build run specs for this task
        run_specs = []
        for method_name, run_id in UNIFIED_BENCHMARK_RUNS.get(task_name, {}).items():
            if not run_id:
                continue
            run_specs.append((
                METHOD_TO_DISPLAY_NAME.get(method_name, method_name),
                run_id,
                method_name in OEL_METHODS,
            ))

        if not run_specs:
            ax.set_visible(False)
            continue

        results = load_results_for_task(
            task_name,
            run_specs,
            force_recompute=force_recompute,
            renormalize_metrics=use_best_returns_normalization,
        )

        metric_name = TASK_TO_METRIC_NAME.get(task_name, "returned_episode_returns")
        heldout_cfg = omegaconf.OmegaConf.to_container(
            GLOBAL_HELDOUT_CONFIG["heldout_set"][task_name], resolve=True
        )
        labels = get_heldout_agent_labels(heldout_cfg)
        groups = group_indices_by_type(labels)

        # Type order: RL agents present in this task (canonical order), heuristic last
        type_order = [t for t in RL_AGENTS if t in groups]
        if HEURISTIC_KEY in groups:
            type_order.append(HEURISTIC_KEY)

        method_names = list(results.keys())
        n_methods = len(method_names)
        n_types = len(type_order)
        bar_w = min(0.8 / n_methods, 0.2)
        colors = plt.cm.tab10(np.arange(n_methods) / 10)
        x_centers = np.arange(n_types)

        for i, method_name in enumerate(method_names):
            per_agent = np.array(results[method_name][metric_name][f"{agg_stat}_per_agent"])
            x_vals, y_vals = [], []
            for j, ag_type in enumerate(type_order):
                indices = groups[ag_type]
                # Filter to valid indices (guard against stale cache size mismatch)
                valid = [k for k in indices if k < len(per_agent)]
                if not valid:
                    continue
                x_vals.append(x_centers[j] + (i - n_methods / 2 + 0.5) * bar_w)
                y_vals.append(float(np.mean(per_agent[valid])))

            ax.bar(x_vals, y_vals, width=bar_w, label=method_name,
                   color=colors[i], alpha=0.7, zorder=3)

        tick_labels = [AGENT_TYPE_DISPLAY.get(t, t) for t in type_order]
        ax.set_xticks(x_centers)
        ax.set_xticklabels(tick_labels, fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(
            f"{agg_stat.capitalize()} Normalized Return",
            fontsize=AXIS_LABEL_FONTSIZE,
        )
        ax.set_title(
            TASK_TO_DISPLAY_NAME.get(task_name, task_name),
            fontsize=TITLE_FONTSIZE,
        )
        ax.yaxis.grid(True, linestyle="--", alpha=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(fontsize=LEGEND_FONTSIZE, loc="center left",
                  bbox_to_anchor=(1.01, 0.5), framealpha=0.8)

    for idx in range(len(task_list), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    plt.tight_layout()
    norm_suffix = "br_norm" if use_best_returns_normalization else "orig_norm"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"by_agent_type_{norm_suffix}.pdf")
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Saved figure to {save_path}")
    if show_plots:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot unified-method performance grouped by heldout agent type"
    )
    parser.add_argument(
        "--tasks", nargs="+",
        help="Tasks to plot (default: all tasks with unified benchmark runs)",
    )
    parser.add_argument(
        "--use_best_returns_normalization", action="store_true",
        help="Renormalize using best observed returns",
    )
    parser.add_argument(
        "--force_recompute", action="store_true",
        help="Recompute summary stats from cached wandb artifacts",
    )
    parser.add_argument(
        "--save_dir", type=str, default=SAVE_DIR,
        help="Directory to save figures (default: %(default)s)",
    )
    parser.add_argument("--show_plots", action="store_true")
    args = parser.parse_args()

    task_list = args.tasks if args.tasks else sorted(UNIFIED_BENCHMARK_RUNS)
    plot_tasks(
        task_list=task_list,
        save_dir=args.save_dir,
        use_best_returns_normalization=args.use_best_returns_normalization,
        force_recompute=args.force_recompute,
        show_plots=args.show_plots,
    )

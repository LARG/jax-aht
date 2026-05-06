"""Plot unified-method performance broken down by heldout agent type as radar charts.

For each task, produces a radar chart where each axis is an agent type
(CoMeDi, LBRDiv, IPPO, BRDiv, OBL, Heuristic) and each trace is a method.
Values are the mean per-method performance over all heldout agents of that type.

Run from repo root: python scripts/paper_vis/plot_by_agent_type.py
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
    TITLE_FONTSIZE,
    AXIS_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
)

HEURISTIC_KEY = "heuristic"

AGENT_TYPE_DISPLAY = {
    "comedi": "CoMeDi",
    "lbrdiv": "LBRDiv",
    "ippo":   "IPPO",
    "brdiv":  "BRDiv",
    "obl":    "OBL",
    HEURISTIC_KEY: "Heuristic",
}


def get_heldout_agent_labels(task_config: dict) -> list[str]:
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
    for rl_type in RL_AGENTS:
        if rl_type in label:
            return rl_type
    return HEURISTIC_KEY


def group_indices_by_type(labels: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        groups[classify_agent(label)].append(i)
    return dict(groups)


def draw_radar_on_ax(
    ax,
    results: dict,
    metric_name: str,
    agg_stat: str,
    type_order: list[str],
    groups: dict[str, list[int]],
    colors,
    plot_title: str,
    show_title: bool,
):
    """Draw a radar chart onto an existing polar axes. Returns the max value plotted."""
    categories = [AGENT_TYPE_DISPLAY.get(t, t) for t in type_order]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    max_value = 0

    for i, (method_name, method_results) in enumerate(results.items()):
        per_agent = np.array(method_results[metric_name][f"{agg_stat}_per_agent"])
        values = []
        for ag_type in type_order:
            indices = groups.get(ag_type, [])
            valid = [k for k in indices if k < len(per_agent)]
            values.append(float(np.mean(per_agent[valid])) if valid else 0.0)

        max_value = max(max_value, max(values))
        closed_values = values + values[:1]

        ax.plot(angles, closed_values, 'o-', linewidth=2, color=colors[i], label=method_name)
        ax.fill(angles, closed_values, alpha=0.15, color=colors[i])

    ax.plot(angles, [1.0] * len(angles), '--', color='dimgray', linewidth=1.5,
            label='_nolegend_')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylim(0, max_value * 1.15)
    ax.yaxis.set_tick_params(labelsize=AXIS_LABEL_FONTSIZE - 4)
    ax.grid(color='#EEEEEE', linewidth=1)
    ax.spines['polar'].set_color('#CCCCCC')

    if show_title:
        ax.set_title(plot_title, fontsize=TITLE_FONTSIZE, pad=20)

    return max_value


def plot_tasks(
    task_list: list[str],
    save_dir: str,
    use_best_returns_normalization: bool,
    force_recompute: bool,
    show_plots: bool,
    show_legend: bool,
):
    agg_stat = str(GLOBAL_HELDOUT_CONFIG["global_heldout_settings"]["AGGREGATE_STAT"])
    norm_suffix = "br_norm" if use_best_returns_normalization else "orig_norm"

    # Collect per-task data first so we know how many valid tasks there are
    task_data = []
    all_method_names = None
    for task_name in task_list:
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
            print(f"No runs found for task {task_name}, skipping.")
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
        type_order = [t for t in RL_AGENTS if t in groups]
        if HEURISTIC_KEY in groups:
            type_order.append(HEURISTIC_KEY)

        task_data.append((task_name, results, metric_name, groups, type_order))
        if all_method_names is None:
            all_method_names = list(results.keys())

    if not task_data:
        print("No tasks to plot.")
        return

    n_tasks = len(task_data)
    ncols = n_tasks
    nrows = 1
    colors = plt.cm.Set2(np.arange(len(all_method_names)) / 10)

    fig = plt.figure(figsize=(ncols * 6, 6))

    for idx, (task_name, results, metric_name, groups, type_order) in enumerate(task_data):
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection='polar')
        draw_radar_on_ax(
            ax=ax,
            results=results,
            metric_name=metric_name,
            agg_stat=agg_stat,
            type_order=type_order,
            groups=groups,
            colors=colors,
            plot_title=TASK_TO_DISPLAY_NAME.get(task_name, task_name),
            show_title=True,
        )

    if show_legend and all_method_names:
        handles = [
            plt.Line2D([0], [0], color=colors[i], linewidth=2, marker='o', label=name)
            for i, name in enumerate(all_method_names)
        ]
        fig.legend(
            handles=handles,
            title="Algorithms",
            title_fontsize=LEGEND_FONTSIZE,
            loc='lower center',
            ncol=len(all_method_names),
            fontsize=LEGEND_FONTSIZE,
            framealpha=0.8,
            bbox_to_anchor=(0.5, -0.1),
        )
        fig.subplots_adjust(bottom=0.12)

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"by_agent_type_{norm_suffix}.pdf")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved combined radar chart to {save_path}")

    if show_plots:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot unified-method performance grouped by heldout agent type (radar charts)"
    )
    parser.add_argument(
        "--tasks", nargs="+",
        help="Tasks to plot (default: all tasks with unified benchmark runs)",
    )
    parser.add_argument(
        "--use_best_returns_normalization", action=argparse.BooleanOptionalAction,
        default=True, help="Renormalize using best observed returns",
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
    parser.add_argument("--show_legend", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    task_list = args.tasks if args.tasks else sorted(UNIFIED_BENCHMARK_RUNS)
    plot_tasks(
        task_list=task_list,
        save_dir=args.save_dir,
        use_best_returns_normalization=args.use_best_returns_normalization,
        force_recompute=args.force_recompute,
        show_plots=args.show_plots,
        show_legend=args.show_legend,
    )

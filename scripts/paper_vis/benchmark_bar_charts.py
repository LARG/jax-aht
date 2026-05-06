import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

from scripts.paper_vis.process_data import load_results_for_task
from scripts.paper_vis.plot_globals import (
    TITLE_FONTSIZE, AXIS_LABEL_FONTSIZE, LEGEND_FONTSIZE,
    TASK_TO_AXIS_DISPLAY_NAME, TASK_TO_METRIC_NAME,
)

plt.rcParams['xtick.labelsize'] = AXIS_LABEL_FONTSIZE
plt.rcParams['ytick.labelsize'] = AXIS_LABEL_FONTSIZE

def plot_single_bar_chart(results, metric_name: str, aggregate_stat_name: str,
                          task_display_name: str, plot_title: str,
                          save: bool, savedir: str, show_plot: bool, savename: str):
    """Bar chart for a single task.

    Each algorithm is a differently-colored bar. Algorithm names appear in a
    legend to the right of the axes; the x-axis is labelled by the task name.
    """
    method_display_names = list(results.keys())
    num_methods = len(method_display_names)
    bar_width = min(0.8 / num_methods, 0.25)

    fig, ax = plt.subplots(figsize=(max(4, num_methods * bar_width * 2 + 4), 5))
    colors = plt.cm.tab10(np.arange(num_methods) / 10)

    star_positions = []
    for i, display_name in enumerate(method_display_names):
        stat_key = f"overall_{aggregate_stat_name}"
        point_estimate = results[display_name][metric_name][stat_key]
        lower_ci = results[display_name][metric_name]["overall_lower_ci"]
        upper_ci = results[display_name][metric_name]["overall_upper_ci"]
        yerr = [[point_estimate - lower_ci], [upper_ci - point_estimate]]
        x = (i - num_methods / 2 + 0.5) * bar_width
        ax.bar([x], [point_estimate], width=bar_width, label=display_name,
               yerr=yerr, alpha=0.7, color=colors[i], ecolor='black', capsize=5, zorder=10)
        if results[display_name].get("_filtered_seeds", False):
            star_positions.append((x, upper_ci))

    for x_s, y_s in star_positions:
        ax.text(x_s, y_s, "*", ha="center", va="bottom", fontsize=LEGEND_FONTSIZE, zorder=11)

    ax.set_xticks([0])
    ax.set_xticklabels([task_display_name], fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xlim(-num_methods * bar_width / 2 - bar_width, num_methods * bar_width / 2 + bar_width)
    ax.set_ylabel(f'{aggregate_stat_name.capitalize()} Normalized Return', fontsize=AXIS_LABEL_FONTSIZE)
    if plot_title:
        ax.set_title(plot_title, fontsize=TITLE_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE, loc='center left', bbox_to_anchor=(1.01, 0.5), framealpha=0.8)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    if save:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        plt.savefig(os.path.join(savedir, f"{savename}.pdf"), bbox_inches='tight')
        print(f"Saved figure to {os.path.join(savedir, f'{savename}.pdf')}")
    if show_plot:
        plt.show()

def plot_all_tasks_bar_chart(all_task_results, metric_name: str, aggregate_stat_name: str,
                   plot_type: str, plot_title: str, save: bool, savedir: str, show_plot: bool, savename: str):
    '''Plots a bar chart where all tasks are plotted as groups on the same bar chart.'''
    tasks = list(all_task_results.keys())
    num_tasks = len(tasks)
    
    # Collect all method names from any task, order preserved (first task first)
    seen: set[str] = set()
    method_display_names: list[str] = []
    for t in tasks:
        for m in all_task_results[t]:
            if m not in seen:
                method_display_names.append(m)
                seen.add(m)
    num_methods = len(method_display_names)
    
    # Width of each bar
    bar_width = min(0.8 / num_methods, 0.35)
    
    # Set up the figure
    if num_methods > 2:
        fig, ax = plt.subplots(figsize=(int(num_tasks * 1.8), 6))
    else:
        fig, ax = plt.subplots(figsize=(num_tasks * 1.3, 6))
    # previously was int(num_tasks * 1.8) = 6*1.8 = 10.8
    # now is int(num_tasks * (num_methods+2) * bar_width) = 6* (6+3) * 0.2 = 10.8
    # previous for ablationsi was 6*1.8 = 10.8
    # now for ablations is 6*4*0.2 = 4.8

    # Set up x positions for tasks and bars within each task group
    task_positions = np.arange(num_tasks)
    
    # Colors for each method
    colors = plt.cm.tab10(np.arange(num_methods) / 10)
    
    # Plot bars for each method across all tasks (skip tasks where method is absent)
    star_positions = []
    for i, method_name in enumerate(method_display_names):
        x_positions = []
        y_values = []
        y_errors = []

        for j, task in enumerate(tasks):
            if method_name not in all_task_results[task]:
                continue
            method_results = all_task_results[task][method_name]
            task_metric_name = TASK_TO_METRIC_NAME.get(task, metric_name) if metric_name == "task_specific" else metric_name

            stat_key = f"overall_{aggregate_stat_name}"
            point_estimate = method_results[task_metric_name][stat_key]
            lower_ci = method_results[task_metric_name]["overall_lower_ci"]
            upper_ci = method_results[task_metric_name]["overall_upper_ci"]

            x_pos = task_positions[j] + (i - num_methods / 2 + 0.5) * bar_width
            x_positions.append(x_pos)
            y_values.append(point_estimate)
            y_errors.append([point_estimate - lower_ci, upper_ci - point_estimate])
            if method_results.get("_filtered_seeds", False):
                star_positions.append((x_pos, upper_ci))

        if not y_values:
            continue

        y_errors_transposed = np.array(y_errors).T
        ax.bar(x_positions, y_values, width=bar_width, label=method_name,
               yerr=y_errors_transposed, alpha=0.7, color=colors[i],
               ecolor='black', capsize=5, zorder=10)

    for x_s, y_s in star_positions:
        ax.text(x_s, y_s, "*", ha="center", va="bottom", fontsize=LEGEND_FONTSIZE, zorder=11)
    
    # Set x-axis tick labels to task names
    task_display_names = [TASK_TO_AXIS_DISPLAY_NAME[task] for task in tasks]
    ax.set_xticks(task_positions)
    ax.set_xticklabels(task_display_names, rotation=0, ha="center", fontsize=AXIS_LABEL_FONTSIZE)

    # Set labels and title
    ax.set_ylabel(f'{aggregate_stat_name.capitalize()} {metric_name.replace("_", " ").title() if metric_name != "task_specific" else "Normalized Return"}', 
                  fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(plot_title, fontsize=TITLE_FONTSIZE)
    
    ax.legend(fontsize=LEGEND_FONTSIZE, loc='center', 
              ncols=1 if plot_type != "core" else 2,
              bbox_to_anchor=(0.73, 0.9), # legend loc if under plot: (0.5, -0.25)
              framealpha=0.8)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        plt.savefig(os.path.join(savedir, f"{savename}.pdf"))
        print(f"Saved figure to {os.path.join(savedir, f'{savename}.pdf')}")
    if show_plot:
        plt.show()


TEAMMATE_HATCH = {
    "fcp_teammates": "",
    "comedi_teammates": "///",
}
TEAMMATE_TYPE_DISPLAY = {
    "fcp_teammates": "FCP",
    "comedi_teammates": "CoMeDi",
}


def plot_all_tasks_ego_bar_chart(
    all_task_results, metric_name: str, aggregate_stat_name: str,
    plot_title: str, save: bool, savedir: str, show_plot: bool, savename: str,
    run_info,
):
    """Bar chart for ego agents across all tasks with per-teammate hatching.

    run_info: ordered list of (display_name, base_method_name, hatch_pattern).
    Bars sharing the same base_method_name get the same color; hatch_pattern
    distinguishes teammate types within each method.
    """
    tasks = list(all_task_results.keys())
    num_tasks = len(tasks)

    seen_base: dict[str, int] = {}
    base_methods: list[str] = []
    for _, base_method, _ in run_info:
        if base_method not in seen_base:
            seen_base[base_method] = len(base_methods)
            base_methods.append(base_method)
    num_base = len(base_methods)
    colors = plt.cm.tab10(np.arange(num_base) / 10)

    num_bars = len(run_info)
    bar_width = min(0.8 / num_bars, 0.25)

    fig, ax = plt.subplots(figsize=(max(4, num_tasks * (num_bars * bar_width * 2 + 1)), 6))
    task_positions = np.arange(num_tasks)

    star_positions = []
    for i, (display_name, base_method, hatch) in enumerate(run_info):
        color = colors[seen_base[base_method]]
        x_positions, y_values, y_errors = [], [], []

        for j, task in enumerate(tasks):
            if display_name not in all_task_results[task]:
                continue
            method_results = all_task_results[task][display_name]
            task_metric_name = (
                TASK_TO_METRIC_NAME.get(task, metric_name)
                if metric_name == "task_specific"
                else metric_name
            )
            stat_key = f"overall_{aggregate_stat_name}"
            point_estimate = method_results[task_metric_name][stat_key]
            lower_ci = method_results[task_metric_name]["overall_lower_ci"]
            upper_ci = method_results[task_metric_name]["overall_upper_ci"]

            x_pos = task_positions[j] + (i - num_bars / 2 + 0.5) * bar_width
            x_positions.append(x_pos)
            y_values.append(point_estimate)
            y_errors.append([point_estimate - lower_ci, upper_ci - point_estimate])
            if method_results.get("_filtered_seeds", False):
                star_positions.append((x_pos, upper_ci))

        if not y_values:
            continue

        ax.bar(x_positions, y_values, width=bar_width, label=display_name,
               yerr=np.array(y_errors).T, alpha=0.7, color=color, hatch=hatch,
               ecolor='black', capsize=5, zorder=10)

    for x_s, y_s in star_positions:
        ax.text(x_s, y_s, "*", ha="center", va="bottom", fontsize=LEGEND_FONTSIZE, zorder=11)

    task_display_names = [TASK_TO_AXIS_DISPLAY_NAME[task] for task in tasks]
    ax.set_xticks(task_positions)
    ax.set_xticklabels(task_display_names, rotation=0, ha="center", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(
        f'{aggregate_stat_name.capitalize()} '
        f'{"Normalized Return" if metric_name == "task_specific" else metric_name.replace("_", " ").title()}',
        fontsize=AXIS_LABEL_FONTSIZE,
    )
    ax.set_title(plot_title, fontsize=TITLE_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE, loc='center', ncols=num_base,
              bbox_to_anchor=(0.5, -0.15), framealpha=0.8)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        plt.savefig(os.path.join(savedir, f"{savename}.pdf"), bbox_inches='tight')
        print(f"Saved figure to {os.path.join(savedir, f'{savename}.pdf')}")
    if show_plot:
        plt.show()


if __name__ == "__main__":
    from scripts.paper_vis.plot_globals import (
        GLOBAL_HELDOUT_CONFIG, SAVE_DIR,
        EGO_BENCHMARK_RUNS, UNIFIED_BENCHMARK_RUNS,
        METHOD_TO_DISPLAY_NAME, OEL_METHODS,
    )

    parser = argparse.ArgumentParser(description="Generate benchmark bar charts")
    parser.add_argument("--plot_type", type=str, default="unified",
                        choices=["unified", "ego"],
                        help="unified: teammate-generation methods; ego: ego-training methods")
    parser.add_argument("--use_best_returns_normalization", action=argparse.BooleanOptionalAction,
                        default=True, help="Renormalize using best observed returns across all methods")
    parser.add_argument("--show_plots", action="store_true",
                        help="Show plots in addition to saving them")
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR,
                        help="Directory to save figures")
    parser.add_argument("--tasks", nargs="+",
                        help="Tasks to plot. Defaults to all tasks with benchmark runs.")
    parser.add_argument("--force_recompute", action="store_true",
                        help="Re-fetch artifacts and recompute summary stats")
    parser.add_argument("--filter_failed_seeds", action="store_true",
                        help="Exclude seeds with near-zero returns across most heldout agents")
    parser.add_argument("--failed_seed_relative_threshold", type=float, default=0.10,
                        help="Seed-vs-teammate mean below this fraction of best seed counts as failed (default: 0.10)")
    parser.add_argument("--failed_seed_breadth_threshold", type=float, default=0.80,
                        help="Fraction of heldout agents that must fail to flag a seed (default: 0.80)")
    args = parser.parse_args()

    # Collect tasks that have at least one benchmark run defined
    all_tasks = sorted(
        set(EGO_BENCHMARK_RUNS.keys()) | set(UNIFIED_BENCHMARK_RUNS.keys())
    )
    task_list = args.tasks if args.tasks else all_tasks

    norm_suffix = "br_normalization" if args.use_best_returns_normalization else "original_normalization"

    all_task_results = {}
    ego_run_info: list[tuple[str, str, str]] = []  # (display_name, base_name, hatch)
    ego_seen: set[str] = set()

    for task_name in task_list:
        run_specs = []

        if args.plot_type == "unified":
            for method_name, run_id in UNIFIED_BENCHMARK_RUNS.get(task_name, {}).items():
                if not run_id:
                    continue
                display_name = METHOD_TO_DISPLAY_NAME.get(method_name, method_name)
                is_oel = method_name in OEL_METHODS
                run_specs.append((display_name, run_id, is_oel))

        elif args.plot_type == "ego":
            for method_name, teammate_runs in EGO_BENCHMARK_RUNS.get(task_name, {}).items():
                for teammate_type, run_id in teammate_runs.items():
                    if not run_id:
                        continue
                    base_name = METHOD_TO_DISPLAY_NAME.get(method_name, method_name)
                    teammate_display = TEAMMATE_TYPE_DISPLAY.get(teammate_type, teammate_type)
                    display_name = f"{base_name} ({teammate_display})"
                    is_oel = method_name in OEL_METHODS
                    run_specs.append((display_name, run_id, is_oel))
                    if display_name not in ego_seen:
                        ego_seen.add(display_name)
                        hatch = TEAMMATE_HATCH.get(teammate_type, "")
                        ego_run_info.append((display_name, base_name, hatch))

        if not run_specs:
            print(f"No benchmark runs configured for {task_name}, skipping.")
            continue

        all_task_results[task_name] = load_results_for_task(
            task_name,
            run_specs,
            force_recompute=args.force_recompute,
            renormalize_metrics=args.use_best_returns_normalization,
            filter_failed_seeds=args.filter_failed_seeds,
            failed_seed_relative_threshold=args.failed_seed_relative_threshold,
            failed_seed_breadth_threshold=args.failed_seed_breadth_threshold,
        )

    if not all_task_results:
        print("No results to plot.")
        raise SystemExit(1)

    agg_stat = GLOBAL_HELDOUT_CONFIG["global_heldout_settings"]["AGGREGATE_STAT"]

    if args.plot_type == "ego":
        plot_all_tasks_ego_bar_chart(
            all_task_results,
            metric_name="task_specific",
            aggregate_stat_name=agg_stat,
            plot_title="",
            save=True,
            savedir=args.save_dir,
            savename=f"all_tasks_{args.plot_type}_{norm_suffix}",
            show_plot=args.show_plots,
            run_info=ego_run_info,
        )
    elif len(all_task_results) == 1:
        task_name = next(iter(all_task_results))
        metric_name = TASK_TO_METRIC_NAME.get(task_name, "returned_episode_returns")
        from scripts.paper_vis.plot_globals import TASK_TO_PLOT_TITLE
        safe_task = task_name.replace("/", "_").replace("-", "_")
        plot_single_bar_chart(
            all_task_results[task_name],
            metric_name=metric_name,
            aggregate_stat_name=agg_stat,
            task_display_name=TASK_TO_AXIS_DISPLAY_NAME.get(task_name, task_name),
            plot_title=TASK_TO_PLOT_TITLE.get(task_name, task_name),
            save=True,
            savedir=args.save_dir,
            savename=f"{safe_task}_{args.plot_type}_{norm_suffix}",
            show_plot=args.show_plots,
        )
    else:
        plot_all_tasks_bar_chart(
            all_task_results,
            metric_name="task_specific",
            aggregate_stat_name=agg_stat,
            plot_type="core",
            save=True,
            savedir=args.save_dir,
            savename=f"all_tasks_{args.plot_type}_{norm_suffix}",
            plot_title="",
            show_plot=args.show_plots,
        )

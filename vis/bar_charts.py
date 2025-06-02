import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

from vis.process_data import load_results_for_task
from vis.plot_globals import TITLE_FONTSIZE, AXIS_LABEL_FONTSIZE, LEGEND_FONTSIZE, TASK_TO_AXIS_DISPLAY_NAME

plt.rcParams['xtick.labelsize'] = AXIS_LABEL_FONTSIZE
plt.rcParams['ytick.labelsize'] = AXIS_LABEL_FONTSIZE

def plot_single_bar_chart(results, metric_name: str, aggregate_stat_name: str,
                   plot_title: str, save: bool, savedir: str, show_plot: bool, savename: str):
    '''Plots a bar chart for a single task, metric, and aggregate stat.'''
    method_display_names = []
    y_values = []
    y_errors = []

    for display_name, method_results in results.items():
        method_display_names.append(display_name)

        stat_key = f"overall_{aggregate_stat_name}"
        point_estimate = method_results[metric_name][stat_key]
        lower_ci = method_results[metric_name]["overall_lower_ci"]
        upper_ci = method_results[metric_name]["overall_upper_ci"]

        y_values.append(point_estimate)
        # y_errors should be in the format [[lower_errors], [upper_errors]]
        y_errors.append([point_estimate - lower_ci, upper_ci - point_estimate])

    num_methods = len(method_display_names)
    x_pos = np.arange(num_methods)

    fig, ax = plt.subplots()

    # ax.axhline(y=1.0, color='dimgray', linestyle='--')

    # Transpose y_errors to match expected format for yerr
    y_errors_transposed = np.array(y_errors).T
    ax.bar(x_pos, y_values, yerr=y_errors_transposed, align='center', alpha=0.7, ecolor='black', capsize=10, zorder=2)

    ax.set_ylabel(f'{aggregate_stat_name.capitalize()} {metric_name.replace("_", " ").title()} (Normalized)', 
                  fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(method_display_names, rotation=0, ha="center", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(plot_title, fontsize=TITLE_FONTSIZE)
    ax.yaxis.grid(True)
    
    plt.tight_layout()
    if save:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        plt.savefig(os.path.join(savedir, f"{savename}.pdf"))
    if show_plot:
        plt.show()

def plot_all_tasks_bar_chart(all_task_results, metric_name: str, aggregate_stat_name: str,
                   plot_type: str, plot_title: str, save: bool, savedir: str, show_plot: bool, savename: str):
    '''Plots a bar chart where all tasks are plotted as groups on the same bar chart.'''
    tasks = list(all_task_results.keys())
    num_tasks = len(tasks)
    
    # Get method names from the first task (assuming all tasks have the same methods)
    first_task = tasks[0]
    method_display_names = list(all_task_results[first_task].keys())
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
    colors = plt.cm.tab10(np.linspace(0, 1, int(num_methods*1.5)))
    
    # Plot bars for each method across all tasks
    for i, method_name in enumerate(method_display_names):
        y_values = []
        y_errors = []
        
        for task in tasks:
            method_results = all_task_results[task][method_name]
            task_metric_name = TASK_TO_METRIC_NAME[task] if metric_name == "task_specific" else metric_name
            
            stat_key = f"overall_{aggregate_stat_name}"
            point_estimate = method_results[task_metric_name][stat_key]
            lower_ci = method_results[task_metric_name]["overall_lower_ci"]
            upper_ci = method_results[task_metric_name]["overall_upper_ci"]
            
            y_values.append(point_estimate)
            y_errors.append([point_estimate - lower_ci, upper_ci - point_estimate])
        
        # Position bars for this method within each task group
        x_positions = task_positions + (i - num_methods/2 + 0.5) * bar_width
        
        # Transpose y_errors to match expected format for yerr
        y_errors_transposed = np.array(y_errors).T
        
        # Plot bars for this method
        ax.bar(x_positions, y_values, width=bar_width, label=method_name, 
               yerr=y_errors_transposed, alpha=0.7, color=colors[i], 
               ecolor='black', capsize=5, zorder=10)
    
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
              bbox_to_anchor=(0.80, 0.9), # legend loc if under plot: (0.5, -0.25)
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


if __name__ == "__main__":
    from vis.plot_globals import OE_BASELINES, TEAMMATE_GEN_BASELINES, \
        GLOBAL_HELDOUT_CONFIG, TASK_TO_PLOT_TITLE, TASK_TO_METRIC_NAME, CACHE_FILENAME
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate bar charts for visualization")
    parser.add_argument("--plot_type", type=str, default="core",
                        help="Type of plot to generate")
    parser.add_argument("--use_original_normalization", action="store_true",
                        help="Use original normalization instead of best-returns normalization")
    parser.add_argument("--show_plots", action="store_true",
                        help="Show plots in addition to saving them")
    parser.add_argument("--save_dir", type=str, default="results/figures",
                        help="Directory to save plots")
    parser.add_argument("--tasks", nargs="+", 
                        help="List of tasks to show best returns for. If not provided, all tasks will be computed.")
    args = parser.parse_args()
    
    if args.plot_type == "core":
        RESULTS_TO_PLOT = {
            **OE_BASELINES,
            **TEAMMATE_GEN_BASELINES
        }
    else:
        # Provide your own results to plot here
        raise ValueError(f"Invalid plot type: {args.plot_type}")

    if not args.tasks:
        task_list = [
            "lbf", 
            "overcooked-v1/cramped_room",
            "overcooked-v1/asymm_advantages",
            "overcooked-v1/counter_circuit",
            "overcooked-v1/coord_ring",
            "overcooked-v1/forced_coord"
        ]
    else:
        task_list = args.tasks
    
    # Add suffix to savename based on normalization method
    norm_suffix = "original_normalization" if args.use_original_normalization else "br_normalization"
    
    all_task_results = {}
    for task_name in task_list:
        # Pass the normalization flag to load_results_for_task
        all_task_results[task_name] = load_results_for_task(
            task_name, 
            RESULTS_TO_PLOT, 
            CACHE_FILENAME, 
            load_from_cache=True,
            renormalize_metrics=not args.use_original_normalization
        )
        metric_name = TASK_TO_METRIC_NAME[task_name]

        # Individual task plots
        # PLOT_ARGS = {
        #     "save": True,
        #     "savedir": f"{args.save_dir}/{task_name}", 
        #     "savename": f"{args.plot_type}_bar_{norm_suffix}",
        #     "plot_title": TASK_TO_PLOT_TITLE[task_name],
        #     "show_plot": args.show_plots
        # }
        
        # plot_single_bar_chart(all_task_results[task_name], 
        #             metric_name=metric_name, 
        #             aggregate_stat_name=GLOBAL_HELDOUT_CONFIG["global_heldout_settings"]["AGGREGATE_STAT"],
        #             **PLOT_ARGS
        #             )
    
    # Plot all tasks together
    ALL_TASKS_PLOT_ARGS = {
        "plot_type": args.plot_type,
        "save": True,
        "savedir": args.save_dir, 
        "savename": f"all_tasks_comparison_{args.plot_type}_{norm_suffix}",
        "plot_title": "",
        "show_plot": args.show_plots
    }
    
    plot_all_tasks_bar_chart(all_task_results, 
                metric_name="task_specific", 
                aggregate_stat_name=GLOBAL_HELDOUT_CONFIG["global_heldout_settings"]["AGGREGATE_STAT"],
                **ALL_TASKS_PLOT_ARGS
                )

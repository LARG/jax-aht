import os
import numpy as np
import plotly
import plotly.graph_objects as go
import argparse

from vis.process_data import load_results_for_task
from vis.plot_globals import get_heldout_agents, TITLE_FONTSIZE
from vis.plot_globals import OE_BASELINES, TEAMMATE_GEN_BASELINES, \
    GLOBAL_HELDOUT_CONFIG, TASK_TO_PLOT_TITLE, TASK_TO_METRIC_NAME, CACHE_FILENAME

plotly.io.kaleido.scope.mathjax = None # disable mathjax to prevent the "loading mathjax" message


def plot_radar_chart(results, metric_name: str, aggregate_stat_name: str,
                   heldout_names: list, plot_title: str, 
                   save: bool, savedir: str, show_plot: bool, savename: str,
                   show_legend: bool, show_title: bool):
    '''Plots radar charts for each algorithm in results, to show the performance against each heldout agent. 
    '''
    fig = go.Figure()
    
    # Define colors for different methods
    colors = plotly.colors.qualitative.Plotly
    max_value = 0    

    for i, (method_name, method_results) in enumerate(results.items()):
        # Get the per-agent performance values
        per_agent_values = method_results[metric_name][f"{aggregate_stat_name}_per_agent"]
        max_value = max(max_value, np.max(per_agent_values))

        # Ensure the polygon closes by explicitly repeating the first point at the end
        radar_values = np.append(per_agent_values, per_agent_values[0])
        radar_categories = heldout_names.copy()
        radar_categories.append(radar_categories[0])  # Repeat the first category at the end

        # Add trace for this method with improved styling
        fig.add_trace(go.Scatterpolar(
            r=radar_values,
            theta=radar_categories,
            fill='toself',
            name=method_name,
            line=dict(color=colors[i % len(colors)], width=2),
            opacity=0.4
        ))
    
    # Add a black circle at radius 1.0
    theta_circle = np.linspace(0, 2*np.pi, 100)
    theta_labels = []
    for angle in theta_circle:
        idx = int(angle / (2*np.pi) * len(heldout_names))
        if idx >= len(heldout_names):
            idx = 0
        theta_labels.append(heldout_names[idx])
    
    fig.add_trace(go.Scatterpolar(
        r=[1.0] * len(theta_labels),
        theta=theta_labels,
        mode='lines',
        name='radius 1.0',
        line=dict(color='dimgray', width=1.5, dash='dot'),
        showlegend=False
    ))
    
    # Update the layout with improved styling
    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(
            l=105, r=50,
            autoexpand=True),
        polar=dict(
            bgcolor='white',
            radialaxis=dict(
                visible=True,
                range=[0, max_value * 1.15],  # Add 10% margin for visibility
                tickfont=dict(size=20),
                gridcolor='#EEEEEE',
                linecolor='#CCCCCC'
            ),
            angularaxis=dict(
                tickfont=dict(size=20, family="sans-serif", color="black"),
                linecolor='#CCCCCC',
                gridcolor='#EEEEEE'
            )
        ),
        title=dict(
            text=plot_title,
            font=dict(size=TITLE_FONTSIZE*2, family="sans-serif", color="black"),
            x=0.5,
            y=0.90
        ) if show_title else None,
        showlegend=show_legend,
        legend=dict(
            font=dict(size=15, family="sans-serif"),
            yanchor="top",
            y=1.20,
            xanchor="left",
            x=-0.20,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1
        )
    )
    
    # Save the figure if requested
    if save:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        
        # Configure Plotly to disable MathJax
        config = {'displayModeBar': False, 'mathjax': None}
        
        # Save as PDF for paper with high resolution
        pdf_path = os.path.join(savedir, f"{savename}.pdf")
        fig.write_image(pdf_path, 
                        # width=1000, height=900
                        )
        print(f"Saved radar chart to {pdf_path}")
    
    # Show the plot if requested
    if show_plot:
        fig.show()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate radar charts for visualization")
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
    parser.add_argument("--show_legend", action="store_true",
                        help="Show legend in the plot")
    parser.add_argument("--show_title", action="store_true",
                        help="Show title in the plot")
    args = parser.parse_args()
    
    if args.plot_type == "core":
        RESULTS_TO_PLOT = {
            **OE_BASELINES,
            **TEAMMATE_GEN_BASELINES
        }
    else:
        # Provide your own results to plot here
        raise ValueError(f"Invalid plot type: {args.plot_type}")
    
    # Add suffix to savename based on normalization method
    norm_suffix = "original_normalization" if args.use_original_normalization else "br_normalization"
    
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
    
    for task in task_list:
        PLOT_ARGS = {
            "save": True,
            "savedir": f"{args.save_dir}/{task}", 
            "savename": f"radar_{args.plot_type}_{norm_suffix}",
            "plot_title": TASK_TO_PLOT_TITLE[task],
            "show_plot": args.show_plots,
            "show_legend": args.show_legend,
            "show_title": args.show_title
        }

        results = load_results_for_task(
            task, 
            RESULTS_TO_PLOT, 
            CACHE_FILENAME, 
            load_from_cache=True,
            renormalize_metrics=not args.use_original_normalization
        )
        metric_name = TASK_TO_METRIC_NAME[task]
        aggregate_stat_name = GLOBAL_HELDOUT_CONFIG["global_heldout_settings"]["AGGREGATE_STAT"]
        heldout_agent_dict = get_heldout_agents(task, task_config_path=f"open_ended_training/configs/task/{task.replace('-v1', '')}.yaml")
        heldout_names = list(heldout_agent_dict.keys())
        plot_radar_chart(results, metric_name, aggregate_stat_name, heldout_names, **PLOT_ARGS)

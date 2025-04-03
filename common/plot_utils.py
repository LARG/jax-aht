import os
from functools import partial
import jax
import jax.numpy as jnp
import seaborn as sns
import matplotlib.pyplot as plt

@partial(jax.jit, static_argnames=['stats', 'num_controlled_agents'])
def get_stats(metrics, stats: tuple, num_controlled_agents: int):
    '''
    Computes mean and std of metrics of interest for each seed and update, 
    using only the final steps of episodes. Note that each rollout contains multiple episodes.

    metrics is a pytree where each leaf has shape 
        (num_seeds, num_updates, rollout_length, num_envs*num_agents)
    stats is a tuple of strings, each corresponding to a metric of interest in metrics
    '''
    num_seeds, num_updates, rollout_len, _ = metrics["returned_episode_lengths"].shape

    # Create mask for final steps of episodes
    mask = metrics["returned_episode_lengths"][..., :num_controlled_agents] > 0
    
    # Initialize output dictionary
    all_stats = {}
    stats = list(stats) # convert to list to correctly iterate if the tuple only has a single element
    for stat_name in stats:
        # Get the metric array
        metric_data = metrics[stat_name][..., :num_controlled_agents]  # Shape: (num_seeds, num_updates, rollout_length, num_envs)

        # Compute means and stds for each seed and update
        # Use masked operations to only consider final episode steps
        means = jnp.where(mask, metric_data, 0).sum(axis=(2, 3)) / mask.sum(axis=(2, 3))
        # For std, first compute masked values
        masked_vals = jnp.where(mask, metric_data, 0)
        squared_diff = (masked_vals - means[..., None, None]) ** 2
        variance = jnp.where(mask, squared_diff, 0).sum(axis=(2, 3)) / mask.sum(axis=(2, 3))
        stds = jnp.sqrt(variance)
        # Stack means and stds
        all_stats[stat_name] = jnp.stack([means, stds], axis=-1)
    
    return all_stats

def plot_train_metrics(all_stats, 
                       num_rollout_steps, num_envs,
                       savedir=None, savename=None,
                       show_plots=False
                       ):
    '''Each key in all_stats is a metric name, and the value is an array of shape (num_seeds, num_updates, 2)'''
    figures = {}
    for stat_name, stats in all_stats.items():
        stat_name = stat_name.replace("_", " ").title()
        num_seeds, num_updates, _ = stats.shape
        for i in range(num_seeds):
            print("Seed: ", i)
            print(f"Mean {stat_name} (Last Episode Step): ", stats[i, -1, 0])
            print(f"Std {stat_name} (Last Episode Step): ", stats[i, -1, 1])
            xs = jnp.arange(num_updates) * num_envs * num_rollout_steps
            means = stats[i, :, 0]
            stds = stats[i, :, 1]

            # Calculate upper and lower bounds for the shaded region
            upper_bound = means + stds
            lower_bound = means - stds

            # Create the plot
            plt.plot(xs, means, label=f"Seed {i}")

            # Shade the region between the bounds
            plt.fill_between(xs, lower_bound, upper_bound, 
                            alpha=0.3)
                        
        plt.xlabel("Time Step")
        plt.ylabel(stat_name)
        plt.title(f"Learning Curve for {stat_name}")
        plt.legend()
        
        # Get the current figure
        fig = plt.gcf()
        
        # Save the figure if requested
        savepath = None
        if savedir is not None and savename is not None:
            savepath = os.path.join(savedir, f"{savename}_{stat_name}.pdf")
            plt.savefig(savepath)
        figures[stat_name] = fig
        if show_plots:
            plt.show()
        
        plt.close(fig)
    
    return figures, savepath

def plot_eval_metrics(eval_metrics, metric_name, higher_is_better=True, agent_idx=0):
    '''
    Note that the FCP agent is always agent 0, the partner is agent 1. 
    
    eval_metrics is a dictionary with keys corresponding to metric names 
    and values as arrays of shape (num_seeds, num_fcp_checkpoints, num_eval_checkpoints, num_episodes, num_agents)
    '''
    # Select agent 0's data and compute mean over seeds and episodes
    heatmap_data = jnp.mean(eval_metrics[metric_name][:, :, :, :, agent_idx], axis=(0, 3))
    
    if higher_is_better:
        colormap="coolwarm_r"
        arrow_str = r" ($\uparrow$)"
    else:    
        colormap="coolwarm"
        arrow_str = r" ($\downarrow$)"
    # Plot as heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(heatmap_data, cmap=colormap, annot=False)
    plt.gca().invert_yaxis()
    plt.xlabel("Eval Checkpoint")
    plt.ylabel("Ego Agent Checkpoint")
    title = f"Average {metric_name.replace('_', ' ').title()}"
    plt.title(title + arrow_str)
    plt.show()
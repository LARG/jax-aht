import os
import numpy as np

import pickle
from common.plot_utils import get_metric_names
from common.save_load_utils import load_train_run
from common.stat_utils import compute_aggregate_stat_and_ci_per_task, compute_aggregate_stat_and_ci
from vis.plot_globals import TASK_TO_ENV_NAME, GLOBAL_HELDOUT_CONFIG, HELDOUT_CURVES_CACHE_FILENAME, get_heldout_agents

# Import from compute_best_returns
from vis.compute_best_returns import load_best_returns, renormalize_eval_metrics, unnormalize_data


def load_results_for_task(task_name, method_dict, cache_filename, 
                          load_from_cache: bool = True, renormalize_metrics: bool = True):
    '''Loads the latest results for the given task, and caches the computed statistics. 
    If the cache does not exist or load_from_cache is False, the results are computed and saved to the cache.
    '''
    env_name = TASK_TO_ENV_NAME[task_name]

    # Load best returns for renormalization if required
    best_returns = None
    if renormalize_metrics:
        best_returns = load_best_returns(task_name)
        print(f"Loaded best returns for {task_name}: {best_returns}")

    results = {}
    for method_subpath, (method_type, display_name) in method_dict.items():
        method_dir = f"results/{task_name}/{method_subpath}"
        # method_dir should have one subdirectory based on the date of the run
        # get all subdirectories in method_dir
        subdirs = [d for d in os.listdir(method_dir) if os.path.isdir(os.path.join(method_dir, d))]
        # get the most recent subdirectory
        most_recent_subdir = max(subdirs, key=lambda x: os.path.getmtime(os.path.join(method_dir, x)))
        
        method_eval_dir = os.path.join(method_dir, most_recent_subdir, "heldout_eval_metrics/")
        cache_file_path = os.path.join(method_eval_dir, cache_filename)

        # Add a suffix to the cache filename if we're renormalizing
        if renormalize_metrics:
            cache_file_path = cache_file_path.replace('.pkl', '_renormalized.pkl')

        summary_data = None
        if load_from_cache and os.path.exists(cache_file_path):
            try:
                with open(cache_file_path, 'rb') as f:
                    summary_data = pickle.load(f)
                print(f"Loaded cached results for {display_name} from {cache_file_path}")
            except Exception as e:
                print(f"Error loading cache for {display_name} from {cache_file_path}: {e}. Recomputing.")
                summary_data = None # Ensure recomputation if cache loading fails

        if summary_data is None:
            print(f"Cache not found or load_from_cache=False for {display_name}. Computing metrics...")
            # First load the original normalized metrics
            eval_metrics = load_train_run(method_eval_dir)
            
            # Renormalize metrics if required
            if renormalize_metrics and best_returns:
                print(f"Renormalizing metrics for {display_name} using best observed returns")
                # The renormalize_eval_metrics function will:
                # 1. Unnormalize the metrics using original performance bounds
                # 2. Normalize them using the best returns
                eval_metrics = renormalize_eval_metrics(eval_metrics, task_name, best_returns)

            if cache_filename == HELDOUT_CURVES_CACHE_FILENAME:
                summary_data = heldout_learning_curves(
                    GLOBAL_HELDOUT_CONFIG, eval_metrics, get_metric_names(env_name)
                )
            else:
                summary_data = heldout_metrics_per_agent(
                    GLOBAL_HELDOUT_CONFIG, eval_metrics, get_metric_names(env_name),
                    oel_method=True if method_type == "open_ended" else False
                )
            # Save the computed summary_data to cache
            if not os.path.exists(method_eval_dir): # Should exist, but good practice
                os.makedirs(method_eval_dir, exist_ok=True)

            with open(cache_file_path, 'wb') as f:
                pickle.dump(summary_data, f)
            print(f"Saved computed results for {display_name} to {cache_file_path}")

        results[display_name] = summary_data
    return results

def heldout_metrics_per_agent(config, eval_metrics, metric_names: tuple, 
                              oel_method: bool):
    '''Treat the first two dimensions of eval_metrics as (seeds, iters, ...) dimensions.
    Computes the aggregate stat and CI for the last iteration of OEL, for each heldout agent.
    '''
    num_heldout_agents = eval_metrics[metric_names[0]].shape[-3]

    summary_data = {}
    aggregate_stat = config["global_heldout_settings"]["AGGREGATE_STAT"]
    
    for metric_name in metric_names:
        if oel_method:
            # shape is (num_seeds, num_oel_iter, num_heldout_agents, num_eval_episodes, num_agents_per_game)
            # we extract the last iteration, take the mean over the num_agents_per_game dimension, and 
            # then reshape to (num_seeds*num_eval_episodes, num_heldout_agents)
            data = eval_metrics[metric_name][:, -1].mean(axis=-1
               ).transpose(0, 2, 1
               ).reshape(-1, num_heldout_agents)
        else:
            # shape is (num_seeds, num_heldout_agents, num_eval_episodes, num_agents_per_game)
            data =  eval_metrics[metric_name].mean(axis=-1
               ).transpose(0, 2, 1
               ).reshape(-1, num_heldout_agents) # final shape (num_seeds*num_eval_episodes, num_heldout_agents)

        data = np.array(data)

        # now compute per-heldout-agent aggregate stat+CIs
        point_est_per_task, interval_ests_per_task = compute_aggregate_stat_and_ci_per_task(data, aggregate_stat, return_interval_est=True)
        lower_ci_per_task = interval_ests_per_task[:, 0]
        upper_ci_per_task = interval_ests_per_task[:, 1]


        # compute aggregate stat+CI over all heldout agents
        point_est_all, interval_ests_all = compute_aggregate_stat_and_ci(data, aggregate_stat, return_interval_est=True)
        lower_ci_all = interval_ests_all[0]
        upper_ci_all = interval_ests_all[1]

        summary_data[metric_name] = {
            f"overall_{aggregate_stat}": point_est_all,
            "overall_lower_ci": lower_ci_all,
            "overall_upper_ci": upper_ci_all,
            f"{aggregate_stat}_per_agent": point_est_per_task,
            "per_agent_lower_ci": lower_ci_per_task,
            "per_agent_upper_ci": upper_ci_per_task,
        }
    return summary_data


def heldout_learning_curves(config, eval_metrics, metric_names: tuple):
    '''
    Computes the aggregate stat and CI for the all iterations of OEL and for all heldout agents.
    Will not work for teammate generation methods, since they don't have OEL iterations.
    '''
    _, num_oel_iter, num_heldout_agents, _, _ = eval_metrics[metric_names[0]].shape

    summary_data = {}
    aggregate_stat = config["global_heldout_settings"]["AGGREGATE_STAT"]
    
    for metric_name in metric_names:
        summary_data[metric_name] = {
            f"overall_{aggregate_stat}": [],
            "overall_lower_ci": [],
            "overall_upper_ci": [],
            f"{aggregate_stat}_per_agent": [],
            "per_agent_lower_ci": [],
            "per_agent_upper_ci": [],
        }

        # shape is (num_seeds, num_oel_iter, num_heldout_agents, num_eval_episodes, num_agents_per_game)
        # we extract iter i's data, take the mean over the num_agents_per_game dimension, and 
        # then reshape to (num_seeds*num_eval_episodes, num_heldout_agents)
        for i in range(num_oel_iter):
            data = eval_metrics[metric_name][:, i].mean(axis=-1
                ).transpose(0, 2, 1
                ).reshape(-1, num_heldout_agents)
            data = np.array(data)

            # now compute per-heldout-agent aggregate stat+CIs
            point_est_per_task, interval_ests_per_task = compute_aggregate_stat_and_ci_per_task(data, aggregate_stat, return_interval_est=True)
            lower_ci_per_task = interval_ests_per_task[:, 0]
            upper_ci_per_task = interval_ests_per_task[:, 1]


            # compute aggregate stat+CI over all heldout agents
            point_est_all, interval_ests_all = compute_aggregate_stat_and_ci(data, aggregate_stat, return_interval_est=True)
            lower_ci_all = interval_ests_all[0]
            upper_ci_all = interval_ests_all[1]

            summary_data[metric_name][f"overall_{aggregate_stat}"].append(point_est_all)
            summary_data[metric_name]["overall_lower_ci"].append(lower_ci_all)
            summary_data[metric_name]["overall_upper_ci"].append(upper_ci_all)
            summary_data[metric_name][f"{aggregate_stat}_per_agent"].append(point_est_per_task)
            summary_data[metric_name]["per_agent_lower_ci"].append(lower_ci_per_task)
            summary_data[metric_name]["per_agent_upper_ci"].append(upper_ci_per_task)

    return summary_data
'''This script evaluates one set of ego agents against a set of heldout agents. 
Based off the generate_xp_matrix.py script.

This evaluation can be run on a specific task using the following command: 
`python evaluation/run.py --config-name human_proxy_eval task=overcooked-v1/cramped_room`

Before running the evaluation, please ensure that the paths to the ego agent set are correctly 
specified in the config file, `configs/human_proxy_eval.yaml`.
'''
import os
from io import StringIO
import numpy as np
from prettytable import PrettyTable
import pandas as pd
import hydra
import jax

from common.agent_loader_from_config import initialize_rl_agent_from_config
from common.tree_utils import tree_stack
from common.plot_utils import get_metric_names
from common.stat_utils import compute_aggregate_stat_and_ci_per_task
from envs import make_env
from envs.log_wrapper import LogWrapper
from evaluation.heldout_evaluator import (
    load_heldout_set,
    eval_egos_vs_heldouts, 
    extract_params, 
    extract_performance_bounds,
    )


def eval_ego_set_vs_heldout_set(config, env, rng, num_episodes, heldout_agents: dict, ego_agents: dict):
    '''Evaluate all heldout agents against each other
    Args: 
        heldout_agents: a dictionary of {agent_name: (policy, params, test_mode, performance_bounds)} 
            for each heldout partner. params might be None for heuristic agents.
        ego_agents: a dictionary of {agent_name: [(policy, params, test_mode, performance_bounds)]} 
            mapping each ego agent type to a list of model checkpoints
    Returns a pytree of shape (num_heldout_agents, num_ego_agents, num_ego_checkpoints, num_episodes, num_agents_per_env)
    '''
    num_agents = env.num_agents
    assert num_agents == 2, "This eval code assumes exactly 2 agents."

    all_metrics = []
    
    # run evaluation of one ego agent type against all heldout agents
    eval_rngs = jax.random.split(rng, len(ego_agents))
    for i, (ego_agent_name, ego_agent_checkpoints) in enumerate(ego_agents.items()):
        # ego_agents_checkpoints is a list of tuples (policy, params, test_mode, performance_bounds)
        # same for all checkpoints
        ego_policy = ego_agent_checkpoints[0][0]
        ego_test_mode = ego_agent_checkpoints[0][2]

        # format params into a single pytree
        ego_param_list = [checkpoint[1] for checkpoint in ego_agent_checkpoints]
        ego_param_stacked = tree_stack(ego_param_list)

        eval_metrics = eval_egos_vs_heldouts(
            config, env, eval_rngs[i], num_episodes, 
            ego_policy, ego_param_stacked, list(heldout_agents.values()), ego_test_mode)
        # eval_metrics shape: (num_ego_checkpoints, num_heldout_agents, num_episodes, num_agents_per_env)
        all_metrics.append(eval_metrics)

    return tree_stack(all_metrics) # shape: (num_ego_agents, num_ego_checkpoints, num_heldout_agents, num_episodes, num_agents_per_env)


def load_ego_set(ego_set_config, env, rng):
    '''Load ego agents from config
    Returns a dictionary of agents with format {agent_name: [(policy, params, test_mode, performance_bounds)]}
    where the list of tuples is for each model checkpoint.
    '''
    ego_agents = {}
    for agent_name, agent_config in ego_set_config.items():
        params_list = None
        idx_labels = None
        test_mode = agent_config.get("test_mode", False)
        # Load RL-based agents
        if "path" in agent_config:
            # ensure that each rl agent has a unique initialization rng
            rng, init_rng = jax.random.split(rng)
            policy, params, init_params, idx_labels = initialize_rl_agent_from_config(agent_config, agent_name, env, init_rng)
            # params contains multiple model checkpoints, so we need to extract each one
            params_list, _ = extract_params(params, init_params, idx_labels)
            performance_bounds_list = extract_performance_bounds(agent_config, len(params_list))

        ego_agents[agent_name] = []
        for i, params_i in enumerate(params_list):
            ego_agents[agent_name].append((policy, params_i, test_mode, performance_bounds_list[i]))
    return ego_agents


def run_human_proxy_evaluation(config, print_metrics=False):
    '''Run heldout evaluation'''
    # Create only one environment instance
    env = make_env(config["ENV_NAME"], config["ENV_KWARGS"])
    env = LogWrapper(env)
    
    rng = jax.random.PRNGKey(config["global_heldout_settings"]["EVAL_SEED"])
    rng, heldout_init_rng, ego_init_rng, eval_rng = jax.random.split(rng, 4)
    
    # load heldout agents
    heldout_cfg = config["heldout_set"][config["TASK_NAME"]]
    heldout_agents = load_heldout_set(heldout_cfg, env, config["TASK_NAME"], config["ENV_KWARGS"], heldout_init_rng)
    
    # load ego agents
    ego_cfg = config["ego_set"][config["TASK_NAME"]]
    ego_agents = load_ego_set(ego_cfg, env, ego_init_rng)

    # run evaluation
    eval_metrics = eval_ego_set_vs_heldout_set(
        config, env, eval_rng, config["global_heldout_settings"]["NUM_EVAL_EPISODES"], 
        heldout_agents, ego_agents)

    if print_metrics:
        # each leaf of eval_metrics has shape (num_ego_agents, num_ego_checkpoints, num_heldout_agents, num_episodes, num_agents_per_env)
        metric_names = get_metric_names(config["ENV_NAME"])
        heldout_names = list(heldout_agents.keys())
        ego_names = list(ego_agents.keys())
        for metric_name in metric_names:
            print_human_proxy_metrics_table(eval_metrics, metric_name, ego_names, heldout_names, 
            config["global_heldout_settings"]["AGGREGATE_STAT"], 
            config["global_heldout_settings"]["NORMALIZE_RETURNS"], save=True, save_format="markdown")
    return eval_metrics


def print_human_proxy_metrics_table(eval_metrics, metric_name, ego_names, heldout_names, 
                        aggregate_stat: str, normalized_metrics: bool, save: bool = False, save_format: str = "csv"):
    '''Generate a table of the aggregate stat and CI of the metric for each ego agent and heldout agent.
    Args:
        eval_metrics: pytree of shape (num_ego_agents, num_ego_checkpoints, num_heldout_agents, num_episodes, num_agents_per_env)
        metric_name: name of the metric to print
        ego_names: list of ego agent names
        heldout_names: list of heldout agent names
        aggregate_stat: string of the aggregate stat to print
        normalized_metrics: boolean of whether the metrics are normalized
        save: boolean of whether to save the table to a csv file
    '''
    # we first take the mean over the num_agents_per_env dimension
    eval_metric_data = np.array(eval_metrics[metric_name]).mean(axis=-1) 
    num_ego_agents, num_ego_checkpoints, num_heldout_agents, num_eval_episodes = eval_metric_data.shape
    table = PrettyTable()
    table.field_names = ["---", *heldout_names]

    for i, ego_name in enumerate(ego_names):
        data = eval_metric_data[i].transpose(0, 2, 1).reshape(-1, num_heldout_agents) # shape (num_ego_checkpoints*num_eval_episodes, num_heldout_agents)
        point_est_all, interval_ests_all = compute_aggregate_stat_and_ci_per_task(data, aggregate_stat, return_interval_est=True)
        lower_ci = interval_ests_all[:, 0]
        upper_ci = interval_ests_all[:, 1]
        row = [ego_name] + [f"{point_est_all[j]:.2f} ({lower_ci[j]:.2f}, {upper_ci[j]:.2f})" for j in range(len(heldout_names))]
        table.add_row(row)
    
    print(f"\n{metric_name} ({aggregate_stat} ± CI):")
    if normalized_metrics:
        print("Metrics are normalized to [lower_bound, upper_bound].")
    print(table)

    if save:
        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        
        # Sanitize metric_name for use in filename
        safe_metric_name = "".join(c if c.isalnum() else "_" for c in metric_name)
        
        table_as_csv_str = table.get_csv_string()
        if save_format == "csv":    
            csv_filename = os.path.join(output_dir, f"{safe_metric_name}_{aggregate_stat}_normalized={normalized_metrics}.csv")
            with open(csv_filename, 'w', newline='') as f_output:
                f_output.write(table_as_csv_str)
            print(f"Table saved to {csv_filename}")
        elif save_format == "markdown":
            markdown_filename = os.path.join(output_dir, f"{safe_metric_name}_{aggregate_stat}_normalized={normalized_metrics}.md")
            df = pd.read_csv(StringIO(table_as_csv_str), sep=",")
            df.to_markdown(markdown_filename, index=False)
            print(f"Table saved to {markdown_filename}")
        else:
            raise ValueError(f"Invalid save format: {save_format}")

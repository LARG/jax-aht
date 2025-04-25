'''This script implements evaluating ego agents against heldout agents. 
Warning: ActorCritic agents that rely on auxiliary information to compute actions are not currently supported.
'''
import jax
import numpy as np
from prettytable import PrettyTable
from functools import partial
import time

from agents.lbf.agent_policy_wrappers import LBFRandomPolicyWrapper, LBFSequentialFruitPolicyWrapper
from agents.overcooked.agent_policy_wrappers import (OvercookedIndependentPolicyWrapper, 
    OvercookedOnionPolicyWrapper,
    OvercookedPlatePolicyWrapper,
    OvercookedStaticPolicyWrapper,
    OvercookedRandomPolicyWrapper)

from evaluation.agent_loader_from_config import initialize_rl_agent_from_config
from common.run_episodes import run_episodes
from common.tree_utils import tree_stack
from common.plot_utils import get_metric_names
from envs import make_env
from envs.log_wrapper import LogWrapper
from envs.overcooked.augmented_layouts import augmented_layouts


def extract_params(params, init_params, idx_labels=None):
    '''params is a pytree of n model checkpoints, where each leaf has an unknown number 
    of checkpoint dimensions, and the last dimension corresponds to the layer dimension. 
    This function extracts each of the n checkpoints and returns a list of n pytrees, 
    where each pytree has the same structure as init_params.

    Args:
        params: pytree of n checkpoints (n >= 1)
        init_params: pytree corresp. to ONE checkpoint. used as a reference for the structure of the output pytrees.
        idx_labels: array of string labels with the same shape as the original checkpoints. If None, numeric indices will be used.

    Returns:
        Tuple of:
            - list of n pytrees with same structure as init_params
            - list of n index labels identifying the original location of each checkpoint
    '''
    assert jax.tree.structure(params) == jax.tree.structure(init_params), "Params and init_params must have the same structure."

    model_list = []
    flattened_idx_labels = []
    params_shape = jax.tree.leaves(params)[0].shape
    init_params_shape = jax.tree.leaves(init_params)[0].shape

    # already matches init_params_shape, no extraction needed
    if params_shape == init_params_shape:
        model_list = [params]
        n_models = 1
        
        if idx_labels is not None:
            flattened_idx_labels = idx_labels
    # multiple models, extract each one
    else:
        # first, flatten the params so that each leaf has shape (..., init_params_shape)
        flattened_params = jax.tree.map(lambda x, y: x.reshape((-1,) + y.shape), params, init_params)        
        # then, extract each model
        n_models = jax.tree.leaves(flattened_params)[0].shape[0]
        
        # Now, flatten the idx_labels to match the flattened parameters
        if idx_labels is not None:
            flattened_idx_labels = np.array(idx_labels).reshape(n_models)
            
        # Extract each model
        for i in range(n_models):
            model_i = jax.tree.map(lambda x: x[i], flattened_params)
            model_list.append(model_i)
    
    if idx_labels is None:
        flattened_idx_labels = [str(i) for i in range(n_models)]
    
    return model_list, flattened_idx_labels

def load_heldout_set(heldout_config, env, task_name, env_kwargs, rng):
    '''Load heldout evaluation agents from config.
    Returns a dictionary of agents with keys as agent names and values as tuples of (policy, params, test_mode).
    '''
    heldout_agents = {}
    for agent_name, agent_config in heldout_config.items():
        params_list = None
        idx_labels = None
        test_mode = agent_config.get("test_mode", False)
        # Load RL-based agents
        if "path" in agent_config:
            # ensure that each rl agent has a unique initialization rng
            rng, init_rng = jax.random.split(rng)
            policy, params, init_params, idx_labels = initialize_rl_agent_from_config(agent_config, agent_name, env, init_rng)
            # params contains multiple model checkpoints, so we need to extract each one
            params_list, idx_labels = extract_params(params, init_params, idx_labels)

        # Load non-RL-based heuristic agents
        elif task_name == 'lbf':
            if agent_config["actor_type"] == 'random_agent':
                policy = LBFRandomPolicyWrapper(using_log_wrapper=True)
            elif agent_config["actor_type"] == 'seq_agent':
                # Get grid size and num fruits from environment
                grid_size = env_kwargs.get("grid_size", 7)
                num_fruits = env_kwargs.get("num_fruits", 3)
                ordering_strategy = agent_config.get("ordering_strategy", "lexicographic")
                policy = LBFSequentialFruitPolicyWrapper(
                    grid_size=grid_size,
                    num_fruits=num_fruits,
                    ordering_strategy=ordering_strategy,
                    using_log_wrapper=True
                )

        elif 'overcooked-v1' in task_name:
            aug_layout_dict = augmented_layouts[env_kwargs["layout"]]
            if agent_config["actor_type"] == 'random_agent':
                policy = OvercookedRandomPolicyWrapper(aug_layout_dict, using_log_wrapper=True)
            elif agent_config["actor_type"] == 'static_agent':
                policy = OvercookedStaticPolicyWrapper(aug_layout_dict, using_log_wrapper=True)
            elif agent_config["actor_type"] == 'independent_agent':
                policy = OvercookedIndependentPolicyWrapper(
                    aug_layout_dict, using_log_wrapper=True, 
                    p_onion_on_counter=agent_config.get("p_onion_on_counter", 0.0), 
                    p_plate_on_counter=agent_config.get("p_plate_on_counter", 0.0))
            elif agent_config["actor_type"] == 'onion_agent':
                policy = OvercookedOnionPolicyWrapper(
                    aug_layout_dict, using_log_wrapper=True, 
                    p_onion_on_counter=agent_config.get("p_onion_on_counter", 0.0))
            elif agent_config["actor_type"] == 'plate_agent':
                policy = OvercookedPlatePolicyWrapper(
                    aug_layout_dict, using_log_wrapper=True, 
                    p_plate_on_counter=agent_config.get("p_plate_on_counter", 0.0))
        else:
            raise ValueError(f"Unknown task: {task_name}")
        
        # Generate agent labels
        if params_list is None: # heuristic agent
            heldout_agents[agent_name] = (policy, None, test_mode)
        else: # rl agent
            for i, params_i in enumerate(params_list):
                if idx_labels is None:
                    agent_label = f'{agent_name} ({i})'
                else:
                    agent_label = f'{agent_name} ({idx_labels[i]})'
                heldout_agents[agent_label] = (policy, params_i, test_mode)

    return heldout_agents


def eval_egos_vs_heldouts(config, env, rng, num_episodes, ego_policy, ego_params, heldout_agent_list, ego_test_mode=False):
    '''Evaluate all ego agents against all heldout partners using vmap over egos.'''
    
    num_agents = env.num_agents
    assert num_agents == 2, "This eval code assumes exactly 2 agents."

    num_ego_agents = jax.tree.leaves(ego_params)[0].shape[0]
    num_partner_total = len(heldout_agent_list)

    def _eval_one_ego_vs_one_partner(single_ego_policy, single_ego_params, rng_for_ego,
                                     heldout_policy, heldout_params, heldout_test_mode):
        return run_episodes(rng_for_ego, env,
                            agent_0_policy=single_ego_policy, agent_0_param=single_ego_params,
                            agent_1_policy=heldout_policy, agent_1_param=heldout_params,
                            max_episode_steps=config["global_heldout_settings"]["MAX_EPISODE_STEPS"],
                            num_eps=num_episodes, 
                            agent_0_test_mode=ego_test_mode,
                            agent_1_test_mode=heldout_test_mode)

    # Outer Python loop over heterogeneous heldout partners
    all_metrics_for_partners = []
    partner_rngs = jax.random.split(rng, num_partner_total)
    start_time = time.time()

    for partner_idx in range(num_partner_total):
        heldout_policy, heldout_params, heldout_test_mode = heldout_agent_list[partner_idx]
        ego_rngs = jax.random.split(partner_rngs[partner_idx], num_ego_agents)

        # Use partial to fix the heldout agent for the function being vmapped
        func_to_vmap = partial(_eval_one_ego_vs_one_partner,
                               heldout_policy=heldout_policy,
                               heldout_params=heldout_params,
                               heldout_test_mode=heldout_test_mode)

        # vmap over the stacked ego agents and their RNGs
        results_for_this_partner = jax.vmap(
            func_to_vmap,
            in_axes=(None, 0, 0) # Map over axis 0 of ego_policies, ego_params, ego_rngs
        )(ego_policy, ego_params, ego_rngs)

        # results_for_this_partner shape: (num_ego_agents, num_episodes, ...)
        all_metrics_for_partners.append(results_for_this_partner)

    end_time = time.time()
    print(f"Time taken for vmap evaluation loop: {end_time - start_time:.2f} seconds")

    # Result shape: (num_partners, num_egos, num_episodes, ...)
    final_metrics = tree_stack(all_metrics_for_partners)
    # Transpose to (num_egos, num_partners, num_episodes, ...)
    final_metrics = jax.tree.map(lambda x: x.transpose(1, 0, 2, 3), final_metrics)

    return final_metrics

def run_heldout_evaluation(config, print_metrics=False):
    '''Run heldout evaluation'''
    # Create only one environment instance
    env = make_env(config["ENV_NAME"], config["ENV_KWARGS"])
    env = LogWrapper(env)
    
    rng = jax.random.PRNGKey(config["global_heldout_settings"]["EVAL_SEED"])
    rng, ego_init_rng, heldout_init_rng, eval_rng = jax.random.split(rng, 4)
    
    # load ego agents
    ego_agent_config = dict(config["ego_agent"])
    ego_test_mode = ego_agent_config.get("test_mode", False)
    ego_policy, ego_params, init_ego_params, ego_idx_labels = initialize_rl_agent_from_config(ego_agent_config, "ego", env, ego_init_rng)
    # flatten ego params and idx labels
    ego_idx_labels = np.array(ego_idx_labels).reshape(-1) # flatten the list of ego agent labels 
    flattened_ego_params = jax.tree.map(lambda x, y: x.reshape((-1,) + y.shape), ego_params, init_ego_params)        
    
    # load heldout agents
    heldout_cfg = config["heldout_set"][config["TASK_NAME"]]
    heldout_agents = load_heldout_set(heldout_cfg, env, config["TASK_NAME"], config["ENV_KWARGS"], heldout_init_rng)
    heldout_agent_list = list(heldout_agents.values())
    
    # run evaluation
    eval_metrics = eval_egos_vs_heldouts(
        config, env, eval_rng, config["global_heldout_settings"]["NUM_EVAL_EPISODES"], 
        ego_policy, flattened_ego_params, heldout_agent_list, ego_test_mode)

    if print_metrics:
        # each leaf of eval_metrics has shape (num_ego_agents, num_heldout_agents, num_eval_episodes, num_agents_per_env)
        metric_names = get_metric_names(config["ENV_NAME"])
        ego_names = [f"ego ({label})" for label in ego_idx_labels]
        heldout_names = list(heldout_agents.keys())
        for metric_name in metric_names:
            print_metrics_table(eval_metrics, metric_name, ego_names, heldout_names)
    return eval_metrics

def print_metrics_table(eval_metrics, metric_name, ego_names, heldout_names):
    '''Print a table of the mean and std of the metric for each ego agent and heldout agent.'''
    num_eval_episodes = eval_metrics[metric_name].shape[-2]
    metric_data_mean = eval_metrics[metric_name].mean(axis=(-2, -1)) # shape (num_ego_agents, num_heldout_agents)
    metric_data_std = eval_metrics[metric_name].std(axis=(-2, -1)) # shape (num_ego_agents, num_heldout_agents)
    metric_data_ci = 1.96 * metric_data_std / np.sqrt(num_eval_episodes) # shape (num_ego_agents, num_heldout_agents)

    table = PrettyTable()
    table.field_names = ["---", *heldout_names]
    for i, ego_name in enumerate(ego_names):
        row = [ego_name] + [f"{metric_data_mean[i, j]:.2f} ± {metric_data_ci[i, j]:.2f}" for j in range(len(heldout_names))]
        table.add_row(row)
    print(f"\n{metric_name} (mean ± CI):")
    print(table)

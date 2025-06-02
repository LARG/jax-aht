''''Implementation of heldout evaluation helper functions used by learners.'''
import logging
import time
from functools import partial
import shutil

import jax
import numpy as np
import hydra

from common.save_load_utils import save_train_run
from common.run_episodes import run_episodes
from common.tree_utils import tree_stack
from common.stat_utils import compute_aggregate_stat_and_ci, compute_aggregate_stat_and_ci_per_task, get_aggregate_stat_fn
from envs import make_env
from envs.log_wrapper import LogWrapper
from evaluation.heldout_evaluator import load_heldout_set, normalize_metrics, eval_egos_vs_heldouts as eval_1d_egos_vs_heldouts

log = logging.getLogger(__name__)   
logging.basicConfig(level=logging.INFO)

def eval_2d_egos_vs_heldouts(config, env, rng, num_episodes, ego_policy, ego_params, 
                          heldout_agent_list, ego_test_mode=False):
    '''Evaluate all ego agents against all heldout partners using vmap over egos.
    Ego_params must be a pytree of shape (num_seeds, num_oel_iters, ...)
    '''
    num_agents = env.num_agents
    assert num_agents == 2, "This eval code assumes exactly 2 agents."

    num_ego_seeds, num_ego_iters = jax.tree.leaves(ego_params)[0].shape[:2]
    tot_ego_agents = num_ego_seeds * num_ego_iters
    num_partner_total = len(heldout_agent_list)

    def _eval_ego_vs_one_partner(rng_for_ego, single_ego_params, single_ego_policy, 
                                 heldout_params, heldout_policy, heldout_test_mode):
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
        heldout_policy, heldout_params, heldout_test_mode, heldout_performance_bounds = heldout_agent_list[partner_idx]
        ego_rngs = jax.random.split(partner_rngs[partner_idx], tot_ego_agents)
        ego_rngs = ego_rngs.reshape(num_ego_seeds, num_ego_iters, 2)

        # Use partial to fix the heldout agent for the function being vmapped
        func_to_vmap = partial(_eval_ego_vs_one_partner,
                               single_ego_policy=ego_policy,
                               heldout_params=heldout_params,
                               heldout_policy=heldout_policy,
                               heldout_test_mode=heldout_test_mode)

        # Inner vmap: Maps over the 'num_oel_iters' dimension.
        # Operates on the partially applied function `eval_partial`.
        vmap_over_iters = jax.vmap(
            func_to_vmap,
            in_axes=(0, 0) # Map over axis 0 of single_ego_params and rng_for_ego
        )

        # Outer vmap: Maps the 'vmap_over_iters' function over the 'num_seeds' dimension.
        vmap_over_seeds_and_iters = jax.vmap(
            vmap_over_iters,
            in_axes=(0, 0) # Map over axis 0 of ego_params and ego_rngs
        )
        # Execute the nested vmap 
        results_for_this_partner = vmap_over_seeds_and_iters(
            ego_rngs, # shape (num_seeds, num_oel_iters, 2)
            ego_params # shape (num_seeds, num_oel_iters, ...)
        )
        # results_for_this_partner shape: (num_seeds, num_oel_iters, num_episodes, ...)
        if config["global_heldout_settings"]["NORMALIZE_RETURNS"]:
            if heldout_performance_bounds is not None:
                results_for_this_partner = normalize_metrics(results_for_this_partner, heldout_performance_bounds)
            else:
                print(f"Warning: no performance bounds provided for {heldout_agent_list[partner_idx]}. Skipping normalization.")
        all_metrics_for_partners.append(results_for_this_partner)

    end_time = time.time()
    print(f"Time taken for vmap evaluation loop: {end_time - start_time:.2f} seconds")

    # Result shape: (num_partners, num_seeds, num_oel_iters, num_episodes, ...)
    final_metrics = tree_stack(all_metrics_for_partners)
    # Transpose to (num_seeds, num_oel_iters, num_partners, num_episodes, ...)
    final_metrics = jax.tree.map(lambda x: x.transpose(1, 2, 0, 3, 4), final_metrics)
    return final_metrics

def run_heldout_evaluation(config, ego_policy, ego_params, init_ego_params, 
                           ego_as_2d: bool, ego_test_mode=False):
    '''Run heldout evaluation given an ego policy, ego params, and init_ego_params.
    Ego_params can be a pytree of shape (num_seeds, num_oel_iters, ...) or (num_seeds, ...).
    Args:
        config: Configuration dictionary
        ego_policy: Policy for the ego agent
        ego_params: Parameters for the ego agent
        init_ego_params: Initial parameters for the ego agent
        ego_as_2d: Whether to treat the ego agent params as a 2D or 1D array of ego agents
        ego_test_mode: Whether the ego agent should run in test mode (default: False)
    '''
    log.info("Running heldout evaluation...")
    env = make_env(config["ENV_NAME"], config["ENV_KWARGS"])
    env = LogWrapper(env)
    
    rng = jax.random.PRNGKey(config["global_heldout_settings"]["EVAL_SEED"])
    rng, heldout_init_rng, eval_rng = jax.random.split(rng, 3)

    if ego_as_2d:
        num_seeds, num_oel_iters = jax.tree.leaves(ego_params)[0].shape[:2]
        ego_names = [f"ego (seed={i}, iter={j})" for i in range(num_seeds) for j in range(num_oel_iters)]
    else:
        # flatten ego params
        ego_params = jax.tree.map(lambda x, y: x.reshape((-1,) + y.shape), ego_params, init_ego_params)      
        num_ego_agents = jax.tree.leaves(ego_params)[0].shape[0]
        ego_names = [f"ego ({i})" for i in range(num_ego_agents)]
    
    # load heldout agents
    heldout_cfg = config["heldout_set"][config["TASK_NAME"]]
    heldout_agents = load_heldout_set(heldout_cfg, env, config["TASK_NAME"], config["ENV_KWARGS"], heldout_init_rng)
    heldout_agent_list = list(heldout_agents.values())
    heldout_names = list(heldout_agents.keys())

    # run evaluation
    if ego_as_2d:
        eval_metrics = eval_2d_egos_vs_heldouts(config, env, eval_rng, config["global_heldout_settings"]["NUM_EVAL_EPISODES"], 
                                            ego_policy, ego_params, heldout_agent_list, ego_test_mode)
    else:
        eval_metrics = eval_1d_egos_vs_heldouts(config, env, eval_rng, config["global_heldout_settings"]["NUM_EVAL_EPISODES"], 
                                            ego_policy, ego_params, heldout_agent_list, ego_test_mode)

    return eval_metrics, ego_names, heldout_names

def log_heldout_metrics(config, logger, eval_metrics, 
        ego_names, heldout_names, metric_names: tuple,
        ego_as_2d: bool):
    '''Log heldout evaluation metrics.'''
    if ego_as_2d:
        table_data = heldout_metrics_2d(config, logger, eval_metrics, ego_names, heldout_names, metric_names)
    else:
        table_data = heldout_metrics_1d(config, logger, eval_metrics, ego_names, heldout_names, metric_names)

    # table_data shape (num_metrics, num_heldout_agents)
    # Add metric name column to the table data
    metric_names_array = np.array(metric_names).reshape(-1, 1)  # Convert to column vector
    
    # Add algo name column to the table data
    algo_name = config["algorithm"]["ALG"]
    algo_name_array = np.full_like(metric_names_array, algo_name)
    
    # Log table
    table_data_with_names = np.hstack((algo_name_array, metric_names_array, table_data))
    aggregate_stat = config["global_heldout_settings"]["AGGREGATE_STAT"]
    logger.log_xp_matrix(f"HeldoutEval/FinalEgoVsHeldout-{aggregate_stat.capitalize()}-CI", table_data_with_names, 
                         columns=["Algorithm", "Metric", f"{aggregate_stat.capitalize()} (all)"] + list(heldout_names), commit=True)

    # Saving artifacts
    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    out_savepath = save_train_run(eval_metrics, savedir, savename="heldout_eval_metrics")
    if config["logger"]["log_eval_out"]:
        logger.log_artifact(name="heldout_eval_metrics", path=out_savepath, type_name="eval_metrics")
    
    # Cleanup locally logged out file
    if not config["local_logger"]["save_eval_out"]:
        shutil.rmtree(out_savepath)

def heldout_metrics_1d(config, logger, eval_metrics, 
                       ego_names, heldout_names, metric_names: tuple):
    '''Treat the first dimension of eval_metrics as (num_seeds, ...). 
    Returns the data for a table where the rows are the metrics and the columns are the heldout agents.
    '''
    num_seeds, num_heldout_agents, num_eval_episodes, _ = eval_metrics[metric_names[0]].shape
    table_data = []
    aggregate_stat = config["global_heldout_settings"]["AGGREGATE_STAT"]

    for metric_name in metric_names:
        # shape of eval_metrics[metric_name] is (num_seeds, num_heldout_agents, num_eval_episodes, num_agents_per_game)
        # we first take the mean over the num_agents_per_game dimension
        data = eval_metrics[metric_name].mean(axis=-1
               ).transpose(0, 2, 1
               ).reshape(-1, num_heldout_agents) # final shape (num_seeds*num_eval_episodes, num_heldout_agents)
        data = np.array(data)
        # compute per-heldout-agent aggregate stat+CIs
        point_est_per_task, interval_ests_per_task = compute_aggregate_stat_and_ci_per_task(data, aggregate_stat, return_interval_est=True)
        lower_ci = interval_ests_per_task[:, 0]
        upper_ci = interval_ests_per_task[:, 1]

        col_strs = [f"{point_est_per_task[i]:.3f} ({lower_ci[i]:.3f}, {upper_ci[i]:.3f})" for i in range(len(point_est_per_task))]

        # compute aggregate stat+CI over all heldout agents
        point_est_all, interval_ests_all = compute_aggregate_stat_and_ci(data, aggregate_stat, return_interval_est=True)
        lower_ci = interval_ests_all[0]
        upper_ci = interval_ests_all[1]

        col_strs.insert(0, f"{point_est_all:.3f} ({lower_ci:.3f}, {upper_ci:.3f})")

        table_data.append(col_strs)
    return np.array(table_data)

def heldout_metrics_2d(config, logger, eval_metrics, 
        ego_names, heldout_names, metric_names: tuple):
    '''Treat the first two dimensions of eval_metrics as (seeds, iters, ...) dimensions.
    Logs a curve for each metric over the iters dimension.
    Returns the data for a table where the rows are the metrics and the columns are the heldout agents.
    '''
    num_seeds, num_oel_iter, num_heldout_agents, \
        num_eval_episodes, num_agents_per_game = eval_metrics[metric_names[0]].shape

    table_data = []
    aggregate_stat = config["global_heldout_settings"]["AGGREGATE_STAT"]
    aggregate_stat_fn = get_aggregate_stat_fn(aggregate_stat)
    for metric_name in metric_names:
        # shape of eval_metrics[metric_name] is 
        # (num_seeds, num_oel_iter, num_heldout_agents, num_eval_episodes, num_agents_per_game)
        for i in range(num_oel_iter):
            # we first take the mean over the num_agents_per_game dimension
            data = eval_metrics[metric_name][:, i].mean(axis=-1
                ).transpose(0, 2, 1
                ).reshape(-1, num_heldout_agents) # final shape (num_seeds*num_eval_episodes, num_heldout_agents)
            data = np.array(data)
            point_est = aggregate_stat_fn(data)
            # log curve aggregated over all heldout agents
            logger.log_item(f"HeldoutEval/AvgEgo_{metric_name}_", point_est, iter=i)        

        # now compute per-heldout-agent aggregate stat+CIs corresponding to the LAST ego iter
        last_iter_data = data
        point_est_per_task, interval_ests_per_task = compute_aggregate_stat_and_ci_per_task(last_iter_data, aggregate_stat, return_interval_est=True)
        lower_ci = interval_ests_per_task[:, 0]
        upper_ci = interval_ests_per_task[:, 1]

        col_strs = [f"{point_est_per_task[i]:.3f} ({lower_ci[i]:.3f}, {upper_ci[i]:.3f})" for i in range(len(point_est_per_task))]

        # compute aggregate stat+CI over all heldout agents
        point_est_all, interval_ests_all = compute_aggregate_stat_and_ci(last_iter_data, aggregate_stat, return_interval_est=True)
        lower_ci = interval_ests_all[0]
        upper_ci = interval_ests_all[1]

        col_strs.insert(0, f"{point_est_all:.3f} ({lower_ci:.3f}, {upper_ci:.3f})")
        table_data.append(col_strs)
    return np.array(table_data)

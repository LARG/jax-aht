"""Entry points for heldout evaluation, plus metric logging/reporting.

Two entry points, differing in where the ego agent comes from:
  - run_heldout_evaluation: ego policy/params passed in by a learner.
  - run_heldout_evaluation_from_config: ego agent loaded from config["ego_agent"].
The evaluation itself lives in evaluation/heldout_core.py.
"""

import logging
import shutil

import hydra
import jax
import numpy as np

from common.agent_loader_from_config import initialize_rl_agent_from_config
from common.plot_utils import get_metric_names
from common.save_load_utils import save_train_run
from common.stat_utils import (
    compute_aggregate_stat_and_ci,
    compute_aggregate_stat_and_ci_per_task,
    get_aggregate_stat_fn,
)
from envs import make_env
from envs.log_wrapper import LogWrapper
from evaluation.heldout_core import (
    eval_egos_vs_heldouts,
    load_heldout_set,
    print_metrics_table,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def run_heldout_evaluation(
    config,
    ego_policy,
    ego_params,
    init_ego_params,
    ego_as_2d: bool,
    ego_test_mode=False,
):
    """Run heldout evaluation given an ego policy, ego params, and init_ego_params.
    Ego_params can be a pytree of shape (num_seeds, num_oel_iters, ...) or (num_seeds, ...).
    Args:
        config: Configuration dictionary
        ego_policy: Policy for the ego agent
        ego_params: Parameters for the ego agent
        init_ego_params: Initial parameters for the ego agent
        ego_as_2d: Whether to treat the ego agent params as a 2D or 1D array of ego agents
        ego_test_mode: Whether the ego agent should run in test mode (default: False)
    """
    log.info("Running heldout evaluation...")
    env = make_env(config["ENV_NAME"], config["ENV_KWARGS"])
    env = LogWrapper(env)

    rng = jax.random.PRNGKey(config["global_heldout_settings"]["EVAL_SEED"])
    rng, heldout_init_rng, eval_rng = jax.random.split(rng, 3)

    if ego_as_2d:
        num_seeds, num_oel_iters = jax.tree.leaves(ego_params)[0].shape[:2]
        ego_names = [
            f"ego (seed={i}, iter={j})"
            for i in range(num_seeds)
            for j in range(num_oel_iters)
        ]
    else:
        # flatten ego params
        ego_params = jax.tree.map(
            lambda x, y: x.reshape((-1,) + y.shape), ego_params, init_ego_params
        )
        num_ego_agents = jax.tree.leaves(ego_params)[0].shape[0]
        ego_names = [f"ego ({i})" for i in range(num_ego_agents)]

    # load heldout agents
    heldout_cfg = config["heldout_set"][config["TASK_NAME"]]
    heldout_agents = load_heldout_set(
        heldout_cfg, env, config["TASK_NAME"], config["ENV_KWARGS"], heldout_init_rng
    )
    heldout_agent_list = list(heldout_agents.values())
    heldout_names = list(heldout_agents.keys())

    # run evaluation
    eval_metrics = eval_egos_vs_heldouts(
        config,
        env,
        eval_rng,
        config["global_heldout_settings"]["NUM_EVAL_EPISODES"],
        ego_policy,
        ego_params,
        heldout_agent_list,
        heldout_names,
        ego_test_mode,
        num_ego_axes=2 if ego_as_2d else 1,
    )

    return eval_metrics, ego_names, heldout_names


def log_heldout_metrics(
    config,
    logger,
    eval_metrics,
    ego_names,
    heldout_names,
    metric_names: tuple,
    ego_as_2d: bool,
):
    """Log heldout evaluation metrics."""
    if ego_as_2d:
        table_data = heldout_metrics_2d(
            config, logger, eval_metrics, ego_names, heldout_names, metric_names
        )
    else:
        table_data = heldout_metrics_1d(
            config, logger, eval_metrics, ego_names, heldout_names, metric_names
        )

    # table_data shape (num_metrics, num_heldout_agents)
    # Add metric name column to the table data
    metric_names_array = np.array(metric_names).reshape(
        -1, 1
    )  # Convert to column vector

    # Add algo name column to the table data
    algo_name = config["algorithm"]["ALG"]
    algo_name_array = np.full_like(metric_names_array, algo_name)

    # Log table
    table_data_with_names = np.hstack((algo_name_array, metric_names_array, table_data))

    # Additionally log each metric separately for parameter sweep analysis
    for i in range(table_data_with_names.shape[0]):
        logger.log_item(
            f"HeldoutEval/FinalEgoVsHeldout/{table_data_with_names[i, 1]}/mean",
            float(table_data_with_names[i, 2].split()[0]),
        )

    aggregate_stat = config["global_heldout_settings"]["AGGREGATE_STAT"]
    logger.log_xp_matrix(
        f"HeldoutEval/FinalEgoVsHeldout-{aggregate_stat.capitalize()}-CI",
        table_data_with_names,
        columns=["Algorithm", "Metric", f"{aggregate_stat.capitalize()} (all)"]
        + list(heldout_names),
        commit=True,
    )

    # Saving artifacts
    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    out_savepath = save_train_run(
        eval_metrics, savedir, savename="heldout_eval_metrics"
    )
    if config["logger"]["log_eval_out"]:
        logger.log_artifact(
            name="heldout_eval_metrics", path=out_savepath, type_name="eval_metrics"
        )

    # Cleanup locally logged out file
    if not config["local_logger"]["save_eval_out"]:
        shutil.rmtree(out_savepath)


def heldout_metrics_1d(
    config, logger, eval_metrics, ego_names, heldout_names, metric_names: tuple
):
    """Treat the first dimension of eval_metrics as (num_seeds, ...).
    Returns the data for a table where the rows are the metrics and the columns are the heldout agents.
    """
    table_data = []
    aggregate_stat = config["global_heldout_settings"]["AGGREGATE_STAT"]

    for metric_name in metric_names:
        # shape of eval_metrics[metric_name] is (num_seeds, num_heldout_agents, num_eval_episodes, num_agents_per_game)
        # average out the episode and agents-per-game dims so that seeds are the
        # bootstrap replication unit and heldout agents are the tasks
        data = eval_metrics[metric_name].mean(
            axis=(-1, -2)
        )  # final shape (num_seeds, num_heldout_agents)
        data = np.array(data)
        # compute per-heldout-agent aggregate stat+CIs
        point_est_per_task, interval_ests_per_task = (
            compute_aggregate_stat_and_ci_per_task(
                data, aggregate_stat, return_interval_est=True
            )
        )
        lower_ci = interval_ests_per_task[:, 0]
        upper_ci = interval_ests_per_task[:, 1]

        col_strs = [
            f"{point_est_per_task[i]:.3f} ({lower_ci[i]:.3f}, {upper_ci[i]:.3f})"
            for i in range(len(point_est_per_task))
        ]

        # compute aggregate stat+CI over all heldout agents
        point_est_all, interval_ests_all = compute_aggregate_stat_and_ci(
            data, aggregate_stat, return_interval_est=True
        )
        lower_ci = interval_ests_all[0]
        upper_ci = interval_ests_all[1]

        col_strs.insert(0, f"{point_est_all:.3f} ({lower_ci:.3f}, {upper_ci:.3f})")

        table_data.append(col_strs)
    return np.array(table_data)


def heldout_metrics_2d(
    config, logger, eval_metrics, ego_names, heldout_names, metric_names: tuple
):
    """Treat the first two dimensions of eval_metrics as (seeds, iters, ...) dimensions.
    Logs a curve for each metric over the iters dimension.
    Returns the data for a table where the rows are the metrics and the columns are the heldout agents.
    """
    num_oel_iter = eval_metrics[metric_names[0]].shape[1]

    table_data = []
    aggregate_stat = config["global_heldout_settings"]["AGGREGATE_STAT"]
    aggregate_stat_fn = get_aggregate_stat_fn(aggregate_stat)
    for metric_name in metric_names:
        # shape of eval_metrics[metric_name] is
        # (num_seeds, num_oel_iter, num_heldout_agents, num_eval_episodes, num_agents_per_game)
        for i in range(num_oel_iter):
            # average out the episode and agents-per-game dims so that seeds are the
            # bootstrap replication unit and heldout agents are the tasks
            data = eval_metrics[metric_name][:, i].mean(
                axis=(-1, -2)
            )  # final shape (num_seeds, num_heldout_agents)
            data = np.array(data)
            point_est = aggregate_stat_fn(data)
            # log curve aggregated over all heldout agents
            logger.log_item(f"HeldoutEval/AvgEgo_{metric_name}_", point_est, iter=i)

        # now compute per-heldout-agent aggregate stat+CIs corresponding to the LAST ego iter
        last_iter_data = data
        point_est_per_task, interval_ests_per_task = (
            compute_aggregate_stat_and_ci_per_task(
                last_iter_data, aggregate_stat, return_interval_est=True
            )
        )
        lower_ci = interval_ests_per_task[:, 0]
        upper_ci = interval_ests_per_task[:, 1]

        col_strs = [
            f"{point_est_per_task[i]:.3f} ({lower_ci[i]:.3f}, {upper_ci[i]:.3f})"
            for i in range(len(point_est_per_task))
        ]

        # compute aggregate stat+CI over all heldout agents
        point_est_all, interval_ests_all = compute_aggregate_stat_and_ci(
            last_iter_data, aggregate_stat, return_interval_est=True
        )
        lower_ci = interval_ests_all[0]
        upper_ci = interval_ests_all[1]

        col_strs.insert(0, f"{point_est_all:.3f} ({lower_ci:.3f}, {upper_ci:.3f})")
        table_data.append(col_strs)
    return np.array(table_data)


def run_heldout_evaluation_from_config(config, print_metrics=False):
    """Run heldout evaluation, loading the ego agent from config["ego_agent"]."""
    # Create only one environment instance
    env = make_env(config["ENV_NAME"], config["ENV_KWARGS"])
    env = LogWrapper(env)

    rng = jax.random.PRNGKey(config["global_heldout_settings"]["EVAL_SEED"])
    rng, ego_init_rng, heldout_init_rng, eval_rng = jax.random.split(rng, 4)

    # load ego agents
    ego_agent_config = dict(config["ego_agent"])
    ego_test_mode = ego_agent_config.get("test_mode", False)
    ego_policy, ego_params, init_ego_params, ego_idx_labels = (
        initialize_rl_agent_from_config(ego_agent_config, "ego", env, ego_init_rng)
    )
    # flatten ego params and idx labels
    ego_idx_labels = np.array(ego_idx_labels).reshape(
        -1
    )  # flatten the list of ego agent labels
    flattened_ego_params = jax.tree.map(
        lambda x, y: x.reshape((-1,) + y.shape), ego_params, init_ego_params
    )

    # load heldout agents
    heldout_cfg = config["heldout_set"][config["TASK_NAME"]]
    heldout_agents = load_heldout_set(
        heldout_cfg, env, config["TASK_NAME"], config["ENV_KWARGS"], heldout_init_rng
    )
    heldout_agent_names = list(heldout_agents.keys())
    heldout_agent_list = list(heldout_agents.values())

    # run evaluation
    eval_metrics = eval_egos_vs_heldouts(
        config,
        env,
        eval_rng,
        config["global_heldout_settings"]["NUM_EVAL_EPISODES"],
        ego_policy,
        flattened_ego_params,
        heldout_agent_list,
        heldout_agent_names,
        ego_test_mode,
    )

    if print_metrics:
        # each leaf of eval_metrics has shape (num_ego_agents, num_heldout_agents, num_eval_episodes, num_agents_per_env)
        metric_names = get_metric_names(config["ENV_NAME"])
        aggregate_stat = config["global_heldout_settings"]["AGGREGATE_STAT"]
        ego_names = [f"ego ({label})" for label in ego_idx_labels]
        heldout_names = list(heldout_agents.keys())
        for metric_name in metric_names:
            print_metrics_table(
                eval_metrics,
                metric_name,
                ego_names,
                heldout_names,
                aggregate_stat,
                config["global_heldout_settings"]["NORMALIZE_RETURNS"],
            )
    return eval_metrics

import shutil
import time
import logging

import jax
import numpy as np
import hydra

from envs import make_env
from envs.log_wrapper import LogWrapper

from ego_agent_training.ppo_ego import train_ppo_ego_agent
from ego_agent_training.utils import initialize_ego_agent
from ego_agent_training.ppo_monitoring import PPOMonitor
from common.plot_utils import get_metric_names, get_stats
from common.save_load_utils import save_train_run

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def train_ego_agent(config, logger, partner_params, partner_population):
    '''
    Train PPO ego agent against a population of partner agents.
    Args:
        config: dict, config for the training
        partner_params: partner parameters pytree with shape (num_seeds, pop_size, ...)
        partner_population: partner population object
    '''
    algorithm_config = config["algorithm"]["ego_train_algorithm"]
    env = make_env(algorithm_config["ENV_NAME"], algorithm_config["ENV_KWARGS"])
    env = LogWrapper(env)

    num_seeds = jax.tree.leaves(partner_params)[0].shape[0]

    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"])
    rng, init_rng = jax.random.split(rng, 2)
    train_rngs = jax.random.split(rng, num_seeds)
    
    
    log.info("Starting ego agent training...")
    start_time = time.time()
    
    # Initialize monitoring if enabled
    monitor = None
    if config.get("enable_ppo_monitoring", False):
        monitoring_dir = config.get("ppo_monitoring_dir", "./ppo_monitoring")
        monitor = PPOMonitor(output_dir=monitoring_dir)
        monitor.start()
    
    def train_ego_fn(rng, partner_params):
        rng, init_rng, train_rng = jax.random.split(rng, 3)
        ego_policy, init_ego_params = initialize_ego_agent(algorithm_config, env, init_rng)

        return train_ppo_ego_agent(
            config=algorithm_config,
            env=env,
            train_rng=train_rng,
            ego_policy=ego_policy,
            init_ego_params=init_ego_params,
            n_ego_train_seeds=algorithm_config["NUM_EGO_TRAIN_SEEDS"], # PER provided partner params
            partner_population=partner_population,
            partner_params=partner_params
        )

    # Run the training
    vmapped_train_fn = jax.jit(jax.vmap(train_ego_fn, in_axes=(0, 0)))
    out = vmapped_train_fn(train_rngs, partner_params)
    
    end_time = time.time()
    total_training_time = end_time - start_time
    log.info(f"Ego agent training completed in {total_training_time:.2f} seconds")
    
    # Prepare ego params and policy for heldout evaluation
    num_seeds, num_ego_train_seeds = jax.tree.leaves(out["final_params"])[0].shape[:2]
    ego_params = jax.tree.map(lambda x: x.reshape(num_seeds*num_ego_train_seeds, *x.shape[2:]), 
                              out["final_params"])
    ego_policy, init_ego_params = initialize_ego_agent(algorithm_config, env, init_rng)

    # Log metrics
    metric_names = get_metric_names(algorithm_config["ENV_NAME"])
    log_ego_metrics(config, out, logger, metric_names, monitor=monitor, total_training_time=total_training_time)

    return ego_params, ego_policy, init_ego_params

def log_ego_metrics(config, out, logger, metric_names: tuple, monitor=None, total_training_time=None):
    '''Log metrics for the ego agent returned by the above train_ego_agent function.
    '''
    train_metrics = out["metrics"]

    # each leaf of out["metrics"] has shape (num_seeds, num_ego_train_seeds, num_updates, ...)
    # we combine the first two dimensions together to get a single seeds dimension, 
    num_seeds, num_ego_train_seeds = train_metrics["returned_episode_returns"].shape[:2]
    train_metrics = jax.tree.map(lambda x: x.reshape(num_seeds * num_ego_train_seeds, *x.shape[2:]), 
                                 train_metrics)

    #### Extract train metrics ####
    train_stats = get_stats(train_metrics, metric_names)
    # each key in train_stats is a metric name, and the value is an array of shape (num_seeds, num_updates, 2)
    # where the last dimension contains the mean and std of the metric
    train_stats = {k: np.mean(np.array(v), axis=0) for k, v in train_stats.items()}
    
    all_ego_value_losses = np.asarray(train_metrics["value_loss"]) # shape (num_seeds, num_updates, num_partners, num_minibatches)
    all_ego_actor_losses = np.asarray(train_metrics["actor_loss"]) # shape (num_seeds, num_updates, num_partners, num_minibatches)
    all_ego_entropy_losses = np.asarray(train_metrics["entropy_loss"]) # shape (num_seeds, num_updates, num_partners, num_minibatches)

    # Process eval return metrics - average across ego seeds, eval episodes,  training partners 
    # and num_agents per game for each checkpoint
    all_ego_returns = np.asarray(train_metrics["eval_ep_last_info"]["returned_episode_returns"]) # shape (num_seeds, num_updates, num_partners, num_eval_episodes, nuM_agents_per_game)
    average_ego_rets_per_iter = np.mean(all_ego_returns, axis=(0, 2, 3, 4))

    # Process loss metrics - average across ego seeds, partners and minibatches dims
    # Loss metrics shape should be (num_seeds, num_updates, ...)
    average_ego_value_losses = np.mean(all_ego_value_losses, axis=(0, 2, 3))
    average_ego_actor_losses = np.mean(all_ego_actor_losses, axis=(0, 2, 3))
    average_ego_entropy_losses = np.mean(all_ego_entropy_losses, axis=(0, 2, 3))
    
    # Log metrics for each update step
    num_updates = len(average_ego_value_losses)
    for step in range(num_updates):
        for stat_name, stat_data in train_stats.items():
            # second dimension contains the mean and std of the metric
            stat_mean = stat_data[step, 0]
            logger.log_item(f"Train/Ego_{stat_name}", stat_mean, train_step=step, commit=True)

        logger.log_item("Eval/EgoReturn", average_ego_rets_per_iter[step], train_step=step, commit=True)
        logger.log_item("Train/EgoValueLoss", average_ego_value_losses[step], train_step=step, commit=True)
        logger.log_item("Train/EgoActorLoss", average_ego_actor_losses[step], train_step=step, commit=True)
        logger.log_item("Train/EgoEntropyLoss", average_ego_entropy_losses[step], train_step=step, commit=True)
        
        logger.commit()
        
        # Record in monitor if available
        if monitor is not None:
            # Reconstruct timing: assume updates were roughly evenly spaced during training
            if total_training_time is not None:
                estimated_elapsed_time = (step / num_updates) * total_training_time
            else:
                estimated_elapsed_time = 0
            
            monitor.record_update_with_time(
                update_step=step,
                ego_return=average_ego_rets_per_iter[step],
                value_loss=average_ego_value_losses[step],
                actor_loss=average_ego_actor_losses[step],
                entropy_loss=average_ego_entropy_losses[step],
                elapsed_time=estimated_elapsed_time
            )
    
    # Save and plot monitoring data if monitor is available
    if monitor is not None:
        monitor.save_data()
        monitor.plot_results()
    
    # Saving artifacts
    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    # TODO: in the future, add video logging feature
    out_savepath = save_train_run(out, savedir, savename="ego_train_run")
    if config["logger"]["log_train_out"]:
        logger.log_artifact(name="ego_train_run", path=out_savepath, type_name="train_run")
        # Cleanup locally logged out file
    if not config["local_logger"]["save_train_out"]:
        shutil.rmtree(out_savepath)
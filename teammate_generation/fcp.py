import shutil
import time
import logging
from functools import partial

import jax
import hydra
import numpy as np
from agents.agent_interface import MLPActorCriticPolicy
from agents.population_interface import AgentPopulation
from envs import make_env
from envs.log_wrapper import LogWrapper
from marl.ippo import make_train as make_ppo_train
from common.plot_utils import get_metric_names, get_stats
from common.save_load_utils import save_train_run

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_fcp_population(config, out, env):
    '''
    For each seeed, flatten the partner pool for for ego training.
    '''
    num_seeds = config["algorithm"]["NUM_SEEDS"]
    fcp_pop_size = config["algorithm"]["PARTNER_POP_SIZE"] * config["algorithm"]["NUM_CHECKPOINTS"]

    partner_params = out['checkpoints'] # shape is (num_seeds, partner_pop_size, num_ckpts, ...)
    flattened_partner_params = jax.tree.map(lambda x: x.reshape(num_seeds, fcp_pop_size, *x.shape[3:]), partner_params)
    
    partner_policy = MLPActorCriticPolicy(
        action_dim=env.action_space(env.agents[1]).n,
        obs_dim=env.observation_space(env.agents[1]).shape[0],
        activation=config["algorithm"].get("ACTIVATION", "tanh")
    )

    # Create partner population
    partner_population = AgentPopulation( 
        pop_size=fcp_pop_size,
        policy_cls=partner_policy
    )

    return flattened_partner_params, partner_population

def train_fcp_partners(rng, env, algorithm_config):
    '''Single seed of training an FCP pool.'''
    rngs = jax.random.split(rng, algorithm_config["PARTNER_POP_SIZE"])
    train_jit = jax.jit(jax.vmap(make_ppo_train(algorithm_config, env)))
    out = train_jit(rngs)
    return out

def run_fcp(config, wandb_logger):
    '''
    Train a pool of partners for FCP. Return checkpoints for all partners.
    Returns out, a dictionary of the final train_state, metrics, and checkpoints.
    '''
    algorithm_config = config["algorithm"]
    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"])
    rngs = jax.random.split(rng, algorithm_config["NUM_SEEDS"])

    env = make_env(algorithm_config["ENV_NAME"], algorithm_config["ENV_KWARGS"])
    env = LogWrapper(env)

    start_time = time.time()
    with jax.disable_jit(False):
        vmapped_train_fn = jax.jit(
            jax.vmap(
            partial(train_fcp_partners, env=env, algorithm_config=algorithm_config)
            )
        )
        out = vmapped_train_fn(rngs)
    end_time = time.time()
    log.info(f"Training FCP partners took {end_time - start_time:.2f} seconds.")
    
    flattened_partner_params, partner_population = get_fcp_population(config, out, env)
    
    # log metrics
    log_metrics(config, out, wandb_logger)
    
    return flattened_partner_params, partner_population

def log_metrics(config, out, logger):
    '''Log statistics, save train run output and log to wandb as artifact.'''   
    metric_names = get_metric_names(config["ENV_NAME"])
    # metrics is a pytree where each leaf has shape 
    # (num_seeds, partner_pop_size, num_partner_updates, rollout_length, agents_per_env * num_envs)
    partner_metrics = out["metrics"]
    num_partner_updates = partner_metrics["returned_episode_returns"].shape[2]
    # Extract partner train stats
    num_controlled_actors = config["algorithm"]["NUM_ENVS"]
    partner_metrics = jax.tree.map(lambda x: x[..., :num_controlled_actors], partner_metrics)
    partner_stats = get_stats(partner_metrics, metric_names) # shape (num_seeds, partner_pop_size, num_partner_updates, 2)
    partner_stat_means = jax.tree.map(lambda x: np.mean(x, axis=(0, 1))[..., 0], partner_stats) # shape (num_partner_updates)

    for step in range(num_partner_updates):
        for stat_name, stat_data in partner_stat_means.items():
            logger.log_item(f"Train/Partner_{stat_name}", stat_data[step], train_step=step)

    logger.commit()

    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    # save artifacts
    out_savepath = save_train_run(out, savedir, savename="saved_train_run")
    if config["logger"]["log_train_out"]:
        logger.log_artifact(name="saved_train_run", path=out_savepath, type_name="train_run")
        # Cleanup locally logged out file
    if not config["local_logger"]["save_train_out"]:
        shutil.rmtree(out_savepath)

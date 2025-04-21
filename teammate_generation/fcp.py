import os
import shutil
import time
import logging
import hydra

import jax

from agents.agent_interface import AgentPopulation, MLPActorCriticPolicy
from envs import make_env
from envs.log_wrapper import LogWrapper
from ppo.ippo import make_train
from common.plot_utils import get_metric_names, get_stats, plot_train_metrics
from common.save_load_utils import save_train_run

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_fcp_population(config, out, env):
    '''
    Get the flattened partner params and partner population for ego training.
    '''
    fcp_pop_size = config["algorithm"]["PARTNER_POP_SIZE"] * config["algorithm"]["NUM_CHECKPOINTS"]

    partner_params = out['checkpoints'] # shape is (num_seeds, num_ckpts, ...)
    flattened_partner_params = jax.tree.map(lambda x: x.reshape(fcp_pop_size, *x.shape[2:]), partner_params)
    
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

def log_metrics(config, out, logger):
    '''Save train run output and log to wandb as artifact.'''
    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    # save artifacts
    out_savepath = save_train_run(out, savedir, savename="saved_train_run")
    if config["logger"]["log_train_out"]:
        logger.log_artifact(name="saved_train_run", path=out_savepath, type_name="train_run")
        # Cleanup locally logged out file
    if not config["local_logger"]["save_train_out"]:
        shutil.rmtree(out_savepath)
   
    metric_names = get_metric_names(config["ENV_NAME"])

    # Generate plots
    all_stats = get_stats(out["metrics"], metric_names)
    figures, _ = plot_train_metrics(all_stats, 
                                    config["algorithm"]["ROLLOUT_LENGTH"], 
                                    config["algorithm"]["NUM_ENVS"],
                                    savedir=savedir if config["local_logger"]["save_figures"] else None,
                                    savename="ippo_train_metrics",
                                    show_plots=False
                                    )
    
    # Log plots to wandb
    for stat_name, fig in figures.items():
        logger.log({f"train_metrics/{stat_name}": fig})


def train_fcp_partners(config, wandb_logger):
    '''
    Train a pool of partners for FCP. Return checkpoints for all partners.
    Returns out, a dictionary of the final train_state, metrics, and checkpoints.
    '''
    algorithm_config = config["algorithm"]
    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"])
    rngs = jax.random.split(rng, algorithm_config["PARTNER_POP_SIZE"])

    env = make_env(algorithm_config["ENV_NAME"], algorithm_config["ENV_KWARGS"])
    env = LogWrapper(env)

    start_time = time.time()
    debug_mode = False
    with jax.disable_jit(debug_mode):
        if debug_mode: 
            out = make_train(config)(rngs)
        else:
            train_jit = jax.jit(jax.vmap(make_train(algorithm_config, env)))
            out = train_jit(rngs)
    end_time = time.time()
    log.info(f"Training partners took {end_time - start_time:.2f} seconds.")
    
    flattened_partner_params, partner_population = get_fcp_population(config, out, env)
    
    # log metrics
    log_metrics(config, out, wandb_logger)
    
    return flattened_partner_params, partner_population
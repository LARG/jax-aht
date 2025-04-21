import time
import logging

import jax

from envs import make_env
from envs.log_wrapper import LogWrapper

from agents.initialize_agents import initialize_s5_agent, initialize_mlp_agent, initialize_rnn_agent
from ego_agent_training.ppo_ego import train_ppo_ego_agent

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def train_ego_agent(config, logger, partner_params, partner_population):
    '''
    Train PPO ego agent against a population of partner agents.
    Args:
        config: dict, config for the training
        partner_params: partner parameters pytree with shape (pop_size, ...)
        partner_population: partner population object
    '''
    # Create only one environment instance
    env = make_env(config["ENV_NAME"], config["ENV_KWARGS"])
    env = LogWrapper(env)

    rng = jax.random.PRNGKey(config["TRAIN_SEED"])
    rng, init_rng, train_rng = jax.random.split(rng, 3)
    
    # Initialize ego agent
    if config["EGO_ACTOR_TYPE"] == "s5":
        ego_policy, init_ego_params = initialize_s5_agent(config, env, init_rng)
    elif config["EGO_ACTOR_TYPE"] == "mlp":
        ego_policy, init_ego_params = initialize_mlp_agent(config, env, init_rng)
    elif config["EGO_ACTOR_TYPE"] == "rnn":
        # WARNING: currently the RNN policy is not working. 
        # TODO: fix this!
        ego_policy, init_ego_params = initialize_rnn_agent(config, env, init_rng)
    
    log.info("Starting ego agent training...")
    start_time = time.time()
    
    # Run the training
    out = train_ppo_ego_agent(
        config=config,
        env=env,
        train_rng=train_rng,
        ego_policy=ego_policy,
        init_ego_params=init_ego_params,
        n_ego_train_seeds=config["NUM_EGO_TRAIN_SEEDS"],
        partner_population=partner_population,
        partner_params=partner_params
    )
    
    log.info(f"Ego agent training completed in {time.time() - start_time:.2f} seconds")
    return out, ego_policy, init_ego_params

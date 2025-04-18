import time
import logging

import jax

from envs import make_env
from envs.log_wrapper import LogWrapper

from common.agent_interface import MLPActorCriticPolicy, AgentPopulation
from common.plot_utils import get_metric_names
from common.initialize_agents import initialize_s5_agent, initialize_mlp_agent, initialize_rnn_agent
from ego_agent_training.ppo_ego import train_ppo_ego_agent, log_metrics

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def train_fcp_ego_agent(config, partner_params, logger):
    '''
    Train FCP agent against a population of MLP partner agents.
    Args:
        config: dict, config for the training
        partner_params: partner parameters pytree with shape (n_seeds, m_ckpts, ...)
        pop_size: int, number of partner agents in the population
    '''
    # Create only one environment instance
    env = make_env(config["ENV_NAME"], config["ENV_KWARGS"])
    env = LogWrapper(env)

    rng = jax.random.PRNGKey(config["SEED"])
    rng, init_rng, train_rng = jax.random.split(rng, 3)

    # prepare partners 
    pop_size = config["PARTNER_POP_SIZE"] * config["NUM_CHECKPOINTS"]
    
    # Flatten partner parameters for AgentPopulation
    flattened_partner_params = jax.tree.map(
        lambda x: x.reshape((pop_size,) + x.shape[2:]), 
        partner_params
    )

    partner_policy = MLPActorCriticPolicy(
        action_dim=env.action_space(env.agents[1]).n,
        obs_dim=env.observation_space(env.agents[1]).shape[0],
    )

    # Create partner population
    partner_population = AgentPopulation(
        pop_size=pop_size,
        policy_cls=partner_policy
    )
    
    # Initialize ego agent
    if config["EGO_ACTOR_TYPE"] == "s5":
        ego_policy, init_params = initialize_s5_agent(config, env, init_rng)
    elif config["EGO_ACTOR_TYPE"] == "mlp":
        ego_policy, init_params = initialize_mlp_agent(config, env, init_rng)
    elif config["EGO_ACTOR_TYPE"] == "rnn":
        # WARNING: currently the RNN policy is not working. 
        # TODO: fix this!
        ego_policy, init_params = initialize_rnn_agent(config, env, init_rng)
    
    log.info("Starting ego agent training...")
    start_time = time.time()
    
    # Run the training
    out = train_ppo_ego_agent(
        config=config,
        env=env,
        train_rng=train_rng,
        ego_policy=ego_policy,
        init_ego_params=init_params,
        n_ego_train_seeds=config["NUM_EGO_TRAIN_SEEDS"],
        partner_population=partner_population,
        partner_params=flattened_partner_params
    )
    
    log.info(f"Training completed in {time.time() - start_time:.2f} seconds")
    
    # process and log metrics
    metric_names = get_metric_names(config["ENV_NAME"])
    log_metrics(config, out, logger, metric_names)
    return out    

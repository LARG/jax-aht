import jax
from common.agent_interface import S5ActorCriticPolicy, \
    MLPActorCriticPolicy, RNNActorCriticPolicy


def initialize_s5_agent(config, env, rng):
    """Initialize an S5 agent with the given config.
    
    Args:
        config: dict, config for the agent
        env: gymnasium environment
        rng: jax.random.PRNGKey, random key for initialization
        
    Returns:
        policy: S5ActorCriticPolicy, the policy object
        params: dict, initial parameters for the agent
    """
    # Create the S5 policy with direct parameters
    policy = S5ActorCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=env.observation_space(env.agents[0]).shape[0],
        d_model=config["S5_D_MODEL"],
        ssm_size=config["S5_SSM_SIZE"],
        n_layers=config["S5_N_LAYERS"],
        blocks=config["S5_BLOCKS"],
        fc_hidden_dim=config["S5_ACTOR_CRITIC_HIDDEN_DIM"],
        s5_activation=config["S5_ACTIVATION"],
        s5_do_norm=config["S5_DO_NORM"],
        s5_prenorm=config["S5_PRENORM"],
        s5_do_gtrxl_norm=config["S5_DO_GTRXL_NORM"],
    )
    
    rng, init_rng = jax.random.split(rng)
    init_params = policy.init_params(init_rng)

    return policy, init_params

def initialize_rnn_agent(config, env, rng):
    """Initialize an RNN agent with the given config.
    
    Args:
        config: dict, config for the agent
        env: gymnasium environment
        rng: jax.random.PRNGKey, random key for initialization
        
    Returns:
        policy: RNNActorCriticPolicy, the policy object
        params: dict, initial parameters for the agent
    """
    # Create the RNN policy
    policy = RNNActorCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=env.observation_space(env.agents[0]).shape[0],
    )
    
    rng, init_rng = jax.random.split(rng)
    init_params = policy.init_params(init_rng)

    return policy, init_params

def initialize_mlp_agent(config, env, rng):
    """
    Initialize an MLP agent with the given config.
    """
    policy = MLPActorCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=env.observation_space(env.agents[0]).shape[0],
    ) 
    rng, init_rng = jax.random.split(rng)
    init_params = policy.init_params(init_rng)

    return policy, init_params
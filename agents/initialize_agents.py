import jax
from agents.agent_interface import S5ActorCriticPolicy, \
    MLPActorCriticPolicy, RNNActorCriticPolicy, ActorWithDoubleCriticPolicy, \
    ActorWithConditionalCriticPolicy


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
        d_model=config.get("S5_D_MODEL", 16),
        ssm_size=config.get("S5_SSM_SIZE", 16),
        n_layers=config.get("S5_N_LAYERS", 2),
        blocks=config.get("S5_BLOCKS", 1),
        fc_hidden_dim=config.get("S5_ACTOR_CRITIC_HIDDEN_DIM", 64),
        s5_activation=config.get("S5_ACTIVATION", "full_glu"),
        s5_do_norm=config.get("S5_DO_NORM", True),
        s5_prenorm=config.get("S5_PRENORM", True),
        s5_do_gtrxl_norm=config.get("S5_DO_GTRXL_NORM", True),
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
        activation=config.get("ACTIVATION", "tanh"),
        fc_hidden_dim=config.get("FC_HIDDEN_DIM", 64),
        gru_hidden_dim=config.get("GRU_HIDDEN_DIM", 64),
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
        activation=config.get("ACTIVATION", "tanh"),
    ) 
    rng, init_rng = jax.random.split(rng)
    init_params = policy.init_params(init_rng)

    return policy, init_params

def initialize_actor_with_double_critic(config, env, rng):
    """Initialize an actor with double critic with the given config."""
    policy = ActorWithDoubleCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=env.observation_space(env.agents[0]).shape[0],
        activation=config.get("ACTIVATION", "tanh"),
    )
    rng, init_rng = jax.random.split(rng)
    init_params = policy.init_params(init_rng)

    return policy, init_params

def initialize_actor_with_conditional_critic(config, env, rng):
    """Initialize an actor with conditional critic with the given config."""
    policy = ActorWithConditionalCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=env.observation_space(env.agents[0]).shape[0],
        pop_size=config["POP_SIZE"],
        activation=config.get("ACTIVATION", "tanh"),
    )
    rng, init_rng = jax.random.split(rng)
    init_params = policy.init_params(init_rng)

    return policy, init_params


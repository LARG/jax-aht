import jax

from agents.dqn_actor_crtic_fqe_agent import DQNActorCriticFQEPolicy
from agents.mlp_actor_critic_agent import MLPActorCriticPolicy
from agents.rnn_actor_critic_agent import RNNActorCriticPolicy
from agents.s5_actor_critic_agent import S5ActorCriticPolicy

def initialize_mlp_agent(config, env, rng, agent_index, observation_type="agent"):
    """
    Initialize an MLP agent with the given config.
    """
    policy = MLPActorCriticPolicy(
        action_dim=env.action_space(env.agents[agent_index]).n,
        obs_dim=env.observation_space(env.agents[agent_index], observation_type=observation_type).shape[0],
        activation=config.get("ACTIVATION", "tanh"),
    )
    rng, init_rng = jax.random.split(rng)
    init_params = policy.init_params(init_rng)

    return policy, init_params

def initialize_rnn_agent(config, env, rng, agent_index, observation_type="agent"):
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
        action_dim=env.action_space(env.agents[agent_index]).n,
        obs_dim=env.observation_space(env.agents[agent_index], observation_type=observation_type).shape[0],
        activation=config.get("ACTIVATION", "tanh"),
        fc_hidden_dim=config.get("FC_HIDDEN_DIM", 64),
        gru_hidden_dim=config.get("GRU_HIDDEN_DIM", 64),
    )

    rng, init_rng = jax.random.split(rng)
    init_params = policy.init_params(init_rng)

    return policy, init_params

def initialize_s5_agent(config, env, rng, agent_index, observation_type="agent"):
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
        action_dim=env.action_space(env.agents[agent_index]).n,
        obs_dim=env.observation_space(env.agents[agent_index], observation_type=observation_type).shape[0],
        d_model=config.get("S5_D_MODEL", 128),
        ssm_size=config.get("S5_SSM_SIZE", 128),
        # d_model=config.get("S5_D_MODEL", 16),
        # ssm_size=config.get("S5_SSM_SIZE", 16),
        ssm_n_layers=config.get("S5_N_LAYERS", 2),
        blocks=config.get("S5_BLOCKS", 1),
        fc_hidden_dim=config.get("S5_ACTOR_CRITIC_HIDDEN_DIM", 1024),
        fc_n_layers=config.get("FC_N_LAYERS", 3),
        # fc_hidden_dim=config.get("S5_ACTOR_CRITIC_HIDDEN_DIM", 64),
        # fc_n_layers=config.get("FC_N_LAYERS", 2),
        s5_activation=config.get("S5_ACTIVATION", "full_glu"),
        s5_do_norm=config.get("S5_DO_NORM", True),
        s5_prenorm=config.get("S5_PRENORM", True),
        s5_do_gtrxl_norm=config.get("S5_DO_GTRXL_NORM", True),
    )

    rng, init_rng = jax.random.split(rng)
    init_params = policy.init_params(init_rng)

    return policy, init_params

def initialize_agent(algorithm_config, env, init_rng, agent_index, observation_type="agent"):
    if algorithm_config["ACTOR_TYPE"] == "mlp":
        policy, init_params = initialize_mlp_agent(algorithm_config, env, init_rng, agent_index, observation_type=observation_type)
    elif algorithm_config["ACTOR_TYPE"] == "rnn":
        policy, init_params = initialize_rnn_agent(algorithm_config, env, init_rng, agent_index, observation_type=observation_type)
    elif algorithm_config["ACTOR_TYPE"] == "s5":
        policy, init_params = initialize_s5_agent(algorithm_config, env, init_rng, agent_index, observation_type=observation_type)
    else:
        raise ValueError(f"Unknown ACTOR_TYPE: {algorithm_config['ACTOR_TYPE']}")
    return policy, init_params

def initialize_dqn_actor_critic_fqe_agent(config, env, rng, actor_critic_policy, agent_index, observation_type="agent"):
    """Initialize the DQN actor-critic FQE agent with the given config.

    Args:
        config: dict, config for the agent
        env: gymnasium environment
        rng: jax.random.PRNGKey, random key for initialization
        actor_critic_policy: the actor-critic policy to be used by the DQN actor-critic FQE policy
    Returns:
        policy: DQNActorCriticFQEPolicy, the policy object
        params: dict, initial parameters for the policy
    """

    policy = DQNActorCriticFQEPolicy(
        action_dim=env.action_space(env.agents[agent_index]).n,
        obs_dim=env.observation_space(env.agents[agent_index], observation_type=observation_type).shape[0],
        actor_critic_policy=actor_critic_policy,
        epsilon_start=config["EPSILON_START"],
        epsilon_finish=config["EPSILON_END"],
        epsilon_anneal_time=config["EPSILON_ANNEAL_TIME"],
    )
    rng, init_rng = jax.random.split(rng)
    init_params = policy.init_params(init_rng)

    return policy, init_params

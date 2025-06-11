import jax
from agents.agent_interface import S5ActorCriticPolicy, \
    MLPActorCriticPolicy, RNNActorCriticPolicy, ActorWithDoubleCriticPolicy, \
    ActorWithConditionalCriticPolicy, PseudoActorWithDoubleCriticPolicy, \
    PseudoActorWithConditionalCriticPolicy, LIAMPolicy

from agents.liam_encoder_decoder import initialize_encoder_decoder

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

def initialize_pseudo_actor_with_double_critic(config, env, rng):
    """Initialize a pseudo actor with double critic with the given config."""
    policy = PseudoActorWithDoubleCriticPolicy(
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

def initialize_pseudo_actor_with_conditional_critic(config, env, rng):
    """Initialize a pseudo actor with conditional critic with the given config."""
    policy = PseudoActorWithConditionalCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=env.observation_space(env.agents[0]).shape[0],
        pop_size=config["POP_SIZE"],
        activation=config.get("ACTIVATION", "tanh"),
    )
    rng, init_rng = jax.random.split(rng)
    init_params = policy.init_params(init_rng)

    return policy, init_params

def initialize_liam_agent(config, env, rng):
    """Initialize the LIAM ego agent with the given config.
    
    Args:
        config: dict, config for the agent
        env: gymnasium environment
        rng: jax.random.PRNGKey, random key for initialization
        
    Returns:
        liam: LIAMPolicy, the policy object
        params: tuple, initial parameters for the {encoder, decoder} and policy
    """
    rng, init_encoder_decoder_rng, init_policy_rng = jax.random.split(rng, 3)

    # Initialize the policy based on the specified type
    if config["EGO_ACTOR_TYPE"] == "s5":
        ego_policy, init_ego_params = initialize_s5_agent(config, env, init_policy_rng)
    elif config["EGO_ACTOR_TYPE"] == "mlp":
        ego_policy, init_ego_params = initialize_mlp_agent(config, env, init_policy_rng)
    elif config["EGO_ACTOR_TYPE"] == "rnn":
        ego_policy, init_ego_params = initialize_rnn_agent(config, env, init_policy_rng)

    # Initialize the encoder and decoder for LIAM
    encoder, decoder, init_encoder_decoder_params = initialize_encoder_decoder(config, env, init_encoder_decoder_rng)

    liam = LIAMPolicy(
        policy=ego_policy,
        encoder=encoder,
        decoder=decoder
    )
    params = {'encoder': init_encoder_decoder_params['encoder'], 
              'decoder': init_encoder_decoder_params['decoder'],
              'policy': init_ego_params}
    return liam, params

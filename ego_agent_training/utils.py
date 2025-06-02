from agents.initialize_agents import initialize_s5_agent, initialize_mlp_agent, initialize_rnn_agent


def initialize_ego_agent(algorithm_config, env, init_rng):
    if algorithm_config["EGO_ACTOR_TYPE"] == "s5":
        ego_policy, init_ego_params = initialize_s5_agent(algorithm_config, env, init_rng)
    elif algorithm_config["EGO_ACTOR_TYPE"] == "mlp":
        ego_policy, init_ego_params = initialize_mlp_agent(algorithm_config, env, init_rng)
    elif algorithm_config["EGO_ACTOR_TYPE"] == "rnn":
        ego_policy, init_ego_params = initialize_rnn_agent(algorithm_config, env, init_rng)
    return ego_policy, init_ego_params

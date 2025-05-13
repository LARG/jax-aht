# TODO: collect trajectories
# @ Johnny

from typing import List
import jax

from common.run_episodes import run_episodes

def collect_n_trajectories(config, agent_1, agent_2, n_episodes):
    # use jax implementation from ippo?
    return None

# TODO: collect trajectories
# @ to-be-assigned

def get_events_from_trajectory(trajectory) -> jax.Array:
    return None

def aggregate_events(data: List[jax.Array]):
    return None


# sample usage
if __name__ == "__main__":

    print("sample trajectory collection for visualization with overcooked")

    # load the sample config
    from omegaconf import OmegaConf
    from hydra import initialize, compose
    from omegaconf import OmegaConf

    with initialize(version_base=None, config_path="../configs"):
        config = compose(config_name="heldout.yaml")

    config = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

    heldout_cfg = config["heldout_set"][config["TASK_NAME"]]

    from evaluation.heldout_evaluator import load_heldout_set
    from envs import make_env
    from envs.log_wrapper import LogWrapper

    rng = jax.random.PRNGKey(config["global_heldout_settings"]["EVAL_SEED"])
    heldout_init_rng, eval_rng = jax.random.split(rng, 2)

    env = make_env(config["ENV_NAME"], config["ENV_KWARGS"])
    env = LogWrapper(env)

    heldout_agents = load_heldout_set(heldout_cfg, env, config["TASK_NAME"], config["ENV_KWARGS"], heldout_init_rng)
    heldout_agent_list = list(heldout_agents.values())

    # just collect 4 trajectories with the first two held_out_agents

    print(len(heldout_agent_list))

    policy1, params1, test_mode1 = heldout_agent_list[0]
    policy2, params2, test_mode2 = heldout_agent_list[1]

    print("running episodes")

    x = run_episodes(eval_rng, env,
        agent_0_policy=policy1, agent_0_param=params1,
        agent_1_policy=policy2, agent_1_param=params2,
        max_episode_steps=config["global_heldout_settings"]["MAX_EPISODE_STEPS"],
        num_eps=4, 
        agent_0_test_mode=test_mode1,
        agent_1_test_mode=test_mode2)

    print(x)

    print("done running episode")

    exit()

    
    
    # # config = 

    # #

    

    # heldout_cfg = config["heldout_set"][config["TASK_NAME"]]

    # print("done")

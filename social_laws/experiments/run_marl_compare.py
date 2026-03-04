'''Main entry point for running MARL algorithms.'''
import hydra
import jax
import random
from omegaconf import OmegaConf, open_dict

import os
import numpy as np

from common.wandb_visualizations import Logger
from social_laws.ippo import run_ippo
from social_laws.common.run_episodes_ippo import run_episodes_vmap

SEEDRANGE = (1, int(1e9))

@hydra.main(version_base=None, config_path="../configs", config_name="base_config_marl")
def ippo(config):

    if config.algorithm.USE_SAME_SEED:
        if config.algorithm.TRAIN_SEED is None:
            config.algorithm.TRAIN_SEED = random.randint(*SEEDRANGE)
            config.algorithm.EVAL_SEED = config.algorithm.TRAIN_SEED
        else:
            config.algorithm.EVAL_SEED = config.algorithm.TRAIN_SEED

    else:
        if config.algorithm.TRAIN_SEED is None:
            config.algorithm.TRAIN_SEED = random.randint(*SEEDRANGE)
        if config.algorithm.EVAL_SEED is None:
            config.algorithm.EVAL_SEED = random.randint(*SEEDRANGE)

    if config.task.get("ENV_KWARGS", {}).get("single_task", False):
        with open_dict(config):
            config.task.ENV_KWARGS.single_task_seed = config.algorithm.TRAIN_SEED

    print(OmegaConf.to_yaml(config, resolve=True))
    wandb_logger = Logger(config)

    if config.algorithm["ALG"] == "ippo":
        out, env, policy = run_ippo(config, wandb_logger)
    else:
        raise NotImplementedError(f"Algorithm {config['ALG']} not implemented.")

    eval_rng = jax.random.PRNGKey(config.algorithm.EVAL_SEED)
    eval_out = run_episodes_vmap(eval_rng, env,
                                agent_param=jax.tree.map(lambda x: x.squeeze(axis=0), out['final_params']),
                                agent_policy=policy,
                                max_episode_steps=env.horizon,
                                num_eps=100,
                                render=True)

    all_returns = np.asarray(eval_out[2]["returned_episode_returns"])
    for num_episode in range(all_returns.shape[0]):
        wandb_logger.log_item(f"Eval/EpisodeReturns", all_returns[num_episode, :].sum(), train_step=num_episode, commit=True)
        for i, agent in enumerate(env.agents):
            wandb_logger.log_item(f"Eval/EpisodeReturns/{agent}", all_returns[num_episode, i], train_step=num_episode, commit=True)

    # shape of render_outs should be (num_train_seeds, num_eps, max_episode_steps, ...)
    # eval_out[0] = stacked carry (9-tuple); eval_out[1] = init_env_state (LogEnvState)
    # animate() expects (seed, episode, ...) shapes, so we add a seed dim via [None]
    eval_render_init_env_state = jax.tree.map(lambda x: x[None], eval_out[1].env_state.env_state)  # (1, num_eps, ...)
    eval_render_env_state = jax.tree.map(lambda x: x[None], eval_out[0][-1]['pre_reset_state'].env_state)  # (1, num_eps, max_steps, ...)
    eval_render_dones = np.array(eval_out[0][4]['__all__'])[None]  # (1, num_eps, max_steps)
    env.animate((eval_render_init_env_state, eval_render_env_state), eval_render_dones, 5, debug=True)

    for eval_ep in range(5):
        wandb_logger.log_video(
            tag=f"Videos/Eval/Episode_{eval_ep}",
            path=os.path.join(env._render_dir, f"{env._render_name}_ep_{eval_ep}.gif")
        )

    wandb_logger.close()

    return out

if __name__ == "__main__":
    out = ippo()


# PYTHONPATH=/home/rolando/GitHub/SOCIAL_LAWS_JAHT JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false python social_laws/experiments/run_marl_compare.py task=rddl/grid_10x10_alternating/toroidal_no_restrictions_single_task algorithm=ippo/rddl/grid_10x10_alternating/toroidal_no_restrictions algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true
'''Main entry point for running agent training algorithms for social laws.'''
import os
# Restrict JAX from taking all GPU memory
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import hydra
import random
from omegaconf import OmegaConf

from common.plot_utils import get_metric_names
from common.wandb_visualizations import Logger
from social_laws.ppo_single_agent_projection import run_training as run_ppo_training
from social_laws.dqn_ppo_value_function_estimation import run_training as run_dqnppo_value_estimation
# from social_laws.drqn_ppo_value_function_estimation import run_training as run_drqnppo_value_estimation
from social_laws.ppo_joint import run_training as run_ppo_joint_training

SEEDRANGE = (1, int(1e9))

@hydra.main(version_base=None, config_path="configs", config_name="base_config")
def run_training(cfg):
    '''Runs the agent training.'''

    if cfg.algorithm.TRAIN_SEED is None:
        cfg.algorithm.TRAIN_SEED = random.randint(*SEEDRANGE)

    if cfg.algorithm.EVAL_SEED is None:
        cfg.algorithm.EVAL_SEED = random.randint(*SEEDRANGE)

    if cfg.value_function.TRAIN_SEED is None:
        cfg.value_function.TRAIN_SEED = random.randint(*SEEDRANGE)

    if cfg.value_function.EVAL_SEED is None:
        cfg.value_function.EVAL_SEED = random.randint(*SEEDRANGE)

    print(OmegaConf.to_yaml(cfg, resolve=True))
    wandb_logger = Logger(cfg)

    # Single agent projection training
    # Creates what is effectively the optimal policy for a single agent in the environment
    if cfg["algorithm"]["ALG"] == "ppo":
        if cfg["value_function"]["ALG"] == "dqnppo":
            assert cfg["algorithm"]["ACTOR_TYPE"] == "mlp", "For DQN PPO value estimation, the PPO policy actor type must be MLP."
            assert cfg["value_function"]["NETWORK_TYPE"] == "mlp", "For DQN PPO value estimation, the DQN network type must be MLP."
        elif cfg["value_function"]["ALG"] == "drqnppo":
            assert cfg["algorithm"]["ACTOR_TYPE"] in ["s5", "rnn"], "For DRQN PPO value estimation, the PPO policy actor type must be s5 or rnn."
            assert cfg["value_function"]["NETWORK_TYPE"] in ["s5", "rnn"], "For DRQN PPO value estimation, the DRQN network type must be s5 or rnn."

        agent_0_params, agent_0_policy, agent_0_init_params = run_ppo_training(cfg, wandb_logger, agent_idx=0)
        agent_1_params, agent_1_policy, agent_1_init_params = run_ppo_training(cfg, wandb_logger, agent_idx=1)

        # Value function estimation for joint policies
        # Creates value functions for joint policies for all agents in the environment
        # conditioned on their single agent projections
        if cfg["value_function"]["ALG"] == "dqnppo":
            agent_0_vf_params, agent_0_vf, agent_0_vf_init_params = run_dqnppo_value_estimation(cfg, wandb_logger, agent_0_params, agent_0_policy, agent_idx=0)
            agent_1_vf_params, agent_1_vf, agent_1_vf_init_params = run_dqnppo_value_estimation(cfg, wandb_logger, agent_1_params, agent_1_policy, agent_idx=1)
        # elif cfg["value_function"]["ALG"] == "drqnppo":
        #     agent_0_vf_params, agent_0_vf, agent_0_vf_init_params = run_drqnppo_value_estimation(cfg, wandb_logger, agent_0_params, agent_0_policy, agent_idx=0)
        #     agent_1_vf_params, agent_1_vf, agent_1_vf_init_params = run_drqnppo_value_estimation(cfg, wandb_logger, agent_1_params, agent_1_policy, agent_idx=1)

    # Joint multi-agent training
    # Creates polices for joint policies for all agents in the environment
    # conditioned on their single agent projections
    if cfg["algorithm"]["ALG"] == "ppo":
        joint_0_params, joint_0_policies, joint_0_init_params = run_ppo_joint_training(cfg, wandb_logger,
                                                                                       (agent_0_params, agent_1_params),
                                                                                       (agent_0_policy, agent_1_policy),
                                                                                       (agent_0_vf_params, agent_1_vf_params),
                                                                                       (agent_0_vf, agent_1_vf),
                                                                                       agent_idx=0)
        joint_1_params, joint_1_policies, joint_1_init_params = run_ppo_joint_training(cfg, wandb_logger,
                                                                                       (agent_0_params, agent_1_params),
                                                                                       (agent_0_policy, agent_1_policy),
                                                                                       (agent_0_vf_params, agent_1_vf_params),
                                                                                       (agent_0_vf, agent_1_vf),
                                                                                       agent_idx=1)

    # Cleanup
    wandb_logger.close()


if __name__ == '__main__':
    run_training()

# PYTHONPATH=/home/rolando/GitHub/SOCIAL_LAWS_JAHT XLA_PYTHON_CLIENT_PREALLOCATE=false python social_laws/run.py task=rddl/grid_4x4 algorithm=ppo/rddl/grid_4x4 value_function=dqnppo/rddl/grid_4x4

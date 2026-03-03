'''Main entry point for running agent training algorithms for social laws.'''
import os
# Restrict JAX from taking all GPU memory
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import hydra
import random
from omegaconf import OmegaConf

from common.plot_utils import get_metric_names
from common.wandb_visualizations import Logger
from social_laws.dqn_single_agent_projection import run_training as run_dqn_training
from social_laws.drqn_single_agent_projection import run_training as run_drqn_training
from social_laws.ppo_joint_from_dqn import run_training as run_ppo_joint_training
from social_laws.ppo_joint_from_dqn_centralized import run_training as run_ppo_joint_centralized_training

SEEDRANGE = (1, int(1e9))

@hydra.main(version_base=None, config_path="configs", config_name="base_config")
def run_training(cfg):
    '''Runs the agent training.'''

    if cfg.algorithm.USE_SAME_SEED:
        if cfg.algorithm.TRAIN_SEED is None:
            cfg.algorithm.TRAIN_SEED = random.randint(*SEEDRANGE)
            cfg.value_function.TRAIN_SEED = cfg.algorithm.TRAIN_SEED
            cfg.algorithm.EVAL_SEED = cfg.algorithm.TRAIN_SEED
            cfg.value_function.EVAL_SEED = cfg.algorithm.TRAIN_SEED
        else:
            cfg.value_function.TRAIN_SEED = cfg.algorithm.TRAIN_SEED
            cfg.algorithm.EVAL_SEED = cfg.algorithm.TRAIN_SEED
            cfg.value_function.EVAL_SEED = cfg.algorithm.TRAIN_SEED
    else:
        if cfg.algorithm.TRAIN_SEED is None:
            cfg.algorithm.TRAIN_SEED = random.randint(*SEEDRANGE)
            cfg.value_function.TRAIN_SEED = cfg.algorithm.TRAIN_SEED

        if cfg.algorithm.EVAL_SEED is None:
            cfg.algorithm.EVAL_SEED = random.randint(*SEEDRANGE)
            cfg.value_function.EVAL_SEED = cfg.algorithm.EVAL_SEED

    # if cfg.value_function.TRAIN_SEED is None:
    #     cfg.value_function.TRAIN_SEED = random.randint(*SEEDRANGE)

    # if cfg.value_function.EVAL_SEED is None:
    #     cfg.value_function.EVAL_SEED = random.randint(*SEEDRANGE)

    print(OmegaConf.to_yaml(cfg, resolve=True))
    wandb_logger = Logger(cfg)

    # Single agent projection training
    # Creates what is effectively the optimal policy for a single agent in the environment
    if cfg["value_function"]["ALG"] == "dqn":
        assert cfg["algorithm"]["ACTOR_TYPE"] == "mlp", "For DQN single agent projection, the Joint PPO policy actor type must be MLP."

        agent_0_params, agent_0_policy, agent_0_init_params = run_dqn_training(cfg, wandb_logger, agent_idx=0)
        agent_1_params, agent_1_policy, agent_1_init_params = run_dqn_training(cfg, wandb_logger, agent_idx=1)

    elif cfg["value_function"]["ALG"] == "drqn":
        assert cfg["algorithm"]["ACTOR_TYPE"] in ["s5", "rnn"], "For DRQN single agent projection, the Joint PPO policy actor type must be s5 or rnn."

        agent_0_params, agent_0_policy, agent_0_init_params = run_drqn_training(cfg, wandb_logger, agent_idx=0)
        agent_1_params, agent_1_policy, agent_1_init_params = run_drqn_training(cfg, wandb_logger, agent_idx=1)

    # Joint multi-agent training
    # Creates polices for joint policies for all agents in the environment
    # conditioned on their single agent projections
    if cfg["algorithm"]["ALG"] == "ppo":
        if cfg["algorithm"]["JOINT_CENTRALIZED"]:
            joint_0_params, joint_0_policy, joint_0_init_params = run_ppo_joint_centralized_training(cfg, wandb_logger,
                                                                                        (agent_0_params, agent_1_params),
                                                                                        (agent_0_policy, agent_1_policy),
                                                                                        agent_idx=0)
            joint_1_params, joint_1_policy, joint_1_init_params = run_ppo_joint_centralized_training(cfg, wandb_logger,
                                                                                        (agent_0_params, agent_1_params),
                                                                                        (agent_0_policy, agent_1_policy),
                                                                                        agent_idx=1)
        else:
            joint_0_params, joint_0_policies, joint_0_init_params = run_ppo_joint_training(cfg, wandb_logger,
                                                                                        (agent_0_params, agent_1_params),
                                                                                        (agent_0_policy, agent_1_policy),
                                                                                        agent_idx=0)
            joint_1_params, joint_1_policies, joint_1_init_params = run_ppo_joint_training(cfg, wandb_logger,
                                                                                        (agent_0_params, agent_1_params),
                                                                                        (agent_0_policy, agent_1_policy),
                                                                                        agent_idx=1)

    # Cleanup
    wandb_logger.close()


if __name__ == '__main__':
    run_training()

# PYTHONPATH=/home/rolando/GitHub/SOCIAL_LAWS_JAHT XLA_PYTHON_CLIENT_PREALLOCATE=false python social_laws/run_dqn.py task=rddl/grid_4x4 algorithm=ppo/rddl/grid_4x4 value_function=dqn/rddl/grid_4x4 algorithm.FIXED_EVAL=true value_function.FIXED_EVAL=true
# PYTHONPATH=/home/rolando/GitHub/SOCIAL_LAWS_JAHT XLA_PYTHON_CLIENT_PREALLOCATE=false python social_laws/run_dqn.py task=rddl/grid_4x4 algorithm=ppo/rddl/grid_4x4 value_function=drqn/rddl/grid_4x4 algorithm.FIXED_EVAL=true value_function.FIXED_EVAL=true algorithm.ACTOR_TYPE=s5

# PYTHONPATH=/home/rolando/GitHub/SOCIAL_LAWS_JAHT XLA_PYTHON_CLIENT_PREALLOCATE=false python social_laws/run_dqn.py task=rddl/grid_4x4_alternating/toroidal_full_restrictions algorithm=ppo/rddl/grid_4x4_alternating/toroidal_full_restrictions value_function=dqn/rddl/grid_4x4_alternating/toroidal_full_restrictions algorithm.FIXED_EVAL=true value_function.FIXED_EVAL=true
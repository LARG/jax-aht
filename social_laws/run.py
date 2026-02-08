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
# from ppo_joint import run_training as run_ppo_joint_training

SEEDRANGE = (1, int(1e9))

@hydra.main(version_base=None, config_path="configs", config_name="base_config")
def run_training(cfg):
    '''Runs the agent training.'''

    if cfg.algorithm.TRAIN_SEED is None:
        cfg.algorithm.TRAIN_SEED = random.randint(*SEEDRANGE)

    if cfg.algorithm.EVAL_SEED is None:
        cfg.algorithm.EVAL_SEED = random.randint(*SEEDRANGE)

    print(OmegaConf.to_yaml(cfg, resolve=True))
    wandb_logger = Logger(cfg)

    # Single agent projection training
    # Creates what is effectively the optimal policy for a single agent in the environment
    if cfg["algorithm"]["ALG"] == "ppo":
        params, policy, init_params = run_ppo_training(cfg, wandb_logger)

    # Joint multi-agent training
    # Creates polices for joint policies for all agents in the environment
    # conditioned on their single agent projections
    # if cfg["algorithm"]["ALG"] == "ppo":
    #     params, policy, init_params = run_ppo_joint_training(cfg, wandb_logger, params, policy, init_params)

    # Cleanup
    wandb_logger.close()


if __name__ == '__main__':
    run_training()

# PYTHONPATH=/home/rolando/GitHub/SOCIAL_LAWS_JAHT XLA_PYTHON_CLIENT_PREALLOCATE=false python social_laws/run.py task=rddl/grid_4x4 algorithm=ppo/rddl/grid_4x4
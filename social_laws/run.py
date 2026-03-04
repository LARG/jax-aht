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
from social_laws.drqn_ppo_value_function_estimation import run_training as run_drqnppo_value_estimation
from social_laws.ppo_joint import run_training as run_ppo_joint_training
from social_laws.ppo_joint_centralized import run_training as run_ppo_joint_centralized_training
from social_laws.reppo_single_agent_projection import run_training as run_reppo_training
from social_laws.reppo_joint import run_training as run_reppo_joint_training
from social_laws.creppo_single_agent_projection import run_training as run_creppo_training
from social_laws.creppo_joint import run_training as run_creppo_joint_training

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

    agent_policies = []
    agent_init_params = []
    agent_params = []
    agent_vf_policies = []
    agent_vf_init_params = []
    agent_vf_params = []
    joint_policies = []
    joint_init_params = []
    joint_params = []
    # Single agent projection training
    # Creates what is effectively the optimal policy for a single agent in the environment
    if cfg["algorithm"]["ALG"] == "ppo":
        if cfg["value_function"]["ALG"] == "dqnppo":
            assert cfg["algorithm"]["ACTOR_TYPE"] == "mlp", "For DQN PPO value estimation, the PPO policy actor type must be MLP."
            assert cfg["value_function"]["NETWORK_TYPE"] == "mlp", "For DQN PPO value estimation, the DQN network type must be MLP."
        elif cfg["value_function"]["ALG"] == "drqnppo":
            assert cfg["algorithm"]["ACTOR_TYPE"] in ["s5", "rnn"], "For DRQN PPO value estimation, the PPO policy actor type must be s5 or rnn."
            assert cfg["value_function"]["NETWORK_TYPE"] in ["s5", "rnn"], "For DRQN PPO value estimation, the DRQN network type must be s5 or rnn."

        for agent_idx in range(cfg.NUM_EXPT_AGENTS):
            agent_param, agent_policy, agent_init_param = run_ppo_training(cfg, wandb_logger, agent_idx=agent_idx)
            agent_policies.append(agent_policy)
            agent_init_params.append(agent_init_param)
            agent_params.append(agent_param)

        # Value function estimation for joint policies
        # Creates value functions for joint policies for all agents in the environment
        # conditioned on their single agent projections
        if cfg["value_function"]["ALG"] == "dqnppo":
            for agent_idx in range(cfg.NUM_EXPT_AGENTS):
                agent_vf_param, agent_vf, agent_vf_init_param = run_dqnppo_value_estimation(cfg, wandb_logger, agent_params[agent_idx], agent_policies[agent_idx], agent_idx=agent_idx)
                agent_vf_params.append(agent_vf_param)
                agent_vf_policies.append(agent_vf)
                agent_vf_init_params.append(agent_vf_init_param)
        elif cfg["value_function"]["ALG"] == "drqnppo":
            for agent_idx in range(cfg.NUM_EXPT_AGENTS):
                agent_vf_param, agent_vf, agent_vf_init_param = run_drqnppo_value_estimation(cfg, wandb_logger, agent_params[agent_idx], agent_policies[agent_idx], agent_idx=agent_idx)
                agent_vf_params.append(agent_vf_param)
                agent_vf_policies.append(agent_vf)
                agent_vf_init_params.append(agent_vf_init_param)

    elif cfg["algorithm"]["ALG"] == "reppo":
        for agent_idx in range(cfg.NUM_EXPT_AGENTS):
            agent_param, agent_policy, agent_init_param = run_reppo_training(cfg, wandb_logger, agent_idx=agent_idx)
            agent_policies.append(agent_policy)
            agent_init_params.append(agent_init_param)
            agent_params.append(agent_param)

    elif cfg["algorithm"]["ALG"] == "creppo":
        for agent_idx in range(cfg.NUM_EXPT_AGENTS):
            agent_param, agent_policy, agent_init_param = run_creppo_training(cfg, wandb_logger, agent_idx=agent_idx)
            agent_policies.append(agent_policy)
            agent_init_params.append(agent_init_param)
            agent_params.append(agent_param)

    # Joint multi-agent training
    # Creates polices for joint policies for all agents in the environment
    # conditioned on their single agent projections
    if cfg["algorithm"]["ALG"] == "ppo":
        if cfg["algorithm"]["JOINT_CENTRALIZED"]:
            for agent_idx in range(cfg.NUM_EXPT_AGENTS):
                joint_param, joint_policy, joint_init_param = run_ppo_joint_centralized_training(cfg, wandb_logger,
                                                                                            agent_params,
                                                                                            agent_policies,
                                                                                            agent_vf_params,
                                                                                            agent_vf_policies,
                                                                                            agent_idx=agent_idx)
                joint_policies.append(joint_policy)
                joint_init_params.append(joint_init_param)
                joint_params.append(joint_param)
        else:
            for agent_idx in range(cfg.NUM_EXPT_AGENTS):
                joint_param, joint_policy, joint_init_param = run_ppo_joint_training(cfg, wandb_logger,
                                                                                    agent_params,
                                                                                    agent_policies,
                                                                                    agent_vf_params,
                                                                                    agent_vf_policies,
                                                                                    agent_idx=agent_idx)
                joint_policies.append(joint_policy)
                joint_init_params.append(joint_init_param)
                joint_params.append(joint_param)

    elif cfg["algorithm"]["ALG"] == "reppo":
        if cfg["algorithm"]["JOINT_CENTRALIZED"]:
            pass
        else:
            for agent_idx in range(cfg.NUM_EXPT_AGENTS):
                joint_param, joint_policy, joint_init_param = run_reppo_joint_training(cfg, wandb_logger,
                                                                                      agent_params,
                                                                                      agent_policies,
                                                                                      agent_idx=agent_idx)
                joint_policies.append(joint_policy)
                joint_init_params.append(joint_init_param)
                joint_params.append(joint_param)

    elif cfg["algorithm"]["ALG"] == "creppo":
        if cfg["algorithm"]["JOINT_CENTRALIZED"]:
            pass
        else:
            for agent_idx in range(cfg.NUM_EXPT_AGENTS):
                joint_param, joint_policy, joint_init_param = run_creppo_joint_training(cfg, wandb_logger,
                                                                                      agent_params,
                                                                                      agent_policies,
                                                                                      agent_idx=agent_idx)
                joint_policies.append(joint_policy)
                joint_init_params.append(joint_init_param)
                joint_params.append(joint_param)

    # Cleanup
    wandb_logger.close()


if __name__ == '__main__':
    run_training()

# PYTHONPATH=/home/rolando/GitHub/SOCIAL_LAWS_JAHT XLA_PYTHON_CLIENT_PREALLOCATE=false python social_laws/run.py task=rddl/grid_4x4 algorithm=ppo/rddl/grid_4x4 value_function=dqnppo/rddl/grid_4x4
# PYTHONPATH=/home/rolando/GitHub/SOCIAL_LAWS_JAHT JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false python social_laws/run.py task=rddl/grid_10x10 algorithm=ppo/rddl/grid_10x10 value_function=dqnppo/rddl/grid_10x10 algorithm.ACTOR_TYPE=s5

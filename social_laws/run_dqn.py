'''Main entry point for running agent training algorithms for social laws.'''
import os
# Restrict JAX from taking all GPU memory
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import hydra
import random
from omegaconf import OmegaConf, open_dict

from common.plot_utils import get_metric_names
from common.wandb_visualizations import Logger
from social_laws.dqn_single_agent_projection import run_training as run_dqn_training
from social_laws.drqn_single_agent_projection import run_training as run_drqn_training
from social_laws.ppo_joint_from_dqn import run_training as run_ppo_joint_training
from social_laws.ppo_joint_from_dqn_centralized import run_training as run_ppo_joint_centralized_training

from envs import make_env
from envs.log_wrapper import LogWrapper
from social_laws.common.run_single_agent_joint_eval import run_single_agent_joint_eval

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

    if cfg.task.get("ENV_KWARGS", {}).get("single_task", False):
        with open_dict(cfg):
            cfg.task.ENV_KWARGS.single_task_seed = cfg.algorithm.TRAIN_SEED

    print(OmegaConf.to_yaml(cfg, resolve=True))
    wandb_logger = Logger(cfg)

    agent_policies = []
    agent_init_params = []
    agent_params = []
    agent_eval_checkpoints = []
    joint_policies = []
    joint_init_params = []
    joint_params = []
    # Single agent projection training
    # Creates what is effectively the optimal policy for a single agent in the environment
    if cfg["value_function"]["ALG"] == "dqn":
        assert cfg["algorithm"]["ACTOR_TYPE"] == "mlp", "For DQN single agent projection, the Joint PPO policy actor type must be MLP."

        for agent_idx in range(cfg.NUM_EXPT_AGENTS):
            agent_param, agent_policy, agent_init_param, eval_checkpoints = run_dqn_training(cfg, wandb_logger, agent_idx=agent_idx)
            agent_policies.append(agent_policy)
            agent_init_params.append(agent_init_param)
            agent_params.append(agent_param)
            agent_eval_checkpoints.append(eval_checkpoints)

        env_kwargs = dict(cfg["algorithm"]["ENV_KWARGS"])
        env_kwargs["render_dir"] = os.path.join("render", "dqn", "Joint_Eval")
        env_kwargs["done_condition"] = "all"  # Joint eval: terminate as soon as all agents take their picture
        env = make_env(cfg["algorithm"]["ENV_NAME"], env_kwargs)
        env = LogWrapper(env)

        run_single_agent_joint_eval(wandb_logger, cfg.algorithm.EVAL_SEED, env, 
                                    agent_eval_checkpoints, agent_policies, env.horizon, 
                                    cfg.algorithm.NUM_EVAL_EPISODES, cfg.algorithm.FIXED_EVAL, 
                                    render=True, agent_test_mode=True)

    elif cfg["value_function"]["ALG"] == "drqn":
        assert cfg["algorithm"]["ACTOR_TYPE"] in ["s5", "rnn"], "For DRQN single agent projection, the Joint PPO policy actor type must be s5 or rnn."

        for agent_idx in range(cfg.NUM_EXPT_AGENTS):
            agent_param, agent_policy, agent_init_param, eval_checkpoints = run_drqn_training(cfg, wandb_logger, agent_idx=agent_idx)
            agent_policies.append(agent_policy)
            agent_init_params.append(agent_init_param)
            agent_params.append(agent_param)
            agent_eval_checkpoints.append(eval_checkpoints)

        env_kwargs = dict(cfg["algorithm"]["ENV_KWARGS"])
        env_kwargs["render_dir"] = os.path.join("render", "drqn", "Joint_Eval")
        env_kwargs["done_condition"] = "all"  # Joint eval: terminate as soon as all agents take their picture
        env = make_env(cfg["algorithm"]["ENV_NAME"], env_kwargs)
        env = LogWrapper(env)

        run_single_agent_joint_eval(wandb_logger, cfg.algorithm.EVAL_SEED, env, 
                                    agent_eval_checkpoints, agent_policies, env.horizon, 
                                    cfg.algorithm.NUM_EVAL_EPISODES, cfg.algorithm.FIXED_EVAL, 
                                    render=True, agent_test_mode=True)

    # Joint multi-agent training
    # Creates polices for joint policies for all agents in the environment
    # conditioned on their single agent projections
    if cfg["algorithm"]["ALG"] == "ppo":
        if cfg["algorithm"]["JOINT_CENTRALIZED"]:
            for agent_idx in range(cfg.NUM_EXPT_AGENTS):
                joint_param, joint_policy, joint_init_param = run_ppo_joint_centralized_training(cfg, wandb_logger,
                                                                                            agent_params,
                                                                                            agent_policies,
                                                                                            agent_idx=agent_idx)
                joint_policies.append(joint_policy)
                joint_init_params.append(joint_init_param)
                joint_params.append(joint_param)

        else:
            for agent_idx in range(cfg.NUM_EXPT_AGENTS):
                joint_param, joint_policy, joint_init_param = run_ppo_joint_training(cfg, wandb_logger,
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

# PYTHONPATH=/home/rolando/GitHub/SOCIAL_LAWS_JAHT XLA_PYTHON_CLIENT_PREALLOCATE=false python social_laws/run_dqn.py task=rddl/grid_4x4 algorithm=ppo/rddl/grid_4x4 value_function=dqn/rddl/grid_4x4 algorithm.FIXED_EVAL=true value_function.FIXED_EVAL=true
# PYTHONPATH=/home/rolando/GitHub/SOCIAL_LAWS_JAHT XLA_PYTHON_CLIENT_PREALLOCATE=false python social_laws/run_dqn.py task=rddl/grid_4x4 algorithm=ppo/rddl/grid_4x4 value_function=drqn/rddl/grid_4x4 algorithm.FIXED_EVAL=true value_function.FIXED_EVAL=true algorithm.ACTOR_TYPE=s5

# PYTHONPATH=/home/rolando/GitHub/SOCIAL_LAWS_JAHT XLA_PYTHON_CLIENT_PREALLOCATE=false python social_laws/run_dqn.py task=rddl/grid_4x4_alternating/toroidal_full_restrictions algorithm=ppo/rddl/grid_4x4_alternating/toroidal_full_restrictions value_function=dqn/rddl/grid_4x4_alternating/toroidal_full_restrictions algorithm.FIXED_EVAL=true value_function.FIXED_EVAL=true
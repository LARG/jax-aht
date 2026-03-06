'''Main entry point for running MARL algorithms.'''
import hydra
import jax
import random
from omegaconf import OmegaConf, open_dict

import os
import numpy as np

from common.wandb_visualizations import Logger
from social_laws.ippo import run_training as run_ippo

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
        params, policy, init_params = run_ippo(config, wandb_logger)
    else:
        raise NotImplementedError(f"Algorithm {config['ALG']} not implemented.")

    wandb_logger.close()

if __name__ == "__main__":
    ippo()

# label="marl_comparison" logger.project=RLC-2026

# PYTHONPATH=/home/rolando/GitHub/SOCIAL_LAWS_JAHT JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false python social_laws/experiments/run_marl_compare.py task=rddl/grid_10x10_alternating/toroidal_no_restrictions_single_task algorithm=ippo/rddl/grid_10x10_alternating/toroidal_no_restrictions algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true

# Single task

# IPPO
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false python social_laws/experiments/run_marl_compare.py task=rddl/grid_10x10_alternating/toroidal_no_restrictions_single_task algorithm=ippo/rddl/grid_10x10_alternating/toroidal_no_restrictions algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true label="marl_comparison" logger.project=RLC-2026
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false python social_laws/experiments/run_marl_compare.py task=rddl/grid_10x10_alternating/toroidal_no_restrictions_3_agents_single_task algorithm=ippo/rddl/grid_10x10_alternating/toroidal_no_restrictions algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true label="marl_comparison" logger.project=RLC-2026
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false python social_laws/experiments/run_marl_compare.py task=rddl/grid_10x10_alternating/toroidal_no_restrictions_4_agents_single_task algorithm=ippo/rddl/grid_10x10_alternating/toroidal_no_restrictions algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true label="marl_comparison" logger.project=RLC-2026

# PPO
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false python social_laws/run.py task=rddl/grid_10x10_alternating/toroidal_no_restrictions_single_task algorithm=ppo/rddl/grid_10x10_alternating/toroidal_full_restrictions value_function=dqnppo/rddl/grid_10x10_alternating/toroidal_full_restrictions algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true value_function.USE_SAME_SEED=true algorithm.FIXED_EVAL=true value_function.FIXED_EVAL=true label="marl_comparison" logger.project=RLC-2026
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false python social_laws/run.py task=rddl/grid_10x10_alternating/toroidal_no_restrictions_3_agents_single_task algorithm=ppo/rddl/grid_10x10_alternating/toroidal_full_restrictions value_function=dqnppo/rddl/grid_10x10_alternating/toroidal_full_restrictions algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true value_function.USE_SAME_SEED=true algorithm.FIXED_EVAL=true value_function.FIXED_EVAL=true NUM_EXPT_AGENTS=3 label="marl_comparison" logger.project=RLC-2026
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false python social_laws/run.py task=rddl/grid_10x10_alternating/toroidal_no_restrictions_4_agents_single_task algorithm=ppo/rddl/grid_10x10_alternating/toroidal_full_restrictions value_function=dqnppo/rddl/grid_10x10_alternating/toroidal_full_restrictions algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true value_function.USE_SAME_SEED=true algorithm.FIXED_EVAL=true value_function.FIXED_EVAL=true NUM_EXPT_AGENTS=4 label="marl_comparison" logger.project=RLC-2026

# CREPPO
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false python social_laws/run.py task=rddl/grid_10x10_alternating/toroidal_no_restrictions_single_task algorithm=creppo/rddl/grid_10x10_alternating/toroidal_full_restrictions algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true algorithm.FIXED_EVAL=true label="marl_comparison" logger.project=RLC-2026
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false python social_laws/run.py task=rddl/grid_10x10_alternating/toroidal_no_restrictions_3_agents_single_task algorithm=creppo/rddl/grid_10x10_alternating/toroidal_full_restrictions algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true algorithm.FIXED_EVAL=true NUM_EXPT_AGENTS=3 label="marl_comparison" logger.project=RLC-2026
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false python social_laws/run.py task=rddl/grid_10x10_alternating/toroidal_no_restrictions_4_agents_single_task algorithm=creppo/rddl/grid_10x10_alternating/toroidal_full_restrictions algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true algorithm.FIXED_EVAL=true NUM_EXPT_AGENTS=4 label="marl_comparison" logger.project=RLC-2026

# PPO social laws (full restrictions)
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false python social_laws/run.py task=rddl/grid_10x10_alternating/toroidal_full_restrictions_single_task algorithm=ppo/rddl/grid_10x10_alternating/toroidal_full_restrictions value_function=dqnppo/rddl/grid_10x10_alternating/toroidal_full_restrictions algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true value_function.USE_SAME_SEED=true algorithm.FIXED_EVAL=true value_function.FIXED_EVAL=true label="marl_comparison" logger.project=RLC-2026
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false python social_laws/run.py task=rddl/grid_10x10_alternating/toroidal_full_restrictions_3_agents_single_task algorithm=ppo/rddl/grid_10x10_alternating/toroidal_full_restrictions value_function=dqnppo/rddl/grid_10x10_alternating/toroidal_full_restrictions algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true value_function.USE_SAME_SEED=true algorithm.FIXED_EVAL=true value_function.FIXED_EVAL=true NUM_EXPT_AGENTS=3 label="marl_comparison" logger.project=RLC-2026
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false python social_laws/run.py task=rddl/grid_10x10_alternating/toroidal_full_restrictions_4_agents_single_task algorithm=ppo/rddl/grid_10x10_alternating/toroidal_full_restrictions value_function=dqnppo/rddl/grid_10x10_alternating/toroidal_full_restrictions algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true value_function.USE_SAME_SEED=true algorithm.FIXED_EVAL=true value_function.FIXED_EVAL=true NUM_EXPT_AGENTS=4 label="marl_comparison" logger.project=RLC-2026

# CREPPO social laws (full restrictions)
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false python social_laws/run.py task=rddl/grid_10x10_alternating/toroidal_full_restrictions_single_task algorithm=creppo/rddl/grid_10x10_alternating/toroidal_full_restrictions algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true algorithm.FIXED_EVAL=true label="marl_comparison" logger.project=RLC-2026
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false python social_laws/run.py task=rddl/grid_10x10_alternating/toroidal_full_restrictions_3_agents_single_task algorithm=creppo/rddl/grid_10x10_alternating/toroidal_full_restrictions algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true algorithm.FIXED_EVAL=true NUM_EXPT_AGENTS=3 label="marl_comparison" logger.project=RLC-2026
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false python social_laws/run.py task=rddl/grid_10x10_alternating/toroidal_full_restrictions_4_agents_single_task algorithm=creppo/rddl/grid_10x10_alternating/toroidal_full_restrictions algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true algorithm.FIXED_EVAL=true NUM_EXPT_AGENTS=4 label="marl_comparison" logger.project=RLC-2026

# multi-task

# IPPO
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false python social_laws/experiments/run_marl_compare.py task=rddl/grid_10x10_alternating/toroidal_no_restrictions algorithm=ippo/rddl/grid_10x10_alternating/toroidal_no_restrictions algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true label="marl_comparison" logger.project=RLC-2026
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false python social_laws/experiments/run_marl_compare.py task=rddl/grid_10x10_alternating/toroidal_no_restrictions_3_agents algorithm=ippo/rddl/grid_10x10_alternating/toroidal_no_restrictions algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true label="marl_comparison" logger.project=RLC-2026
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false python social_laws/experiments/run_marl_compare.py task=rddl/grid_10x10_alternating/toroidal_no_restrictions_4_agents algorithm=ippo/rddl/grid_10x10_alternating/toroidal_no_restrictions algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true label="marl_comparison" logger.project=RLC-2026

# PPO
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false python social_laws/run.py task=rddl/grid_10x10_alternating/toroidal_no_restrictions algorithm=ppo/rddl/grid_10x10_alternating/toroidal_full_restrictions value_function=dqnppo/rddl/grid_10x10_alternating/toroidal_full_restrictions algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true value_function.USE_SAME_SEED=true algorithm.FIXED_EVAL=true value_function.FIXED_EVAL=true label="marl_comparison" logger.project=RLC-2026
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false python social_laws/run.py task=rddl/grid_10x10_alternating/toroidal_no_restrictions_3_agents algorithm=ppo/rddl/grid_10x10_alternating/toroidal_full_restrictions value_function=dqnppo/rddl/grid_10x10_alternating/toroidal_full_restrictions algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true value_function.USE_SAME_SEED=true algorithm.FIXED_EVAL=true value_function.FIXED_EVAL=true NUM_EXPT_AGENTS=3 label="marl_comparison" logger.project=RLC-2026
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false python social_laws/run.py task=rddl/grid_10x10_alternating/toroidal_no_restrictions_4_agents algorithm=ppo/rddl/grid_10x10_alternating/toroidal_full_restrictions value_function=dqnppo/rddl/grid_10x10_alternating/toroidal_full_restrictions algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true value_function.USE_SAME_SEED=true algorithm.FIXED_EVAL=true value_function.FIXED_EVAL=true NUM_EXPT_AGENTS=4 label="marl_comparison" logger.project=RLC-2026

# CREPPO
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false python social_laws/run.py task=rddl/grid_10x10_alternating/toroidal_no_restrictions algorithm=creppo/rddl/grid_10x10_alternating/toroidal_full_restrictions algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true algorithm.FIXED_EVAL=true label="marl_comparison" logger.project=RLC-2026
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false python social_laws/run.py task=rddl/grid_10x10_alternating/toroidal_no_restrictions_3_agents algorithm=creppo/rddl/grid_10x10_alternating/toroidal_full_restrictions algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true algorithm.FIXED_EVAL=true NUM_EXPT_AGENTS=3 label="marl_comparison" logger.project=RLC-2026
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false python social_laws/run.py task=rddl/grid_10x10_alternating/toroidal_no_restrictions_4_agents algorithm=creppo/rddl/grid_10x10_alternating/toroidal_full_restrictions algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true algorithm.FIXED_EVAL=true NUM_EXPT_AGENTS=4 label="marl_comparison" logger.project=RLC-2026

# PPO social laws (full restrictions)
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false python social_laws/run.py task=rddl/grid_10x10_alternating/toroidal_full_restrictions algorithm=ppo/rddl/grid_10x10_alternating/toroidal_full_restrictions value_function=dqnppo/rddl/grid_10x10_alternating/toroidal_full_restrictions algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true value_function.USE_SAME_SEED=true algorithm.FIXED_EVAL=true value_function.FIXED_EVAL=true label="marl_comparison" logger.project=RLC-2026
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false python social_laws/run.py task=rddl/grid_10x10_alternating/toroidal_full_restrictions_3_agents algorithm=ppo/rddl/grid_10x10_alternating/toroidal_full_restrictions value_function=dqnppo/rddl/grid_10x10_alternating/toroidal_full_restrictions algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true value_function.USE_SAME_SEED=true algorithm.FIXED_EVAL=true value_function.FIXED_EVAL=true NUM_EXPT_AGENTS=3 label="marl_comparison" logger.project=RLC-2026
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false python social_laws/run.py task=rddl/grid_10x10_alternating/toroidal_full_restrictions_4_agents algorithm=ppo/rddl/grid_10x10_alternating/toroidal_full_restrictions value_function=dqnppo/rddl/grid_10x10_alternating/toroidal_full_restrictions algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true value_function.USE_SAME_SEED=true algorithm.FIXED_EVAL=true value_function.FIXED_EVAL=true NUM_EXPT_AGENTS=4 label="marl_comparison" logger.project=RLC-2026

# CREPPO social laws (full restrictions)
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false python social_laws/run.py task=rddl/grid_10x10_alternating/toroidal_full_restrictions algorithm=creppo/rddl/grid_10x10_alternating/toroidal_full_restrictions algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true algorithm.FIXED_EVAL=true label="marl_comparison" logger.project=RLC-2026
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false python social_laws/run.py task=rddl/grid_10x10_alternating/toroidal_full_restrictions_3_agents algorithm=creppo/rddl/grid_10x10_alternating/toroidal_full_restrictions algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true algorithm.FIXED_EVAL=true NUM_EXPT_AGENTS=3 label="marl_comparison" logger.project=RLC-2026
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false python social_laws/run.py task=rddl/grid_10x10_alternating/toroidal_full_restrictions_4_agents algorithm=creppo/rddl/grid_10x10_alternating/toroidal_full_restrictions algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true algorithm.FIXED_EVAL=true NUM_EXPT_AGENTS=4 label="marl_comparison" logger.project=RLC-2026

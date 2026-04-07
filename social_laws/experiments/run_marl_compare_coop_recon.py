'''Main entry point for running MARL algorithms on continuous coop recon.'''
import hydra
import jax
import random
from omegaconf import OmegaConf, open_dict

import os
import numpy as np

from common.wandb_visualizations import Logger
from social_laws.ippo import run_training as run_ippo
from social_laws.ippo_centralized import run_training as run_ippo_centralized

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
        if config["algorithm"].get("CENTRALIZED", False):
            params, policy, init_params = run_ippo_centralized(config, wandb_logger)
        else:
            params, policies, init_params = run_ippo(config, wandb_logger)
    else:
        raise NotImplementedError(f"Algorithm {config.algorithm['ALG']} not implemented.")

    wandb_logger.close()

if __name__ == "__main__":
    ippo()

# Command Reference for MARL Comparison (Section 8)

# === N=2 AGENTS ===
# IPPO (Baseline, No Law)
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_DEFAULT_MATMUL_PRECISION=highest python social_laws/experiments/run_marl_compare_coop_recon.py task=continuous/coop_recon_compare_no_law_2_agent algorithm=ippo/continuous/coop_recon algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true algorithm.FIXED_EVAL=true label="marl_comparison" logger.project=NEURIPS-2026
# IPPO (Social Law)
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_DEFAULT_MATMUL_PRECISION=highest python social_laws/experiments/run_marl_compare_coop_recon.py task=continuous/coop_recon_compare_law_2_agent algorithm=ippo/continuous/coop_recon algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true algorithm.FIXED_EVAL=true label="marl_comparison" logger.project=NEURIPS-2026
# PPO (Baseline, No Law)
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_DEFAULT_MATMUL_PRECISION=highest python social_laws/run.py task=continuous/coop_recon_compare_no_law_2_agent algorithm=ppo/continuous/coop_recon value_function=dqnppo/continuous/coop_recon algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true value_function.USE_SAME_SEED=true algorithm.FIXED_EVAL=true value_function.FIXED_EVAL=true NUM_EXPT_AGENTS=2 label="marl_comparison" logger.project=NEURIPS-2026
# PPO (Social Law)
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_DEFAULT_MATMUL_PRECISION=highest python social_laws/run.py task=continuous/coop_recon_compare_law_2_agent algorithm=ppo/continuous/coop_recon value_function=dqnppo/continuous/coop_recon algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true value_function.USE_SAME_SEED=true algorithm.FIXED_EVAL=true value_function.FIXED_EVAL=true NUM_EXPT_AGENTS=2 label="marl_comparison" logger.project=NEURIPS-2026
# CREPPO (Baseline, No Law)
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_DEFAULT_MATMUL_PRECISION=highest python social_laws/run.py task=continuous/coop_recon_compare_no_law_2_agent algorithm=creppo/continuous/coop_recon algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true algorithm.FIXED_EVAL=true NUM_EXPT_AGENTS=2 label="marl_comparison" logger.project=NEURIPS-2026
# CREPPO (Social Law)
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_DEFAULT_MATMUL_PRECISION=highest python social_laws/run.py task=continuous/coop_recon_compare_law_2_agent algorithm=creppo/continuous/coop_recon algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true algorithm.FIXED_EVAL=true NUM_EXPT_AGENTS=2 label="marl_comparison" logger.project=NEURIPS-2026

# === N=3 AGENTS ===
# IPPO (Baseline, No Law)
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_DEFAULT_MATMUL_PRECISION=highest python social_laws/experiments/run_marl_compare_coop_recon.py task=continuous/coop_recon_compare_no_law_3_agent algorithm=ippo/continuous/coop_recon algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true algorithm.FIXED_EVAL=true label="marl_comparison" logger.project=NEURIPS-2026
# IPPO (Social Law)
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_DEFAULT_MATMUL_PRECISION=highest python social_laws/experiments/run_marl_compare_coop_recon.py task=continuous/coop_recon_compare_law_3_agent algorithm=ippo/continuous/coop_recon algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true algorithm.FIXED_EVAL=true label="marl_comparison" logger.project=NEURIPS-2026
# PPO (Baseline, No Law)
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_DEFAULT_MATMUL_PRECISION=highest python social_laws/run.py task=continuous/coop_recon_compare_no_law_3_agent algorithm=ppo/continuous/coop_recon value_function=dqnppo/continuous/coop_recon algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true value_function.USE_SAME_SEED=true algorithm.FIXED_EVAL=true value_function.FIXED_EVAL=true NUM_EXPT_AGENTS=3 label="marl_comparison" logger.project=NEURIPS-2026
# PPO (Social Law)
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_DEFAULT_MATMUL_PRECISION=highest python social_laws/run.py task=continuous/coop_recon_compare_law_3_agent algorithm=ppo/continuous/coop_recon value_function=dqnppo/continuous/coop_recon algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true value_function.USE_SAME_SEED=true algorithm.FIXED_EVAL=true value_function.FIXED_EVAL=true NUM_EXPT_AGENTS=3 label="marl_comparison" logger.project=NEURIPS-2026
# CREPPO (Baseline, No Law)
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_DEFAULT_MATMUL_PRECISION=highest python social_laws/run.py task=continuous/coop_recon_compare_no_law_3_agent algorithm=creppo/continuous/coop_recon algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true algorithm.FIXED_EVAL=true NUM_EXPT_AGENTS=3 label="marl_comparison" logger.project=NEURIPS-2026
# CREPPO (Social Law)
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_DEFAULT_MATMUL_PRECISION=highest python social_laws/run.py task=continuous/coop_recon_compare_law_3_agent algorithm=creppo/continuous/coop_recon algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true algorithm.FIXED_EVAL=true NUM_EXPT_AGENTS=3 label="marl_comparison" logger.project=NEURIPS-2026

# === N=4 AGENTS ===
# IPPO (Baseline, No Law)
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_DEFAULT_MATMUL_PRECISION=highest python social_laws/experiments/run_marl_compare_coop_recon.py task=continuous/coop_recon_compare_no_law_4_agent algorithm=ippo/continuous/coop_recon algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true algorithm.FIXED_EVAL=true label="marl_comparison" logger.project=NEURIPS-2026
# IPPO (Social Law)
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_DEFAULT_MATMUL_PRECISION=highest python social_laws/experiments/run_marl_compare_coop_recon.py task=continuous/coop_recon_compare_law_4_agent algorithm=ippo/continuous/coop_recon algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true algorithm.FIXED_EVAL=true label="marl_comparison" logger.project=NEURIPS-2026
# PPO (Baseline, No Law)
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_DEFAULT_MATMUL_PRECISION=highest python social_laws/run.py task=continuous/coop_recon_compare_no_law_4_agent algorithm=ppo/continuous/coop_recon value_function=dqnppo/continuous/coop_recon algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true value_function.USE_SAME_SEED=true algorithm.FIXED_EVAL=true value_function.FIXED_EVAL=true NUM_EXPT_AGENTS=4 label="marl_comparison" logger.project=NEURIPS-2026
# PPO (Social Law)
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_DEFAULT_MATMUL_PRECISION=highest python social_laws/run.py task=continuous/coop_recon_compare_law_4_agent algorithm=ppo/continuous/coop_recon value_function=dqnppo/continuous/coop_recon algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true value_function.USE_SAME_SEED=true algorithm.FIXED_EVAL=true value_function.FIXED_EVAL=true NUM_EXPT_AGENTS=4 label="marl_comparison" logger.project=NEURIPS-2026
# CREPPO (Baseline, No Law)
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_DEFAULT_MATMUL_PRECISION=highest python social_laws/run.py task=continuous/coop_recon_compare_no_law_4_agent algorithm=creppo/continuous/coop_recon algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true algorithm.FIXED_EVAL=true NUM_EXPT_AGENTS=4 label="marl_comparison" logger.project=NEURIPS-2026
# CREPPO (Social Law)
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_DEFAULT_MATMUL_PRECISION=highest python social_laws/run.py task=continuous/coop_recon_compare_law_4_agent algorithm=creppo/continuous/coop_recon algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true algorithm.FIXED_EVAL=true NUM_EXPT_AGENTS=4 label="marl_comparison" logger.project=NEURIPS-2026

# === N=5 AGENTS ===
# IPPO (Baseline, No Law)
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_DEFAULT_MATMUL_PRECISION=highest python social_laws/experiments/run_marl_compare_coop_recon.py task=continuous/coop_recon_compare_no_law_5_agent algorithm=ippo/continuous/coop_recon algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true algorithm.FIXED_EVAL=true label="marl_comparison" logger.project=NEURIPS-2026
# IPPO (Social Law)
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_DEFAULT_MATMUL_PRECISION=highest python social_laws/experiments/run_marl_compare_coop_recon.py task=continuous/coop_recon_compare_law_5_agent algorithm=ippo/continuous/coop_recon algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true algorithm.FIXED_EVAL=true label="marl_comparison" logger.project=NEURIPS-2026
# PPO (Baseline, No Law)
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_DEFAULT_MATMUL_PRECISION=highest python social_laws/run.py task=continuous/coop_recon_compare_no_law_5_agent algorithm=ppo/continuous/coop_recon value_function=dqnppo/continuous/coop_recon algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true value_function.USE_SAME_SEED=true algorithm.FIXED_EVAL=true value_function.FIXED_EVAL=true NUM_EXPT_AGENTS=5 label="marl_comparison" logger.project=NEURIPS-2026
# PPO (Social Law)
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_DEFAULT_MATMUL_PRECISION=highest python social_laws/run.py task=continuous/coop_recon_compare_law_5_agent algorithm=ppo/continuous/coop_recon value_function=dqnppo/continuous/coop_recon algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true value_function.USE_SAME_SEED=true algorithm.FIXED_EVAL=true value_function.FIXED_EVAL=true NUM_EXPT_AGENTS=5 label="marl_comparison" logger.project=NEURIPS-2026
# CREPPO (Baseline, No Law)
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_DEFAULT_MATMUL_PRECISION=highest python social_laws/run.py task=continuous/coop_recon_compare_no_law_5_agent algorithm=creppo/continuous/coop_recon algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true algorithm.FIXED_EVAL=true NUM_EXPT_AGENTS=5 label="marl_comparison" logger.project=NEURIPS-2026
# CREPPO (Social Law)
# PYTHONPATH=/work/05187/rfern/stampede3/GitHub/jax-aht JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_DEFAULT_MATMUL_PRECISION=highest python social_laws/run.py task=continuous/coop_recon_compare_law_5_agent algorithm=creppo/continuous/coop_recon algorithm.TRAIN_SEED=72128 algorithm.USE_SAME_SEED=true algorithm.FIXED_EVAL=true NUM_EXPT_AGENTS=5 label="marl_comparison" logger.project=NEURIPS-2026

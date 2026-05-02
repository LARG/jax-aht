"""Stores hard-coded variables for paper plots."""

import os


SAVE_DIR = "results/figures"
HYPERPARAM_DEFAULT_METRIC = "HeldoutEval/FinalEgoVsHeldout/returned_episode_returns/mean"
HYPERPARAM_PROJECT = "aht-parameter-sweep"
ENTITY = "aht-project"

METHOD_TO_DISPLAY_NAME = {
    "ppo_ego": "PPO",
    "liam": "LIAM",
    "meliba": "MeLIBA",
    "fcp": "FCP",
    "brdiv": "BRDiv",
    "lbrdiv": "LBRDiv",
    "comedi": "CoMeDi",
    "rotate": "ROTATE",
    "cole": "COLE",
    "trajedi": "TrajeDi",
}

TASK_LEGACY_NAMES = {
    "lbf/lbf_7x7_nolevels": "lbf",
}

TASK_TO_DISPLAY_NAME = {
    "lbf/lbf_7x7_nolevels": "LBF 7x7 (No Levels)",
    "overcooked-v1/coord_ring": "Overcooked: Coordination Ring",
}

ALGO_TO_ENTRY_POINT = {
    "fcp": "teammate_generation",
    "brdiv": "teammate_generation",
    "lbrdiv": "teammate_generation",
    "comedi": "teammate_generation",
    "rotate": "open_ended_training",
    "cole": "open_ended_training",
    "trajedi": "open_ended_training",
    "ppo_ego": "ego_agent_training",
    "liam": "ego_agent_training",
    "meliba": "ego_agent_training",
}

HYPERPARAM_SWEEPS = {
    "lbf/lbf_7x7_nolevels": {
        "ppo_ego": "yje7een6", "liam": "xqiaed80", "meliba": "y4ddadn8",
        "fcp": "22cojezv", "brdiv": "d3e7c0fx", "lbrdiv": "rni853js",
        "comedi": "d1dt0arj", "rotate": "slpf9grh", "cole": "pr0fwbdp", "trajedi": "dqsezvy1",
    },
    "overcooked-v1/coord_ring": {
        "ppo_ego": "qeafl8r7", "liam": "pbq863zp", "meliba": "i532vemb",
        "fcp": "fubwmomo", "brdiv": "wgapxysb", "lbrdiv": "eu0g1orm",
        "comedi": "xeikmue5", "rotate": "lgqrnsmt", "cole": "irstlaiv", "trajedi": "j26xa39y",
    },
}

# values that were mistakenly included in the
# hyperparameter sweep that now need to be excluded
FILTERED_HYPERPARAMETER_KV = {
    "trajedi": {
        "TRAJEDI_COEF": [0.0]
    }
}
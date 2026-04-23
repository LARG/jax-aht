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
    "lbrdiv": "LB-RDiv",
    "comedi": "CoMeDi",
    "rotate": "ROTATE",
    "cole": "COLE",
    "trajedi": "TrajeDi",
}

TASK_TO_DISPLAY_NAME = {
    "lbf": "LBF",
    "overcooked-v1/coord_ring": "Overcooked: Coordination Ring",
}

EGO_HYPERPARAM_SWEEPS = {
    "lbf": {"ppo_ego": "yje7een6", "liam": "xqiaed80", "meliba": "y4ddadn8"},
    "overcooked-v1/coord_ring": {"ppo_ego": "qeafl8r7", "liam": "pbq863zp", "meliba": "i532vemb"}
}

UNIFIED_HYPERPARAM_SWEEPS = {
    "lbf": {
        "fcp": "22cojezv",
        "brdiv": "d3e7c0fx",
        "lbrdiv": "rni853js",
        "comedi": "d1dt0arj",
        "rotate": "slpf9grh",
        "cole": "pr0fwbdp",
    },
    "overcooked-v1/coord_ring": {
        "fcp": "fubwmomo",
        "brdiv": "wgapxysb",
        "lbrdiv": "eu0g1orm",
        # "comedi": "", #. PENDING
        "rotate": "lgqrnsmt",
        "cole": "irstlaiv",
    }
}

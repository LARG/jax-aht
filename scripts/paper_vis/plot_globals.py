"""Stores hard-coded variables for paper plots."""
import omegaconf

SAVE_DIR = "results/figures"
ENTITY = "aht-project"
####### AGENT TYPE SUBSTRINGS #### 
HEURISTIC_AGENTS = [
    # LBF
    "seq_agent*", "entitled_agent", "greedy_",
    # Overcooked
    "independent", "onion", "plate", 
    # Hanabi
    "iggi", "piers", "flawed", "outer", "van_den_bergh", 
    "smartbot", "cautious", "internal"
]
RL_AGENTS = [
    "comedi", "lbrdiv", "ippo", "brdiv", "obl"
]
HUMAN_PROXY_AGENTS = [
    "human_proxy"
]
####### BENCHMARK RUNS #######
BENCHMARK_PROJECT = "aht-benchmark"
EGO_BENCHMARK_RUNS = {
    "lbf/lbf_7x7_nolevels": {
        "ppo_ego": {
            "fcp_teammates": "d411alba",
            "comedi_teammates": "2e5x3uko"
        }, 
        "liam": {
            "fcp_teammates": "zq6yx6kf",
            "comedi_teammates": "0kh751xl"
        }, 
        "meliba": {
            "fcp_teammates": "47gw238c",
            "comedi_teammates": "bc1legak"
        }, 
    },
    "lbf/lbf_12x12": {
        "ppo_ego": {
            "fcp_teammates": "otxy993u",
            "comedi_teammates": "jna0irje"
        }, 
        "liam": {
            "fcp_teammates": "j9adbki8",
            "comedi_teammates": "3h6fcrni"
        }, 
        "meliba": {
            "fcp_teammates": "pifmvl5o",
            "comedi_teammates": "pwvgckir"
        }, 
    },
    "overcooked-v1/cramped_room": {
        "ppo_ego": {
            "fcp_teammates": "rm5bx4ui",
            "comedi_teammates": "7jhrdpxc"
        }, 
        "liam": {
            "fcp_teammates": "xuvmlpmi",
            "comedi_teammates": "8486vdnp"
        }, 
        "meliba": {
            "fcp_teammates": ["lghgomp3", "v5ey4lkt"],
            "comedi_teammates": ["en1wbqt4", "3e6wffoi"]
        }, 
    },
    "overcooked-v1/coord_ring": {
        "ppo_ego": {
            "fcp_teammates": "0wzr6nbv",
            "comedi_teammates": "5njknt2q"
        }, 
        "liam": {
            "fcp_teammates": "e8y4gy49",
            "comedi_teammates": "pmm9u33p"
        }, 
        "meliba": {
            "fcp_teammates": "x6gmp9uc",
            "comedi_teammates": ["7a19mtio", "bd6w70tr"]
        },
    },
}

UNIFIED_BENCHMARK_RUNS = {
    "lbf/lbf_7x7_nolevels": {
        "fcp": "1bhjc1ri", 
        "brdiv": "3i68f80z", 
        "lbrdiv": "jj2ycq8o",
        "comedi": "us4jl8ch", 
        "cole": "9mmd6r45", 
        "trajedi": "ilklfw2f",
    },
    "lbf/lbf_12x12": {
        "fcp": ["1c0um2ls", "52wp5amm"],
        "brdiv": "b3xozodw", 
        "lbrdiv": "dft2f0do",
        "comedi": "7qizaam5", 
        "cole": "wom49daz", 
        "trajedi": "ipsuzp4z",
    },
    "overcooked-v1/cramped_room": {
        "fcp": "n1mplxeg", 
        "brdiv": "0ruf65cb", 
        "lbrdiv": "kfiwwxbu",
        "comedi": ["8k97saxv", "fqw407x0"],
        "cole": "ga4gg8bs", 
        "trajedi": "2xtvi3x6",
    },
    "overcooked-v1/coord_ring": {
        "fcp": "ikrlj1qe", 
        "brdiv": "r4a3ncl2", 
        "lbrdiv": "x012q7qc",
        "comedi": "ehz0njkk", 
        "cole": "zxwuozip",
        "trajedi": "bkuq6gvy", # TODO: update!
    }
}

####### HYPERPARAMETER SWEEPS #######
HYPERPARAM_DEFAULT_METRIC = "HeldoutEval/FinalEgoVsHeldout/returned_episode_returns/mean"
HYPERPARAM_PROJECT = "aht-parameter-sweep"

HYPERPARAM_SWEEPS = {
    "lbf/lbf_7x7_nolevels": {
        "ppo_ego": "yje7een6", 
        "liam": "xqiaed80", 
        "meliba": "y4ddadn8",
        "fcp": "22cojezv", 
        "brdiv": "d3e7c0fx", 
        "lbrdiv": "rni853js",
        "comedi": "d1dt0arj", 
        # "rotate": "slpf9grh", # done but excluded
        "cole": "pr0fwbdp", 
        "trajedi": "dqsezvy1",
    },
    "lbf/lbf_12x12": {
        "ppo_ego": "k2giuu4l", 
        "liam": "yibcruuz", 
        "meliba": "6jdo5rjv",
        "fcp": "nivg4xvf", 
        "brdiv": "fdg6dw1n", 
        "lbrdiv": "y23unh8y",
        "comedi": "7e9yf5zg",
        # "rotate": "t4hufinh", # running
        "cole": "kbvghubr", 
        "trajedi": "mavnqv7e",
    },
    "overcooked-v1/cramped_room": {
        "ppo_ego": "vexvuss8", 
        "liam": "zz9lkwdz", 
        "meliba": "dva0ffdq",
        "fcp": "e23khyjt", 
        "brdiv": "19gmzemf", 
        "lbrdiv": "w8abf056",
        "comedi": "vt0xnwxc",
        # "rotate": "hrer8x3c",  # done but excluded
        "cole": "jg6700d6", 
        "trajedi": "dua855jm",
    },
    "overcooked-v1/coord_ring": {
        "ppo_ego": "qeafl8r7", 
        "liam": "pbq863zp", 
        "meliba": "i532vemb",
        "fcp": "fubwmomo", 
        "brdiv": "wgapxysb", 
        "lbrdiv": "eu0g1orm",
        "comedi": "xeikmue5", 
        # "rotate": "lgqrnsmt", 
        "cole": "irstlaiv", 
        "trajedi": "j26xa39y",
    },
    "mini-hanabi": {
        "ppo_ego": "y2w21bej",
        "liam": "8t2so38u", 
        "meliba": "q4z3szuh",
        # "fcp": "oku0yyg0", 
        # "brdiv": "wnnhav1m", 
        # "lbrdiv": "uvvpc05r",
        # "comedi": "s745q3lg", 
        # "cole": "158to6y5",
        # "trajedi": "4my4n564",
    }
}

# values that were mistakenly included in the
# hyperparameter sweep that now need to be excluded
FILTERED_HYPERPARAMETER_KV = {
    "trajedi": {
        "TRAJEDI_COEF": [0.0]
    }
}

####### PLOTTING SETTINGS #######
TASK_TO_PLOT_TITLE = {
    "lbf/lbf_7x7_nolevels": "LBF 7x7",
    "overcooked-v1/cramped_room": "Cramped Room (Overcooked)",
    "overcooked-v1/asymm_advantages": "Asymmetric Advantages (Overcooked)",
    "overcooked-v1/forced_coord": "Forced Coordination (Overcooked)",
    "overcooked-v1/counter_circuit": "Counter Circuit (Overcooked)",
    "overcooked-v1/coord_ring": "Coordination Ring (Overcooked)",
}

TASK_TO_AXIS_DISPLAY_NAME = {
    "lbf/lbf_7x7_nolevels": "LBF 7x7",
    "lbf/lbf_12x12": "LBF 12x12",
    "overcooked-v1/cramped_room": "CR",
    "overcooked-v1/asymm_advantages": "AA",
    "overcooked-v1/forced_coord": "FC",
    "overcooked-v1/counter_circuit": "CC",
    "overcooked-v1/coord_ring": "CoR",
}

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
    "lbf/lbf_7x7_nolevels": "LBF 7x7",
    "lbf/lbf_12x12": "LBF 12x12",
    "overcooked-v1/cramped_room": "Cramped Room (Overcooked)",
    "overcooked-v1/coord_ring": "Coordination Ring (Overcooked)",
    "overcooked-v1/asymm_advantages": "Asymmetric Advantages (Overcooked)",
    "overcooked-v1/forced_coord": "Forced Coordination (Overcooked)",
    "overcooked-v1/counter_circuit": "Counter Circuit (Overcooked)",
}

TASK_TO_ENV_NAME = {
    "lbf/lbf_7x7_nolevels": "lbf",
    "lbf/lbf_12x12": "lbf",
    "overcooked-v1/cramped_room": "overcooked-v1",
    "overcooked-v1/asymm_advantages": "overcooked-v1",
    "overcooked-v1/forced_coord": "overcooked-v1",
    "overcooked-v1/counter_circuit": "overcooked-v1",
    "overcooked-v1/coord_ring": "overcooked-v1",
}

TASK_TO_METRIC_NAME = {
    "lbf/lbf_7x7_nolevels": "returned_episode_returns",
    "lbf/lbf_12x12": "returned_episode_returns",
    "overcooked-v1/cramped_room": "returned_episode_returns",
    "overcooked-v1/asymm_advantages": "returned_episode_returns",
    "overcooked-v1/forced_coord": "returned_episode_returns",
    "overcooked-v1/counter_circuit": "returned_episode_returns",
    "overcooked-v1/coord_ring": "returned_episode_returns",
}

# Methods that use open-ended learning (OEL); these have 5D eval metrics
# shape (num_seeds, num_oel_iter, num_heldout_agents, num_eval_episodes, num_agents_per_game)
OEL_METHODS = ["rotate"]

GLOBAL_HELDOUT_CONFIG = omegaconf.OmegaConf.load("evaluation/configs/global_heldout_settings.yaml")
CACHE_FILENAME = "cached_summary_metrics.pkl"
HELDOUT_CURVES_CACHE_FILENAME = "cached_heldout_curves.pkl"
TITLE_FONTSIZE = 20
AXIS_LABEL_FONTSIZE = 18
LEGEND_FONTSIZE = 14

# def get_heldout_agents(task_name, task_config_path):
    # rng = jax.random.PRNGKey(0)
    # heldout_cfg = GLOBAL_HELDOUT_CONFIG["heldout_set"][task_name]
    # env_config = omegaconf.OmegaConf.load(task_config_path)
    # env_name = env_config["ENV_NAME"]
    # env_kwargs = env_config["ENV_KWARGS"]

    # env = make_env(env_name, env_kwargs)
    # heldout_agents = load_heldout_set(heldout_cfg, env, task_name, env_kwargs, rng)

    # return heldout_agents

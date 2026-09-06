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
# Run ids by campaign wandb tag (project aht-benchmark), refreshed 2026-09-06.
# Seed counts are DISTINCT seeds: distinct (TRAIN_SEED, seed_index) pairs across
# the listed runs. jax.random.split is prefix-stable, so chunks sharing a
# TRAIN_SEED overlap and chunk sizes must not be summed.
EGO_BENCHMARK_RUNS = {
    "lbf/lbf_7x7_nolevels": {
        "ppo_ego": {
            "fcp_teammates": "lerlzvpv",  # iclr26, 5 seeds
            "comedi_teammates": "i124n1hl",  # iclr26, 5 seeds
        },
        "liam": {
            "fcp_teammates": "3byp3td0",  # iclr26, 5 seeds
            "comedi_teammates": "2opu8s3x",  # iclr26, 5 seeds
        },
        "meliba": {
            "fcp_teammates": ["ernydfeu", "isx6v5kf", "m04si2f5"],  # iclr26, 10 seeds
            "comedi_teammates": ["2gs98nrk", "sadapb3o", "pxsx1ahd"],  # iclr26, 10 seeds
        },
    },
    "lbf/lbf_12x12": {
        "ppo_ego": {
            "fcp_teammates": "otxy993u",  # neurips
            "comedi_teammates": "jna0irje",  # neurips
        },
        "liam": {
            "fcp_teammates": "j9adbki8",  # neurips
            "comedi_teammates": "3h6fcrni",  # neurips
        },
        "meliba": {
            "fcp_teammates": ["t8v125ba", "2swcrf3y", "af08h0sw"],  # iclr26, 10 seeds
            "comedi_teammates": ["lw7cmntl", "gxk4t9ee", "xgi1oqro"],  # iclr26, 10 seeds
        },
    },
    "overcooked-v1/cramped_room": {
        "ppo_ego": {
            "fcp_teammates": "rm5bx4ui",  # neurips
            "comedi_teammates": "7jhrdpxc",  # neurips
        },
        "liam": {
            "fcp_teammates": "xuvmlpmi",  # neurips
            "comedi_teammates": "8486vdnp",  # neurips
        },
        "meliba": {
            "fcp_teammates": ["sjkmr2co", "9kvtltg8", "7gqbc2gd"],  # iclr26, 10 seeds
            "comedi_teammates": ["ikbqpjaq", "6hyan548", "5mx43y3q"],  # iclr26, 10 seeds
        },
    },
    "overcooked-v1/coord_ring": {
        "ppo_ego": {
            "fcp_teammates": "we10tivv",  # iclr26, 5 seeds
            "comedi_teammates": "1tq4878f",  # iclr26, 5 seeds
        },
        "liam": {
            "fcp_teammates": "a7usto7b",  # iclr26, 5 seeds
            "comedi_teammates": "qsjbzty8",  # iclr26, 5 seeds
        },
        "meliba": {
            "fcp_teammates": ["vjvs7uug", "0b27ry3j", "8bpn86ss"],  # iclr26, 10 seeds
            "comedi_teammates": ["vt2edaa7", "p0rv1ege", "ins39ls1"],  # iclr26, 10 seeds
        },
    },
    "overcooked-v1/asymm_advantages": {
        "ppo_ego": {
            "fcp_teammates": "nydruntw",  # iclr26, 5 seeds
            "comedi_teammates": "5z39jj4k",  # iclr26, 5 seeds
        },
        "liam": {
            "fcp_teammates": "ryq0tsxe",  # iclr26, 5 seeds
            "comedi_teammates": "iwu9233c",  # iclr26, 5 seeds
        },
        "meliba": {
            "fcp_teammates": "4m8tmxj3",  # iclr26, 5 seeds
            "comedi_teammates": "kekiners",  # iclr26, 5 seeds
        },
    },
    "overcooked-v1/forced_coord": {
        "ppo_ego": {
            "fcp_teammates": "tf6xkzpu",  # iclr26, 5 seeds
            "comedi_teammates": "5x1rtnii",  # iclr26, 5 seeds
        },
        "liam": {
            "fcp_teammates": "vuwcr5yh",  # iclr26, 5 seeds
            "comedi_teammates": "seepwt80",  # iclr26, 5 seeds
        },
        "meliba": {
            "fcp_teammates": "1d2kx3tq",  # iclr26, 5 seeds
            "comedi_teammates": "0cd2w1ec",  # iclr26, 5 seeds
        },
    },
    "overcooked-v1/counter_circuit": {
        "ppo_ego": {
            "fcp_teammates": "9wekrbxo",  # iclr26, 5 seeds
            "comedi_teammates": "udklgvdq",  # iclr26, 5 seeds
        },
        "liam": {
            "fcp_teammates": "4rwoutix",  # iclr26, 5 seeds
            "comedi_teammates": "vyxn9dxp",  # iclr26, 5 seeds
        },
        "meliba": {
            "fcp_teammates": ["obtdbm9y", "z7pmvhj7", "tv16ua4v"],  # iclr26, 10 seeds
            "comedi_teammates": ["r5e8vlp7", "giim6lb3", "gnocl630"],  # iclr26, 10 seeds
        },
    },
    "mini-hanabi": {
        "ppo_ego": {
            "fcp_teammates": ["hkoidsa0", "rflrtlzo", "719okso0", "6s91gqf5"],  # neurips, 5 seeds
            "comedi_teammates": ["g61afu52", "7rpmb6ie", "70d2c8v2", "38pn9msz"],  # neurips
        },
        "liam": {
            "fcp_teammates": ["gr03b106", "5ab05qxj", "0hwgmfj3", "lwpl8w6x"],  # neurips, 5 seeds
            "comedi_teammates": ["x0ckfnlv", "b2by5off", "72at5w6x", "tq4fcjh4", "6sig81om"],  # neurips
        },
        "meliba": {
            "fcp_teammates": ["3pu1t8gw", "azrmgwkw", "es6t0hbh"],  # iclr26, 10 seeds
            "comedi_teammates": ["lza9gtk4", "mtx7yx4l", "9vh5ul08"],  # iclr26, 10 seeds
        },
    },
}

# BC heldout-eval wandb runs (64-eps variant — eps count matches the training-time
# heldout-eval runs in EGO_BENCHMARK_RUNS so artifacts can concat along the
# partner axis without an eps mismatch). Each cell evaluates one ego against
# the BC partner(s) for its task: 5 BC partners (overcooked) or 1 (LBF).
# When --include-bc is passed to benchmark_bar_charts.py, these artifacts are
# merged with the standard heldout-eval ones before the reducer runs.
# Generated 2026-05-06 via evaluation/run_heldout_ego_bc.py at NUM_EVAL_EPISODES=64.
BC_BENCHMARK_RUNS = {
    "lbf/lbf_7x7_nolevels": {
        "ppo_ego": {"fcp_teammates": "shov7vtt", "comedi_teammates": "5bzzvtvr"},
        "liam":    {"fcp_teammates": "rodfedo5", "comedi_teammates": "kbmgv1qy"},
        "meliba":  {"fcp_teammates": "yfqq4r7c", "comedi_teammates": "d66eh0tt"},
    },
    "lbf/lbf_12x12": {
        "ppo_ego": {"fcp_teammates": "lgqum8kt", "comedi_teammates": "t7idtlm9"},
        "liam":    {"fcp_teammates": "dhr9hjgo", "comedi_teammates": "pwuy5p46"},
        "meliba":  {"fcp_teammates": "u5yu6xvn", "comedi_teammates": "ij48l33l"},
    },
    "overcooked-v1/coord_ring": {
        "ppo_ego": {"fcp_teammates": "v8x87epc", "comedi_teammates": "1k2kz0ge"},
        "liam":    {"fcp_teammates": "4k8zkd5e", "comedi_teammates": "p3a0x1gb"},
        "meliba":  {"fcp_teammates": "gna0budi", "comedi_teammates": "0eqejvkm"},
    },
    "overcooked-v1/cramped_room": {
        "ppo_ego": {"fcp_teammates": "durax862", "comedi_teammates": "86yovjo6"},
        "liam":    {"fcp_teammates": "9ni4tvq3", "comedi_teammates": "tsfrosfr"},
        "meliba":  {"fcp_teammates": "b612ne9c", "comedi_teammates": "kvygcmqc"},
    },
}

UNIFIED_BENCHMARK_RUNS = {
    "lbf/lbf_7x7_nolevels": {
        "fcp": "1bhjc1ri",  # neurips
        "brdiv": "nqbhn0x8",  # iclr26, 5 seeds
        "lbrdiv": "jj2ycq8o",  # neurips
        "comedi": "us4jl8ch",  # neurips
        "rotate": "0vt39b0r",  # neurips
        "cole": "k7uf5ndx",  # iclr26, 5 seeds
        "trajedi": "733qoihr",  # iclr26, 5 seeds
    },
    "lbf/lbf_12x12": {
        "fcp": ["1c0um2ls", "52wp5amm"],  # neurips; 2+3 split at same TRAIN_SEED=20374 -> 3 seeds, not 5
        "brdiv": "b3xozodw",  # neurips
        "lbrdiv": "dft2f0do",  # neurips
        "comedi": "7qizaam5",  # neurips
        "rotate": "9a280lft",  # neurips
        "cole": "62dhrovw",  # iclr26, 5 seeds
        "trajedi": "mbz2pbmy",  # iclr26, 5 seeds
    },
    "overcooked-v1/cramped_room": {
        "fcp": "n1mplxeg",  # neurips
        "brdiv": "u1hihvk2",  # iclr26, 5 seeds
        "lbrdiv": "kfiwwxbu",  # neurips
        "comedi": ["8k97saxv", "fqw407x0"],  # neurips; 2+3 split at same TRAIN_SEED=20374 -> 3 seeds, not 5
        "rotate": "egwn4951",  # neurips
        "cole": "mj2zzmzq",  # iclr26, 5 seeds
        "trajedi": "4ubrdpio",  # iclr26, 5 seeds
    },
    "overcooked-v1/coord_ring": {
        "fcp": "ikrlj1qe",  # neurips
        "brdiv": "1yag7vj1",  # iclr26, 5 seeds
        "lbrdiv": "x012q7qc",  # neurips
        "comedi": "ont3iesc",  # iclr26, 5 seeds
        "rotate": "x7cynboy",  # neurips
        "cole": "zrc2agdk",  # iclr26, 5 seeds
        "trajedi": ["r55lnqcb", "p8lnwq93"],  # iclr26, 5 seeds
    },
    "overcooked-v1/asymm_advantages": {
        "fcp": "waolda91",  # iclr26, 5 seeds
        "brdiv": "zmnouxfd",  # iclr26, 5 seeds
        "lbrdiv": ["e7ui3d6b", "ua0spahs", "95xkc9e3", "6q6ixj7x", "1pr9nuhi"],  # iclr26, 5 seeds
        "comedi": "zeuzmm3p",  # iclr26, 5 seeds
        "rotate": "l9q0kchn",  # iclr26, 5 seeds
        "cole": "162sq4py",  # iclr26, 5 seeds
        "trajedi": ["1v1ofunu", "p6uvw44b"],  # iclr26, 5 seeds
    },
    "overcooked-v1/forced_coord": {
        "fcp": "n57xsw4x",  # iclr26, 5 seeds
        "brdiv": "1aj5wy5a",  # iclr26, 5 seeds
        "lbrdiv": ["otb5ba9q", "4al8en97"],  # iclr26, 5 seeds
        "comedi": "e6p1h9f5",  # iclr26, 5 seeds
        "rotate": "ci72un83",  # iclr26, 5 seeds
        "cole": "o9x9z9bb",  # iclr26, 5 seeds
        "trajedi": ["awrppuvi", "qhszyk5x"],  # iclr26, 5 seeds
    },
    "overcooked-v1/counter_circuit": {
        "fcp": "5v7kd2ok",  # iclr26, 5 seeds
        "brdiv": "f8tvok12",  # iclr26, 5 seeds
        "lbrdiv": ["5iyo3iw6", "2r9gen4f", "jcg3b9c0", "ct5luwgo", "u021qxlz"],  # iclr26, 5 seeds
        "comedi": "um47gdei",  # iclr26, 5 seeds
        "rotate": "8nsgv1wk",  # iclr26, 5 seeds
        "cole": "upb63g1n",  # iclr26, 5 seeds
        "trajedi": ["pmecpq57", "8ritukil"],  # iclr26, 5 seeds
    },
    "mini-hanabi": {
        "fcp": "c5kukiyx",  # neurips
        "brdiv": "0az0sa6t",  # neurips
        "lbrdiv": ["wv7j92rh", "fig22cpk"],  # neurips; 2+3 split at same TRAIN_SEED=20374 -> 3 seeds, not 5
        "comedi": "acp6wglt",  # neurips
        "rotate": "mw1jdo7s",  # neurips
        "cole": "ffb5g4tu",  # iclr26, 5 seeds
        "trajedi": "qj2yxrp2",  # iclr26, 5 seeds
    },
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
        "rotate": "44c1kwu0",
        "cole": "pr0fwbdp",
        "trajedi": "13umuekr",
    },
    "lbf/lbf_12x12": {
        "ppo_ego": "k2giuu4l", 
        "liam": "yibcruuz", 
        "meliba": "6jdo5rjv",
        "fcp": "nivg4xvf", 
        "brdiv": "fdg6dw1n", 
        "lbrdiv": "y23unh8y",
        "comedi": "7e9yf5zg",
        "rotate": "csg80xwm",
        "cole": "kbvghubr",
        "trajedi": "ik9juu5l",
    },
    "overcooked-v1/cramped_room": {
        "ppo_ego": "vexvuss8", 
        "liam": "zz9lkwdz", 
        "meliba": "dva0ffdq",
        "fcp": "e23khyjt", 
        "brdiv": "19gmzemf", 
        "lbrdiv": "w8abf056",
        "comedi": "vt0xnwxc",
        "rotate": "aki2rypl",
        "cole": "jg6700d6",
        "trajedi": "okerlren",
    },
    "overcooked-v1/coord_ring": {
        "ppo_ego": "qeafl8r7", 
        "liam": "pbq863zp", 
        "meliba": "i532vemb",
        "fcp": "fubwmomo", 
        "brdiv": "wgapxysb", 
        "lbrdiv": "eu0g1orm",
        "comedi": "xeikmue5", 
        "rotate": "df4m613k",
        "cole": "irstlaiv",
        "trajedi": "awdjju8b",
    },
    "overcooked-v1/asymm_advantages": {
        "ppo_ego": "w2iy974t",
        "liam": "aytvhvup",
        "meliba": "nat9mb01",
        "fcp": "if4vnx8p",
        "brdiv": "h51mmj4h",
        "lbrdiv": "ysp4xt5x",
        "comedi": "z3prs3tb",
        "rotate": "4euj1b9o",
        "cole": "gsy9hwi6",
        "trajedi": "geyaapnl",
    },
    "overcooked-v1/counter_circuit": {
        # no ppo_ego / liam / meliba sweeps were run for this task
        "fcp": "ip6k5tfx",
        "brdiv": "m1fe94fp",
        "lbrdiv": "pka05plw",
        "comedi": "2lf5fd8x",
        "rotate": "bgxh7r5y",
        "cole": "mso4wpns",
        "trajedi": "ujckbg0k",
    },
    "overcooked-v1/forced_coord": {
        "ppo_ego": "gg5rax9p",
        "liam": "a1aewb0l",
        "meliba": "ckydg4og",
        "fcp": "eh469hgb",
        "brdiv": "9gerqh3l",
        "lbrdiv": "zmtqkcqa",
        "comedi": "4zysz591",
        "rotate": "fx48ltux",
        "cole": "hqdudasd",
        "trajedi": "nze0cuft",
    },
    "mini-hanabi": {
        "ppo_ego": "y2w21bej",
        "liam": "8t2so38u", 
        "meliba": "q4z3szuh",
        "fcp": "oku0yyg0", 
        "brdiv": "wnnhav1m", 
        "lbrdiv": "uvvpc05r",
        "comedi": "s745q3lg",
        "rotate": "ehxr5cyx",
        "cole": "158to6y5",
        "trajedi": "4uw6liu5",
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
# Tasks included in this paper revision's figures (the NeurIPS task set). Runs for other
# tasks stay in the tables above; pass --tasks explicitly to plot them.
PAPER_TASKS = [
    "lbf/lbf_12x12",
    "lbf/lbf_7x7_nolevels",
    "mini-hanabi",
    "overcooked-v1/coord_ring",
    "overcooked-v1/cramped_room",
]


def paper_tasks(*run_tables):
    """PAPER_TASKS (in order) that appear in at least one of the given run tables."""
    available = set()
    for table in run_tables:
        available |= set(table)
    return [t for t in PAPER_TASKS if t in available]


TASK_TO_PLOT_TITLE = {
    "lbf/lbf_7x7_nolevels": "LBF 7x7",
    "lbf/lbf_12x12": "LBF 12x12",
    "overcooked-v1/cramped_room": "Cramped Room (Overcooked)",
    "overcooked-v1/asymm_advantages": "Asymmetric Advantages (Overcooked)",
    "overcooked-v1/forced_coord": "Forced Coordination (Overcooked)",
    "overcooked-v1/counter_circuit": "Counter Circuit (Overcooked)",
    "overcooked-v1/coord_ring": "Coordination Ring (Overcooked)",
    "hanabi": "Hanabi",
    "mini-hanabi": "Mini-Hanabi",
}

TASK_TO_AXIS_DISPLAY_NAME = {
    "lbf/lbf_7x7_nolevels": "LBF 7x7",
    "lbf/lbf_12x12": "LBF 12x12",
    "overcooked-v1/cramped_room": "CR",
    "overcooked-v1/asymm_advantages": "AA",
    "overcooked-v1/forced_coord": "FC",
    "overcooked-v1/counter_circuit": "CC",
    "overcooked-v1/coord_ring": "CoR",
    "mini-hanabi": "Mini Hanabi"
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
    "mini-hanabi": "Mini Hanabi"
}

TASK_TO_ENV_NAME = {
    "lbf/lbf_7x7_nolevels": "lbf",
    "lbf/lbf_12x12": "lbf",
    "overcooked-v1/cramped_room": "overcooked-v1",
    "overcooked-v1/asymm_advantages": "overcooked-v1",
    "overcooked-v1/forced_coord": "overcooked-v1",
    "overcooked-v1/counter_circuit": "overcooked-v1",
    "overcooked-v1/coord_ring": "overcooked-v1",
    "mini-hanabi": "hanabi"
}

TASK_TO_METRIC_NAME = {
    "lbf/lbf_7x7_nolevels": "returned_episode_returns",
    "lbf/lbf_12x12": "returned_episode_returns",
    "overcooked-v1/cramped_room": "returned_episode_returns",
    "overcooked-v1/asymm_advantages": "returned_episode_returns",
    "overcooked-v1/forced_coord": "returned_episode_returns",
    "overcooked-v1/counter_circuit": "returned_episode_returns",
    "overcooked-v1/coord_ring": "returned_episode_returns",
    "mini-hanabi": "returned_episode_returns",
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

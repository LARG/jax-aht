import jax
import omegaconf
from evaluation.heldout_evaluator import load_heldout_set
from envs import make_env

TASK_TO_ENV_NAME = {
    "lbf": "lbf",
    "overcooked-v1/cramped_room": "overcooked-v1",
    "overcooked-v1/asymm_advantages": "overcooked-v1",
    "overcooked-v1/forced_coord": "overcooked-v1",
    "overcooked-v1/counter_circuit": "overcooked-v1",
    "overcooked-v1/coord_ring": "overcooked-v1",
}

TASK_TO_PLOT_TITLE = {
    "lbf": "Level-Based Foraging",
    "overcooked-v1/cramped_room": "Cramped Room (Overcooked)",
    "overcooked-v1/asymm_advantages": "Asymmetric Advantages (Overcooked)",
    "overcooked-v1/forced_coord": "Forced Coordination (Overcooked)",
    "overcooked-v1/counter_circuit": "Counter Circuit (Overcooked)",
    "overcooked-v1/coord_ring": "Coordination Ring (Overcooked)",
}

TASK_TO_AXIS_DISPLAY_NAME = {
    "lbf": "LBF",
    "overcooked-v1/cramped_room": "CR",
    "overcooked-v1/asymm_advantages": "AA",
    "overcooked-v1/forced_coord": "FC",
    "overcooked-v1/counter_circuit": "CC",
    "overcooked-v1/coord_ring": "CoR",
}

TASK_TO_METRIC_NAME = {
    "lbf": "percent_eaten",
    "overcooked-v1/cramped_room": "base_return",
    "overcooked-v1/asymm_advantages": "base_return",
    "overcooked-v1/forced_coord": "base_return",
    "overcooked-v1/counter_circuit": "base_return",
    "overcooked-v1/coord_ring": "base_return",
}

OE_BASELINES = { # path_to_method_directory: (method_type, display_name)
    "open_ended_minimax/paper-v0": ("open_ended", "Minimax"),
}

TEAMMATE_GEN_BASELINES = { # path_to_method_directory: (method_type, display_name)
    "fcp/paper-v0": ("teammate_generation", "FCP"),
}


GLOBAL_HELDOUT_CONFIG = omegaconf.OmegaConf.load("evaluation/configs/global_heldout_settings.yaml")
CACHE_FILENAME = "cached_summary_metrics.pkl"
HELDOUT_CURVES_CACHE_FILENAME = "cached_heldout_curves.pkl"
TITLE_FONTSIZE = 20
AXIS_LABEL_FONTSIZE = 18
LEGEND_FONTSIZE = 14

def get_heldout_agents(task_name, task_config_path):
    rng = jax.random.PRNGKey(0)
    heldout_cfg = GLOBAL_HELDOUT_CONFIG["heldout_set"][task_name]
    env_config = omegaconf.OmegaConf.load(task_config_path)
    env_name = env_config["ENV_NAME"]
    env_kwargs = env_config["ENV_KWARGS"]

    env = make_env(env_name, env_kwargs)
    heldout_agents = load_heldout_set(heldout_cfg, env, task_name, env_kwargs, rng)

    return heldout_agents

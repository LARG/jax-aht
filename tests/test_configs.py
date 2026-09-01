from pathlib import Path

import pytest
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
YAML_ROOTS = (
    "agents",
    "ego_agent_training",
    "evaluation",
    "marl",
    "open_ended_training",
    "teammate_generation",
)
HYDRA_CONFIGS = (
    ("marl/configs", "base_config_marl"),
    ("teammate_generation/configs", "base_config_teammate"),
    ("ego_agent_training/configs", "base_config_ego"),
    ("open_ended_training/configs", "base_config_oel"),
    ("evaluation/configs", "heldout_ego"),
    ("evaluation/configs", "heldout_xp"),
)


def test_yaml_configs_parse():
    config_paths = []
    for root_name in YAML_ROOTS:
        root = REPO_ROOT / root_name
        config_paths.extend(root.rglob("*.yaml"))
        config_paths.extend(root.rglob("*.yml"))

    assert config_paths
    for config_path in config_paths:
        yaml.safe_load(config_path.read_text())


@pytest.mark.parametrize(("config_dir", "config_name"), HYDRA_CONFIGS)
def test_entrypoint_configs_compose(config_dir, config_name):
    with initialize_config_dir(
        version_base=None,
        config_dir=str(REPO_ROOT / config_dir),
    ):
        config = compose(config_name=config_name)
        OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

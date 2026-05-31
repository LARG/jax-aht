from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from agents.bc.bc_lstm import BCLSTMConfig


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BCCheckpointConfig:
    model: BCLSTMConfig
    lbf_feature_mode: str = "none"
    lbf_grid_size: int = 0
    lbf_num_food: int = 0

    @property
    def obs_dim(self) -> int:
        return self.model.obs_dim

    @property
    def action_dim(self) -> int:
        return self.model.action_dim

    @property
    def preprocess_dim(self) -> int:
        return self.model.preprocess_dim

    @property
    def lstm_dim(self) -> int:
        return self.model.lstm_dim

    @property
    def postprocess_dim(self) -> int:
        return self.model.postprocess_dim

    @property
    def dropout_rate(self) -> float:
        return self.model.dropout_rate


def load_bc_config(path: str) -> BCCheckpointConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return BCCheckpointConfig(
        model=BCLSTMConfig(
            obs_dim=int(raw["obs_dim"]),
            action_dim=int(raw["action_dim"]),
            preprocess_dim=int(raw.get("preprocess_dim", 256)),
            lstm_dim=int(raw.get("lstm_dim", 128)),
            postprocess_dim=int(raw.get("postprocess_dim", 64)),
            dropout_rate=float(raw.get("dropout_rate", 0.0)),
        ),
        lbf_feature_mode=str(raw.get("lbf_feature_mode", "none")),
        lbf_grid_size=int(raw.get("lbf_grid_size", 0)),
        lbf_num_food=int(raw.get("lbf_num_food", 0)),
    )


def with_lbf_env_config(config: BCCheckpointConfig, env_kwargs) -> BCCheckpointConfig:
    if config.lbf_feature_mode == "none":
        return config
    return replace(
        config,
        lbf_grid_size=env_kwargs["grid_size"],
        lbf_num_food=env_kwargs["num_food"],
    )


def resolve_first_existing_path(paths: list[str]) -> str:
    for path in paths:
        if (REPO_ROOT / path).exists() or Path(path).exists():
            return path
    raise FileNotFoundError(
        f"None of the candidate checkpoint paths exist: {paths}"
    )

from pathlib import Path

import yaml

from agents.bc.bc_lstm import BCLSTMConfig


REPO_ROOT = Path(__file__).resolve().parents[1]

LBF_CONFIGS = {
    "grid7_food3_nolevels": {
        "grid_size": 7,
        "num_food": 3,
        "different_levels": False,
    },
    "grid7_food3_levels": {
        "grid_size": 7,
        "num_food": 3,
        "different_levels": True,
    },
    "grid12_food6_nolevels": {
        "grid_size": 12,
        "num_food": 6,
        "different_levels": False,
    },
    "grid12_food6_levels": {
        "grid_size": 12,
        "num_food": 6,
        "different_levels": True,
    },
}


def load_bc_config(path: str) -> BCLSTMConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return BCLSTMConfig(
        obs_dim=int(raw["obs_dim"]),
        action_dim=int(raw["action_dim"]),
        preprocess_dim=int(raw.get("preprocess_dim", 256)),
        lstm_dim=int(raw.get("lstm_dim", 128)),
        postprocess_dim=int(raw.get("postprocess_dim", 64)),
        dropout_rate=float(raw.get("dropout_rate", 0.0)),
        lbf_feature_mode=str(raw.get("lbf_feature_mode", "none")),
        lbf_grid_size=int(raw.get("lbf_grid_size", 0)),
        lbf_num_food=int(raw.get("lbf_num_food", 0)),
    )


def resolve_first_existing_path(paths: list[str]) -> str:
    for path in paths:
        if (REPO_ROOT / path).exists() or Path(path).exists():
            return path
    raise FileNotFoundError(
        f"None of the candidate checkpoint paths exist: {paths}"
    )

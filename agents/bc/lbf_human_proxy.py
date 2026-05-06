"""Loader for the LBF BC-LSTM human-proxy policies hosted at
huggingface.co/datasets/jaxaht/lbf-human-proxy-policies.

Layouts: "lbf_7x7" (grid7_food3_levels) and "lbf_12x12" (grid12_food6_levels).
The two HF entries point at the same checkpoint (same SHA256) — one BC was
trained on combined data. `BCLSTMPolicyWrapper.get_action` zero-pads obs from
15→24 automatically when the env emits the smaller 7x7 obs.
"""
import shutil
from pathlib import Path

import yaml

from agents.bc.bc_lstm import BCLSTMAgent, BCLSTMConfig, BCLSTMPolicyWrapper

HF_REPO_ID = "jaxaht/lbf-human-proxy-policies"
HF_REPO_TYPE = "dataset"
SUPPORTED_LAYOUTS = ("lbf_7x7", "lbf_12x12")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_BASE = _REPO_ROOT / "eval_teammates" / "lbf"


def _ensure_cached(layout: str) -> tuple[str, str]:
    """Make sure policy.safetensors + policy.yaml for `layout` are mirrored under
    `eval_teammates/lbf/<layout>/human_proxy/`. Returns (weights_path, yaml_path)."""
    if layout not in SUPPORTED_LAYOUTS:
        raise ValueError(
            f"Unknown LBF layout '{layout}'; expected one of {SUPPORTED_LAYOUTS}"
        )
    local_dir = _LOCAL_BASE / layout / "human_proxy"
    local_dir.mkdir(parents=True, exist_ok=True)
    targets = {
        "policy.safetensors": local_dir / "policy.safetensors",
        "policy.yaml":        local_dir / "policy.yaml",
    }
    missing = {n: t for n, t in targets.items() if not t.exists()}
    if missing:
        from huggingface_hub import hf_hub_download
        for fname, dst in missing.items():
            src = hf_hub_download(
                HF_REPO_ID,
                filename=f"{layout}/{fname}",
                repo_type=HF_REPO_TYPE,
            )
            shutil.copyfile(src, dst)
    return str(targets["policy.safetensors"]), str(targets["policy.yaml"])


def load_lbf_human_proxy(layout: str):
    """Return (BCLSTMPolicyWrapper, params) for the LBF human-proxy on `layout`.

    Use as:
        policy, params = load_lbf_human_proxy("lbf_12x12")
        hstate = policy.init_hstate(batch_size=NUM_ENVS)
        action, hstate = policy.get_action(
            params=params, obs=..., done=..., avail_actions=...,
            hstate=hstate, rng=..., test_mode=True,
        )
    """
    weights_path, yaml_path = _ensure_cached(layout)
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    config = BCLSTMConfig(
        obs_dim=int(raw["obs_dim"]),
        action_dim=int(raw["action_dim"]),
        preprocess_dim=int(raw.get("preprocess_dim", 256)),
        lstm_dim=int(raw.get("lstm_dim", 128)),
        postprocess_dim=int(raw.get("postprocess_dim", 64)),
        dropout_rate=float(raw.get("dropout_rate", 0.0)),
    )
    agent = BCLSTMAgent(config, weight_path=weights_path)
    policy = BCLSTMPolicyWrapper(config)
    return policy, agent.params

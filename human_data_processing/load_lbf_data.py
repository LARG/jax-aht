"""Load processed human episode data as JAX arrays for behavior cloning.

See human_data_processing/README.md for full documentation on data format,
filtering criteria, available configs, and usage examples.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from safetensors.numpy import load_file as st_load
from safetensors import safe_open


# ---------- Data structures ----------

class BCDataset(NamedTuple):
    """Flat BC dataset — all episodes concatenated along the time axis.

    Fields:
        obs:           (N, obs_dim)  float32 — VectorObserver obs before each action
        actions:       (N,)          int32   — human action taken (0=noop,1=up,2=down,3=left,4=right,5=load)
        ai_actions:    (N,)          int32   — AI teammate action
        rewards:       (N,)          float32 — per-step reward for the human agent
        dones:         (N,)          bool    — True on the last timestep of each episode
        avail_actions: (N, 6)        bool    — which of the 6 actions are valid
        episode_ids:   (N,)          int32   — episode index each timestep belongs to
    """
    obs: jnp.ndarray
    actions: jnp.ndarray
    ai_actions: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray
    avail_actions: jnp.ndarray
    episode_ids: jnp.ndarray


class BCDatasetPadded(NamedTuple):
    """Padded BC dataset — each episode is a row, padded to max episode length.

    Fields:
        obs:           (E, T, obs_dim)  float32
        actions:       (E, T)           int32
        ai_actions:    (E, T)           int32
        rewards:       (E, T)           float32
        dones:         (E, T)           bool    — True on last real step AND all padding
        avail_actions: (E, T, 6)        bool
        mask:          (E, T)           bool    — True where data is real, False where padded
        agent_types:   list[str]               — agent type label per episode (length E)
    """
    obs: jnp.ndarray
    actions: jnp.ndarray
    ai_actions: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray
    avail_actions: jnp.ndarray
    mask: jnp.ndarray
    agent_types: list


# ---------- Loading helpers ----------

PROCESSED_DIR = Path(__file__).parent / "processed"


def _load_safetensors(path: str) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Load tensors and metadata from a safetensors file."""
    tensors = st_load(path)
    with safe_open(path, framework="numpy") as f:
        metadata = f.metadata()
    return tensors, metadata or {}


def _filter_by_uncertainty(tensors: dict, metadata: dict, format: str) -> tuple[dict, dict]:
    """Remove episodes flagged as step0_uncertain."""
    uncertain = json.loads(metadata.get("step0_uncertain", "[]"))
    keep = [i for i, u in enumerate(uncertain) if not u]

    if len(keep) == len(uncertain):
        return tensors, metadata

    if format == "flat":
        episode_ids = tensors["episode_ids"]
        keep_set = set(keep)
        mask = np.array([eid in keep_set for eid in episode_ids])

        # Remap episode ids to be contiguous
        id_map = {old: new for new, old in enumerate(keep)}
        new_tensors = {}
        for k, v in tensors.items():
            if k == "episode_ids":
                new_tensors[k] = np.array([id_map[eid] for eid in v[mask]], dtype=np.int32)
            else:
                new_tensors[k] = v[mask]

        # Update metadata lists
        new_meta = dict(metadata)
        for list_key in ["agent_types", "session_ids", "step0_reconstructed", "step0_uncertain"]:
            if list_key in metadata:
                vals = json.loads(metadata[list_key])
                new_meta[list_key] = json.dumps([vals[i] for i in keep])
        new_meta["num_episodes"] = str(len(keep))
        new_meta["num_timesteps"] = str(int(np.sum(mask)))
        return new_tensors, new_meta

    else:  # padded
        keep_idx = np.array(keep)
        new_tensors = {k: v[keep_idx] for k, v in tensors.items()}

        new_meta = dict(metadata)
        for list_key in ["agent_types", "session_ids", "step0_reconstructed", "step0_uncertain"]:
            if list_key in metadata:
                vals = json.loads(metadata[list_key])
                new_meta[list_key] = json.dumps([vals[i] for i in keep])
        new_meta["num_episodes"] = str(len(keep))
        return new_tensors, new_meta


# ---------- Public API ----------

def load_bc_data(config_name: str, exclude_uncertain: bool = False) -> BCDataset:
    """Load a flat BC dataset for one env config.

    Args:
        config_name: One of "grid7_food3_nolevels", "grid7_food3_levels",
                     "grid12_food6_nolevels", "grid12_food6_levels".
        exclude_uncertain: If True, drop episodes whose step-0 reconstruction
                           is flagged as uncertain.

    Returns:
        BCDataset with all episodes concatenated.
    """
    path = str(PROCESSED_DIR / config_name / "flat.safetensors")
    tensors, metadata = _load_safetensors(path)
    if exclude_uncertain:
        tensors, metadata = _filter_by_uncertainty(tensors, metadata, "flat")

    return BCDataset(
        obs=jnp.array(tensors["obs"]),
        actions=jnp.array(tensors["actions"]),
        ai_actions=jnp.array(tensors["ai_actions"]),
        rewards=jnp.array(tensors["rewards"]),
        dones=jnp.array(tensors["dones"]),
        avail_actions=jnp.array(tensors["avail_actions"]),
        episode_ids=jnp.array(tensors["episode_ids"]),
    )


def load_bc_data_by_agent(config_name: str,
                          exclude_uncertain: bool = False) -> dict[str, BCDataset]:
    """Load flat BC datasets grouped by AI teammate type.

    Returns:
        Dict mapping agent_type string -> BCDataset.
    """
    path = str(PROCESSED_DIR / config_name / "flat.safetensors")
    tensors, metadata = _load_safetensors(path)
    if exclude_uncertain:
        tensors, metadata = _filter_by_uncertainty(tensors, metadata, "flat")

    agent_types = json.loads(metadata["agent_types"])
    episode_ids = tensors["episode_ids"]

    # Group episode indices by agent type
    agent_to_ep_ids: dict[str, set[int]] = defaultdict(set)
    for ep_idx, at in enumerate(agent_types):
        agent_to_ep_ids[at].add(ep_idx)

    result = {}
    for agent, ep_ids in sorted(agent_to_ep_ids.items()):
        mask = np.array([eid in ep_ids for eid in episode_ids])
        id_map = {old: new for new, old in enumerate(sorted(ep_ids))}
        result[agent] = BCDataset(
            obs=jnp.array(tensors["obs"][mask]),
            actions=jnp.array(tensors["actions"][mask]),
            ai_actions=jnp.array(tensors["ai_actions"][mask]),
            rewards=jnp.array(tensors["rewards"][mask]),
            dones=jnp.array(tensors["dones"][mask]),
            avail_actions=jnp.array(tensors["avail_actions"][mask]),
            episode_ids=jnp.array([id_map[eid] for eid in episode_ids[mask]], dtype=np.int32),
        )
    return result


def load_bc_data_padded(config_name: str,
                        exclude_uncertain: bool = False) -> BCDatasetPadded:
    """Load a padded BC dataset for one env config (for recurrent models).

    Args:
        config_name: One of the 4 config names.
        exclude_uncertain: If True, drop uncertain-reconstruction episodes.

    Returns:
        BCDatasetPadded with shape (num_episodes, max_ep_length, ...).
    """
    path = str(PROCESSED_DIR / config_name / "padded.safetensors")
    tensors, metadata = _load_safetensors(path)
    if exclude_uncertain:
        tensors, metadata = _filter_by_uncertainty(tensors, metadata, "padded")

    agent_types = json.loads(metadata["agent_types"])

    return BCDatasetPadded(
        obs=jnp.array(tensors["obs"]),
        actions=jnp.array(tensors["actions"]),
        ai_actions=jnp.array(tensors["ai_actions"]),
        rewards=jnp.array(tensors["rewards"]),
        dones=jnp.array(tensors["dones"]),
        avail_actions=jnp.array(tensors["avail_actions"]),
        mask=jnp.array(tensors["mask"]),
        agent_types=agent_types,
    )


# Backward-compatible names used by human_data_processing/README.md.
load_lbf_data = load_bc_data
load_lbf_data_by_agent = load_bc_data_by_agent
load_lbf_data_padded = load_bc_data_padded


# ---------- Quick test ----------

if __name__ == "__main__":
    for cfg in ["grid7_food3_nolevels", "grid7_food3_levels",
                "grid12_food6_nolevels", "grid12_food6_levels"]:
        data = load_bc_data(cfg)
        padded = load_bc_data_padded(cfg)
        print(f"{cfg}:")
        print(f"  Flat:   obs {data.obs.shape}, actions {data.actions.shape}")
        print(f"  Padded: obs {padded.obs.shape}, mask {padded.mask.shape}")
        print(f"  Agent types: {len(set(padded.agent_types))}")
        print()

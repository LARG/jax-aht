"""Load processed human episode data as JAX arrays for behavior cloning.

Usage:
    from human_data.load_bc_data import load_bc_data, load_bc_data_by_agent

    # --- Flat dataset (all episodes concatenated) ---
    data = load_bc_data("grid7_food3_nolevels")
    # data is a BCDataset (NamedTuple):
    #   obs:            jnp.ndarray (N, obs_dim)    float32  — observations before each action
    #   actions:        jnp.ndarray (N,)            int32    — human actions (0-5)
    #   ai_actions:     jnp.ndarray (N,)            int32    — AI teammate actions
    #   rewards:        jnp.ndarray (N,)            float32  — per-step human reward
    #   dones:          jnp.ndarray (N,)            bool     — True on last step of each episode
    #   avail_actions:  jnp.ndarray (N, 6)          bool     — valid action mask
    #   episode_ids:    jnp.ndarray (N,)            int32    — which episode each timestep belongs to
    #
    # where N = total timesteps across all episodes, obs_dim = 15 (7x7) or 24 (12x12)

    # --- Grouped by agent type ---
    by_agent = load_bc_data_by_agent("grid7_food3_nolevels")
    # dict mapping agent_type (str) -> BCDataset

    # --- Padded episodes (for recurrent / sequence models) ---
    padded = load_bc_data_padded("grid7_food3_nolevels")
    # padded is a BCDatasetPadded (NamedTuple):
    #   obs:            jnp.ndarray (E, T, obs_dim)  float32
    #   actions:        jnp.ndarray (E, T)            int32
    #   ai_actions:     jnp.ndarray (E, T)            int32
    #   rewards:        jnp.ndarray (E, T)            float32
    #   dones:          jnp.ndarray (E, T)            bool
    #   avail_actions:  jnp.ndarray (E, T, 6)         bool
    #   mask:           jnp.ndarray (E, T)             bool  — True for real timesteps, False for padding
    #   agent_types:    list[str]                             — agent type label per episode
    #
    # where E = number of episodes, T = max episode length (padded)

Available config names:
    "grid7_food3_nolevels"   — 7x7 grid, 3 food, same levels     (obs_dim=15)
    "grid7_food3_levels"     — 7x7 grid, 3 food, different levels (obs_dim=15)
    "grid12_food6_nolevels"  — 12x12 grid, 6 food, same levels   (obs_dim=24)
    "grid12_food6_levels"    — 12x12 grid, 6 food, different levels (obs_dim=24)
"""

import pickle
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple, Optional

import jax.numpy as jnp
import numpy as np


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
        dones:         (E, T)           bool
        avail_actions: (E, T, 6)        bool
        mask:          (E, T)           bool  — True where data is real, False where padded
        agent_types:   list[str]              — agent type label per episode (length E)
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


def _load_raw(config_name: str, exclude_uncertain: bool = False) -> list[dict]:
    """Load the raw list of episode dicts from a config's trajectories.pkl."""
    pkl_path = PROCESSED_DIR / config_name / "trajectories.pkl"
    with open(pkl_path, "rb") as f:
        episodes = pickle.load(f)
    if exclude_uncertain:
        episodes = [ep for ep in episodes if not ep["step0_uncertain"]]
    return episodes


def _episodes_to_flat(episodes: list[dict]) -> BCDataset:
    """Concatenate a list of episode dicts into a flat BCDataset."""
    obs_parts = []
    act_parts = []
    ai_act_parts = []
    rew_parts = []
    done_parts = []
    avail_parts = []
    eid_parts = []

    for i, ep in enumerate(episodes):
        T = len(ep["actions"])
        obs_parts.append(ep["obs"])
        act_parts.append(ep["actions"])
        ai_act_parts.append(ep["ai_actions"])
        rew_parts.append(ep["rewards"])
        done_parts.append(ep["dones"])
        avail_parts.append(ep["avail_actions"])
        eid_parts.append(np.full(T, i, dtype=np.int32))

    return BCDataset(
        obs=jnp.array(np.concatenate(obs_parts)),
        actions=jnp.array(np.concatenate(act_parts)),
        ai_actions=jnp.array(np.concatenate(ai_act_parts)),
        rewards=jnp.array(np.concatenate(rew_parts)),
        dones=jnp.array(np.concatenate(done_parts)),
        avail_actions=jnp.array(np.concatenate(avail_parts)),
        episode_ids=jnp.array(np.concatenate(eid_parts)),
    )


def _episodes_to_padded(episodes: list[dict]) -> BCDatasetPadded:
    """Pad episodes to uniform length and stack into a BCDatasetPadded."""
    max_len = max(len(ep["actions"]) for ep in episodes)
    E = len(episodes)
    obs_dim = episodes[0]["obs"].shape[1]

    obs = np.zeros((E, max_len, obs_dim), dtype=np.float32)
    actions = np.zeros((E, max_len), dtype=np.int32)
    ai_actions = np.zeros((E, max_len), dtype=np.int32)
    rewards = np.zeros((E, max_len), dtype=np.float32)
    dones = np.zeros((E, max_len), dtype=bool)
    avail_actions = np.zeros((E, max_len, 6), dtype=bool)
    mask = np.zeros((E, max_len), dtype=bool)
    agent_types = []

    for i, ep in enumerate(episodes):
        T = len(ep["actions"])
        obs[i, :T] = ep["obs"]
        actions[i, :T] = ep["actions"]
        ai_actions[i, :T] = ep["ai_actions"]
        rewards[i, :T] = ep["rewards"]
        dones[i, :T] = ep["dones"]
        avail_actions[i, :T] = ep["avail_actions"]
        mask[i, :T] = True
        agent_types.append(ep["agent_type"])

    return BCDatasetPadded(
        obs=jnp.array(obs),
        actions=jnp.array(actions),
        ai_actions=jnp.array(ai_actions),
        rewards=jnp.array(rewards),
        dones=jnp.array(dones),
        avail_actions=jnp.array(avail_actions),
        mask=jnp.array(mask),
        agent_types=agent_types,
    )


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
    episodes = _load_raw(config_name, exclude_uncertain=exclude_uncertain)
    return _episodes_to_flat(episodes)


def load_bc_data_by_agent(config_name: str,
                          exclude_uncertain: bool = False) -> dict[str, BCDataset]:
    """Load flat BC datasets grouped by AI teammate type.

    Returns:
        Dict mapping agent_type string -> BCDataset.
    """
    episodes = _load_raw(config_name, exclude_uncertain=exclude_uncertain)
    grouped = defaultdict(list)
    for ep in episodes:
        grouped[ep["agent_type"]].append(ep)
    return {agent: _episodes_to_flat(eps) for agent, eps in sorted(grouped.items())}


def load_bc_data_padded(config_name: str,
                        exclude_uncertain: bool = False) -> BCDatasetPadded:
    """Load a padded BC dataset for one env config (for recurrent models).

    Args:
        config_name: One of the 4 config names.
        exclude_uncertain: If True, drop uncertain-reconstruction episodes.

    Returns:
        BCDatasetPadded with shape (num_episodes, max_ep_length, ...).
    """
    episodes = _load_raw(config_name, exclude_uncertain=exclude_uncertain)
    return _episodes_to_padded(episodes)


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

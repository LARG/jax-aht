"""Process raw Prolific episode JSON files into pickle files for behavior cloning.

For each of the 4 LBF env configs, produces a .pkl file containing a list of
episode dicts with JAX-compatible arrays matching the VectorObserver format
used during IPPO training.

Episodes without a recorded step 0 have their initial state reconstructed by
reversing the actions at step 1. These are flagged with ``step0_reconstructed``
and, when the reversal may be incorrect (blocked move indistinguishable from
real move), ``step0_uncertain``.

Usage:
    python human_data/process_episodes.py
"""

import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np


# ---------- Env config helpers ----------

def config_key(grid_size: int, num_fruits: int, different_levels: bool) -> str:
    """Human-readable directory name for an env config."""
    levels_str = "levels" if different_levels else "nolevels"
    return f"grid{grid_size}_food{num_fruits}_{levels_str}"


# ---------- Observation reconstruction ----------

def state_to_obs(state: dict, agent_idx: int, num_agents: int, num_food: int) -> np.ndarray:
    """Reconstruct the VectorObserver observation for a given agent.

    With fov == grid_size (full observability), the VectorObserver returns
    absolute positions.  The observation vector has length 3*(num_food + num_agents):
        [food_0_row, food_0_col, food_0_level, ...,   # num_food triplets
         self_row, self_col, self_level,                # current agent
         other_0_row, other_0_col, other_0_level, ...]  # num_agents-1 triplets
    Eaten food is encoded as (-1, -1, 0).
    """
    obs = []

    # Food triplets
    for i in range(num_food):
        if state["food_eaten"][i]:
            obs.extend([-1, -1, 0])
        else:
            obs.extend([
                state["food_positions"][i][0],
                state["food_positions"][i][1],
                state["food_levels"][i],
            ])

    # Current agent triplet
    obs.extend([
        state["agent_positions"][agent_idx][0],
        state["agent_positions"][agent_idx][1],
        state["agent_levels"][agent_idx],
    ])

    # Other agents
    for i in range(num_agents):
        if i == agent_idx:
            continue
        obs.extend([
            state["agent_positions"][i][0],
            state["agent_positions"][i][1],
            state["agent_levels"][i],
        ])

    return np.array(obs, dtype=np.float32)


def compute_action_mask(state: dict, agent_idx: int, num_agents: int, grid_size: int) -> np.ndarray:
    """Compute the 6-element boolean action mask for a given agent.

    Actions: 0=noop, 1=up, 2=down, 3=left, 4=right, 5=load
    Moves:   noop=(0,0), up=(-1,0), down=(1,0), left=(0,-1), right=(0,1), load=(0,0)
    """
    MOVES = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]])
    agent_pos = np.array(state["agent_positions"][agent_idx])
    next_positions = agent_pos + MOVES  # (6, 2)

    mask = np.ones(6, dtype=bool)

    for action_i in range(6):
        if action_i == 5:
            # Load: allowed only if adjacent to uneaten food
            continue
        npos = next_positions[action_i]

        # Out of bounds
        if np.any(npos < 0) or np.any(npos >= grid_size):
            mask[action_i] = False
            continue

        # Occupied by another agent
        for j in range(num_agents):
            if j == agent_idx:
                continue
            if np.array_equal(npos, state["agent_positions"][j]):
                mask[action_i] = False
                break

        if not mask[action_i]:
            continue

        # Occupied by uneaten food
        for fi in range(len(state["food_positions"])):
            if not state["food_eaten"][fi] and np.array_equal(npos, state["food_positions"][fi]):
                mask[action_i] = False
                break

    # Load action: check adjacency to any uneaten food
    has_adj_food = False
    for fi in range(len(state["food_positions"])):
        if state["food_eaten"][fi]:
            continue
        dist = np.abs(np.array(state["food_positions"][fi]) - agent_pos)
        if np.sum(dist) == 1:
            has_adj_food = True
            break
    if not has_adj_food:
        mask[5] = False

    return mask


# ---------- Step-0 reconstruction ----------

# noop, up, down, left, right, load
MOVE_DELTAS = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]])


def reconstruct_step0(state1: dict, human_action: int, ai_action: int,
                       grid_size: int, num_food: int) -> tuple[dict, bool]:
    """Reconstruct the step-0 state by reversing actions applied at step 1.

    Returns (reconstructed_state, uncertain) where *uncertain* is True when the
    reversal might be wrong because we cannot detect whether the action was
    blocked (e.g. agent at grid edge tried to move further out — reverse gives
    a valid interior cell).
    """
    uncertain = False

    agent_positions = [list(p) for p in state1["agent_positions"]]

    for agent_idx, action in enumerate([human_action, ai_action]):
        delta = MOVE_DELTAS[action]
        if delta[0] == 0 and delta[1] == 0:
            # noop or load — agent didn't move, position is correct
            continue

        cur_pos = np.array(agent_positions[agent_idx])
        reversed_pos = cur_pos - delta

        # Detectable block: reversed position out of bounds
        if np.any(reversed_pos < 0) or np.any(reversed_pos >= grid_size):
            # Agent was definitely blocked — keep state1 position
            continue

        # Detectable block: reversed position on uneaten food
        on_food = False
        for fi in range(num_food):
            if not state1["food_eaten"][fi] and \
               np.array_equal(reversed_pos, state1["food_positions"][fi]):
                on_food = True
                break
        if on_food:
            continue

        # Undetectable: could be a real move or a blocked move that reverses
        # to a valid cell.  We accept the reversal but flag uncertainty.
        # Check if the agent *could* have been blocked (boundary, food, other agent).
        # If current position is at boundary in the action direction, or adjacent
        # to food/agent in that direction, the action may have been a no-op.
        blocked_pos = cur_pos + delta  # where would agent go if at reversed_pos?
        might_be_blocked = False

        # Would moving from reversed_pos have been blocked by boundary?
        if np.any(blocked_pos < 0) or np.any(blocked_pos >= grid_size):
            # No — that would be cur_pos going further, not reversed_pos moving
            pass

        # Actually, the simple heuristic: we can't distinguish real moves from
        # blocked moves when reverse gives a valid cell. Flag as uncertain
        # only when we actually change the position (movement action applied).
        uncertain = True
        agent_positions[agent_idx] = reversed_pos.tolist()

    return {
        "agent_positions": agent_positions,
        "agent_levels": state1["agent_levels"],
        "food_positions": state1["food_positions"],
        "food_levels": state1["food_levels"],
        "food_eaten": [False] * num_food,  # step 0 always has no eaten food
        "step_count": 0,
    }, uncertain


# ---------- Episode processing ----------

def process_episode(filepath: str) -> dict | None:
    """Load a single episode JSON and convert to arrays for BC.

    Returns None only for truly empty episodes.
    Returns a dict with:
        - obs:                 (T, obs_dim)  float32  — observation BEFORE each action
        - actions:             (T,)          int32    — human actions
        - ai_actions:          (T,)          int32    — AI teammate actions
        - rewards:             (T,)          float32  — per-step reward for the human
        - dones:               (T,)          bool     — episode termination flag
        - avail_actions:       (T, 6)        bool     — action mask BEFORE each action
        - step0_reconstructed: bool  — True if step 0 was reconstructed (not recorded)
        - step0_uncertain:     bool  — True if the reconstructed step 0 may be wrong
        - agent_type:          str           — AI teammate label
        - config:              dict          — {grid_size, num_fruits, different_levels}
        - session_id:          str
        - total_rewards:       dict
        - duration:            float
    """
    with open(filepath, "r") as f:
        ep = json.load(f)

    traj = ep["trajectory"]
    if not traj:
        return None

    grid_size = ep["grid_size"]
    num_food = ep["num_fruits"]
    num_agents = 2
    human_idx = 0  # human is always agent_0

    step0_reconstructed = False
    step0_uncertain = False

    has_step0 = (traj[0]["step"] == 0)

    if not has_step0:
        # Reconstruct step 0 from step 1 by reversing actions
        step1 = traj[0]  # first entry is step 1
        reconstructed_state, uncertain = reconstruct_step0(
            step1["state"], step1["human_action"], step1["ai_action"],
            grid_size, num_food,
        )
        # Prepend a synthetic step 0 entry
        traj = [{"step": 0, "state": reconstructed_state}] + traj
        step0_reconstructed = True
        step0_uncertain = uncertain

    obs_list = []
    action_list = []
    ai_action_list = []
    reward_list = []
    done_list = []
    avail_list = []

    # Build (obs_before_action, action) pairs
    # obs for action at step t comes from state at step t-1
    for i in range(1, len(traj)):
        prev_state = traj[i - 1]["state"]
        cur_step = traj[i]

        obs = state_to_obs(prev_state, human_idx, num_agents, num_food)
        amask = compute_action_mask(prev_state, human_idx, num_agents, grid_size)

        obs_list.append(obs)
        avail_list.append(amask)
        action_list.append(cur_step["human_action"])
        ai_action_list.append(cur_step["ai_action"])
        reward_list.append(cur_step["rewards"]["agent_0"])
        # Last step is done (episode was saved because it ended)
        done_list.append(i == len(traj) - 1)

    T = len(obs_list)
    if T == 0:
        return None

    return {
        "obs": np.stack(obs_list),                           # (T, obs_dim)
        "actions": np.array(action_list, dtype=np.int32),    # (T,)
        "ai_actions": np.array(ai_action_list, dtype=np.int32),
        "rewards": np.array(reward_list, dtype=np.float32),  # (T,)
        "dones": np.array(done_list, dtype=bool),            # (T,)
        "avail_actions": np.stack(avail_list),                # (T, 6)
        "step0_reconstructed": step0_reconstructed,
        "step0_uncertain": step0_uncertain,
        "agent_type": ep["agent_type"],
        "config": {
            "grid_size": grid_size,
            "num_fruits": num_food,
            "different_levels": ep["different_levels"],
        },
        "session_id": ep["session_id"],
        "total_rewards": ep["total_rewards"],
        "duration": ep.get("duration", 0.0),
    }


# ---------- Player filtering ----------

NOOP_ACTION = 0
LOAD_ACTION = 5
REQUIRED_GAMES = 8


def load_all_raw_episodes(input_dir: Path) -> list[dict]:
    """Load all raw JSON episode files."""
    episodes = []
    for fpath in sorted(input_dir.glob("*.json")):
        with open(fpath, "r") as f:
            ep = json.load(f)
        ep["_filepath"] = str(fpath)
        episodes.append(ep)
    return episodes


def compute_player_stats(player_episodes: list[dict]) -> dict:
    """Compute engagement stats for a player across all their games.

    Returns dict with:
        num_games, total_actions, wait_count, wasted_load_count,
        wait_or_wasted_pct, wait_pct, avg_score, zero_score_games
    """
    total_actions = 0
    wait_count = 0
    wasted_load_count = 0
    scores = []

    for ep in player_episodes:
        score = ep["total_rewards"].get("agent_0", 0.0)
        scores.append(score)

        for step in ep["trajectory"]:
            action = step.get("human_action")
            if action is None:
                continue  # step 0 has no action
            total_actions += 1
            if action == NOOP_ACTION:
                wait_count += 1
            elif action == LOAD_ACTION:
                reward = step["rewards"]["agent_0"]
                if reward == 0.0:
                    wasted_load_count += 1

    wait_or_wasted = wait_count + wasted_load_count
    return {
        "num_games": len(player_episodes),
        "total_actions": total_actions,
        "wait_count": wait_count,
        "wasted_load_count": wasted_load_count,
        "wait_or_wasted_pct": wait_or_wasted / total_actions if total_actions > 0 else 0,
        "wait_pct": wait_count / total_actions if total_actions > 0 else 0,
        "avg_score": sum(scores) / len(scores) if scores else 0,
        "zero_score_games": sum(1 for s in scores if s == 0.0),
    }


def filter_players(all_episodes: list[dict]) -> tuple[set[str], dict]:
    """Determine which prolific_pids pass quality filters.

    Returns:
        (clean_pids, filter_report) where filter_report has counts per reason.
    """
    # Group by player
    by_player: dict[str, list] = defaultdict(list)
    for ep in all_episodes:
        pid = ep.get("prolific_pid", "")
        by_player[pid].append(ep)

    total_players = len(by_player)
    disqualified = set()   # didn't complete 8 games
    disengaged = set()     # flagged for low engagement
    disengage_reasons: dict[str, list[str]] = {}

    for pid, eps in by_player.items():
        if len(eps) < REQUIRED_GAMES:
            disqualified.add(pid)
            continue

        stats = compute_player_stats(eps)

        reasons = []
        if stats["wait_or_wasted_pct"] > 0.35:
            reasons.append(f"wait+wasted_load={stats['wait_or_wasted_pct']:.1%}")
        if stats["wait_pct"] > 0.15:
            reasons.append(f"wait={stats['wait_pct']:.1%}")
        if stats["avg_score"] == 0.0:
            reasons.append("avg_score=0")
        if stats["zero_score_games"] >= 5:
            reasons.append(f"zero_score_games={stats['zero_score_games']}")

        if reasons:
            disengaged.add(pid)
            disengage_reasons[pid] = reasons

    clean_pids = set(by_player.keys()) - disqualified - disengaged

    report = {
        "total_players": total_players,
        "disqualified": len(disqualified),
        "disengaged": len(disengaged),
        "clean": len(clean_pids),
        "total_episodes": len(all_episodes),
        "clean_episodes": sum(len(by_player[pid]) for pid in clean_pids),
        "disengage_reasons": disengage_reasons,
    }
    return clean_pids, report


# ---------- Main ----------

def main():
    input_dir = Path(__file__).parent / "collected_data_prolific"
    output_dir = Path(__file__).parent / "processed"
    output_dir.mkdir(exist_ok=True)

    # --- Pass 1: load all raw episodes and filter players ---
    print("Loading raw episodes...")
    all_raw = load_all_raw_episodes(input_dir)
    print(f"Found {len(all_raw)} episode files")

    clean_pids, report = filter_players(all_raw)
    print(f"\nPlayer filtering:")
    print(f"  Total players:  {report['total_players']}")
    print(f"  Disqualified (< {REQUIRED_GAMES} games): {report['disqualified']}")
    print(f"  Disengaged:     {report['disengaged']}")
    print(f"  Clean:          {report['clean']}")
    print(f"  Episodes kept:  {report['clean_episodes']} / {report['total_episodes']}")
    if report["disengage_reasons"]:
        print(f"  Disengage details:")
        for pid, reasons in sorted(report["disengage_reasons"].items()):
            print(f"    {pid}: {', '.join(reasons)}")
    print()

    # --- Pass 2: process clean episodes ---
    # Build set of filepaths for clean players
    clean_files = set()
    for ep in all_raw:
        if ep.get("prolific_pid", "") in clean_pids:
            clean_files.add(ep["_filepath"])

    grouped: dict[str, list] = defaultdict(list)
    skipped = 0
    errors = 0

    for fpath in sorted(input_dir.glob("*.json")):
        if str(fpath) not in clean_files:
            skipped += 1
            continue

        try:
            result = process_episode(str(fpath))
        except Exception as e:
            print(f"  ERROR processing {fpath.name}: {e}")
            errors += 1
            continue

        if result is None:
            skipped += 1
            continue

        key = config_key(
            result["config"]["grid_size"],
            result["config"]["num_fruits"],
            result["config"]["different_levels"],
        )
        grouped[key].append(result)

    # Save per-config pkl files
    total_processed = sum(len(v) for v in grouped.values())
    total_reconstructed = sum(1 for v in grouped.values() for ep in v if ep["step0_reconstructed"])
    total_uncertain = sum(1 for v in grouped.values() for ep in v if ep["step0_uncertain"])
    print(f"Errors: {errors}")
    print(f"Processed: {total_processed}")
    print(f"  step0 from recording: {total_processed - total_reconstructed}")
    print(f"  step0 reconstructed:  {total_reconstructed} ({total_uncertain} uncertain)")
    print()

    for key, episodes in sorted(grouped.items()):
        cfg_dir = output_dir / key
        cfg_dir.mkdir(exist_ok=True)
        out_path = cfg_dir / "trajectories.pkl"

        with open(out_path, "wb") as f:
            pickle.dump(episodes, f)

        # Stats
        agent_types = defaultdict(int)
        total_steps = 0
        n_reconstructed = sum(1 for ep in episodes if ep["step0_reconstructed"])
        n_uncertain = sum(1 for ep in episodes if ep["step0_uncertain"])
        for ep in episodes:
            agent_types[ep["agent_type"]] += 1
            total_steps += len(ep["actions"])

        print(f"{key}:")
        print(f"  Episodes: {len(episodes)}, Total steps: {total_steps}")
        print(f"  Reconstructed step0: {n_reconstructed} ({n_uncertain} uncertain)")
        print(f"  Agent types:")
        for at, count in sorted(agent_types.items()):
            print(f"    {at}: {count}")
        print()

    print(f"Output saved to {output_dir}")


if __name__ == "__main__":
    main()

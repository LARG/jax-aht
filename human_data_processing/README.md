# Human Data Processing

Converts raw Prolific episode JSON files into JAX-compatible arrays for behavior cloning.

## Data Quality Filtering

Applied during `process_episodes.py`, before pkl creation:

1. **Disqualified**: Players who did not complete all 8 games are removed.
2. **Disengaged**: Among players who completed all 8 games, a player is flagged and ALL their games are removed if ANY of the following hold (computed across all 8 games):
   - \>35% of actions were wait (action 0) or unsuccessful load (action 5 with 0 reward that step)
   - \>15% of actions were pure wait (action 0)
   - Average score (agent\_0 total\_rewards) across all 8 games is zero
   - Scored zero in 5 or more of their 8 games

## Step-0 Reconstruction

Episodes without a recorded step 0 have their initial state reconstructed by reversing the actions at step 1. These are flagged with `step0_reconstructed=True` and, when the reversal may be incorrect (blocked move indistinguishable from real move), `step0_uncertain=True`.

## Usage

```python
from human_data_processing.load_bc_data import load_bc_data, load_bc_data_by_agent, load_bc_data_padded
```

### Flat dataset (all episodes concatenated)

```python
data = load_bc_data("grid7_food3_nolevels")
```

Returns a `BCDataset` (NamedTuple):

| Field | Shape | Dtype | Description |
|-------|-------|-------|-------------|
| `obs` | `(N, obs_dim)` | float32 | VectorObserver observation before each action |
| `actions` | `(N,)` | int32 | Human actions (0=noop, 1=up, 2=down, 3=left, 4=right, 5=load) |
| `ai_actions` | `(N,)` | int32 | AI teammate actions |
| `rewards` | `(N,)` | float32 | Per-step reward for the human agent |
| `dones` | `(N,)` | bool | True on the last timestep of each episode |
| `avail_actions` | `(N, 6)` | bool | Which of the 6 actions are valid |
| `episode_ids` | `(N,)` | int32 | Which episode each timestep belongs to |

Where `N` = total timesteps across all episodes, `obs_dim` = 15 (7x7) or 24 (12x12).

### Grouped by agent type

```python
by_agent = load_bc_data_by_agent("grid7_food3_nolevels")
# dict mapping agent_type (str) -> BCDataset
```

### Padded episodes (for recurrent / sequence models)

```python
padded = load_bc_data_padded("grid7_food3_nolevels")
```

Returns a `BCDatasetPadded` (NamedTuple):

| Field | Shape | Dtype | Description |
|-------|-------|-------|-------------|
| `obs` | `(E, T, obs_dim)` | float32 | |
| `actions` | `(E, T)` | int32 | |
| `ai_actions` | `(E, T)` | int32 | |
| `rewards` | `(E, T)` | float32 | |
| `dones` | `(E, T)` | bool | |
| `avail_actions` | `(E, T, 6)` | bool | |
| `mask` | `(E, T)` | bool | True for real timesteps, False for padding |
| `agent_types` | list[str] | — | Agent type label per episode (length E) |

Where `E` = number of episodes, `T` = max episode length (padded).

### Options

All loading functions accept `exclude_uncertain=True` to drop episodes whose step-0 reconstruction is flagged as uncertain.

## Available Configs

| Config name | Grid | Food | Levels | obs\_dim |
|-------------|------|------|--------|---------|
| `grid7_food3_nolevels` | 7x7 | 3 | same | 15 |
| `grid7_food3_levels` | 7x7 | 3 | different | 15 |
| `grid12_food6_nolevels` | 12x12 | 6 | same | 24 |
| `grid12_food6_levels` | 12x12 | 6 | different | 24 |

## Observation Format

The observation uses Jumanji's `VectorObserver` with `fov = grid_size` (full observability). Positions are absolute. The vector is structured as:

```
[food_0_row, food_0_col, food_0_level,   # num_food triplets
 ...
 self_row, self_col, self_level,          # current agent
 other_0_row, other_0_col, other_0_level] # other agents
```

Eaten food is encoded as `(-1, -1, 0)`.

## Per-Episode Metadata (in pkl)

Each episode dict in the pkl also contains:

| Field | Type | Description |
|-------|------|-------------|
| `agent_type` | str | AI teammate, e.g. `"GreedyHeuristicAgent(closest_avg)"` |
| `config` | dict | `{grid_size, num_fruits, different_levels}` |
| `session_id` | str | UUID for the game session |
| `total_rewards` | dict | `{agent_0: float, agent_1: float}` |
| `duration` | float | Wall-clock seconds |
| `step0_reconstructed` | bool | True if step 0 was reverse-engineered |
| `step0_uncertain` | bool | True if reconstructed step 0 may be wrong |

## Scripts

- **`process_episodes.py`** — Reads raw JSON from `human_data_collecting/collected_data_prolific/`, applies player filtering, converts to arrays, writes `.pkl` files to `processed/`.
- **`load_bc_data.py`** — Loads `.pkl` files and returns JAX arrays ready for training.

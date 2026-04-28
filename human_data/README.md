# LBF Human Interaction Web Application

A Flask-based web application for collecting human interaction data with the Level-Based Foraging (LBF) environment. Human players play cooperatively with a heuristic AI agent.

## Features

- **Interactive Web Interface**: Play LBF through your browser with real-time visualization
- **Cooperative Gameplay**: Work together with a heuristic AI agent to maximize team score
- **Keyboard Controls**: Intuitive WASD/Arrow key controls for movement and space bar for food collection
- **Data Collection**: Automatically saves episode trajectories including actions, rewards, and states
- **Real-time Stats**: Track your score, AI score, and team total as you play
- **Multiple Sessions**: Support for multiple concurrent players

## Game Rules

- **Objective**: Collect food items by working cooperatively with the AI agent
- **Movement**: Use WASD or arrow keys to move around the grid
- **Collection**: Press SPACE when adjacent to food to attempt collection
- **Level Requirement**: Food can only be collected if combined agent levels ≥ food level
- **Cooperation**: Some food items require both agents to work together!

## Controls

| Key | Action |
|-----|--------|
| W / ↑ | Move Up |
| S / ↓ | Move Down |
| A / ← | Move Left |
| D / → | Move Right |
| SPACE | Load/Collect Food |
| Q | No Operation (Wait) |

## Installation

1. Make sure you have the main project dependencies installed found in
```bash
jax-aht/pyproject.toml
```

2. Install dependencies found in
```bash
jax-aht/human_data/requirements.txt
```

## Running the Application

1. Navigate to the human_data directory:

2. Start the Flask server:
```bash
python app.py
```

3. Open your web browser and navigate to the port the flask app is serving to

## Using Prolific

Use the base URL + /prolific to have an entrypoint that takes optional URL query parameters to collect prolific_id, experiment_id, etc.  

Prolific has completion codes which is provided at the end of the survey for the participant to submit back to Prolific to verify that they have completed the study. You can change the completion code provided by this website by changing the hardcoded field. 

## Collected Data

Episode data is **automatically saved** to `human_data/collected_data/` when an episode completes (reaches max steps or game ends). Each file contains:

- **Player Information**: Player name and session ID
- **Timing Information**: When the game actually started/ended and elapsed duration
- **Episode Metadata**: Total steps, rewards, grid configuration
- **Full Trajectory**: Step-by-step actions, rewards, states, and elapsed time

Data format example:
```json
{
  "player_name": "Anonymous",
  "session_id": "uuid-string",
  "timestamp": "20251023_143052",
  "start_time": 1700000000.123,
  "end_time": 1700000012.456,
  "duration": 12.333,
  "total_steps": 30,
  "total_rewards": {
    "agent_0": 0.5,
    "agent_1": 0.5
  },
  "grid_size": 7,
  "num_fruits": 3,
  "trajectory": [
    {
      "step": 1,
      "human_action": 1,
      "ai_action": 4,
      "rewards": {"agent_0": 0.0, "agent_1": 0.0},
      "state": { ... },
      "timestamp": 1700000000.123,
      "elapsed": 0.0
    },
    ...
  ]
}
```

Filenames include timestamp, session ID prefix, and step count: `episode_20251023_143052_e4619939_30steps.json`

## Human Data Visualizer

`human_data/visualize.py` analyzes collected LBF gameplay sessions from Prolific,
flags low-effort or bad-faith players, and produces two interactive HTML dashboards.

### Usage

```bash
python human_data/visualize.py
```

Outputs are saved to `human_data/plots/`:
- `effort_analysis.html` — per-session dashboard with episode replay
- `reaction_time_scatter.html` — reaction time vs score scatter plot

### Terminology

| Term | Meaning |
|------|---------|
| **Session** | One player sitting down to play, identified by a UUID session ID |
| **Game** | One round within a session = one JSON file (up to 8 per session) |
| **Trajectory** | The step-by-step sequence of actions and states within a game |

### Flagging Logic

Sessions are assigned one of three statuses:

- **OK** — normal player, data is usable
- **FLAGGED** — suspicious play, review before including in analysis
- **DISQUALIFIED** — incomplete session (fewer than 8 games), exclude from analysis

A session is flagged if any of the following are true:

| Check | Threshold | Rationale |
|-------|-----------|-----------|
| High idle rate | ≥ 35% of actions were Q key or unproductive SPACE presses | Player is not meaningfully engaging |
| High Q-key rate | ≥ 15% of actions were pure Q (wait) presses | Unambiguous idling |
| Zero average score | avg score = 0.0 | Never collected any food |
| Many zero-score games | 5 or more out of 8 games scored zero | Consistent non-performance across session |

**Notes on idle rate:** The idle rate counts both Q key presses (true waits) and SPACE presses that
didn't result in food collection. SPACE near an unreachable fruit still counts as wasted — this is
intentional, as a player genuinely attempting collection will reposition rather than spam SPACE.
However, SPACE alone is not used as a flag; only the combined idle rate matters.

**Idle sequence detection** (`⟳` indicator in the dashboard) is computed and shown per game
but does **not** affect flagging status — it was found to produce too many false positives
on normal gameplay near high-level fruits.

### Tunable Constants

All thresholds are at the top of the script:

```python
FLAG_SCORE   = 0.0   # avg score at or below this → flagged
FLAG_IDLE    = 0.35  # idle rate at or above this → flagged
REQUIRED_GAMES = 8   # fewer games than this → disqualified
```

### Output 1: Effort Analysis Dashboard (`effort_analysis.html`)

Each row is one session. The columns are:

- **Session ID** — first 8 chars shown; hover for full UUID. Status badge (OK / FLAGGED / DISQUALIFIED) shown inline.
- **Avg score per game** — color bar from red (0) to green (0.5 = max possible)
- **Idle rate** — fraction of actions that were Q or unproductive SPACE. Red = above threshold.
- **Games played** — out of 8 required
- **Score std dev** — variance across games within the session
- **Per-game scores** — one colored dot per game. Fill = score, border = idle rate.

Click any row to expand it and see:
- Full session UUID (click to select/copy)
- Flagging reason(s)
- Median reaction time and 10th percentile RT for the session
- Per-game breakdown table (score, idle rate, Q/SPACE counts, steps, duration, median RT, grid config)
- **▶ Replay all games** button — opens a modal showing all 8 game boards simultaneously,
  with play/pause, step controls, speed selector, and a per-game action strip.
  - `H` = human agent (blue), `A` = AI agent (red), orange squares = food
  - Action strip pips: red = Q/wait, blue = movement, green = SPACE/collect
  - Keyboard: `←/→` to step, `Space` to play/pause, `Esc` to close

Use the **filter buttons** (All / Flagged / Disqualified / OK) and **sort dropdown** to navigate sessions.

### Output 2: Reaction Time Scatter Plot (`reaction_time_scatter.html`)

Visualizes the relationship between how fast players think and how well they score.

**Reaction time** is computed as the elapsed time between consecutive human actions within a game,
derived from the `elapsed` field in each trajectory step. Values are in milliseconds.

**Axes:**
- X axis — median reaction time per session (switchable to 10th percentile via dropdown)
- Y axis — average score per game

**Visual encoding:**
- Dot color: green = OK, red = flagged, purple = disqualified
- Dot size: larger = more games played
- Red ring around a dot: 10th percentile RT < 150ms (suspiciously fast — possible key-holding or bot)
- Blue dashed trend line: linear regression on OK sessions only, with Pearson r shown

**10th percentile RT** is the reaction time that 90% of a player's actions were slower than —
i.e. their fastest 10% of actions. A very low p10 RT means some actions were taken
inhumanly quickly, even if the median looks normal.

Use the **X max slider** to zoom in on the dense cluster of fast players.
Hover any dot for full session stats.

**Stats cards** at the top show: total sessions with RT data, median RT across all sessions,
median RT for OK sessions only, and count of suspiciously fast sessions.

The reaction time metric is currently **informational only** — it does not affect
OK/Flagged/Disqualified status. Use the scatter plot to identify sessions worth
manually reviewing via the replay tool.

## Configuration

You can modify game parameters in the `app.py` file or pass them when starting a new game:

- `max_steps`: Maximum steps per episode (default: 50)
- `grid_size`: Size of the grid (default: 7)
- `num_fruits`: Number of food items (default: 3)

The AI agent strategy can be changed by modifying the `ordering_strategy` parameter:
- `'lexicographic'`: Top-to-bottom, left-to-right
- `'nearest_agent'`: Closest food first (default)
- `'farthest_agent'`: Farthest food first
- `'column_major'`: Column-wise ordering
- And more...

## Architecture

### Backend (`app.py`)
- Flask server handling game logic and state management
- JAX-based environment simulation
- Heuristic agent integration
- Session management and data persistence

### Frontend (`templates/index.html`)
- Canvas-based game visualization
- Real-time keyboard input handling
- REST API communication
- Responsive design

## API Endpoints

- `POST /api/new_game`: Start a new game session
- `POST /api/step`: Execute a step with player action
- `POST /api/reset`: Reset current game
- `POST /api/save_episode`: Save episode data to file
- `GET /api/controls`: Get control scheme information

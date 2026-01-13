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


## Collected Data

Episode data is **automatically saved** to `human_data/collected_data/` when an episode completes (reaches max steps or game ends). Each file contains:

- **Player Information**: Player name and session ID
- **Episode Metadata**: Total steps, rewards, grid configuration
- **Full Trajectory**: Step-by-step actions, rewards, and environment states

Data format example:
```json
{
  "player_name": "Anonymous",
  "session_id": "uuid-string",
  "timestamp": "20251023_143052",
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
      "state": { ... }
    },
    ...
  ]
}
```

Filenames include timestamp, session ID prefix, and step count: `episode_20251023_143052_e4619939_30steps.json`

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

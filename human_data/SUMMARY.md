# LBF Human Interaction Web Application - Summary

## What Was Created

A complete Flask-based web application for collecting human interaction data in the Level-Based Foraging (LBF) environment.

## File Structure

```
human_data/
├── app.py                      # Main Flask application
├── templates/
│   └── index.html             # Web interface with game visualization
├── collected_data/            # Directory for saved episodes (auto-created)
├── README.md                  # Comprehensive documentation
├── requirements.txt           # Python dependencies
├── start_server.sh           # Startup script
├── test_app.py               # Component testing script
└── .gitignore                # Git ignore rules
```

## Key Features

### 1. Flask Backend (`app.py`)
- **Session Management**: Each player gets a unique session with isolated game state
- **Environment Integration**: Seamlessly integrates with existing LBF environment and agents
- **AI Opponent**: Uses SequentialFruitAgent with 'nearest_agent' strategy
- **Data Collection**: Automatically saves full episode trajectories
- **REST API**: Clean API endpoints for game control

### 2. Web Interface (`templates/index.html`)
- **Real-time Visualization**: Canvas-based rendering of the game grid
- **Visual Elements**:
  - Player (you): Red square with 👤 icon
  - AI Agent: Teal square with 🤖 icon
  - Food items: Light teal with 🍎 and level number
  - Grid lines and borders
- **Statistics Display**:
  - Current step / max steps
  - Individual scores (player & AI)
  - Team total score
- **Responsive Design**: Works on different screen sizes
- **Keyboard Controls**: Intuitive WASD/Arrow keys + Space

### 3. Controls Defined

| Key | Action | Description |
|-----|--------|-------------|
| **W** / **↑** | Move Up | Move player up one cell |
| **S** / **↓** | Move Down | Move player down one cell |
| **A** / **←** | Move Left | Move player left one cell |
| **D** / **→** | Move Right | Move player right one cell |
| **SPACE** | Load Food | Attempt to collect adjacent food |
| **Q** | Wait/NOOP | No operation (wait one turn) |

### 4. Game Mechanics

- **Cooperative Play**: Human plays as agent_0, AI as agent_1
- **Level System**: Agents and food have levels
- **Collection Rule**: Can only collect food if agent level ≥ food level
- **Shared Rewards**: Team-based scoring encourages cooperation
- **Time Limit**: Configurable max steps (default: 50)
- **Highlighted Player**: Red border shows which agent is the human

## API Endpoints

```
POST /api/new_game     - Start new game session
POST /api/step         - Execute action and advance game
POST /api/reset        - Reset current game
POST /api/save_episode - Save episode data to JSON
GET  /api/controls     - Get control scheme information
```

## Data Collection Format

Episodes are **automatically saved** when completed. Saved episodes include:
```json
{
  "player_name": "Anonymous",
  "session_id": "uuid",
  "timestamp": "20251023_143052",
  "total_steps": 30,
  "total_rewards": {"agent_0": 0.5, "agent_1": 0.5},
  "grid_size": 7,
  "num_fruits": 3,
  "trajectory": [
    {
      "step": 1,
      "human_action": 1,      // UP = 1
      "ai_action": 4,          // RIGHT = 4
      "rewards": {...},
      "state": {
        "agent_positions": [[row, col], ...],
        "agent_levels": [2, 2],
        "food_positions": [[row, col], ...],
        "food_levels": [1, 2, 2],
        "food_eaten": [false, false, false]
      }
    },
    ...
  ]
}
```

Filenames: `episode_20251023_143052_e4619939_30steps.json`

## How to Use

### Quick Start
```bash
# 1. Navigate to directory
cd /scratch/cluster/jyliu/Documents/jax-aht/human_data

# 2. Install dependencies (if needed)
pip install flask flask-cors

# 3. Start server
./start_server.sh
# or
python app.py

# 4. Open browser
http://localhost:5000
```

### For Remote Server
```bash
# On local machine, set up SSH tunnel:
ssh -L 5000:localhost:5000 user@remote-server

# Then access:
http://localhost:5000
```

## Configuration Options

Easily customizable in `app.py`:

```python
# Game parameters
max_steps = 50          # Episode length
grid_size = 7           # Grid dimensions
num_fruits = 3          # Number of food items

# AI agent strategy options:
ordering_strategy = 'nearest_agent'  # Default
# Other options: 'lexicographic', 'farthest_agent', 
#                'column_major', 'reverse_column_major'
```

## Design Decisions

1. **Human as Agent 0**: Player is always agent_0 (highlighted in red)
2. **CPU-only JAX**: Avoids GPU conflicts, suitable for web server
3. **Session-based**: Multiple users can play simultaneously
4. **Auto-save**: Episodes automatically saved when completed
5. **Simple UI**: Focus on gameplay, minimal distractions
6. **Cooperative**: Encourages teamwork with shared rewards

## Testing

Run the test suite:
```bash
python test_app.py
```

Tests verify:
- Flask dependencies installed
- Environment initialization
- Agent creation and execution
- Episode running end-to-end

## Next Steps

To extend the application:
1. Add different AI difficulty levels
2. Implement episode replay viewer
3. Add tutorial mode for new players
4. Create leaderboard/statistics page
5. Support 2-human multiplayer
6. Add voice/text chat for research studies
7. Integrate with data analysis pipeline

## Technical Notes

- **JAX JIT**: Disabled by default in test mode for easier debugging
- **State Management**: Uses in-memory dict (use Redis for production)
- **Serialization**: JAX arrays converted to lists for JSON
- **CORS**: Enabled for development (restrict in production)
- **Error Handling**: Basic error messages, can be enhanced

## Troubleshooting

Common issues and solutions documented in README.md:
- Port conflicts
- JAX/GPU issues  
- Import errors
- Environment activation

---

**Created**: 2025-10-23
**Purpose**: Human data collection for LBF environment
**Technology Stack**: Flask, JAX, HTML5 Canvas, Vanilla JavaScript

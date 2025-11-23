"""
Flask web application for human interaction with the Level-Based Foraging (LBF) environment.
Human player plays against a heuristic LBF agent.
"""
import os
import sys
import json
import time
from functools import wraps
from flask import Flask, render_template, jsonify, request, session
from flask_cors import CORS
import jax
import jax.numpy as jnp
import numpy as np
import asyncio
import uuid
import threading
from asyncio import run_coroutine_threadsafe

# Add parent directory to path to import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs import make_env
from agents.lbf import SequentialFruitAgent

app = Flask(__name__)
app.secret_key = '<FILL_THIS_IN>'  # Change this in production
CORS(app)

# Global game state storage (in production, use Redis or similar)
game_sessions = {}

# Action constants
NOOP = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
LOAD = 5

def simple_timer(func):
    """
    Simple timing decorator that always runs.
    Prints execution time for the function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"⏱️  {func.__name__} took {elapsed_time:.4f}s")
        return result
    return wrapper

class GameSession:
    """Manages a single game session with environment state and agent state."""
    
    def __init__(self, session_id, max_steps=50, grid_size=7, num_fruits=3):
        self.session_id = session_id
        self.max_steps = max_steps
        self.grid_size = grid_size
        self.num_fruits = num_fruits
        
        # Initialize environment
        self.env = make_env(
            env_name="lbf", 
            env_kwargs={
                "time_limit": max_steps,
                "grid_size": grid_size,
                "num_agents": 2,
                "num_food": num_fruits,
                "highlight_agent_idx": 0  # Highlight human player
            }
        )
        
        # Initialize heuristic agent (agent 1 - computer)
        self.ai_agent = SequentialFruitAgent(
            grid_size=grid_size, 
            num_fruits=num_fruits, 
            ordering_strategy='nearest_agent'
        )
        
        # Initialize JAX random key
        self.rng = jax.random.PRNGKey(np.random.randint(0, 1000000))
        
        # Game state
        self.obs = None
        self.state = None
        self.done = False
        self.rewards = {"agent_0": 0.0, "agent_1": 0.0}
        self.total_rewards = {"agent_0": 0.0, "agent_1": 0.0}
        self.step_count = 0
        self.ai_agent_state = None
        self.episode_history = []  # Store episode trajectory
        
        # Initialize the game
        self.reset()
        
        # Pre-compile JAX functions with a warmup step
        self._warmup_jit_compilation()
    
    def _warmup_jit_compilation(self):
        """
        Pre-compile JAX functions by doing warmup steps.
        This ensures the first actual game step is fast.
        """
        print(f"🔥 Warming up JIT compilation for session {self.session_id[:8]}...")
        warmup_start = time.time()
        
        # Save current state
        saved_obs = self.obs
        saved_state = self.state
        saved_ai_agent_state = self.ai_agent_state
        saved_rng = self.rng
        
        # Do a few warmup steps to trigger compilation
        for _ in range(3):
            # Get AI agent action (triggers agent compilation)
            self.rng, ai_rng = jax.random.split(self.rng)
            ai_action, self.ai_agent_state = self.ai_agent.get_action(
                self.obs["agent_1"], 
                self.state, 
                self.ai_agent_state, 
                ai_rng
            )
            
            # Prepare actions
            actions = {
                "agent_0": jnp.array(0, dtype=jnp.int32),  # NOOP
                "agent_1": ai_action
            }
            
            # Step environment (triggers env.step compilation)
            self.rng, step_key = jax.random.split(self.rng)
            self.obs, self.state, self.rewards, done_dict, info = self.env.step(
                step_key, self.state, actions
            )
            
            # If done, reset for next warmup iteration
            if done_dict["__all__"]:
                self.rng, reset_key = jax.random.split(self.rng)
                self.obs, self.state = self.env.reset(reset_key)
                self.ai_agent_state = self.ai_agent.init_agent_state(1)
        
        # Block until all JAX operations complete
        jax.block_until_ready(self.obs)
        
        # Restore original state
        self.obs = saved_obs
        self.state = saved_state
        self.ai_agent_state = saved_ai_agent_state
        self.rng = saved_rng
        
        warmup_time = time.time() - warmup_start
        print(f"✅ JIT compilation complete ({warmup_time:.2f}s)")
    
    def reset(self):
        """Reset the game environment."""
        self.rng, subkey = jax.random.split(self.rng)
        self.obs, self.state = self.env.reset(subkey)
        
        # Initialize AI agent state
        self.ai_agent_state = self.ai_agent.init_agent_state(1)  # Agent 1 is AI
        
        self.done = False
        self.rewards = {"agent_0": 0.0, "agent_1": 0.0}
        self.total_rewards = {"agent_0": 0.0, "agent_1": 0.0}
        self.step_count = 0
        self.episode_history = []
        
        return self.get_state_dict()
    
    def step(self, human_action):
        """Execute one step with human action and AI agent action."""
        if self.done:
            return self.get_state_dict()
        
        # Get AI agent action
        self.rng, ai_rng = jax.random.split(self.rng)
        ai_action, self.ai_agent_state = self.ai_agent.get_action(
            self.obs["agent_1"], 
            self.state, 
            self.ai_agent_state, 
            ai_rng
        )
        
        # Combine actions
        actions = {
            "agent_0": jnp.array(human_action, dtype=jnp.int32),
            "agent_1": ai_action
        }
        
        # Step environment
        self.rng, step_key = jax.random.split(self.rng)
        self.obs, self.state, self.rewards, done_dict, info = self.env.step(
            step_key, self.state, actions
        )
        
        # Update state
        self.done = done_dict["__all__"]
        self.step_count += 1
        
        # Update total rewards
        for agent in self.env.agents:
            self.total_rewards[agent] += float(self.rewards[agent])
        
        # Store in history
        self.episode_history.append({
            "step": self.step_count,
            "human_action": int(human_action),
            "ai_action": int(ai_action),
            "rewards": {k: float(v) for k, v in self.rewards.items()},
            "state": self._serialize_state()
        })
        
        # Auto-save when episode is done
        if self.done:
            self.save_episode()
        
        return self.get_state_dict()
    
    def _serialize_state(self):
        """Serialize environment state for storage."""
        # Extract key state information
        return {
            "agent_positions": self.state.env_state.agents.position.tolist(),
            "agent_levels": self.state.env_state.agents.level.tolist(),
            "food_positions": self.state.env_state.food_items.position.tolist(),
            "food_levels": self.state.env_state.food_items.level.tolist(),
            "food_eaten": self.state.env_state.food_items.eaten.tolist(),
            "step_count": int(self.step_count)
        }
    
    def get_state_dict(self):
        """Get current game state as a dictionary for JSON serialization."""
        state_data = self._serialize_state()
        
        # Add available actions for human player
        avail_actions = self.state.avail_actions["agent_0"].tolist()
        
        return {
            "done": bool(self.done),
            "step_count": int(self.step_count),
            "max_steps": self.max_steps,
            "rewards": {k: float(v) for k, v in self.rewards.items()},
            "total_rewards": {k: float(v) for k, v in self.total_rewards.items()},
            "avail_actions": avail_actions,
            "state": state_data
        }
    
    def save_episode(self, player_name="Anonymous"):
        """Save episode data to file."""
        if not self.episode_history:
            return None
        
        # Add timestamp for uniqueness
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        episode_data = {
            "player_name": player_name,
            "session_id": self.session_id,
            "timestamp": timestamp,
            "total_steps": self.step_count,
            "total_rewards": {k: float(v) for k, v in self.total_rewards.items()},
            "grid_size": self.grid_size,
            "num_fruits": self.num_fruits,
            "trajectory": self.episode_history
        }
        
        # Create data directory if it doesn't exist
        data_dir = os.path.join(os.path.dirname(__file__), "collected_data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Save to file with timestamp for uniqueness
        filename = f"episode_{timestamp}_{self.session_id[:8]}_{self.step_count}steps.json"
        filepath = os.path.join(data_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(episode_data, f, indent=2)
        
        return filepath

# -- PREWARMING -- 
# Prewarm a queue of game sessions that are ready to go

PREWARMED_GAMES = []

def get_game():
    max_steps = 50
    grid_size = 7
    num_fruits = 3
    game = GameSession(str(uuid.uuid4()), max_steps, grid_size, num_fruits)
    return game

async def add_prewarmed_game():
    print("Adding prewarmed game session...")
    game = get_game()
    print("Added prewarmed game session")
    PREWARMED_GAMES.append(game)

    return game


@app.route('/')
def index():
    """Serve the main game page."""
    return render_template('index.html')


@app.route('/api/new_game', methods=['POST'])
def new_game():
    """Start a new game session."""
    # Generate session ID
    session_id = str(uuid.uuid4())
    
    print("Prewarmed games:", len(PREWARMED_GAMES))

    if PREWARMED_GAMES:
        game = PREWARMED_GAMES.pop(0)
        run_coroutine_threadsafe(add_prewarmed_game(), BACKGROUND_LOOP)
    else:
        game = get_game()

    game.session_id = session_id
    game_sessions[session_id] = game
    
    # Store session ID in Flask session
    session['session_id'] = session_id
    
    return jsonify({
        "success": True,
        "session_id": session_id,
        "state": game.get_state_dict()
    })


@app.route('/api/step', methods=['POST'])
def step():
    """Execute a step with the human player's action."""
    data = request.get_json()
    session_id = session.get('session_id')
    action = data.get('action')
    
    if not session_id or session_id not in game_sessions:
        return jsonify({"success": False, "error": "No active game session"}), 400
    
    if action is None or not isinstance(action, int) or action < 0 or action > 5:
        return jsonify({"success": False, "error": "Invalid action"}), 400
    
    game = game_sessions[session_id]
    state = game.step(action)
    
    return jsonify({
        "success": True,
        "state": state
    })


@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset the current game."""
    session_id = session.get('session_id')
    
    if not session_id or session_id not in game_sessions:
        return jsonify({"success": False, "error": "No active game session"}), 400
    
    game = game_sessions[session_id]
    state = game.reset()
    
    return jsonify({
        "success": True,
        "state": state
    })


@app.route('/api/save_episode', methods=['POST'])
def save_episode():
    """Save the current episode data."""
    data = request.get_json()
    session_id = session.get('session_id')
    player_name = data.get('player_name', 'Anonymous')
    
    if not session_id or session_id not in game_sessions:
        return jsonify({"success": False, "error": "No active game session"}), 400
    
    game = game_sessions[session_id]
    filepath = game.save_episode(player_name)
    
    if filepath:
        return jsonify({
            "success": True,
            "message": f"Episode saved to {os.path.basename(filepath)}"
        })
    else:
        return jsonify({
            "success": False,
            "error": "No episode data to save"
        }), 400


@app.route('/api/controls', methods=['GET'])
def get_controls():
    """Return the control scheme for the game."""
    controls = {
        "keyboard": {
            "w": {"action": UP, "name": "Move Up"},
            "s": {"action": DOWN, "name": "Move Down"},
            "a": {"action": LEFT, "name": "Move Left"},
            "d": {"action": RIGHT, "name": "Move Right"},
            "space": {"action": LOAD, "name": "Load/Collect Food"},
            "q": {"action": NOOP, "name": "No Operation (Wait)"}
        },
        "actions": {
            NOOP: "No Operation (Wait)",
            UP: "Move Up",
            DOWN: "Move Down",
            LEFT: "Move Left",
            RIGHT: "Move Right",
            LOAD: "Load/Collect Food"
        }
    }
    return jsonify(controls)


global BACKGROUND_LOOP
BACKGROUND_LOOP = asyncio.new_event_loop()
def start_background_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

if __name__ == '__main__':
    print("=" * 60)
    print("LBF Human Interaction Server")
    print("=" * 60)
    print("\nControls:")
    print("  W/↑    : Move Up")
    print("  S/↓    : Move Down")
    print("  A/←    : Move Left")
    print("  D/→    : Move Right")
    print("  SPACE  : Load/Collect Food")
    print("  Q      : No Operation (Wait)")
    print("\nStarting server at http://localhost:8998")
    print("=" * 60)
    
    threading.Thread(target=start_background_loop, args=(BACKGROUND_LOOP,), daemon=True).start()

    for _ in range(2):
        print("[Startup] prewarming")
        run_coroutine_threadsafe(add_prewarmed_game(), BACKGROUND_LOOP)

    app.run(debug=False, host='0.0.0.0', port=8998)

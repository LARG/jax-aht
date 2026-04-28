"""
Flask web application for human interaction with the Level-Based Foraging (LBF) environment.
Human player plays against a heuristic LBF agent.
"""
import os
import sys
import json
import time
import random
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
from agents.lbf import GreedyHeuristicAgent
from agents.lbf import EntitledAgent
from common.agent_loader_from_config import initialize_rl_agent_from_config
from evaluation.heldout_evaluator import extract_params

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', os.urandom(32).hex())
CORS(app)

# Global game state storage (in production, use Redis or similar)
game_sessions = {}
game_sessions_lock = threading.Lock()


# -- RL POLICY AGENT SUPPORT --

class RLPolicyAgentWrapper:
    """Wraps an AgentPolicy + params to match the heuristic BaseAgent interface
    so GameSession can use RL policy agents interchangeably with heuristic agents."""

    def __init__(self, policy, params, test_mode=False, name="rl_agent"):
        self.policy = policy
        self.params = params
        self.test_mode = test_mode
        self._name = name

    def init_agent_state(self, agent_id):
        return self.policy.init_hstate(1, aux_info={"agent_id": agent_id})

    def get_action(self, obs, env_state, agent_state, rng):
        obs_reshaped = obs.reshape(1, 1, -1)
        done = jnp.zeros((1, 1), dtype=bool)
        avail_actions = env_state.avail_actions["agent_1"].astype(jnp.float32)
        action, new_hstate = self.policy.get_action(
            params=self.params,
            obs=obs_reshaped,
            done=done,
            avail_actions=avail_actions,
            hstate=agent_state,
            rng=rng,
            env_state=env_state,
            test_mode=self.test_mode,
        )
        return action.squeeze(), new_hstate

    def get_name(self):
        return self._name


# Pool of loaded RL agents per game variant (populated at startup)
# Key: (grid_size, different_levels) -> list of RLPolicyAgentWrapper
RL_AGENTS = {}

# Configs for RL teammates stored in human_data/teammates/, keyed by (grid_size, different_levels)
_RL_AGENT_CONFIGS = {
    (7, False): [
        {
            "name": "ippo_mlp",
            "path": "human_data/teammates/ippo-lbf-7/saved_train_run",
            "actor_type": "mlp",
            "ckpt_key": "final_params",
            "idx_list": [0],
            "test_mode": False,
        },
        {
            "name": "ippo_mlp_s2c0",
            "path": "human_data/teammates/ippo-lbf-7/saved_train_run",
            "actor_type": "mlp",
            "idx_list": [[2, 0]],  # seed 2, checkpoint 0 — needs "checkpoints" key (default)
            "test_mode": False,
        },
        # {
        #     "name": "brdiv-conf1",
        #     "path": "human_data/teammates/brdiv-lbf-7-1/saved_train_run",
        #     "actor_type": "actor_with_conditional_critic",
        #     "ckpt_key": "final_params_conf",
        #     "idx_list": [0, 1, 2],
        #     "POP_SIZE": 5,
        #     "test_mode": False,
        # },
        # {
        #     "name": "brdiv-conf2",
        #     "path": "human_data/teammates/brdiv-lbf-7-2/saved_train_run",
        #     "actor_type": "actor_with_conditional_critic",
        #     "ckpt_key": "final_params_conf",
        #     "idx_list": [0, 1],
        #     "POP_SIZE": 3,
        #     "test_mode": False,
        # },
    ],
    (7, True): [
        {
            "name": "ippo_mlp_7_levels",
            "path": "human_data/teammates/ippo-lbf-7-levels/saved_train_run",
            "actor_type": "mlp",
            "ckpt_key": "final_params",
            "idx_list": [0],
            "test_mode": False,
        },
    ],
    (12, False): [
        {
            "name": "ippo_mlp_12",
            "path": "human_data/teammates/ippo-lbf-12/saved_train_run",
            "actor_type": "mlp",
            "ckpt_key": "final_params",
            "idx_list": [0],
            "test_mode": False,
        },
    ],
    (12, True): [
        {
            "name": "ippo_mlp_12_levels",
            "path": "human_data/teammates/ippo-lbf-12-levels/saved_train_run",
            "actor_type": "mlp",
            "ckpt_key": "final_params",
            "idx_list": [0],
            "test_mode": False,
        },
    ],
}


_rl_load_lock = threading.Lock()
_rl_load_rng = jax.random.PRNGKey(0)


def _load_rl_agents_for_variant(variant_key):
    """Load RL agents for a single (grid_size, different_levels) variant. Thread-safe, idempotent."""
    global _rl_load_rng
    import logging

    if variant_key in RL_AGENTS:
        return  # Already loaded

    with _rl_load_lock:
        if variant_key in RL_AGENTS:
            return  # Double-check after acquiring lock

        grid_size, different_levels = variant_key
        cfg_list = _RL_AGENT_CONFIGS.get(variant_key, [])
        if not cfg_list:
            RL_AGENTS[variant_key] = []
            return

        cpu = jax.devices('cpu')[0]
        num_food = 3 if grid_size == 7 else 6
        env = make_env(env_name="lbf", env_kwargs={
            "grid_size": grid_size, "num_agents": 2, "num_food": num_food,
            "different_levels": different_levels,
        })

        root_logger = logging.getLogger()
        prev_level = root_logger.level
        root_logger.setLevel(logging.CRITICAL)

        agents = []
        try:
            for cfg in cfg_list:
                cfg = dict(cfg)
                name = cfg.pop("name")
                test_mode = cfg.pop("test_mode")

                _rl_load_rng, init_rng = jax.random.split(_rl_load_rng)
                cfg["custom_loader"] = {"name": "partial_load"}
                policy, params, init_params, idx_labels = initialize_rl_agent_from_config(
                    cfg, name, env, init_rng
                )
                params = jax.tree.map(lambda x: jax.device_put(x, cpu), params)
                params_list, idx_labels = extract_params(params, init_params, idx_labels)
                del init_params

                for i, p in enumerate(params_list):
                    label = f"{name}({idx_labels[i]})" if len(params_list) > 1 else name
                    param_bytes = sum(x.nbytes for x in jax.tree.leaves(p))
                    print(f"  Agent '{label}': {param_bytes / 1024:.1f} KB ({len(jax.tree.leaves(p))} arrays)")
                    agents.append(RLPolicyAgentWrapper(policy, p, test_mode, label))
        finally:
            root_logger.setLevel(prev_level)

        RL_AGENTS[variant_key] = agents
        print(f"Loaded {len(agents)} RL policy agents for {grid_size}x{grid_size} (different_levels={different_levels})")


def load_rl_agents():
    """Eagerly load RL agents for the (7, 'same') variant at startup. Others load lazily on first use."""
    _load_rl_agents_for_variant((7, False))
    _load_rl_agents_for_variant((7, True))
    _load_rl_agents_for_variant((12, True))
    _load_rl_agents_for_variant((12, False))

# Action constants
NOOP = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
LOAD = 5

class GameSession:
    """Manages a single game session with environment state and agent state."""

    def __init__(self, session_id, max_steps=50, env_kwargs: dict = None, is_warmup: bool = False, data_source: str = "",
                 prolific_pid=None, study_id=None, prolific_session_id=None):
        self.is_warmup = is_warmup
        self.data_source = data_source
        self.prolific_pid = prolific_pid
        self.study_id = study_id
        self.prolific_session_id = prolific_session_id
        self.session_id = session_id
        self.max_steps = max_steps
        self.env_kwargs = env_kwargs or {}
        self.grid_size = self.env_kwargs.get("grid_size", 7)
        self.num_fruits = self.env_kwargs.get("num_food", 3 if self.grid_size == 7 else 6)
        self.different_levels = self.env_kwargs.get("different_levels", False)

        # Choose agent first
        self.ai_agent = self._choose_agent()

        # Initialize environment
        env_args = dict(self.env_kwargs)
        env_args.update({
            "time_limit": max_steps,
            "num_agents": 2,
            "highlight_agent_idx": 0,
        })
        self.env = make_env(env_name="lbf", env_kwargs=env_args)
        
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
    
    def _choose_agent(self):
        # Build a flat list of all possible agent factories, then pick uniformly
        candidates = []

        # RL policy agents
        variant_key = (self.grid_size, self.different_levels)
        if variant_key not in RL_AGENTS:
            _load_rl_agents_for_variant(variant_key)
        for agent in RL_AGENTS.get(variant_key, []):
            candidates.append(lambda a=agent: a)

        # SequentialFruitAgent — one per ordering strategy
        for strategy in SequentialFruitAgent.VALID_ORDERING_STRATEGIES:
            candidates.append(lambda s=strategy: SequentialFruitAgent(
                grid_size=self.grid_size,
                num_fruits=self.num_fruits,
                ordering_strategy=s,
            ))

        # GreedyHeuristicAgent — one per heuristic
        heuristics = GreedyHeuristicAgent.VALID_HEURISTICS_LEVELS if self.different_levels else GreedyHeuristicAgent.VALID_HEURISTICS
        for heuristic in heuristics:
            candidates.append(lambda h=heuristic: GreedyHeuristicAgent(
                grid_size=self.grid_size,
                num_fruits=self.num_fruits,
                heuristic=h,
            ))

        # EntitledAgent — weighted 1.5x relative to any other single candidate
        entitled_factory = lambda: EntitledAgent(
            grid_size=self.grid_size,
            num_fruits=self.num_fruits,
        )
        weights = [1.0] * len(candidates) + [1.5]
        candidates.append(entitled_factory)

        return random.choices(candidates, weights=weights, k=1)[0]()

    def _warmup_jit_compilation(self):
        """
        Pre-compile JAX functions by doing warmup steps.
        This ensures the first actual game step is fast.
        """
        # print(f"🔥 Warming up JIT compilation for session {self.session_id[:8]}...")
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
        # print(f"✅ JIT compilation complete ({warmup_time:.2f}s)")
    
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
        # track timing for the episode
        self.start_time = time.time()
        self.end_time = None

        self.episode_history = []

        return self.get_state_dict()
    
    def step(self, human_action):
        """Execute one step with human action and AI agent action."""
        if self.done:
            # TODO: Record endtime
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
        
        # Block until all JAX operations are complete before proceeding
        # This ensures responsive gameplay without lag
        jax.block_until_ready(self.obs)
        
        # Update state
        self.done = done_dict["__all__"]
        self.step_count += 1
        
        # Update total rewards
        for agent in self.env.agents:
            self.total_rewards[agent] += float(self.rewards[agent])
        
        # Store in history (with elapsed time since start)
        now = time.time()
        elapsed = now - self.start_time if hasattr(self, 'start_time') else None
        self.episode_history.append({
            "step": self.step_count,
            "human_action": int(human_action),
            "ai_action": int(ai_action),
            "rewards": {k: float(v) for k, v in self.rewards.items()},
            "state": self._serialize_state(),
            # include wall-clock timestamp and elapsed seconds
            "timestamp": now,
            "elapsed": elapsed
        })
        
        # Auto-save when episode is done
        if self.done:
            # record end time before saving
            self.end_time = time.time()
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
            "state": state_data,
            "grid_size": self.grid_size,
            "num_fruits": self.num_fruits,
            "different_levels": self.different_levels
        }
    
    def save_episode(self, player_name="Anonymous"):
        """Save episode data to file."""
        if not self.episode_history:
            return None
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # make sure end_time is set if game is being saved after completion
        if getattr(self, 'end_time', None) is None and self.done:
            self.end_time = time.time()

        episode_data = {
            "player_name": player_name,
            "session_id": self.session_id,
            "timestamp": timestamp,
            "prolific_pid": self.prolific_pid,
            "study_id": self.study_id,
            "prolific_session_id": self.prolific_session_id,
            "total_steps": self.step_count,
            "total_rewards": {k: float(v) for k, v in self.total_rewards.items()},
            "grid_size": self.grid_size,
            "num_fruits": self.num_fruits,
            "different_levels": self.different_levels,
            "agent_type": self.ai_agent.get_name(),
            # include game start/end timing information
            "start_time": getattr(self, "start_time", None),
            "end_time": getattr(self, "end_time", None),
            "duration": (self.end_time - self.start_time) if (self.start_time and self.end_time) else None,
            "trajectory": self.episode_history
        }
        
        # Create data directory if it doesn't exist
        # Warmup games go to a separate folder; data_source adds a suffix (e.g. "_prolific")
        suffix = f"_{self.data_source}" if self.data_source else ""
        folder_name = f"collected_data_warmup{suffix}" if self.is_warmup else f"collected_data{suffix}"
        data_dir = os.path.join(os.path.dirname(__file__), folder_name)
        os.makedirs(data_dir, exist_ok=True)
        
        # Save to file with timestamp for uniqueness
        filename = f"episode_{timestamp}_{self.session_id[:8]}_{self.step_count}steps.json"
        filepath = os.path.join(data_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(episode_data, f, indent=2)
        
        return filepath

# Number of warmup games at the start of each session
NUM_WARMUP_GAMES = 2
NUM_REAL_GAMES = 8


class MultiGameSession:
    """Manages a sequence of GameSession instances played sequentially with lazy loading."""
    def __init__(self, session_id, game_configs, first_game=None, num_warmup=NUM_WARMUP_GAMES, data_source="",
                 prolific_pid=None, study_id=None, prolific_session_id=None):
        self.session_id = session_id
        self.config_list = game_configs  # All configs (used for lazy creation)
        self.games = [None] * len(game_configs)  # Placeholder for all games
        self.current_idx = 0
        self.num_warmup = num_warmup  # First N games are warmup
        self.data_source = data_source
        self.prolific_pid = prolific_pid
        self.study_id = study_id
        self.prolific_session_id = prolific_session_id
        self.session_complete = False  # True when all games are done
        self.last_activity = time.time()

        # Set first game if provided (already prewarmed or created)
        if first_game:
            self.games[0] = first_game
            first_game.session_id = session_id
            first_game.is_warmup = (0 < num_warmup)
            first_game.data_source = data_source
            first_game.prolific_pid = prolific_pid
            first_game.study_id = study_id
            first_game.prolific_session_id = prolific_session_id

    def _ensure_game_loaded(self, idx):
        """Ensure game at index is loaded; fetch or create if needed."""
        if self.games[idx] is not None:
            return  # Already loaded

        cfg = self.config_list[idx]
        game = get_or_create_game(cfg)
        game.session_id = self.session_id
        game.is_warmup = (idx < self.num_warmup)
        game.data_source = self.data_source
        game.prolific_pid = self.prolific_pid
        game.study_id = self.study_id
        game.prolific_session_id = self.prolific_session_id
        self.games[idx] = game

    def _current(self):
        self._ensure_game_loaded(self.current_idx)
        return self.games[self.current_idx]

    def step(self, human_action):
        self.last_activity = time.time()
        cur = self._current()
        prev_idx = self.current_idx
        cur.step(human_action)
        # If current finished, advance to next game
        if cur.done and self.current_idx < len(self.games) - 1:
            self.current_idx += 1
            # Release the completed game to free JAX arrays and env state
            self.games[prev_idx] = None
            # Ensure next game is loaded before returning its state
            self._ensure_game_loaded(self.current_idx)
            result = self.get_state_dict()
            result['game_just_advanced'] = True
            result['prev_game_index'] = prev_idx
            return result
        # If all games are done, mark session complete
        if cur.done and self.current_idx >= len(self.games) - 1:
            self.session_complete = True
        result = self.get_state_dict()
        result['game_just_advanced'] = False
        return result

    def get_state_dict(self):
        state = self._current().get_state_dict()
        # add multi-game metadata
        state['multi'] = True
        state['current_game_index'] = self.current_idx
        state['total_games'] = len(self.games)
        state['num_warmup'] = self.num_warmup
        state['is_warmup'] = self.current_idx < self.num_warmup
        state['session_complete'] = self.session_complete
        state['game_configs'] = self.config_list
        return state

    def save_all(self, player_name='Anonymous'):
        paths = []
        for game in self.games:
            if game is not None:
                p = game.save_episode(player_name)
                if p:
                    paths.append(p)
        return paths


# -- PREWARMING --
# Prewarm a pool of games; each env_kwargs config is prewarmed independently.
# User gets a prewarmed game if available from the pool, otherwise created on-demand.

PREWARMED_GAMES_POOL = {}  # {config_key: [GameSession, ...]}
PREWARMING_LOCK = threading.Lock()

# The 4 base env_kwargs configurations
BASE_ENV_CONFIGS = [
    {"grid_size": 7, "num_food": 3, "different_levels": False},
    {"grid_size": 7, "num_food": 3, "different_levels": True},
    {"grid_size": 12, "num_food": 6, "different_levels": False},
    {"grid_size": 12, "num_food": 6, "different_levels": True},
]

def _config_key(env_kwargs):
    """Hashable key for an env_kwargs dict."""
    return (env_kwargs.get("grid_size", 7), env_kwargs.get("different_levels", False))

def generate_session_configs(num_warmup=NUM_WARMUP_GAMES, num_real=NUM_REAL_GAMES):
    """Generate the full list of game configs (env_kwargs dicts) for a session.

    Returns a list of dicts: first `num_warmup` are warmup, rest are real games.
    Warmup games use randomly chosen configs.
    Real games use 2 copies of each of the 4 base configs, shuffled.
    """
    # Warmup: pick randomly from base configs
    warmup = [random.choice(BASE_ENV_CONFIGS).copy() for _ in range(num_warmup)]

    # Real games: 2 copies of each base config, shuffled
    real = [cfg.copy() for cfg in BASE_ENV_CONFIGS for _ in range(num_real // len(BASE_ENV_CONFIGS))]
    # If num_real isn't perfectly divisible, pad with random picks
    while len(real) < num_real:
        real.append(random.choice(BASE_ENV_CONFIGS).copy())
    random.shuffle(real)

    return warmup + real

def get_or_create_game(env_kwargs):
    """Get a prewarmed game or create one on-demand."""
    key = _config_key(env_kwargs)
    with PREWARMING_LOCK:
        if key in PREWARMED_GAMES_POOL and PREWARMED_GAMES_POOL[key]:
            game = PREWARMED_GAMES_POOL[key].pop(0)
            # When returning a prewarmed game, reset to ensure start/end times are for the actual human episode
            game.reset()
            return game

    # Fallback: create on-demand
    return GameSession(str(uuid.uuid4()), 50, env_kwargs=env_kwargs)

SESSION_TTL = 30 * 60  # 30 minutes

async def prewarm_games():
    """Continuously prewarm games in background, keeping a pool of each config type."""
    while True:
        try:
            # Clean up stale game sessions
            now = time.time()
            with game_sessions_lock:
                stale = [sid for sid, gs in game_sessions.items()
                         if hasattr(gs, 'last_activity') and now - gs.last_activity > SESSION_TTL]
                for sid in stale:
                    del game_sessions[sid]
            if stale:
                print(f"Cleaned up {len(stale)} stale game session(s)")

            for cfg in BASE_ENV_CONFIGS:
                key = _config_key(cfg)
                with PREWARMING_LOCK:
                    if key not in PREWARMED_GAMES_POOL:
                        PREWARMED_GAMES_POOL[key] = []
                    needs_prewarm = len(PREWARMED_GAMES_POOL[key]) < 3

                if not needs_prewarm:
                    continue

                gs = GameSession(str(uuid.uuid4()), 50, env_kwargs=cfg.copy())

                with PREWARMING_LOCK:
                    if len(PREWARMED_GAMES_POOL[key]) < 3:
                        PREWARMED_GAMES_POOL[key].append(gs)
                    else:
                        del gs  # Pool was replenished by another thread

            await asyncio.sleep(1)
        except Exception as e:
            print(f"Error prewarming: {e}")
            await asyncio.sleep(1)


@app.route('/')
def index():
    """Serve the main game page."""
    return render_template('index.html', data_source='')


@app.route('/prolific')
def prolific_index():
    """Serve the game page for Prolific participants (data saved to prolific-specific folders)."""
    import base64
    _completion_url = "https://app.prolific.com/submissions/complete?cc=C13XLNJ0"
    _encoded = base64.b64encode(_completion_url.encode()).decode()
    return render_template('index.html', data_source='prolific', prolific_completion=_encoded)


@app.route('/api/new_game', methods=['POST'])
def new_game():
    """Start a new game session with 2 warmup + 8 real games. Only first game is loaded upfront."""
    data = request.get_json() or {}
    # Sanitize data_source to prevent path traversal — only allow alphanumeric and hyphens
    raw_source = data.get('data_source', '')
    data_source = raw_source if raw_source.isalnum() or all(c.isalnum() or c == '-' for c in raw_source) else ''

    prolific_pid = data.get('prolific_pid') or None
    study_id = data.get('study_id') or None
    prolific_session_id = data.get('prolific_session_id') or None

    session_id = str(uuid.uuid4())

    # Generate full config list: 2 warmup + 8 real
    conf_list = generate_session_configs()

    # Only get/create the FIRST game immediately
    first_cfg = conf_list[0]
    first_game = get_or_create_game(first_cfg)

    # Create MultiGameSession with lazy loading for remaining games
    multi = MultiGameSession(session_id, conf_list, first_game=first_game, num_warmup=NUM_WARMUP_GAMES,
                             data_source=data_source, prolific_pid=prolific_pid,
                             study_id=study_id, prolific_session_id=prolific_session_id)
    with game_sessions_lock:
        game_sessions[session_id] = multi

    # Store session ID in Flask session
    session['session_id'] = session_id

    return jsonify({
        "success": True,
        "session_id": session_id,
        "state": multi.get_state_dict()
    })



@app.route('/api/step', methods=['POST'])
def step():
    """Execute a step with the human player's action."""
    data = request.get_json()
    session_id = session.get('session_id')
    action = data.get('action')
    
    if action is None or not isinstance(action, int) or action < 0 or action > 5:
        return jsonify({"success": False, "error": "Invalid action"}), 400

    with game_sessions_lock:
        if not session_id or session_id not in game_sessions:
            return jsonify({"success": False, "error": "No active game session"}), 400
        game = game_sessions[session_id]
    # capture reference to underlying current game before stepping (so we can report last actions)
    pre_game = game._current() if hasattr(game, '_current') else game
    state = game.step(action)
    # last actions come from the pre-step game
    last_step = pre_game.episode_history[-1] if pre_game.episode_history else None
    
    return jsonify({
        "success": True,
        "state": state,
        "human_action": last_step["human_action"] if last_step else None,
        "ai_action": last_step["ai_action"] if last_step else None
    })


@app.route('/api/save_episode', methods=['POST'])
def save_episode():
    """Save the current episode data."""
    data = request.get_json()
    session_id = session.get('session_id')
    player_name = data.get('player_name', 'Anonymous')
    
    with game_sessions_lock:
        if not session_id or session_id not in game_sessions:
            return jsonify({"success": False, "error": "No active game session"}), 400
        game = game_sessions.pop(session_id)

    # If this is a multi-game session, save all games
    if hasattr(game, 'save_all'):
        paths = game.save_all(player_name)
        if paths:
            names = [os.path.basename(p) for p in paths]
            return jsonify({"success": True, "message": f"Saved episodes: {', '.join(names)}"})
        else:
            return jsonify({"success": False, "error": "No episode data to save"}), 400
    else:
        filepath = game.save_episode(player_name)
        if filepath:
            return jsonify({"success": True, "message": f"Episode saved to {os.path.basename(filepath)}"})
        else:
            return jsonify({"success": False, "error": "No episode data to save"}), 400


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


BACKGROUND_LOOP = asyncio.new_event_loop()
def start_background_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

if __name__ == '__main__':
    
    threading.Thread(target=start_background_loop, args=(BACKGROUND_LOOP,), daemon=True).start()

    # Load RL policy agents from checkpoints (once at startup)
    print("[Startup] Loading RL policy agents...")
    load_rl_agents()

    # Start background prewarming loop
    print("[Startup] Starting background prewarming loop...")
    run_coroutine_threadsafe(prewarm_games(), BACKGROUND_LOOP)

    app.run(debug=False, host='0.0.0.0', port=8998)

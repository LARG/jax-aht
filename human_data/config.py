"""
Configuration file for LBF Human Interaction Web Application
Modify these settings to customize the game experience
"""

# ============================================
# Server Configuration
# ============================================
SERVER_HOST = '0.0.0.0'  # '0.0.0.0' for all interfaces, 'localhost' for local only
SERVER_PORT = 8998
DEBUG_MODE = True  # Set to False in production
SECRET_KEY = 'your-secret-key-change-in-production'  # IMPORTANT: Change this!

# ============================================
# Game Environment Configuration
# ============================================
DEFAULT_MAX_STEPS = 50  # Maximum steps per episode
DEFAULT_GRID_SIZE = 7   # Size of the game grid (7x7)
DEFAULT_NUM_FRUITS = 3  # Number of food items
DEFAULT_NUM_AGENTS = 2  # Always 2 (human + AI)

# LBF-specific settings
DEFAULT_FOV = 7              # Field of view for agents
DEFAULT_MAX_AGENT_LEVEL = 2  # Maximum agent level
DEFAULT_FORCE_COOP = True    # Force cooperation for food collection

# Highlighting (0 = human player highlighted, None = no highlighting)
HIGHLIGHT_AGENT_IDX = 0

# ============================================
# AI Agent Configuration
# ============================================
AI_AGENT_TYPE = 'SequentialFruitAgent'  # Type of AI agent to use

# AI Strategy Options:
# - 'lexicographic': Top-to-bottom, left-to-right order
# - 'reverse_lexicographic': Bottom-to-top, right-to-left
# - 'column_major': Left-to-right, top-to-bottom
# - 'reverse_column_major': Right-to-left, bottom-to-top
# - 'nearest_agent': Closest food first (recommended)
# - 'farthest_agent': Farthest food first
AI_ORDERING_STRATEGY = 'nearest_agent'

# ============================================
# Data Collection Configuration
# ============================================
DATA_DIR = 'collected_data'  # Directory to save episode data
SAVE_FORMAT = 'json'         # Format for saved episodes (json)
AUTO_CREATE_DATA_DIR = True  # Automatically create data directory
AUTO_SAVE_EPISODES = True    # Automatically save episodes when they complete

# Filename pattern for saved episodes
# Available variables: {session_id}, {steps}, {timestamp}
EPISODE_FILENAME_PATTERN = 'episode_{timestamp}_{session_id}_{steps}steps.json'

# ============================================
# Session Configuration
# ============================================
# Maximum number of concurrent sessions (0 = unlimited)
MAX_CONCURRENT_SESSIONS = 0

# Session timeout in seconds (0 = no timeout)
SESSION_TIMEOUT = 0

# ============================================
# JAX Configuration
# ============================================
# Force JAX to use CPU (recommended for web server)
USE_CPU_ONLY = True

# JAX debug settings
DISABLE_JIT = False  # Set to True for debugging

# ============================================
# UI Configuration
# ============================================
# Canvas size in pixels
CANVAS_WIDTH = 600
CANVAS_HEIGHT = 600

# Colors (RGB normalized 0-1)
COLORS = {
    'player': (1.0, 0.42, 0.42),      # Red - Human player
    'ai': (0.31, 0.80, 0.77),         # Teal - AI agent
    'food': (0.58, 0.88, 0.83),       # Light teal - Food items
    'grid': (0.94, 0.94, 0.94),       # Light gray - Grid background
    'grid_lines': (0.87, 0.87, 0.87), # Grid lines
}

# ============================================
# Logging Configuration
# ============================================
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE = None     # None for console only, or specify path

# ============================================
# Advanced Settings
# ============================================
# Enable CORS (Cross-Origin Resource Sharing)
ENABLE_CORS = True

# Share rewards between agents
SHARE_REWARDS = True

# Allow reset during episode
ALLOW_RESET = True

# Require player name for saving episodes
REQUIRE_PLAYER_NAME = False

# ============================================
# Experimental Features
# ============================================
# Enable experimental features (not fully tested)
ENABLE_REPLAY_MODE = False
ENABLE_MULTIPLAYER = False
ENABLE_DIFFICULTY_LEVELS = False

# ============================================
# Custom Game Variants
# ============================================
# You can define custom game configurations here
GAME_VARIANTS = {
    'easy': {
        'max_steps': 100,
        'grid_size': 5,
        'num_fruits': 2,
        'max_agent_level': 2,
    },
    'standard': {
        'max_steps': 50,
        'grid_size': 7,
        'num_fruits': 3,
        'max_agent_level': 2,
    },
    'hard': {
        'max_steps': 30,
        'grid_size': 10,
        'num_fruits': 5,
        'max_agent_level': 3,
    },
}

# Default variant to use
DEFAULT_VARIANT = 'standard'

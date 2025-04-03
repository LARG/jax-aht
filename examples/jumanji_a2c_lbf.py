import os
import requests
import warnings

warnings.filterwarnings("ignore")

from jumanji.training.train import train
from hydra import compose, initialize

env = "lbf"  # @param ['bin_pack', 'cleaner', 'connector', 'cvrp', 'game_2048', 'graph_coloring', 'job_shop', 'knapsack', 'lbf', 'maze', 'minesweeper', 'mmst', 'multi_cvrp', 'robot_warehouse', 'rubiks_cube', 'snake', 'sudoku', 'tetris', 'tsp']
agent = "a2c"  # @param ['random', 'a2c']

def download_file(url: str, file_path: str) -> None:
    # Send an HTTP GET request to the URL
    response = requests.get(url)
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        with open(file_path, "wb") as f:
            f.write(response.content)
    else:
        print("Failed to download the file.")

os.makedirs("jumanji_configs", exist_ok=True)
config_url = "https://raw.githubusercontent.com/instadeepai/jumanji/main/jumanji/training/configs/config.yaml"
download_file(config_url, "jumanji_configs/config.yaml")
env_url = f"https://raw.githubusercontent.com/instadeepai/jumanji/main/jumanji/training/configs/env/{env}.yaml"
os.makedirs("jumanji_configs/env", exist_ok=True)
download_file(env_url, f"jumanji_configs/env/{env}.yaml")

# training code
# config path located at continual-aht/jumanji_configs
with initialize(version_base=None, config_path="../jumanji_configs"):
    cfg = compose(
        config_name="config.yaml",
        overrides=[
            f"env={env}",
            f"agent={agent}",
            "logger.type=terminal",
            "logger.save_checkpoint=true",
        ],
    )

train(cfg)
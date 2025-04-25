import hydra
from omegaconf import OmegaConf

from ppo_ego import run_ego_training
from train_br_vs_single_agent import run_br_training
from common.wandb_visualizations import Logger

@hydra.main(version_base=None, config_path="configs", config_name="default")
def run_training(cfg):
    '''Runs the ego agent training against a fixed partner population. This script is 
    mostly just used for debugging.'''
    print(OmegaConf.to_yaml(cfg, resolve=True))
    wandb_logger = Logger(cfg)
    if cfg["algorithm"]["ALG"] == "ppo_ego":
        run_ego_training(cfg, wandb_logger)
    elif cfg["algorithm"]["ALG"] == "ppo_br":
        run_br_training(cfg, wandb_logger)
    # Cleanup
    wandb_logger.close()


if __name__ == '__main__':
    run_training()
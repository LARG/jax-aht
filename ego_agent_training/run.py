'''Main entry point for running ego agent training algorithms against a fixed partner population.'''
import hydra
from omegaconf import OmegaConf

from common.plot_utils import get_metric_names
from common.wandb_visualizations import Logger
from evaluation.heldout_eval import run_heldout_evaluation, log_heldout_metrics
from ppo_ego import run_ego_training as run_ego_ppo_training
from liam_ego import run_ego_training as run_ego_liam_training
from meliba_ego import run_ego_training as run_ego_meliba_training
from ego_agent_training.ppo_br import run_br_training


@hydra.main(version_base=None, config_path="configs", config_name="base_config_ego")
def run_training(cfg):
    '''Runs the ego agent training against a fixed partner population.'''
    print(OmegaConf.to_yaml(cfg, resolve=True))
    wandb_logger = Logger(cfg)

    if cfg["algorithm"]["ALG"] == "ppo_ego":
        ego_params, ego_policy, init_ego_params = run_ego_ppo_training(cfg, wandb_logger)
    elif cfg["algorithm"]["ALG"] == "ppo_br":
        ego_params, ego_policy, init_ego_params = run_br_training(cfg, wandb_logger)
    elif cfg["algorithm"]["ALG"] == "liam_ego":
        ego_params, ego_policy, init_ego_params = run_ego_liam_training(cfg, wandb_logger)
    elif cfg["algorithm"]["ALG"] == "meliba_ego":
        ego_params, ego_policy, init_ego_params = run_ego_meliba_training(cfg, wandb_logger)

    if cfg["run_heldout_eval"]:
        metric_names = get_metric_names(cfg["ENV_NAME"])
        eval_metrics, ego_names, heldout_names = run_heldout_evaluation(cfg, ego_policy, ego_params, init_ego_params, ego_as_2d=False)
        log_heldout_metrics(cfg, wandb_logger, eval_metrics, ego_names, heldout_names, metric_names, ego_as_2d=False)

    # Cleanup
    wandb_logger.close()


if __name__ == '__main__':
    run_training()
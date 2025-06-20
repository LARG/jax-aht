'''Main entry point for running open-ended training algorithms.'''
import hydra
from omegaconf import OmegaConf

from evaluation.heldout_eval import run_heldout_evaluation, log_heldout_metrics
from common.plot_utils import get_metric_names
from common.wandb_visualizations import Logger
from open_ended_training.rotate import run_rotate
from open_ended_minimax import run_minimax
from paired import run_paired

@hydra.main(version_base=None, config_path="configs", config_name="base_config_oel")
def run_training(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True))
    wandb_logger = Logger(cfg)

    if cfg.algorithm["ALG"] == "rotate":
        ego_policy, final_ego_params, init_ego_params = run_rotate(cfg, wandb_logger)
    elif cfg.algorithm["ALG"] == "open_ended_minimax":
        ego_policy, final_ego_params, init_ego_params = run_minimax(cfg, wandb_logger)
    elif cfg.algorithm["ALG"] == "paired":
        ego_policy, final_ego_params, init_ego_params = run_paired(cfg, wandb_logger)
    else:
        raise NotImplementedError("Selected method not implemented.")

    if cfg["run_heldout_eval"]:
        metric_names = get_metric_names(cfg["task"]["ENV_NAME"])
        ego_as_2d = False if cfg.algorithm["ALG"] in ["paired"] else True
        eval_metrics, ego_names, heldout_names = run_heldout_evaluation(
            cfg, ego_policy, final_ego_params, init_ego_params, ego_as_2d=ego_as_2d
        )
        log_heldout_metrics(
            cfg, wandb_logger, eval_metrics, ego_names, heldout_names, metric_names, ego_as_2d=ego_as_2d
        )

    wandb_logger.close()

if __name__ == '__main__':
    run_training()
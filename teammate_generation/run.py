'''Main entry point for running teammate generation algorithms.'''
import hydra
from omegaconf import OmegaConf

from evaluation.heldout_eval import run_heldout_evaluation, log_heldout_metrics
from common.plot_utils import get_metric_names
from common.wandb_visualizations import Logger
from teammate_generation.BRDiv import run_brdiv
from teammate_generation.LBRDiv import run_lbrdiv
from teammate_generation.CoMeDi import run_comedi
from teammate_generation.fcp import run_fcp
from teammate_generation.train_ego import train_ego_agent


@hydra.main(version_base=None, config_path="configs", config_name="base_config_teammate")
def run_training(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True))
    wandb_logger = Logger(cfg)
    cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # train partner population
    if cfg["algorithm"]["ALG"] == "brdiv":
        partner_params, partner_population = run_brdiv(cfg, wandb_logger)
    elif cfg["algorithm"]["ALG"] == "fcp":
        partner_params, partner_population = run_fcp(cfg, wandb_logger)
    elif cfg["algorithm"]["ALG"] == "lbrdiv":
        partner_params, partner_population = run_lbrdiv(cfg, wandb_logger)
    elif cfg["algorithm"]["ALG"] == "comedi":
        partner_params, partner_population = run_comedi(cfg, wandb_logger)
    else:
        raise NotImplementedError("Selected method not implemented.")
    
    metric_names = get_metric_names(cfg["task"]["ENV_NAME"])
    if cfg["train_ego"]:
        ego_params, ego_policy, init_ego_params = train_ego_agent(cfg, wandb_logger, partner_params, partner_population)
    
    if cfg["run_heldout_eval"]:
        eval_metrics, ego_names, heldout_names = run_heldout_evaluation(cfg, ego_policy, ego_params, init_ego_params, ego_as_2d=False)
        log_heldout_metrics(cfg, wandb_logger, eval_metrics, ego_names, heldout_names, metric_names, ego_as_2d=False)
    wandb_logger.close()

if __name__ == '__main__':
    run_training()

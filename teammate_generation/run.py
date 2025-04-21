import hydra
from omegaconf import OmegaConf
from common.wandb_visualizations import Logger

from BRDiv import run_brdiv
from fcp import train_fcp_partners
from ego_agent_training.ppo_ego import log_metrics as log_ego_metrics
from evaluation.heldout_eval import run_heldout_evaluation, log_heldout_metrics
from common.plot_utils import get_metric_names
from train_ego import train_ego_agent


@hydra.main(version_base=None, config_path="configs", config_name="default")
def run_training(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True))
    wandb_logger = Logger(cfg)
    cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # train partner population
    if cfg["algorithm"]["ALG"] == "brdiv":
        partner_params, partner_population = run_brdiv(cfg, wandb_logger)
    elif cfg["algorithm"]["ALG"] == "fcp":
        partner_params, partner_population = train_fcp_partners(cfg, wandb_logger)
    else:
        raise NotImplementedError("Selected method not implemented.")
    
    metric_names = get_metric_names(cfg["ENV_NAME"])
    if cfg["train_ego"]:
        out, ego_policy, init_ego_params = train_ego_agent(cfg["ego_train_algorithm"], wandb_logger, partner_params, partner_population)
        log_ego_metrics(cfg, out, wandb_logger, metric_names)
    
    if cfg["heldout_eval"]:
        eval_metrics, ego_names, heldout_names = run_heldout_evaluation(cfg, ego_policy, out['checkpoints'], init_ego_params)
        log_heldout_metrics(cfg, wandb_logger, eval_metrics, ego_names, heldout_names, metric_names)
    wandb_logger.close()

if __name__ == '__main__':
    run_training()
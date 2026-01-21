#!/usr/bin/env python
"""
Example script showing how to run BRDiv with monitoring enabled.

This script demonstrates how to:
1. Enable BRDiv monitoring to track wall-clock time and returns
2. Run the BRDiv algorithm
3. Automatically save the data to a JSON file
4. Generate plots of time vs returns

Usage:
    python teammate_generation/run_brdiv_monitored.py \
        algorithm=brdiv/lbf \
        task=lbf \
        label=test_brdiv_monitored \
        enable_brdiv_monitoring=true \
        brdiv_monitoring_dir=./brdiv_monitoring_output \
        run_heldout_eval=false \
        train_ego=false
"""

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
    """Run teammate generation with monitoring enabled."""
    print(OmegaConf.to_yaml(cfg, resolve=True))
    
    # Convert to container to enable monitoring parameters
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    
    # Enable monitoring for BRDiv
    cfg_dict["enable_brdiv_monitoring"] = cfg_dict.get("enable_brdiv_monitoring", True)
    cfg_dict["brdiv_monitoring_dir"] = cfg_dict.get("brdiv_monitoring_dir", "./brdiv_monitoring")
    
    wandb_logger = Logger(cfg)

    # train partner population
    if cfg_dict["algorithm"]["ALG"] == "brdiv":
        partner_params, partner_population = run_brdiv(cfg_dict, wandb_logger)
    elif cfg_dict["algorithm"]["ALG"] == "fcp":
        partner_params, partner_population = run_fcp(cfg_dict, wandb_logger)
    elif cfg_dict["algorithm"]["ALG"] == "lbrdiv":
        partner_params, partner_population = run_lbrdiv(cfg_dict, wandb_logger)
    elif cfg_dict["algorithm"]["ALG"] == "comedi":
        partner_params, partner_population = run_comedi(cfg_dict, wandb_logger)
    else:
        raise NotImplementedError("Selected method not implemented.")
    
    metric_names = get_metric_names(cfg_dict["task"]["ENV_NAME"])
    
    if cfg_dict.get("train_ego", False):
        ego_params, ego_policy, init_ego_params = train_ego_agent(
            cfg_dict, wandb_logger, partner_params, partner_population
        )
    
    if cfg_dict.get("run_heldout_eval", False):
        eval_metrics, ego_names, heldout_names = run_heldout_evaluation(
            cfg_dict, ego_policy, ego_params, init_ego_params, ego_as_2d=False
        )
        log_heldout_metrics(
            cfg_dict, wandb_logger, eval_metrics, ego_names, heldout_names,
            metric_names, ego_as_2d=False
        )
    
    wandb_logger.close()
    print(f"\nBRDiv monitoring output saved to: {cfg_dict['brdiv_monitoring_dir']}")


if __name__ == '__main__':
    run_training()

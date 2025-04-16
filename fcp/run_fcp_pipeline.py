'''
This script implements the full training and evaluation pipeline for Fictitious Co-Play (FCP). 
The steps implemented are: 
1. Generate training and testing teammates using IPPO. 
2. Train FCP agent using generated training teammates. 
3. Evaluate FCP agent against training and testing teammates.
'''
import hydra
import logging
from omegaconf import OmegaConf

from common.save_load_utils import save_train_run, load_checkpoints
from common.plot_utils import get_metric_names
from common.wandb_visualizations import Logger
from fcp.fcp_eval import main as eval_main
from fcp.train_partners import train_partners_in_parallel
from fcp.fcp_train import train_fcp_ego_agent

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@hydra.main(version_base=None, config_path="configs", config_name="fcp_master")
def fcp_pipeline(config):
    '''Runs the full FCP training and evaluation pipeline.'''
    # initialize config
    logger = Logger(config)
    print(OmegaConf.to_yaml(config, resolve=True))
    
    config = OmegaConf.to_container(config, resolve=True)
    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    # Train/Load partner policies
    train_partner_path = config["PARTNER_TRAIN_ARGS"]["TRAIN_PATH"]
    if train_partner_path is not None:
        train_partner_ckpts = load_checkpoints(train_partner_path)
        log.info(f"Loaded training partners from {train_partner_path}")
    else:
        log.info("Training the training partners...")
        train_out = train_partners_in_parallel(config["PARTNER_TRAIN_ARGS"], config["PARTNER_TRAIN_ARGS"]["TRAIN_SEED"])
        savepath = save_train_run(train_out, savedir, savename="train_partners")
        train_partner_ckpts = train_out["checkpoints"]
        log.info(f"Saved train partner data to {savepath}")

    eval_partner_path = config["PARTNER_TRAIN_ARGS"]["EVAL_PATH"]
    if eval_partner_path is not None:
        eval_partner_ckpts = load_checkpoints(eval_partner_path)
        log.info(f"Loaded testing partners from {eval_partner_path}")
    else:
        log.info("Training the testing partners...")
        eval_out = train_partners_in_parallel(config["PARTNER_TRAIN_ARGS"], config["PARTNER_TRAIN_ARGS"]["EVAL_SEED"])
        eval_partner_ckpts = eval_out["checkpoints"]
        savepath = save_train_run(eval_out, savedir, savename="eval_partners")
        log.info(f"Saved test partner data to {savepath}")

    # Train FCP agent
    log.info("Training FCP Agent...")
    train_cfg = config["FCP_TRAIN_ARGS"]
    fcp_out = train_fcp_ego_agent(train_cfg, train_partner_ckpts, logger)
    
    # Perform evaluation
    metric_names = get_metric_names(config["ENV_NAME"])
    log.info("Evaluating FCP Agent...")
    eval_main(config=config["FCP_EVAL_ARGS"],
              ego_config=config["FCP_TRAIN_ARGS"],
              partner_config=config["PARTNER_TRAIN_ARGS"],
              eval_savedir=savedir, 
              ego_ckpts=fcp_out["checkpoints"], 
              train_partner_ckpts=train_partner_ckpts, 
              test_partner_ckpts=eval_partner_ckpts, 
              num_episodes=config["NUM_EVAL_EPISODES"],
              metric_names=metric_names)

    return fcp_out

if __name__ == "__main__":
    fcp_pipeline()
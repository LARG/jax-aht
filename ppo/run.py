import hydra
from omegaconf import OmegaConf

from common.wandb_visualizations import Logger as WandBLogger
from ippo import run_ippo


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(config):
    print(OmegaConf.to_yaml(config, resolve=True))
    logger = WandBLogger(config)

    config = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    
    run_ippo(config, logger)
        
    logger.close()

if __name__ == "__main__":
    main()
import hydra
from omegaconf import OmegaConf

from common.wandb_visualizations import Logger
from ippo import run_ippo


@hydra.main(version_base=None, config_path="configs", config_name="base_config_marl")
def main(config):
    print(OmegaConf.to_yaml(config, resolve=True))
    wandb_logger = Logger(config)

    if config.algorithm["ALG"] == "ippo":
        run_ippo(config, wandb_logger)
    else:
        raise NotImplementedError(f"Algorithm {config['ALG']} not implemented.")
        
    wandb_logger.close()

if __name__ == "__main__":
    main()
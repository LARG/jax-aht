import hydra
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path=".", config_name="test_config")
def main(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print(OmegaConf.to_container(cfg, resolve=True))
    breakpoint()

if __name__ == "__main__":
    main()

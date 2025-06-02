import hydra
from omegaconf import OmegaConf
import logging

from regret_evaluator import run_regret_evaluation
from heldout_evaluator import run_heldout_evaluation
from evaluation.generate_xp_matrix import run_heldout_xp_evaluation

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="configs", config_name="default")
def run_evaluation(cfg):
    '''Run evaluation. 
    All evaluators assume that the path to the ego agent is provided at config["ego_agent"]["path"]
    and that all information necessary to properly initialize the ego agent is provided at config["ego_agent"]
    '''
    print(OmegaConf.to_yaml(cfg, resolve=True))

    if "regret" in cfg["name"]:
        run_regret_evaluation(cfg)

    elif "heldout_ego" in cfg["name"]:
        run_heldout_evaluation(cfg, print_metrics=True)

    elif "heldout_xp" in cfg["name"]:
        run_heldout_xp_evaluation(cfg, print_metrics=True)

    else: 
        raise ValueError(f"Evaluator {cfg['name']} not found.")

if __name__ == '__main__':
    run_evaluation()
import hydra
from omegaconf import OmegaConf
import jax
import logging

from regret_evaluator import run_regret_evaluation
from common.save_load_utils import load_checkpoints

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="configs", config_name="default")
def run_evaluation(cfg):
    '''Run regret-maximizing evaluation. Assumes all information necessary to properly 
    initialize the ego agent is provided at config["algorithm"]'''

    print(OmegaConf.to_yaml(cfg, resolve=True))

    #  Load ego agent
    ego_agent_path = cfg["EGO_AGENT_PATH"]
    ego_agent_ckpt = load_checkpoints(ego_agent_path)

    n_seeds, m_ckpts = jax.tree.leaves(ego_agent_ckpt)[0].shape[:2]
    log.info(f"Ego agent checkpoint has {n_seeds} seeds and {m_ckpts} checkpoints.")
    log.info("Selecting zeroth seed and last checkpoint for evaluation.")
    ego_agent_params = jax.tree.map(lambda x: x[0, -1], ego_agent_ckpt)

    run_regret_evaluation(cfg, ego_agent_params)


if __name__ == '__main__':
    run_evaluation()
"""Evaluate a finished benchmark run's ego agent against the human proxy teammate.

Downloads the ``ego_train_run`` artifact of a teammate-generation benchmark run,
rebuilds the ego policy from the run's own ``ego_train_algorithm`` config, and runs
the standard heldout evaluation restricted to ``human_proxy`` (or any subset of
partners). Results are logged to wandb exactly like a training run's heldout eval
(``HeldoutEval/FinalEgoVsHeldout-*-CI`` table + ``heldout_eval_metrics`` artifact),
so scripts/paper_vis can merge them by partner name.

Run from repo root, e.g.:
    PYTHONPATH=. python evaluation/run_heldout_ego_human_proxy.py \
        task=overcooked-v1/cramped_room source_run_id=n1mplxeg
"""

import logging
import os
import tempfile

import hydra
import jax
import wandb
from omegaconf import OmegaConf, open_dict

from common.plot_utils import get_metric_names
from common.save_load_utils import load_train_run
from common.wandb_visualizations import Logger
from ego_agent_training.utils import initialize_ego_agent
from envs import make_env
from envs.log_wrapper import LogWrapper
from evaluation.heldout_runner import log_heldout_metrics, run_heldout_evaluation

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _download_ego_train_run(run, root):
    for artifact in run.logged_artifacts():
        if artifact.type == "train_run" and artifact.name.startswith("ego_train_run"):
            log.info(f"Downloading {artifact.name} ({artifact.size / 1e6:.0f} MB)")
            return artifact.download(root=root)
    raise ValueError(f"Run {run.id} has no ego_train_run artifact.")


@hydra.main(
    version_base=None, config_path="configs", config_name="heldout_ego_human_proxy"
)
def main(cfg):
    api = wandb.Api()
    source = api.run(f"{cfg.source_entity}/{cfg.source_project}/{cfg.source_run_id}")
    source_cfg = source.config
    if source_cfg["TASK_NAME"] != cfg.TASK_NAME:
        raise ValueError(
            f"task={cfg.TASK_NAME} but source run {source.id} is {source_cfg['TASK_NAME']}"
        )
    ego_cfg = source_cfg["algorithm"]["ego_train_algorithm"]

    with open_dict(cfg):
        cfg.method = source_cfg["algorithm"]["ALG"]
        cfg.name = f"{cfg.TASK_NAME}/heldout_ego_bc/{cfg.method}/{cfg.label}"
        cfg.source_run_name = source.name
        # log_heldout_metrics reads algorithm.ALG; keep the full source algorithm config
        # for provenance.
        cfg.algorithm = source_cfg["algorithm"]
        cfg.heldout_set = {
            cfg.TASK_NAME: {p: cfg.heldout_set[cfg.TASK_NAME][p] for p in cfg.partners}
        }
    print(OmegaConf.to_yaml(cfg, resolve=True))

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_dir = _download_ego_train_run(source, tmp)
        out = load_train_run(os.path.abspath(ckpt_dir))
    ego_params = out["final_params"]
    num_seeds, num_ego_train_seeds = jax.tree.leaves(ego_params)[0].shape[:2]
    log.info(f"Loaded ego params: {num_seeds} seeds x {num_ego_train_seeds} ego seeds")

    # Older runs did not copy the env settings into the ego sub-config (and some
    # have no ENV_KWARGS at all); fall back to the source run, then the task config.
    env_name = ego_cfg.get("ENV_NAME") or source_cfg["ENV_NAME"]
    env_kwargs = (
        ego_cfg.get("ENV_KWARGS") or source_cfg.get("ENV_KWARGS") or cfg.ENV_KWARGS
    )
    env_kwargs = (
        OmegaConf.to_container(env_kwargs, resolve=True)
        if OmegaConf.is_config(env_kwargs)
        else env_kwargs
    )
    log.info(f"Ego env: {env_name} {env_kwargs}")
    env = LogWrapper(make_env(env_name, env_kwargs))
    ego_policy, init_ego_params = initialize_ego_agent(
        ego_cfg, env, jax.random.PRNGKey(0)
    )

    wandb_logger = Logger(cfg)
    cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    eval_metrics, ego_names, heldout_names = run_heldout_evaluation(
        cfg,
        ego_policy,
        ego_params,
        init_ego_params,
        ego_as_2d=False,
        ego_test_mode=cfg["ego_test_mode"],
    )
    log_heldout_metrics(
        cfg,
        wandb_logger,
        eval_metrics,
        ego_names,
        heldout_names,
        get_metric_names(cfg["ENV_NAME"]),
        ego_as_2d=False,
    )
    wandb_logger.close()


if __name__ == "__main__":
    main()

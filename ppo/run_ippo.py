import hydra
from omegaconf import OmegaConf
import os
import jax

from common.save_load_utils import save_train_run
from common.wandb_visualizations import Logger as WandBLogger
from common.plot_utils import plot_train_metrics, get_stats
from ippo import make_train


@hydra.main(version_base=None, config_path="configs", config_name="ippo_master")
def main(config):
    print(OmegaConf.to_yaml(config, resolve=True))
    
    # initialize logger
    logger = WandBLogger(config)

    config_dict = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    rng = jax.random.PRNGKey(config_dict["SEED"])
    rngs = jax.random.split(rng, config_dict["NUM_SEEDS"])
    with jax.disable_jit(False):
        train_jit = jax.jit(jax.vmap(make_train(config_dict)))
        out = train_jit(rngs)
    
    # save train run output as pickle file and to wandb as artifact
    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    out_savepath = save_train_run(out, savedir, savename="ippo_train_run")
    logger.log_artifact(name="ippo_train_run", path=out_savepath, type_name="train_run")

    if config["ENV_NAME"] == "lbf":
        metric_names = ("percent_eaten", "returned_episode_returns")
    elif config["ENV_NAME"] == "overcooked-v2":
        metric_names = ("shaped_reward", "returned_episode_returns")
    else: 
        metric_names = ("returned_episode_returns")
    
    # Generate plots
    all_stats = get_stats(out["metrics"], metric_names, num_controlled_agents=1)
    figures, _ = plot_train_metrics(all_stats, 
                                    config_dict["NUM_STEPS"], 
                                    config_dict["NUM_ENVS"],
                                    savedir=savedir if config["local_logger"]["save_figures"] else None,
                                    savename="ippo_train_metrics",
                                    show_plots=False
                                    )
    
    # Log plots to wandb
    for stat_name, fig in figures.items():
        logger.log({f"train_metrics/{stat_name}": fig})
        
    # Cleanup locally logged out file
    if not config["local_logger"]["save_results"]:
        os.remove(out_savepath)

    # TODO: in the future, add video logging feature

    logger.close()

if __name__ == "__main__":
    main()
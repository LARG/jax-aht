import hydra
from omegaconf import OmegaConf
import os
import jax

from common.save_load_utils import save_train_run
from common.wandb_visualizations import Logger as WandBLogger
from common.plot_utils import plot_train_metrics, get_stats, get_metric_names
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
    
    # save train run output and log to wandb as artifact
    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    # TODO: in the future, add video logging feature
    out_savepath = save_train_run(out, savedir, savename="saved_train_run")
    if config["logger"]["log_train_out"]:
        logger.log_artifact(name="saved_train_run", path=out_savepath, type_name="train_run")
        # Cleanup locally logged out file
    if not config["local_logger"]["save_train_out"]:
        os.remove(out_savepath)
   

    metric_names = get_metric_names(config["ENV_NAME"])

    # Generate plots
    all_stats = get_stats(out["metrics"], metric_names)
    figures, _ = plot_train_metrics(all_stats, 
                                    config_dict["ROLLOUT_LENGTH"], 
                                    config_dict["NUM_ENVS"],
                                    savedir=savedir if config["local_logger"]["save_figures"] else None,
                                    savename="ippo_train_metrics",
                                    show_plots=False
                                    )
    
    # Log plots to wandb
    for stat_name, fig in figures.items():
        logger.log({f"train_metrics/{stat_name}": fig})
        
    logger.close()

if __name__ == "__main__":
    main()
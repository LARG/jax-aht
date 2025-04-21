''''Implementation of heldout evaluation helper functions used by learners.'''
import jax
import numpy as np
import shutil
import hydra

from common.save_load_utils import save_train_run
from envs import make_env
from envs.log_wrapper import LogWrapper
from evaluation.heldout_evaluator import eval_egos_vs_heldouts, load_heldout_set


def run_heldout_evaluation(config, ego_policy, ego_params, init_ego_params):
    '''Run heldout evaluation given an ego policy, ego params, and init_ego_params.'''
    env = make_env(config["ENV_NAME"], config["ENV_KWARGS"])
    env = LogWrapper(env)
    
    rng = jax.random.PRNGKey(config["EVAL_SEED"])
    rng, heldout_init_rng, eval_rng = jax.random.split(rng, 3)
    
    # flatten ego checkpoints and idx labels
    flattened_ego_params = jax.tree.map(lambda x, y: x.reshape((-1,) + y.shape), ego_params, init_ego_params)      
    num_ego_agents = jax.tree.leaves(flattened_ego_params)[0].shape[0]
    ego_names = [f"ego ({i})" for i in range(num_ego_agents)]
    
    # load heldout agents
    heldout_cfg = config["heldout_set"][config["ENV_NAME"]]
    heldout_agents = load_heldout_set(heldout_cfg, env, config["ENV_NAME"], config["ENV_KWARGS"], heldout_init_rng)
    heldout_agent_list = list(heldout_agents.values())
    heldout_names = list(heldout_agents.keys())

    # run evaluation
    eval_metrics = eval_egos_vs_heldouts(config, env, eval_rng, config["NUM_EVAL_EPISODES"], 
                                        ego_policy, flattened_ego_params, heldout_agent_list)
    
    return eval_metrics, ego_names, heldout_names

def log_heldout_metrics(config, logger, eval_metrics, 
        ego_names, heldout_names, metric_names: tuple, 
        log_dim0_as_curve: bool = False):
    '''Log heldout evaluation metrics.'''
    num_oel_iter, _, num_eval_episodes, _ = eval_metrics[metric_names[0]].shape
    table_data = []
    for metric_name in metric_names:
        # shape of eval_metrics is (num_iter/num_seeds, num_heldout_agents, num_eval_episodes, 2)
        metric_mean = eval_metrics[metric_name][..., 0].mean(axis=(-1)) # shape (num_iter/num_seeds, num_heldout_agents)
        metric_std = eval_metrics[metric_name][..., 0].std(axis=(-1)) # shape (num_iter/num_seeds, num_heldout_agents)
        metric_ci = 1.96 * metric_std / np.sqrt(num_eval_episodes) # shape (num_iter/num_seeds, num_heldout_agents)
        # log curve
        if log_dim0_as_curve:
            for i in range(num_oel_iter):
                logger.log_item(f"HeldoutEval/AvgEgo_{metric_name}", metric_mean[i].mean(), iter=i)        
        mean0, ci0 = metric_mean[-1], metric_ci[-1]

        mean_and_ci_str = [f"{mean0[i]:.3f} ± {ci0[i]:.3f}" for i in range(len(mean0))]
        table_data.append(mean_and_ci_str)
        
    # log table where the columns are the metric names and the rows are the heldout agents vs the last ego agent
    table_data = np.array(table_data) # shape (num_metrics, num_heldout_agents)
    logger.log_xp_matrix("HeldoutEval/FinalEgoVsHeldout-Mean-CI", table_data.T, 
                         columns=list(metric_names), rows=heldout_names)
    logger.commit()

    # Saving artifacts
    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    out_savepath = save_train_run(eval_metrics, savedir, savename="heldout_eval_metrics")
    if config["logger"]["log_eval_out"]:
        logger.log_artifact(name="heldout_eval_metrics", path=out_savepath, type_name="eval_metrics")
    
    # Cleanup locally logged out file
    if not config["local_logger"]["save_eval_out"]:
        shutil.rmtree(out_savepath)


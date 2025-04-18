import os
import logging

import jax

from envs import make_env
from envs.log_wrapper import LogWrapper
from common.agent_interface import MLPActorCriticPolicy
from common.save_load_utils import load_checkpoints, save_train_run
from common.plot_utils import plot_xp_from_eval_metrics
from common.run_episodes import run_episodes
from common.initialize_agents import initialize_s5_agent, initialize_mlp_agent, initialize_rnn_agent

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def eval_ego_agent(config, ego_config, partner_config, ego_params, partner_params, 
                   num_episodes: int
                   ):
    '''
    config: config dict specifying evaluation parameters
    ego_config: config dict specifying ego agent parameters
    partner_config: config dict specifying partner agent parameters
    ego_params: a pytree where each leaf contains N seeds, and M checkpoints of trained ego (FCP) agents 
                    (e.g. {"params": <pytree with leaves of shape (N, M, ...)>})
    partner_params: a pytree where each leaf contains N seeds, and M checkpoints of trained IPPO agents.
                    All eval checkpoints are treated as one pool of partners.
    For each ego agent (each seed, each checkpoint) we evaluate it against every eval checkpoint partner in the pool,
    running num_episodes per pairing.
        
    Returns a dictionary with key "episode_returns" whose value is a jnp.array of shape 
       (num_ego_seeds, num_ego_ckpts, num_partner_total, num_episodes)
    '''
    # --- 1. Prepare the environment ---
    env = make_env(config["ENV_NAME"], config["ENV_KWARGS"])
    env = LogWrapper(env)

    num_agents = env.num_agents
    assert num_agents == 2, "This eval code assumes exactly 2 agents."

    base_rng = jax.random.PRNGKey(config["SEED"])
    rng, init_ego_rng, init_partner_rng = jax.random.split(base_rng, 3)

    # Initialize partner policy
    partner_policy = MLPActorCriticPolicy(
        action_dim=env.action_space(env.agents[1]).n,
        obs_dim=env.observation_space(env.agents[1]).shape[0],
    )

    # Initialize ego agent
    if ego_config["EGO_ACTOR_TYPE"] == "s5":
        ego_policy, init_params = initialize_s5_agent(ego_config, env, init_ego_rng)
    elif ego_config["EGO_ACTOR_TYPE"] == "mlp":
        ego_policy, init_params = initialize_mlp_agent(ego_config, env, init_ego_rng)
    elif ego_config["EGO_ACTOR_TYPE"] == "rnn":
        # WARNING: currently the RNN policy is not working. 
        # TODO: fix this!
        ego_policy, init_params = initialize_rnn_agent(ego_config, env, init_ego_rng)

    sample_leaf = jax.tree_util.tree_leaves(ego_params)[0]
    n_ego, m_ego = sample_leaf.shape[:2]

    # For eval checkpoints, flatten the first two dims to form a partner pool.
    def flatten_ckpt(x):
        return x.reshape(-1, *x.shape[2:])
    
    flattened_partner_params = jax.tree.map(flatten_ckpt, partner_params)
    sample_leaf_eval = jax.tree_util.tree_leaves(flattened_partner_params)[0]
    num_partner_total = sample_leaf_eval.shape[0]

    def eval_pair_fn(rng, ego_param, partner_param):
        return run_episodes(rng, env, 
                            agent_0_param=ego_param, agent_0_policy=ego_policy,
                            agent_1_param=partner_param, agent_1_policy=partner_policy,
                            max_episode_steps=ego_config["ROLLOUT_LENGTH"],
                            num_eps=num_episodes)

    # --- 5. Outer function to be jitted and vmapped over ego agent seeds ---
    @jax.jit
    def outer_eval(rngs):
        """
        rngs: an array of shape (n_ego, key), one key per ego agent seed.
        Returns a pytree with leaves of shape (n_ego, m_ego, num_partner_total, num_episodes, ...).
        """
        # For each seed, use its corresponding fcp parameters.
        # Here, ego_params has shape (n_ego, m_ego, ...).
        def eval_ego_seed(rng, ego_params_seed):
            '''Evaluate all checkpoints (m_ego) for a single ego agent seed against 
            all eval partners (num_partner_total).'''
            # For each checkpoint (m_ego) we need a set of rng keys for evaluation against all eval partners.
            total_rngs = jax.random.split(rng, m_ego * num_partner_total)
            total_rngs = total_rngs.reshape(m_ego, num_partner_total, -1)
            # For one checkpoint, evaluate over the partner pool defined by flattened_partner_params, 
            # which has shape (num_partner_total, ...)
            def eval_for_checkpoint(ego_param, rngs_ckpt):
                return jax.vmap(
                    lambda rng, partner_param: eval_pair_fn(rng, ego_param, partner_param),
                    in_axes=(0, 0)
                )(rngs_ckpt, flattened_partner_params)
            # Vmap over the m_ego checkpoints.
            return jax.vmap(eval_for_checkpoint, in_axes=(0, 0))(ego_params_seed, total_rngs)
        return jax.vmap(eval_ego_seed, in_axes=(0, 0))(rngs, ego_params)

    # --- 6. JIT-compile and run ---
    # Prepare one RNG per FCP seed (n_ego total).
    rngs = jax.random.split(rng, n_ego)
    with jax.disable_jit(False):
        eval_metrics = outer_eval(rngs)
    # eval_metrics has shape: (n_ego, m_ego, num_partner_total, num_episodes, ...info)
    return eval_metrics

def main(config, ego_config, partner_config,
         eval_savedir, ego_ckpts, train_partner_ckpts, test_partner_ckpts=None, 
         num_episodes=32, metric_names=("returned_episode_returns", "returned_episode_lengths")):
    '''
    config: config dict specifying evaluation parameters
    ego_config: config dict specifying ego agent parameters
    partner_config: config dict specifying partner agent parameters
    eval_savedir: path to save eval metrics
    ego_ckpts: pytree of ego agent checkpoints
    train_partner_ckpts: pytree of train partner checkpoints
    test_partner_ckpts: pytree of test partner checkpoints
    metric_names: tuple of str, names of metrics to evaluate
    '''
    eval_res = {}
    eval_res["train"] = eval_ego_agent(config, ego_config, partner_config,
                                      ego_ckpts, train_partner_ckpts,
                                      num_episodes=num_episodes)
    if test_partner_ckpts is not None:
        eval_res["test"] = eval_ego_agent(config, ego_config, partner_config,
                                         ego_ckpts, test_partner_ckpts,
                                         num_episodes=num_episodes)
    
    for k, eval_metrics in eval_res.items():
        # save metric data
        savepath = save_train_run(eval_metrics, eval_savedir, savename=f"{k}_eval_metrics")
        log.info(f"Saved {k} eval metrics to {savepath}")
        # each submetric shape is (num_ego_seeds, num_ego_ckpts, num_partner_ckpts, episodes, num_agents)
        # the ego agent is always agent 0, the partner is agent 1
        for metric_name in metric_names:
            plot_xp_from_eval_metrics(eval_metrics, 
                metric_name=metric_name, agent_idx=0,
                savedir=eval_savedir, savename=f"{k}_xp_matrix",
                show_plots=False)

        # print metrics
        num_ego_ckpts = eval_metrics["returned_episode_returns"].shape[1]
        for ego_ckpt_idx in range(num_ego_ckpts):
            for metric_name in metric_names:
                metric_arr = eval_metrics[metric_name][:, ego_ckpt_idx, :, :, 0]
                log.info(f"Ego agent ckpt {ego_ckpt_idx}. {metric_name}: {metric_arr.mean()}, std: {metric_arr.std()}")
            log.info("#####\n")


if __name__ == "__main__":
    # set hyperparameters:
    config = {
        "ENV_NAME": "lbf",
        "ENV_KWARGS": {
        },
        "SEED": 12345,
        "ROLLOUT_LENGTH": 128,
    }
    ego_config = {
        "ROLLOUT_LENGTH": 128, # max episode steps
        "SEED": 12345,
        "EGO_ACTOR_TYPE": "s5",
        "S5_ACTOR_CRITIC_HIDDEN_DIM": 64,
        "S5_D_MODEL": 16,
        "S5_SSM_SIZE": 16,
        "S5_N_LAYERS": 2,
        "S5_BLOCKS": 1,
        "S5_ACTIVATION": "full_glu",
        "S5_DO_NORM": True,
        "S5_PRENORM": True,
        "S5_DO_GTRXL_NORM": True,
    }
    partner_config = {}

    # load checkpoints
    train_partner_path = "results/lbf/ippo/2025-04-10_20-21-47/ippo_train_run"
    train_partner_ckpts = load_checkpoints(train_partner_path)

    # eval_partner_path = "results/lbf/debug/2025-03-17_23-12-43/train_partners.pkl"
    # test_partner_ckpts = load_checkpoints(eval_partner_path)

    ego_path = "results/lbf/fcp_s5/2025-04-13_18-21-52/saved_train_run"
    
    ego_ckpts = load_checkpoints(ego_path)

    # perform eval
    ego_basedir = os.path.dirname(ego_path)
    main(config,
         ego_config, partner_config,
         ego_basedir, 
         ego_ckpts, train_partner_ckpts, 
         test_partner_ckpts=None, 
         num_episodes=32)
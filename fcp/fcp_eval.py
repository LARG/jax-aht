import os
import logging

import jax
import jax.numpy as jnp

from envs import make_env
from envs.log_wrapper import LogWrapper
from common.mlp_actor_critic import ActorCritic
from common.s5_actor_critic import S5ActorCritic, StackedEncoderModel, init_S5SSM, make_DPLR_HiPPO
from common.rnn_actor_critic import RNNActorCritic, ScannedRNN
from common.save_load_utils import load_checkpoints, save_train_run
from common.plot_utils import plot_eval_metrics

log = logging.getLogger(__name__)

def eval_ego_agent(base_config, ego_config, partner_config, 
                   ego_checkpoints, partner_checkpoints, 
                   num_episodes: int, ego_net_type: str = "s5"
                   ):
    '''
    ego_checkpoints: a pytree where each leaf contains N seeds, and M checkpoints of trained ego (FCP) agents 
                    (e.g. {"params": <pytree with leaves of shape (N, M, ...)>})
    partner_checkpoints: a pytree where each leaf contains N seeds, and M checkpoints of trained IPPO agents.
                    All eval checkpoints are treated as one pool of partners.
    For each ego agent (each seed, each checkpoint) we evaluate it against every eval checkpoint partner in the pool,
    running num_episodes per pairing.
    
    ego_net_type: str, one of ["mlp", "rnn", "s5"] to specify which actor critic architecture to use for the ego agent.
    
    Returns a dictionary with key "episode_returns" whose value is a jnp.array of shape 
       (num_ego_seeds, num_ego_ckpts, num_partner_total, num_episodes)
    '''
    # --- 1. Prepare the environment ---
    env = make_env(base_config["ENV_NAME"], base_config["ENV_KWARGS"])
    env = LogWrapper(env)
    num_agents = env.num_agents
    assert num_agents == 2, "This eval code assumes exactly 2 agents."

    # --- 2. Build the networks ---
    if ego_net_type == "mlp":
        ego_net = ActorCritic(env.action_space(env.agents[0]).n)
    elif ego_net_type == "rnn":
        ego_net = RNNActorCritic(action_dim=env.action_space(env.agents[0]).n,
                                fc_hidden_dim=ego_config["FC_HIDDEN_DIM"],
                                gru_hidden_dim=ego_config["GRU_HIDDEN_DIM"])
    elif ego_net_type == "s5":
        # Initialize S5 specific parameters
        d_model = ego_config["S5_D_MODEL"]
        ssm_size = ego_config["S5_SSM_SIZE"]
        n_layers = ego_config["S5_N_LAYERS"]
        blocks = ego_config["S5_BLOCKS"]
        block_size = int(ssm_size / blocks)

        Lambda, _, _, V,  _ = make_DPLR_HiPPO(ssm_size)
        block_size = block_size // 2
        ssm_size = ssm_size // 2
        Lambda = Lambda[:block_size]
        V = V[:, :block_size]
        Vinv = V.conj().T

        ssm_init_fn = init_S5SSM(H=d_model,
                                 P=ssm_size,
                                 Lambda_re_init=Lambda.real,
                                 Lambda_im_init=Lambda.imag,
                                 V=V,
                                 Vinv=Vinv)
        
        ego_net = S5ActorCritic(env.action_space(env.agents[0]).n, 
                               config=ego_config, 
                               ssm_init_fn=ssm_init_fn,
                               fc_hidden_dim=ego_config["S5_ACTOR_CRITIC_HIDDEN_DIM"],
                               ssm_hidden_dim=ego_config["S5_SSM_SIZE"])
    else:
        raise ValueError(f"Unknown ego_net_type: {ego_net_type}")
    
    partner_net = ActorCritic(env.action_space(env.agents[1]).n)

    # --- 3. Process checkpoints ---
    # ego_checkpoints: each leaf shape (n_ego, m_ego, ...); extract dimensions.
    ego_params = ego_checkpoints
    sample_leaf = jax.tree_util.tree_leaves(ego_params)[0]
    n_ego, m_ego = sample_leaf.shape[:2]

    # For eval checkpoints, flatten the first two dims to form a partner pool.
    def flatten_ckpt(x):
        return x.reshape(-1, *x.shape[2:])
    
    eval_params_flat = jax.tree.map(flatten_ckpt, partner_checkpoints["params"])
    sample_leaf_eval = jax.tree_util.tree_leaves(eval_params_flat)[0]
    num_partner_total = sample_leaf_eval.shape[0]

    # We'll use a fixed maximum step count per episode.
    max_episode_steps = base_config["NUM_STEPS"] 

    # --- 4. Inner evaluation functions (without jit) ---
    def run_single_episode(rng, ego_param, partner_param):
        # Reset the env.
        rng, reset_rng = jax.random.split(rng)
        obs, env_state = env.reset(reset_rng)
        # Do one step to get a dummy info structure.
        rng, act_rng, part_rng, step_rng = jax.random.split(rng, 4)
        
        # Get available actions for agent 0 from environment state
        avail_actions = env.get_avail_actions(env_state.env_state)
        avail_actions = jax.lax.stop_gradient(avail_actions)
        avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
        avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)
        
        # Initialize hidden state if needed
        if ego_net_type in ["rnn", "s5"]:
            if ego_net_type == "rnn":
                init_hstate_0 = ScannedRNN.initialize_carry(1, ego_config["GRU_HIDDEN_DIM"])
            else:  # s5
                init_hstate_0 = StackedEncoderModel.initialize_carry(1, ssm_size, n_layers)
            
            # Prepare inputs for ego agent
            rnn_input_0 = (
                obs["agent_0"].reshape(1, 1, -1),
                jnp.zeros((1, 1), dtype=bool),
                avail_actions_0
            )
            hstate_0, pi0, _ = ego_net.apply(ego_param, init_hstate_0, rnn_input_0)
            act0 = pi0.sample(seed=act_rng).squeeze()
        else:  # mlp
            pi0, _ = ego_net.apply(ego_param, (obs["agent_0"], avail_actions_0))
            act0 = pi0.sample(seed=act_rng)
        
        # Get partner action
        pi1, _ = partner_net.apply({'params': partner_param}, (obs["agent_1"], avail_actions_1))
        act1 = pi1.sample(seed=part_rng)
        
        # Step environment
        both_actions = [act0, act1]
        env_act = {k: both_actions[i] for i, k in enumerate(env.agents)}
        _, _, _, _, dummy_info = env.step(step_rng, env_state, env_act)

        # We'll use a scan to iterate steps until the episode is done.
        ep_ts = 1
        if ego_net_type in ["rnn", "s5"]:
            init_carry = (ep_ts, env_state, obs, rng, jnp.array(False), hstate_0, dummy_info)
        else:
            init_carry = (ep_ts, env_state, obs, rng, jnp.array(False), dummy_info)

        def scan_step(carry, _):
            def take_step(carry_step):
                if ego_net_type in ["rnn", "s5"]:
                    ep_ts, env_state, obs, rng, done_flag, hstate_0, last_info = carry_step
                else:
                    ep_ts, env_state, obs, rng, done_flag, last_info = carry_step
                
                rng, act_rng, part_rng, step_rng = jax.random.split(rng, 4)
                
                # Get available actions
                avail_actions = env.get_avail_actions(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(avail_actions)
                avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
                avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)
                
                if ego_net_type in ["rnn", "s5"]:
                    # Prepare inputs for ego agent
                    rnn_input_0 = (
                        obs["agent_0"].reshape(1, 1, -1),
                        jnp.zeros((1, 1), dtype=bool),
                        avail_actions_0
                    )
                    hstate_0, pi0, _ = ego_net.apply(ego_param, hstate_0, rnn_input_0)
                    act0 = pi0.sample(seed=act_rng).squeeze()
                else:  # mlp
                    pi0, _ = ego_net.apply(ego_param, (obs["agent_0"], avail_actions_0))
                    act0 = pi0.sample(seed=act_rng)
                
                pi1, _ = partner_net.apply({'params': partner_param}, (obs["agent_1"], avail_actions_1))
                act1 = pi1.sample(seed=part_rng)
                both_actions = [act0, act1]
                env_act = {k: both_actions[i] for i, k in enumerate(env.agents)}
                obs_next, env_state_next, reward, done, info = env.step(step_rng, env_state, env_act)

                if ego_net_type in ["rnn", "s5"]:
                    return (ep_ts + 1, env_state_next, obs_next, rng, done["__all__"], hstate_0, info)
                else:
                    return (ep_ts + 1, env_state_next, obs_next, rng, done["__all__"], info)
            
            if ego_net_type in ["rnn", "s5"]:
                _, _, _, _, done_flag, _, _ = carry
            else:
                _, _, _, _, done_flag, _ = carry
                
            new_carry = jax.lax.cond(
                done_flag,
                # if done, return the carry. Else, take a step.
                lambda curr_carry: curr_carry,
                take_step,
                operand=carry
            )
            return new_carry, None

        final_carry, _ = jax.lax.scan(
            scan_step, init_carry, None, length=max_episode_steps)
        # Return the final info (which includes the episode return via LogWrapper).
        final_info = final_carry[-1]
        return final_info

    def run_episodes(rng, ego_param, partner_param, num_eps):
        def body_fn(carry, _):
            rng = carry
            rng, ep_rng = jax.random.split(rng)
            ep_info = run_single_episode(ep_rng, ego_param, partner_param)
            return rng, ep_info
        rng, ep_infos = jax.lax.scan(body_fn, rng, None, length=num_eps)
        return ep_infos  # each leaf has shape (num_eps, ...)

    def eval_pair_fn(rng, ego_param, partner_param):
        return run_episodes(rng, ego_param, partner_param, num_episodes)

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
            # For one checkpoint, evaluate over the partner pool defined by eval_params_flat, 
            # which has shape (num_partner_total, ...)
            def eval_for_checkpoint(ego_param, rngs_ckpt):
                return jax.vmap(
                    lambda rng, partner_param: eval_pair_fn(rng, ego_param, partner_param),
                    in_axes=(0, 0)
                )(rngs_ckpt, eval_params_flat)
            # Vmap over the m_ego checkpoints.
            return jax.vmap(eval_for_checkpoint, in_axes=(0, 0))(ego_params_seed, total_rngs)
        return jax.vmap(eval_ego_seed, in_axes=(0, 0))(rngs, ego_params)

    # --- 6. JIT-compile and run ---
    base_rng = jax.random.PRNGKey(base_config["SEED"])
    # Prepare one RNG per FCP seed (n_ego total).
    rngs = jax.random.split(base_rng, n_ego)
    with jax.disable_jit(False):
        eval_metrics = outer_eval(rngs)
    # eval_metrics has shape: (n_ego, m_ego, num_partner_total, num_episodes, ...info)
    return eval_metrics

def main(base_config, ego_config, partner_config, 
         eval_savedir, ego_ckpts, train_partner_ckpts, test_partner_ckpts=None, 
         num_episodes=32, ego_net_type="s5", metric_names=("returned_episode_returns")):
    '''
    base_config: config dict specifying evaluation parameters
    ego_config: config dict containing parameters to initialize ego agent
    partner_config: config dict containing parameters to initialize partner agent
    eval_savedir: path to save eval metrics
    ego_ckpts: pytree of ego agent checkpoints
    train_partner_ckpts: pytree of train partner checkpoints
    test_partner_ckpts: pytree of test partner checkpoints
    ego_net_type: str, one of ["mlp", "rnn", "s5"] to specify which actor critic architecture to use for the ego agent.
    metric_names: tuple of str, names of metrics to evaluate
    '''
    eval_res = {}
    eval_res["train"] = eval_ego_agent(base_config, ego_config, partner_config, ego_ckpts, train_partner_ckpts, 
                                       num_episodes=num_episodes, ego_net_type=ego_net_type)
    if test_partner_ckpts is not None:
        eval_res["test"] = eval_ego_agent(base_config, ego_config, partner_config, ego_ckpts, test_partner_ckpts, 
        num_episodes=num_episodes, ego_net_type=ego_net_type)
    
    for k, eval_metrics in eval_res.items():
        # save metric data
        savepath = save_train_run(eval_metrics, eval_savedir, savename=f"{k}_eval_metrics")
        log.info(f"Saved {k} eval metrics to {savepath}")
        # each submetric shape is (num_ego_seeds, num_ego_ckpts, num_partner_ckpts, episodes, num_agents)
        # the ego agent is always agent 0, the partner is agent 1
        for metric_name in metric_names:
            plot_eval_metrics(eval_metrics, metric_name=metric_name, agent_idx=0)

        # print metrics
        num_ego_ckpts = eval_metrics["returned_episode_returns"].shape[1]
        for ego_ckpt_idx in range(num_ego_ckpts):
            for metric_name in metric_names:
                metric_arr = eval_metrics[metric_name][:, ego_ckpt_idx, :, :, 0]
                log.info(f"Ego agent ckpt {ego_ckpt_idx}. {metric_name}: {metric_arr.mean()}, std: {metric_arr.std()}")
            log.info("#####\n")


if __name__ == "__main__":
    # set hyperparameters:
    base_config = {
        "ENV_NAME": "lbf",
        "ENV_KWARGS": {
        },
        "NUM_STEPS": 128, # max episode steps
        "SEED": 12345
    }
    partner_config = {}
    ego_config = {
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

    # load checkpoints
    train_partner_path = "results/lbf/debug/2025-03-17_23-12-43/train_partners.pkl"
    train_partner_ckpts = load_checkpoints(train_partner_path)

    # eval_partner_path = "results/lbf/debug/2025-03-17_23-12-43/train_partners.pkl"
    # test_partner_ckpts = load_checkpoints(eval_partner_path)

    ego_path = "results/lbf/fcp_s5/2025-03-31_15-02-23/fcp_train.pkl"
    ego_net_type = "s5"

    # ego_path = "results/lbf/fcp_mlp/2025-03-31_16-08-42/fcp_train.pkl"
    # ego_net_type = "mlp"
    
    ego_ckpts = load_checkpoints(ego_path)

    # perform eval
    ego_basedir = os.path.dirname(ego_path)
    main(base_config, ego_config, partner_config, 
         ego_basedir, 
         ego_ckpts, train_partner_ckpts, 
         test_partner_ckpts=None, 
         num_episodes=32,
         ego_net_type=ego_net_type)
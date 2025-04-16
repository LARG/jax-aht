"""
Based on PureJaxRL Implementation of PPO. 
Script adapted from JaxMARL IPPO RNN Smax script.
"""
import os
import time
from datetime import datetime
import logging

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from common.mlp_actor_critic import ActorCritic
from common.s5_actor_critic import S5ActorCritic, StackedEncoderModel, init_S5SSM, make_DPLR_HiPPO
from common.save_load_utils import load_checkpoints, save_train_run
from common.plot_utils import get_stats, plot_train_metrics
from envs import make_env
from envs.log_wrapper import LogWrapper
from fcp.utils import unbatchify, Transition
from fcp.train_partners import train_partners_in_parallel

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def train_fcp_agent(config, checkpoints):
    '''
    Train an FCP agent using the given partner checkpoints and IPPO.
    Return model checkpoints and metrics. 
    '''
    # ------------------------------
    # 1) Flatten partner checkpoints into shape (N, ...) if desired
    #    but we can also keep them as (n_seeds, m_ckpts, ...).
    #    We'll just do gather via dynamic indexing in a jittable way.
    # ------------------------------
    partner_params = checkpoints["params"]  # This is the full PyTree
    n_seeds, m_ckpts = partner_params["Dense_0"]["kernel"].shape[:2]
    num_total_partners = n_seeds * m_ckpts

    # We can define a small helper to gather the correct slice for each environment
    # from shape (n_seeds, m_ckpts, ...) -> (num_envs, ...)
    # We'll do an integer mapping from [0, num_total_partners) -> (seed_idx, ckpt_idx).
    def unravel_partner_idx(idx):
        """Given a scalar in [0, n_seeds*m_ckpts), return (seed_idx, ckpt_idx)."""
        # seed_idx = idx // m_ckpts
        # ckpt_idx = idx % m_ckpts
        # We'll do jax-friendly approach:
        seed_idx = jnp.floor_divide(idx, m_ckpts)
        ckpt_idx = jnp.mod(idx, m_ckpts)
        return seed_idx, ckpt_idx

    def gather_partner_params(partner_params_pytree, idx_vec):
        """
        partner_params_pytree: pytree with all partner params. Each leaf has shape (n_seeds, m_ckpts, ...).
        idx_vec: a vector of indices with shape (num_envs,) each in [0, n_seeds*m_ckpts).

        Return a new pytree where each leaf has shape (num_envs, ...). Each leaf has a sampled
        partner's parameters for each environment.
        """
        # We'll define a function that gathers from each leaf
        # where leaf has shape (n_seeds, m_ckpts, ...), we want [idx_vec[i]] for each i.
        # We'll vmap a slicing function.
        def gather_leaf(leaf):
            # leaf shape: (n_seeds, m_ckpts, ...)
            # We'll define a function that slices out a single index:
            def slice_one(idx):
                seed_idx, ckpt_idx = unravel_partner_idx(idx)
                return leaf[seed_idx, ckpt_idx]  # shape (...)
            return jax.vmap(slice_one)(idx_vec)

        return jax.tree.map(gather_leaf, partner_params_pytree)

    # ------------------------------
    # 3) Build the FCP training function, closely mirroring `make_train(...)`.
    # ------------------------------
    def make_fcp_train(config, partner_params):
        # ------------------------------
        # 2) Prepare environment (same as IPPO).
        #    We'll assume exactly 2 agents: agent_0 = trainable, agent_1 = partner.
        # ------------------------------
        env = make_env(config["ENV_NAME"], config["ENV_KWARGS"])
        env = LogWrapper(env)

        num_agents = env.num_agents
        assert num_agents == 2, "This FCP snippet assumes exactly 2 agents."

        config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
        config["NUM_UNCONTROLLED_ACTORS"] = config["NUM_ENVS"] # assumption: we control 1 agent
        config["NUM_CONTROLLED_ACTORS"] = config["NUM_ENVS"] # assumption: we control 1 agent
        config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["ROLLOUT_LENGTH"] // config["NUM_ENVS"]
        config["NUM_ACTIONS"] = env.action_space(env.agents[0]).n
        
        # S5 specific parameters
        d_model = config["S5_D_MODEL"]
        ssm_size = config["S5_SSM_SIZE"]
        n_layers = config["S5_N_LAYERS"]
        blocks = config["S5_BLOCKS"]
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

        def linear_schedule(count):
            frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
            return config["LR"] * frac

        def train(rng):
            # --------------------------
            # 3a) Init agent_0 network
            # --------------------------
            agent0_net = S5ActorCritic(env.action_space(env.agents[0]).n, 
                                       ssm_init_fn=ssm_init_fn,
                                       fc_hidden_dim=config["S5_ACTOR_CRITIC_HIDDEN_DIM"],
                                       ssm_hidden_dim=config["S5_SSM_SIZE"],
                                       s5_d_model=config["S5_D_MODEL"],
                                       s5_n_layers=config["S5_N_LAYERS"],
                                       s5_activation=config["S5_ACTIVATION"],
                                       s5_do_norm=config["S5_DO_NORM"],
                                       s5_prenorm=config["S5_PRENORM"],
                                       s5_do_gtrxl_norm=config["S5_DO_GTRXL_NORM"],
                                       )

            rng, init_rng = jax.random.split(rng)
            init_x = (
                # init obs, dones, avail_actions
                jnp.zeros((1, config["NUM_CONTROLLED_ACTORS"], env.observation_space(env.agents[0]).shape[0])),
                jnp.zeros((1, config["NUM_CONTROLLED_ACTORS"])),
                jnp.ones((1, config["NUM_CONTROLLED_ACTORS"], env.action_space(env.agents[0]).n)),
            )
            init_hstate_0 = StackedEncoderModel.initialize_carry(config["NUM_CONTROLLED_ACTORS"], ssm_size, n_layers)

            init_params = agent0_net.init(init_rng, init_hstate_0, init_x)

            if config["ANNEAL_LR"]:
                tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(learning_rate=linear_schedule, eps=1e-5),
                )
            else:
                tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(config["LR"], eps=1e-5),
                )
            train_state = TrainState.create(
                apply_fn=agent0_net.apply,
                params=init_params,
                tx=tx,
            )

            # --------------------------
            # 3b) Init envs & partner indices
            # --------------------------
            rng, reset_rng = jax.random.split(rng)
            reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
            obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rngs)

            # Initialize hidden state for RNN
            init_hstate_0 = StackedEncoderModel.initialize_carry(config["NUM_CONTROLLED_ACTORS"], ssm_size, n_layers)

            # Each environment picks a partner index in [0, n_seeds*m_ckpts)
            rng, partner_rng = jax.random.split(rng)
            partner_indices = jax.random.randint(
                key=partner_rng,
                shape=(config["NUM_UNCONTROLLED_ACTORS"],),
                minval=0,
                maxval=num_total_partners
            )

            # --------------------------
            # 3c) Define env step
            # --------------------------
            def _env_step(runner_state, unused):
                """
                runner_state = (train_state, env_state, last_obs, partner_indices, rng)
                Returns updated runner_state, and a Transition for agent_0.
                """
                train_state, env_state, prev_obs, prev_done, hstate_0, partner_indices, rng = runner_state
                rng, actor_rng, partner_rng, step_rng = jax.random.split(rng, 4)
                
                # Prepare inputs for agent 0 (RNN)
                obs_0 = prev_obs["agent_0"]
                # Get available actions for agent 0 from environment state
                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(avail_actions)
                avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
                avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)
                # obs, done should have shape (sequence_length, num_actors, features) for the RNN
                # hstate should have shape (1, num_actors, hidden_dim)
                rnn_input_0 = (
                    obs_0.reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                    prev_done.reshape(1, config["NUM_CONTROLLED_ACTORS"]), 
                    avail_actions_0
                )

                # Agent_0 action
                hstate_0, pi_0, val_0 = agent0_net.apply(train_state.params, hstate_0, rnn_input_0)
                act_0 = pi_0.sample(seed=actor_rng).squeeze()
                logp_0 = pi_0.log_prob(act_0).squeeze()
                val_0 = val_0.squeeze()

                # Agent_1 (partner) action
                # Gather correct partner params for each env -> shape (num_envs, ...)
                # Note that partner idxs are resampled after every update
                gathered_params = gather_partner_params(partner_params, partner_indices)
                # We'll vmap the partner net apply
                def apply_partner(p, input_x, rng_):
                    # p: single-partner param dictionary
                    # input_x: single obs vector
                    # rng_: single environment's RNG
                    pi, _ = ActorCritic(env.action_space(env.agents[1]).n).apply({'params': p}, input_x)
                    return pi.sample(seed=rng_)

                rng_partner = jax.random.split(partner_rng, config["NUM_UNCONTROLLED_ACTORS"])
                obs_1 = prev_obs["agent_1"]
                partner_input = (obs_1, avail_actions_1)
                act_1 = jax.vmap(apply_partner)(gathered_params, partner_input, rng_partner)

                # Combine actions into the env format
                combined_actions = jnp.concatenate([act_0, act_1], axis=0)  # shape (2*num_envs,)
                env_act = unbatchify(combined_actions, env.agents, config["NUM_ENVS"], num_agents)
                env_act = {k: v.flatten() for k, v in env_act.items()}

                # Step env
                step_rngs = jax.random.split(step_rng, config["NUM_ENVS"])
                obs_next, env_state_next, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0))(
                    step_rngs, env_state, env_act
                )
                # note that num_actors = num_envs * num_agents
                done_0 = done["agent_0"]
                info_0 = jax.tree.map(lambda x: x[:, 0], info)

                # Store agent_0 data in transition
                transition = Transition(
                    done=done_0,
                    action=act_0,
                    value=val_0,
                    reward=reward["agent_0"],
                    log_prob=logp_0,
                    obs=obs_0,
                    info=info_0,
                    avail_actions=avail_actions_0
                )
                new_runner_state = (train_state, env_state_next, obs_next, done_0, hstate_0, partner_indices, rng)
                return new_runner_state, transition

            # --------------------------
            # 3d) GAE & update step
            # --------------------------
            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )

                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate_0, traj_batch, advantages, returns = batch_info
                    def _loss_fn(params, init_hstate_0, traj_batch, gae, target_v):
                        rnn_input_0 = (
                            traj_batch.obs, # shape (rollout_len, num_actors/num_minibatches, feat_size) =  (128, 4, 15)
                            traj_batch.done, # shape (rollout_len, num_actors/num_minibatches) = (128, 4)
                            traj_batch.avail_actions # shape (rollout_len, num_agents, num_actions) = (128, 4, 6)
                        )
                        _, pi, value = agent0_net.apply(
                            params, 
                            init_hstate_0,
                            rnn_input_0
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # Value loss
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                            ).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - target_v)
                        value_losses_clipped = jnp.square(value_pred_clipped - target_v)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # Policy gradient loss
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae_norm = (gae - gae.mean()) / (gae.std() + 1e-8)
                        pg_loss_1 = ratio * gae_norm
                        pg_loss_2 = jnp.clip(
                            ratio, 
                            1.0 - config["CLIP_EPS"], 
                            1.0 + config["CLIP_EPS"]) * gae_norm
                        pg_loss = -jnp.mean(jnp.minimum(pg_loss_1, pg_loss_2))

                        # Entropy
                        entropy = jnp.mean(pi.entropy())

                        total_loss = pg_loss + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
                        return total_loss, (value_loss, pg_loss, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (loss_val, aux_vals), grads = grad_fn(
                        train_state.params, init_hstate_0, traj_batch, advantages, returns)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, (loss_val, aux_vals)

                (
                    train_state,
                    init_hstate_0,
                    traj_batch,
                    advantages,
                    targets,
                    rng
                 ) = update_state
                rng, perm_rng = jax.random.split(rng)
                
                # batch_size is now config["NUM_ENVS"]
                permutation = jax.random.permutation(perm_rng, config["NUM_CONTROLLED_ACTORS"])

                batch = (
                    init_hstate_0, # shape (1, num_agents, hidden_dim) = (1, 16, 64)
                    traj_batch, # pytree: obs is shape (rollout_len, num_actors, feat_shape) = (128, 16, 15)
                    advantages, # shape (rollout_len, num_agents) = (128, 16)
                    targets # shape (rollout_len, num_agents) = (128, 16)
                )
                # each leaf of shuffled batch has shape (rollout_len, num_agents, feat_shape)
                # except for init_hstate_0 which has shape (1, num_agents, hidden_dim)
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )
                # each leaf has shape (num_minibatches, rollout_len, num_agents/num_minibatches, feat_shape)
                # except for init_hstate_0 which has shape (num_minibatches, 1, num_agents/num_minibatches, hidden_dim)
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1] 
                            + list(x.shape[2:]),
                    ), 1, 0,),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state, 
                    init_hstate_0,
                    traj_batch, 
                    advantages, 
                    targets, 
                    rng
                )
                return update_state, total_loss

            def _update_step(update_runner_state, unused):
                """
                1. Collect rollouts
                2. Compute advantage
                3. PPO updates
                """
                (train_state, env_state, last_obs, last_done, last_hstate_0, partner_indices, rng, update_steps) = update_runner_state
                # 1) rollout
                runner_state = (
                    train_state, 
                    env_state, 
                    last_obs, 
                    last_done, 
                    last_hstate_0, 
                    partner_indices, 
                    rng
                )
                runner_state, traj_batch = jax.lax.scan(
                    _env_step, runner_state, None, config["ROLLOUT_LENGTH"])
                (train_state, env_state, last_obs, last_done, last_hstate_0, partner_indices, rng) = runner_state

                # 2) advantage
                last_obs_batch_0 = last_obs["agent_0"]
                # Get available actions for agent 0 from environment state
                avail_actions_0 = jax.vmap(env.get_avail_actions)(env_state.env_state)["agent_0"].astype(jnp.float32)
                input_0 = (
                    last_obs_batch_0.reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                    last_done.reshape(1, config["NUM_CONTROLLED_ACTORS"]),
                    jax.lax.stop_gradient(avail_actions_0)
                )
                _, _, last_val = agent0_net.apply(train_state.params, last_hstate_0, input_0)
                last_val = last_val.squeeze()
                advantages, targets = _calculate_gae(traj_batch, last_val)
            
                # 3) PPO update
                update_state = (
                    train_state,
                    init_hstate_0, # shape is (num_controlled_actors, gru_hidden_dim) with all-0s value
                    traj_batch, # obs has shape (rollout_len, num_controlled_actors, -1)
                    advantages,
                    targets,
                    rng
                )
                update_state, _ = jax.lax.scan(
                    _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
                )
                train_state = update_state[0]

                # Resample partner for each env for next rollout
                # Note that we reset the hidden state after resampling partners by returning init_hstate_0
                rng, p_rng = jax.random.split(rng)
                new_partner_idx = jax.random.randint(
                    key=p_rng, shape=(config["NUM_UNCONTROLLED_ACTORS"],),
                    minval=0, maxval=num_total_partners
                )                
                # Reset environment due to partner change
                rng, reset_rng = jax.random.split(rng)
                reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
                obs, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rngs)
                init_done = jnp.zeros((config["NUM_CONTROLLED_ACTORS"]), dtype=bool)

                # Metrics
                metric = traj_batch.info
                metric["update_steps"] = update_steps
                new_runner_state = (train_state, env_state, obs, init_done, init_hstate_0, new_partner_idx, rng, update_steps + 1)
                return (new_runner_state, metric)

            # --------------------------
            # 3e) PPO Update and Checkpoint saving
            # --------------------------
            checkpoint_interval = max(1, config["NUM_UPDATES"] // config["NUM_CHECKPOINTS"])
            num_ckpts = config["NUM_CHECKPOINTS"]

            # Build a PyTree that holds parameters for all FCP checkpoints
            def init_ckpt_array(params_pytree):
                return jax.tree.map(
                    lambda x: jnp.zeros((num_ckpts,) + x.shape, x.dtype), 
                    params_pytree)

            def _update_step_with_ckpt(state_with_ckpt, unused):
                ((train_state, env_state, last_obs, last_done, hstate_0, partner_idx, rng, update_steps),
                 checkpoint_array, ckpt_idx) = state_with_ckpt

                # Single PPO update
                (new_runner_state, metric) = _update_step(
                    (train_state, env_state, last_obs, last_done, hstate_0, partner_idx, rng, update_steps),
                    None
                )
                (train_state, env_state, last_obs, last_done, hstate_0, partner_idx, rng, update_steps) = new_runner_state

                # Decide if we store a checkpoint
                to_store = jnp.equal(jnp.mod(update_steps, checkpoint_interval), 0)

                def store_ckpt(args):
                    ckpt_arr, cidx = args
                    new_ckpt_arr = jax.tree.map(
                        lambda c_arr, p: c_arr.at[cidx].set(p),
                        ckpt_arr, train_state.params
                    )
                    return (new_ckpt_arr, cidx + 1)

                def skip_ckpt(args):
                    return args

                (checkpoint_array, ckpt_idx) = jax.lax.cond(
                    to_store, store_ckpt, skip_ckpt, (checkpoint_array, ckpt_idx)
                )

                return ((train_state, env_state, last_obs, last_done, hstate_0, partner_idx, rng, update_steps),
                        checkpoint_array, ckpt_idx), metric

            # init checkpoint array
            checkpoint_array = init_ckpt_array(train_state.params)
            ckpt_idx = 0

            # initial runner state for scanningp
            update_steps = 0
            init_done = jnp.zeros((config["NUM_CONTROLLED_ACTORS"]), dtype=bool)
            update_runner_state = (
                train_state,
                env_state,
                obsv,
                init_done,
                init_hstate_0,
                partner_indices,
                rng,
                update_steps
            )
            state_with_ckpt = (update_runner_state, checkpoint_array, ckpt_idx)

            # run training
            state_with_ckpt, metrics = jax.lax.scan(
                _update_step_with_ckpt,
                state_with_ckpt,
                xs=None,
                length=config["NUM_UPDATES"]
            )
            (final_runner_state, checkpoint_array, final_ckpt_idx) = state_with_ckpt

            out = {
                "final_params": final_runner_state[0].params,
                "metrics": metrics,  # shape (NUM_UPDATES, ...)
                "checkpoints": checkpoint_array,
            }
            return out

        return train
    # ------------------------------
    # 4) Actually run the FCP training
    # ------------------------------
    start_time = time.time()
    # training is vmapped across multiple seeds
    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    
    debug_mode = False
    with jax.disable_jit(debug_mode):
        if debug_mode:
            out = make_fcp_train(config, partner_params)(rngs)
        else:
            fcp_train_fn = jax.jit(jax.vmap(make_fcp_train(config, partner_params)))
            out = fcp_train_fn(rngs)
    
    end_time = time.time()
    log.info(f"Training FCP agent took {end_time - start_time:.2f} seconds.")
    return out

if __name__ == "__main__":
    # set hyperparameters:
    config = {
        "TOTAL_TIMESTEPS": 1e6,
        "LR": 1.e-4,
        "NUM_ENVS": 16,
        "ROLLOUT_LENGTH": 128,
        "UPDATE_EPOCHS": 15,
        "NUM_MINIBATCHES": 8,
        "NUM_CHECKPOINTS": 5,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.05,
        "ENT_COEF": 0.01,
        "VF_COEF": 1.0,
        "MAX_GRAD_NORM": 1.0,
        "ANNEAL_LR": True,

        "S5_ACTOR_CRITIC_HIDDEN_DIM": 64,
        "S5_D_MODEL": 16,
        "S5_SSM_SIZE": 16,
        "S5_N_LAYERS": 2,
        "S5_BLOCKS": 1,
        "S5_ACTIVATION": "full_glu",
        "S5_DO_NORM": True,
        "S5_PRENORM": True,
        "S5_DO_GTRXL_NORM": True,

        "ENV_NAME": "lbf",
        "ENV_KWARGS": {},
        "SEED": 38410, 
        "PARTNER_SEED": 112358,
        "NUM_SEEDS": 3,
        "RESULTS_PATH": "results/lbf/fcp_s5/"
    }
    
    curr_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    savedir = os.path.join(config["RESULTS_PATH"], curr_datetime) 

    train_partner_path = "results/lbf/ippo/2025-04-10_20-21-47/ippo_train_run"
    if train_partner_path != "":
        train_partner_ckpts = load_checkpoints(train_partner_path)
    else:
        train_out = train_partners_in_parallel(config, config["PARTNER_SEED"])
        savepath = save_train_run(train_out, savedir, savename="train_partners")
        train_partner_ckpts = train_out["checkpoints"]
        print(f"Saved train partner data to {savepath}")

    fcp_out = train_fcp_agent(config, train_partner_ckpts)
    savepath = save_train_run(fcp_out, savedir, savename="fcp_train")
    print(f"Saved FCP training data to {savepath}")
    
    #################################
    # visualize results!
    # metrics values shape is (num_seeds, num_updates, num_rollout_steps, num_envs, num_agents)
    metrics = fcp_out["metrics"]
    if config["ENV_NAME"] == "lbf":
        all_stats = get_stats(metrics, ("percent_eaten", "returned_episode_returns"))
    elif config["ENV_NAME"] == "overcooked-v1":
        all_stats = get_stats(metrics, ("shaped_reward", "returned_episode_returns"))
    else: 
        all_stats = get_stats(metrics, ("returned_episode_returns", "returned_episode_lengths"))
    plot_train_metrics(all_stats, 
                       config["TOTAL_TIMESTEPS"], 
                       config["NUM_CONTROLLED_ACTORS"],
                       savedir=config["RESULTS_PATH"] + f"/{curr_datetime}",
                       savename="fcp_s5_train")
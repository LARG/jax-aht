'''
Based on the IPPO implementation from JaxMarl. Trains a parameter-shared, MLP IPPO agent on a
fully cooperative multi-agent environment. Note that this code is only compatible with MLP policies.
'''
import os
import shutil
import time
import logging
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import hydra
from flax.training.train_state import TrainState

from social_laws.common.run_episodes_ippo import run_episodes_vmap
from agents.initialize_agents import initialize_s5_agent, initialize_mlp_agent, \
    initialize_rnn_agent, initialize_pseudo_actor_with_double_critic, initialize_pseudo_actor_with_conditional_critic
from common.plot_utils import get_stats, get_metric_names
from common.save_load_utils import save_train_run
from envs import make_env
from envs.log_wrapper import LogWrapper
from marl.ppo_utils import Transition, batchify, unbatchify, _create_minibatches

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def initialize_agent(actor_type, algorithm_config, env, init_rng):
    if actor_type == "s5":
        policy, init_params = initialize_s5_agent(algorithm_config, env, init_rng)
    elif actor_type == "mlp":
        policy, init_params = initialize_mlp_agent(algorithm_config, env, init_rng)
    elif actor_type == "rnn":
        policy, init_params = initialize_rnn_agent(algorithm_config, env, init_rng)
    elif actor_type == "pseudo_actor_with_double_critic":
        policy, init_params = initialize_pseudo_actor_with_double_critic(algorithm_config, env, init_rng)
    elif actor_type == "pseudo_actor_with_conditional_critic":
        policy, init_params = initialize_pseudo_actor_with_conditional_critic(algorithm_config, env, init_rng)
    return policy, init_params


def train_ippo_agent(config, env, train_rng,
                    policies, init_params):
    '''
    Train IPPO the given initial parameters.

    Args:
        config: dict, config for the training
        env: gymnasium environment
        train_rng: jax.random.PRNGKey, random key for training
        policies: list[AgentPolicy], policies for the agents
        init_params: list[dict], initial parameters for the agents
    '''
    # ------------------------------
    # Build the IPPO training function
    # ------------------------------
    def make_ippo_train(config):
        num_agents = env.num_agents
        config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
        config["NUM_CONTROLLED_ACTORS"] = config["NUM_ENVS"] # assumption: we control 1 agent
        config["NUM_UPDATES"] = (
            config["TOTAL_TIMESTEPS"] // config["ROLLOUT_LENGTH"] // config["NUM_ENVS"]
        )
        config["MINIBATCH_SIZE"] = (
            config["NUM_ACTORS"] * config["ROLLOUT_LENGTH"] // config["NUM_MINIBATCHES"]
        )

        def linear_schedule(count):
            frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
            return config["LR"] * frac

        def train(rng):
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

            train_states = [
                TrainState.create(
                    apply_fn=policies[i].network.apply,
                    params=init_params[i],
                    tx=tx,
                )
                for i in range(num_agents)
            ]

            # Create functions to get agent-specific data based on agent_idx
            # This avoids indexing with traced values
            def get_agent_data(data_dict, idx):
                """Select data for agent based on index using switch for any number of agents"""
                branches = [lambda d=data_dict, k=agent_key: d[k] for agent_key in env.agents]
                return jax.lax.switch(idx, branches)

            def set_agent_data(data_dict, idx, value):
                """Set data for agent based on index using switch for any number of agents"""
                branches = [lambda d=data_dict, k=agent_key, v=value: {**d, k: v} for agent_key in env.agents]
                return jax.lax.switch(idx, branches)

            #  Init hstates
            init_hstates = [policies[i].init_hstate(config["NUM_CONTROLLED_ACTORS"]) for i in range(num_agents)]

            def _env_step(runner_state, unused):
                """
                One step of the environment:
                1. Get observations, sample actions from all agents
                2. Step environment using sampled actions
                3. Return state, reward, ...
                """
                train_states, env_state, prev_obs, prev_done, hstates, rng = runner_state
                rng, *actor_rngs, step_rng = jax.random.split(rng, num_agents + 2)

                # Get available actions for the agent from environment state
                # Get available actions for all agents
                avail_actions = env.get_avail_actions(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(avail_actions)

                prev_obs_per_agent = [get_agent_data(prev_obs, i).reshape(1, config["NUM_CONTROLLED_ACTORS"], -1) for i in range(num_agents)]
                prev_done_per_agent = [get_agent_data(prev_done, i).reshape(1, config["NUM_CONTROLLED_ACTORS"]) for i in range(num_agents)]
                avail_actions_per_agent = [get_agent_data(avail_actions, i).astype(jnp.float32) for i in range(num_agents)]

                # Note that we do not need to reset the hidden states for the agents
                # as the recurrent states are automatically reset when done is True.

                acts, vals, logps, new_hstates = [], [], [], []
                for i in range(num_agents):
                    act_i, val_i, pi_i, new_hs_i = policies[i].get_action_value_policy(
                        params=train_states[i].params,
                        obs=prev_obs_per_agent[i],
                        done=prev_done_per_agent[i],
                        avail_actions=avail_actions_per_agent[i],
                        hstate=hstates[i],
                        rng=actor_rngs[i]
                    )
                    acts.append(act_i.squeeze(axis=0))
                    vals.append(val_i.squeeze(axis=0))
                    logps.append(pi_i.log_prob(act_i).squeeze(axis=0))
                    new_hstates.append(new_hs_i)

                # Combine actions into the env format
                combined_actions = jnp.concatenate(acts, axis=0)  # shape (num_agents*num_envs,)
                env_act = unbatchify(combined_actions, env.agents, config["NUM_ENVS"], num_agents)
                env_act = {k: v.flatten() for k, v in env_act.items()}

                # Step env
                step_rngs = jax.random.split(step_rng, config["NUM_ENVS"])
                (obs_next, obs_full_next), env_state_next, reward, done_next, info = jax.vmap(env.step, in_axes=(0,0,0))(
                    step_rngs, env_state, env_act
                )

                # Filter out rendering-only fields (pre_reset_state, pre_reset_obs) which
                # have incompatible shapes for the per-actor reshape
                info = {k: v for k, v in info.items() if k not in ('pre_reset_state', 'pre_reset_obs')}
                info = jax.tree.map(lambda x: x.reshape((config["NUM_CONTROLLED_ACTORS"])), info)

                # Store per-agent transitions
                transitions = tuple(
                    Transition(
                        done=get_agent_data(done_next, i),
                        action=acts[i],
                        value=vals[i],
                        reward=get_agent_data(reward, i),
                        log_prob=logps[i],
                        obs=get_agent_data(prev_obs, i),
                        info=info,
                        avail_actions=avail_actions_per_agent[i],
                    )
                    for i in range(num_agents)
                )

                new_runner_state = (train_states, env_state_next, obs_next, done_next, new_hstates, rng)
                return new_runner_state, transitions

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

            def _make_update_minbatch(agent_i):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, returns = batch_info
                    def _loss_fn(params, init_hstate, traj_batch, gae, target_v):
                        _, value, pi, _ = policies[agent_i].get_action_value_policy(
                            params=params,
                            obs=traj_batch.obs,
                            done=traj_batch.done,
                            avail_actions=traj_batch.avail_actions,
                            hstate=init_hstate,
                            rng=jax.random.PRNGKey(0)
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # Value loss
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                            ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - target_v)
                        value_losses_clipped = jnp.square(value_pred_clipped - target_v)
                        value_loss = jnp.maximum(value_losses, value_losses_clipped).mean()

                        # Policy gradient loss
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae_norm = (gae - gae.mean()) / (gae.std() + 1e-8)
                        pg_loss_1 = ratio * gae_norm
                        pg_loss_2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae_norm
                        pg_loss = -jnp.mean(jnp.minimum(pg_loss_1, pg_loss_2))

                        entropy = jnp.mean(pi.entropy())
                        total_loss = pg_loss + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
                        return total_loss, (value_loss, pg_loss, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (loss_val, aux_vals), grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, returns)
                    train_state = train_state.apply_gradients(grads=grads)

                    grad_l2_norms = jax.tree.map(lambda g: jnp.linalg.norm(g.astype(jnp.float32)), grads)
                    sum_of_grad_norms = jax.tree.reduce(lambda x, y: x + y, grad_l2_norms)
                    n_elements = len(jax.tree.leaves(grad_l2_norms))
                    avg_grad_norm = sum_of_grad_norms / n_elements

                    return train_state, (loss_val, aux_vals, avg_grad_norm)
                return _update_minbatch

            def _make_update_epoch(agent_i):
                update_minbatch_fn = _make_update_minbatch(agent_i)
                def _update_epoch(update_state, unused):
                    train_state, init_hstate, traj_batch, advantages, targets, rng = update_state
                    rng, perm_rng = jax.random.split(rng)
                    minibatches = _create_minibatches(traj_batch, advantages, targets, init_hstate, config["NUM_CONTROLLED_ACTORS"], config["NUM_MINIBATCHES"], perm_rng)
                    train_state, losses_and_grads = jax.lax.scan(update_minbatch_fn, train_state, minibatches)
                    update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
                    return update_state, losses_and_grads
                return _update_epoch

            update_epoch_fns = [_make_update_epoch(i) for i in range(num_agents)]

            def _update_step(update_runner_state, unused):
                """
                1. Collect rollouts
                2. Compute advantage
                3. PPO updates
                """
                (train_states, rng, rng_eval, update_steps) = update_runner_state
                # Init envs & partner indices
                rng, reset_rng = jax.random.split(rng, 2)
                reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
                (init_obs, init_full_obs), init_env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rngs)
                init_done = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}

                # 1) rollout
                runner_state = (train_states, init_env_state, init_obs, init_done, init_hstates, rng)

                runner_state, traj_batch = jax.lax.scan(
                    _env_step, runner_state, None, config["ROLLOUT_LENGTH"])
                (train_states, env_state, obs, done, hstates, rng) = runner_state

                # 2) advantage
                # Get available actions for agent 0 from environment state
                avail_actions = env.get_avail_actions(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(avail_actions)
                rng, *update_rngs = jax.random.split(rng, num_agents + 1)

                agent_metrics = {}
                for i in range(num_agents):
                    obs_i = get_agent_data(obs, i).reshape(1, config["NUM_CONTROLLED_ACTORS"], -1)
                    done_i = get_agent_data(done, i).reshape(1, config["NUM_CONTROLLED_ACTORS"])
                    avail_i = get_agent_data(avail_actions, i).astype(jnp.float32)

                    _, last_val_i, _, _ = policies[i].get_action_value_policy(
                        params=train_states[i].params,
                        obs=obs_i,
                        done=done_i,
                        avail_actions=avail_i,
                        hstate=hstates[i],
                        rng=jax.random.PRNGKey(0)
                    )
                    last_val_i = last_val_i.squeeze(axis=0)

                    advantages_i, targets_i = _calculate_gae(traj_batch[i], last_val_i)

                    update_state_i = (
                        train_states[i], init_hstates[i], traj_batch[i],
                        advantages_i, targets_i, update_rngs[i]
                    )
                    update_state_i, losses_and_grads_i = jax.lax.scan(
                        update_epoch_fns[i], update_state_i, None, config["UPDATE_EPOCHS"])
                    train_states[i] = update_state_i[0]
                    _, loss_terms_i, avg_grad_norm_i = losses_and_grads_i
                    agent_metrics[f"agent_{i}/actor_loss"] = loss_terms_i[1]
                    agent_metrics[f"agent_{i}/value_loss"] = loss_terms_i[0]
                    agent_metrics[f"agent_{i}/entropy_loss"] = loss_terms_i[2]
                    agent_metrics[f"agent_{i}/avg_grad_norm"] = avg_grad_norm_i

                metric = traj_batch[0].info
                metric["update_steps"] = update_steps
                metric.update(agent_metrics)
                new_runner_state = (train_states, rng, rng_eval, update_steps + 1)
                return (new_runner_state, metric)

            # IPPO Update and Checkpoint saving
            ckpt_and_eval_interval = config["NUM_UPDATES"] // max(1, config["NUM_CHECKPOINTS"] - 1)  # -1 because we store a ckpt at the last update
            num_ckpts = config["NUM_CHECKPOINTS"]

            # Build a PyTree that holds parameters for all FCP checkpoints
            def init_ckpt_array(params_pytree):
                return jax.tree.map(
                    lambda x: jnp.zeros((num_ckpts,) + x.shape, x.dtype),
                    params_pytree)

            max_episode_steps = env.horizon # config["ROLLOUT_LENGTH"]

            def _update_step_with_ckpt(state_with_ckpt, unused):
                (update_state, checkpoint_arrays, ckpt_idx, init_ckpt_eval_last_info, init_eval_last_info) = state_with_ckpt

                # Single IPPO update
                new_update_state, metric = _update_step(
                    update_state,
                    None
                )
                (train_states, rng, rng_eval, update_steps) = new_update_state

                # update steps is 1-indexed because it was incremented at the end of the update step
                to_store = jnp.logical_or(jnp.equal(jnp.mod(update_steps-1, ckpt_and_eval_interval), 0),
                                        jnp.equal(update_steps, config["NUM_UPDATES"]))


                def store_and_eval_ckpt(args):
                    ckpt_arrs, cidx, rng, rng_eval, prev_ckpt_eval_ret_info, prev_eval_ret_info = args
                    new_ckpt_arrs = [
                        jax.tree.map(lambda c_arr, p: c_arr.at[cidx].set(p), ckpt_arrs[i], train_states[i].params)
                        for i in range(num_agents)
                    ]

                    if config["FIXED_EVAL"]:
                        eval_rng = rng_eval
                    else:
                        rng_eval, eval_rng = jax.random.split(rng_eval, 2)
                    ckpt_eval_eps_last_infos = run_episodes_vmap(
                        eval_rng, env,
                        agent_params=[train_state.params for train_state in train_states],
                        agent_policies=policies,
                        max_episode_steps=max_episode_steps,
                        num_eps=config["NUM_EVAL_EPISODES"])
                    return (new_ckpt_arrs, cidx + 1, rng, rng_eval, ckpt_eval_eps_last_infos, ckpt_eval_eps_last_infos)

                def skip_ckpt_and_eval(args):
                    def do_eval(eval_args):
                        ckpt_arrs, cidx, rng, rng_eval, prev_ckpt_eval_ret_info, prev_eval_ret_info = eval_args
                        if config["FIXED_EVAL"]:
                            eval_rng = rng_eval
                        else:
                            rng_eval, eval_rng = jax.random.split(rng_eval, 2)
                        eval_eps_last_infos = run_episodes_vmap(
                            eval_rng, env,
                            agent_params=[train_state.params for train_state in train_states],
                            agent_policies=policies,
                            max_episode_steps=max_episode_steps,
                            num_eps=config["NUM_EVAL_EPISODES"])
                        return (ckpt_arrs, cidx, rng, rng_eval, prev_ckpt_eval_ret_info, eval_eps_last_infos)

                    def skip_eval(eval_args):
                        return eval_args

                    (ckpt_arrs, cidx, rng, rng_eval, prev_ckpt_eval_ret_info, eval_eps_last_infos) = jax.lax.cond(
                        config["TRAIN_EVAL"],
                        do_eval,
                        skip_eval,
                        args
                    )

                    return (ckpt_arrs, cidx, rng, rng_eval, prev_ckpt_eval_ret_info, eval_eps_last_infos)

                (checkpoint_arrays, ckpt_idx, rng, rng_eval, ckpt_eval_eps_last_infos, eval_eps_last_infos) = jax.lax.cond(
                    to_store, store_and_eval_ckpt, skip_ckpt_and_eval, (checkpoint_arrays, ckpt_idx, rng, rng_eval, init_ckpt_eval_last_info, init_eval_last_info)
                )

                metric["ckpt_eval_ep_last_info"] = ckpt_eval_eps_last_infos
                metric["eval_ep_last_info"] = eval_eps_last_infos
                return ((train_states, rng, rng_eval, update_steps),
                         checkpoint_arrays, ckpt_idx, ckpt_eval_eps_last_infos, eval_eps_last_infos), metric

            checkpoint_arrays = [init_ckpt_array(train_states[i].params) for i in range(num_agents)]
            ckpt_idx = 0

            rng, rng_train = jax.random.split(rng, 2)

            rng_eval = jax.random.PRNGKey(config["EVAL_SEED"])
            rng_eval, eval_rng = jax.random.split(rng_eval, 2)
            if config["FIXED_EVAL"]:
                eval_rng = rng_eval

            # Init eval return infos
            eval_eps_last_infos = run_episodes_vmap(eval_rng, env,
                                    agent_params=[train_state.params for train_state in train_states],
                                    agent_policies=policies,
                                    max_episode_steps=max_episode_steps,
                                    num_eps=config["NUM_EVAL_EPISODES"])

            # initial runner state for scanning
            update_steps = 0

            update_runner_state = (train_states, rng_train, rng_eval, update_steps)
            state_with_ckpt = (update_runner_state, checkpoint_arrays, ckpt_idx, eval_eps_last_infos, eval_eps_last_infos)

            state_with_ckpt, metrics = jax.lax.scan(
                _update_step_with_ckpt,
                state_with_ckpt,
                xs=None,
                length=config["NUM_UPDATES"]
            )

            (final_runner_state, final_checkpoint_arrays, final_ckpt_idx, ckpt_eval_eps_last_infos, eval_eps_last_infos) = state_with_ckpt
            final_agent_ts = final_runner_state[0]
            out = {
                **{f"final_params_agent_{i}": final_agent_ts[i].params for i in range(num_agents)},
                "metrics": metrics,
                **{f"checkpoints_agent_{i}": final_checkpoint_arrays[i] for i in range(num_agents)},
            }

            if env._render:
                # Collect final eval gifs for logging
                rng_eval = final_runner_state[2] # extract final rng_eval from the final runner state after training
                if config["FIXED_EVAL"]:
                    eval_rng = rng_eval
                else:
                    rng_eval, eval_rng = jax.random.split(rng_eval, 2)

                out["render_outs"] = run_episodes_vmap(eval_rng, env,
                                                    agent_params=[final_agent_ts[i].params for i in range(num_agents)],
                                                    agent_policies=policies,
                                                    max_episode_steps=env.horizon,
                                                    num_eps=5, render=True)

            return out
        return train

    # ------------------------------
    # Actually run the IPPO training
    # ------------------------------
    rngs = jax.random.split(train_rng, config["NUM_TRAIN_SEEDS"])

    # Run training seeds in parallel using vmap
    train_fn = make_ippo_train(config)
    out = jax.vmap(train_fn, in_axes=(0))(rngs)
    return out

def run_training(config, wandb_logger):
    '''Run IPPO training.

    Args:
        config: dict, config for the training
    '''
    algorithm_config = dict(config["algorithm"])

    # Create only one environment instance
    env_kwargs = dict(algorithm_config["ENV_KWARGS"])
    env = make_env(algorithm_config["ENV_NAME"], env_kwargs)
    env = LogWrapper(env)

    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"])
    _, init_rng, train_rng = jax.random.split(rng, 3)

    # Initialize agent
    policy, init_params = initialize_agent(algorithm_config["ACTOR_TYPE"], algorithm_config, env, init_rng)

    num_agents = env.num_agents

    # Initialize one joint policy per agent
    policies, init_params = map(list, zip(*[
        initialize_agent(algorithm_config["ACTOR_TYPE"], algorithm_config, env, init_rng)
        for i in range(num_agents)
    ]))

    log.info(f"Starting IPPO training ...")
    start_time = time.perf_counter()

    # Run the training
    out = train_ippo_agent(
        config=algorithm_config,
        env=env,
        train_rng=train_rng,
        policies=policies,
        init_params=init_params
    )

    elapsed_time = time.perf_counter() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, rem = divmod(rem, 60)
    seconds, rem = divmod(rem, 1)
    milliseconds = int(rem * 1000)
    microseconds = int((rem * 1_000_000) % 1000)
    log.info(f"IPPO Training completed in {elapsed_time:.2f}s")
    log.info(f"IPPO Training completed in {int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s {milliseconds:03d}ms {microseconds:03d}µs")

    # process and log metrics
    log.info(f"Starting IPPO logging ...")
    start_time = time.perf_counter()
    metric_names = get_metric_names(config["ENV_NAME"])
    log_metrics(env, config, out, wandb_logger, metric_names)
    elapsed_time = time.perf_counter() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, rem = divmod(rem, 60)
    seconds, rem = divmod(rem, 1)
    milliseconds = int(rem * 1000)
    microseconds = int((rem * 1_000_000) % 1000)
    log.info(f"IPPO Logging completed in {elapsed_time:.2f}s")
    log.info(f"IPPO Logging completed in {int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s {milliseconds:03d}ms {microseconds:03d}µs")

    return out["final_params"], policies, init_params

def log_metrics(env, config, train_out, logger, metric_names: tuple):
    """Process training metrics and log them using the provided logger.

    Args:
        env: the environment used for training, needed for logging videos
        config: dict, the training configuration
        train_out: dict, the logs from training
        logger: Logger, instance to log metrics
        metric_names: tuple, names of metrics to extract from training logs
    """
    train_metrics = train_out["metrics"]

    #### Extract train metrics ####
    train_stats = get_stats(train_metrics, metric_names)
    # each key in train_stats is a metric name, and the value is an array of shape (num_seeds, num_updates, 2)
    # where the last dimension contains the mean and std of the metric
    train_stats = {k: np.mean(np.array(v), axis=0) for k, v in train_stats.items()}

    all_agent_value_losses = np.asarray(train_metrics["value_loss"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    all_agent__actor_losses = np.asarray(train_metrics["actor_loss"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    all_agent_entropy_losses = np.asarray(train_metrics["entropy_loss"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    all_agent_grad_norms = np.asarray(train_metrics["avg_grad_norm"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)

    # Process eval return metrics - average across train seeds, eval episodes, and num_agents per game for each checkpoint
    all_ckpt_returns = np.asarray(train_metrics["ckpt_eval_ep_last_info"]["returned_episode_returns"]) # shape (n_train_seeds, num_updates, num_eval_episodes, num_agents_per_game)
    all_returns = np.asarray(train_metrics["eval_ep_last_info"]["returned_episode_returns"]) # shape (n_train_seeds, num_updates, num_eval_episodes, num_agents_per_game)
    average_ckpt_rets_per_iter = np.mean(np.sum(all_ckpt_returns, axis=(3)), axis=(0, 2)) # shape (num_updates,)
    average_rets_per_iter = np.mean(np.sum(all_returns, axis=(3)), axis=(0, 2)) # shape (num_updates,)
    average_ckpt_agent_rets_per_iter = np.mean(all_ckpt_returns, axis=(0, 2)) # shape (num_updates,)
    average_agent_rets_per_iter = np.mean(all_returns, axis=(0, 2)) # shape (num_updates,)

    # Process loss metrics - average across train seeds, partners and minibatches dims
    # Loss metrics shape should be (n_train_seeds, num_updates, ...)
    average_agent_value_losses = np.mean(all_agent_value_losses, axis=(0, 2, 3))  # shape (num_updates,)
    average_agent_actor_losses = np.mean(all_agent__actor_losses, axis=(0, 2, 3)) # shape (num_updates,)
    average_agent_entropy_losses = np.mean(all_agent_entropy_losses, axis=(0, 2, 3)) # shape (num_updates,)
    average_agent_grad_norms = np.mean(all_agent_grad_norms, axis=(0, 2, 3)) # shape (num_updates,)

    # Log metrics for each update step
    num_agents = env.num_agents
    num_updates = len(average_agent_value_losses)
    for step in range(num_updates):
        for stat_name, stat_data in train_stats.items():
            # second dimension contains the mean and std of the metric
            stat_mean = stat_data[step, 0]
            logger.log_item(f"Train/{stat_name}", stat_mean, train_step=step, commit=True)

        for agent_id in range(num_agents):
             logger.log_item(f"Eval/Agent_{agent_id}/Return", average_agent_rets_per_iter[step, agent_id], train_step=step, commit=True)
             logger.log_item(f"Eval/Agent_{agent_id}/CheckpointReturn", average_ckpt_agent_rets_per_iter[step, agent_id], train_step=step, commit=True)

        logger.log_item(f"Eval/Return", average_rets_per_iter[step], train_step=step, commit=True)
        logger.log_item(f"Eval/CheckpointReturn", average_ckpt_rets_per_iter[step], train_step=step, commit=True)
        logger.log_item(f"Train/ValueLoss", average_agent_value_losses[step], train_step=step, commit=True)
        logger.log_item(f"Train/ActorLoss", average_agent_actor_losses[step], train_step=step, commit=True)
        logger.log_item(f"Train/EntropyLoss", average_agent_entropy_losses[step], train_step=step, commit=True)
        logger.log_item(f"Train/GradNorm", average_agent_grad_norms[step], train_step=step, commit=True)
        logger.commit()

    # Saving artifacts
    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    if env._render:
        # shape of render_outs should be (num_train_seeds, num_eps, max_episode_steps, ...)
        eval_render_init_env_state = train_out['render_outs'][1].env_state.env_state # LogEnvState
        eval_render_env_state = train_out['render_outs'][0][-1]['pre_reset_state'].env_state # WrappedEnvState
        eval_render_dones = train_out['render_outs'][0][4]['__all__']
        num_episodes = eval_render_dones.shape[1]
        env.animate((eval_render_init_env_state, eval_render_env_state), eval_render_dones, num_episodes, debug=True)

        for eval_ep in range(num_episodes):
            logger.log_video(
                tag=f"Videos/Episode_{eval_ep}",
                path=os.path.join(env._render_dir, f"{env._render_name}_ep_{eval_ep}.gif")
            )

    out_savepath = save_train_run(train_out, savedir, savename=f"IPPO_Agent_Train_Run")
    if config["logger"]["log_train_out"]:
        logger.log_artifact(name=f"IPPO_Agent_Train_Run", path=out_savepath, type_name="train_run")
        # Cleanup locally logged out file
    if not config["local_logger"]["save_train_out"]:
        shutil.rmtree(out_savepath)

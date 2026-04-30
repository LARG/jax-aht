'''
Based on the IPPO implementation from JaxMarl. Trains a parameter-shared, MLP IPPO agent on a
fully cooperative multi-agent environment. Note that this code is only compatible with MLP policies.
'''
import os
import shutil
import time
import logging
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import hydra
from flax.training.train_state import TrainState

from social_laws.common.run_episodes_ippo_centralized import run_episodes_vmap
from agents.initialize_agents import initialize_s5_agent, initialize_mlp_agent, \
    initialize_rnn_agent, initialize_pseudo_actor_with_double_critic, initialize_pseudo_actor_with_conditional_critic
from common.plot_utils import get_stats, get_metric_names
from common.save_load_utils import save_train_run
from envs import make_env
from envs.log_wrapper import LogWrapper
from marl.ppo_utils import Transition, batchify, unbatchify, _create_minibatches

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class CurriculumState(NamedTuple):
    use_no_law_env: jnp.ndarray
    ema_eval_return: jnp.ndarray
    stable_updates: jnp.ndarray

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


def train_ippo_agent(config, env, no_law_env, train_rng,
                    policy, init_params):
    '''
    Train IPPO the given initial parameters.

    Args:
        config: dict, config for the training
        env: gymnasium environment
        train_rng: jax.random.PRNGKey, random key for training
        policy: AgentPolicy, policy for the agent
        init_params: dict, initial parameters for the agent
    '''
    # ------------------------------
    # Build the IPPO training function
    # ------------------------------
    def make_ippo_train(config):
        num_agents = env.num_agents
        config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
        config["NUM_UPDATES"] = (
            config["TOTAL_TIMESTEPS"] // config["ROLLOUT_LENGTH"] // config["NUM_ENVS"]
        )
        config["MINIBATCH_SIZE"] = (
            config["NUM_ACTORS"] * config["ROLLOUT_LENGTH"] // config["NUM_MINIBATCHES"]
        )

        # Curriculum switch settings.
        convergence_tol = float(config.get("CURRICULUM_CONVERGENCE_TOL", 1e-4))
        convergence_patience = int(config.get("CURRICULUM_CONVERGENCE_PATIENCE", 5))
        min_updates_before_switch = int(config.get("CURRICULUM_MIN_UPDATES_BEFORE_SWITCH", 30))
        switch_on_min_updates = bool(config.get("CURRICULUM_SWITCH_ON_MIN_UPDATES", True))
        target_eval_return = float(config.get("CURRICULUM_TARGET_EVAL_RETURN", -jnp.inf))
        ema_alpha = float(config.get("CURRICULUM_EMA_ALPHA", 0.1))
        law_ent_coef = float(config["ENT_COEF"])
        no_law_ent_coef_cfg = config.get("NO_LAW_ENT_COEF", None)
        no_law_ent_coef = law_ent_coef if no_law_ent_coef_cfg is None else float(no_law_ent_coef_cfg)

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

            train_state = TrainState.create(
                apply_fn=policy.network.apply,
                params=init_params,
                tx=tx,
            )

            #  Init hstates
            init_hstate = policy.init_hstate(config["NUM_ACTORS"])

            def _select_env_data(use_no_law_env, law_value, no_law_value):
                return jax.tree.map(
                    lambda law_x, no_law_x: jax.lax.select(use_no_law_env, no_law_x, law_x),
                    law_value,
                    no_law_value,
                )

            def _env_step(runner_state, unused):
                """
                One step of the environment:
                1. Get observations, sample actions from all agents
                2. Step environment using sampled actions
                3. Return state, reward, ...
                """
                train_state, env_state, prev_obs, prev_done, hstate, use_no_law_env, rng = runner_state
                rng, actor_rng, step_rng = jax.random.split(rng, 3)

                 # Get available actions for the agent from environment state
                avail_actions_law = env.get_avail_actions(env_state.env_state)
                avail_actions_no_law = no_law_env.get_avail_actions(env_state.env_state)
                avail_actions = _select_env_data(use_no_law_env, avail_actions_law, avail_actions_no_law)
                avail_actions = jax.lax.stop_gradient(batchify(avail_actions,
                    env.agents, config["NUM_ACTORS"]).astype(jnp.float32))

                prev_obs_batch = batchify(prev_obs, env.agents, config["NUM_ACTORS"])
                prev_done_batch = batchify(prev_done, env.agents, config["NUM_ACTORS"])

                # Note that we do not need to reset the hidden states for the agents
                # as the recurrent states are automatically reset when done is True.

                # Controlled Agent action, value, log_prob
                act, val, pi, new_hstate = policy.get_action_value_policy(
                    params=train_state.params,
                    obs=prev_obs_batch.reshape(1, config["NUM_ACTORS"], -1),
                    done=prev_done_batch.reshape(1, config["NUM_ACTORS"]),
                    avail_actions=avail_actions,
                    hstate=hstate,
                    rng=actor_rng
                )
                logp = pi.log_prob(act)

                act = act.squeeze(axis=0)
                logp = logp.squeeze(axis=0)
                val = val.squeeze(axis=0)

                env_act = unbatchify(act, env.agents, config["NUM_ENVS"], env.num_agents)
                env_act = {k:v.flatten() for k,v in env_act.items()}

                # Step env
                step_rngs = jax.random.split(step_rng, config["NUM_ENVS"])
                law_step_out = jax.vmap(env.step, in_axes=(0,0,0))(
                    step_rngs, env_state, env_act
                )
                no_law_step_out = jax.vmap(no_law_env.step, in_axes=(0,0,0))(
                    step_rngs, env_state, env_act
                )
                (obs_next, obs_full_next), env_state_next, reward, done_next, info = _select_env_data(
                    use_no_law_env,
                    law_step_out,
                    no_law_step_out,
                )

                # Filter out rendering-only fields (pre_reset_state, pre_reset_obs) which
                # have incompatible shapes for the per-actor reshape
                info = {k: v for k, v in info.items() if k not in ('pre_reset_state', 'pre_reset_obs')}
                info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)

                # Store agent_idx data in transition
                transition = Transition(
                    done=batchify(done_next, env.agents, config["NUM_ACTORS"]).squeeze(), # shape (num_envs,)
                    action=act, # shape (num_envs,)
                    value=val, # shape (num_envs,)
                    reward=batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(), # shape (num_envs,)
                    log_prob=logp, # shape (num_envs,)
                    obs=prev_obs_batch, # shape (num_envs, obs_dim)
                    info=info,
                    avail_actions=avail_actions # shape (num_envs, num_actions)
                )

                new_runner_state = (train_state, env_state_next, obs_next, done_next, new_hstate, use_no_law_env, rng)
                return new_runner_state, transition

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

            def _update_minbatch(train_state, batch_info):
                init_hstate, traj_batch, advantages, returns, current_ent_coef = batch_info
                def _loss_fn(params, init_hstate, traj_batch, gae, target_v):
                    _, value, pi, _ = policy.get_action_value_policy(
                        params=params,
                        obs=traj_batch.obs,
                        done=traj_batch.done,
                        avail_actions=traj_batch.avail_actions,
                        hstate=init_hstate,
                        rng=jax.random.PRNGKey(0) # only used for action sampling, which is unused here
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
                        jnp.maximum(value_losses, value_losses_clipped).mean()
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

                    total_loss = pg_loss + config["VF_COEF"] * value_loss - current_ent_coef * entropy
                    return total_loss, (value_loss, pg_loss, entropy)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                (loss_val, aux_vals), grads = grad_fn(
                    train_state.params, init_hstate, traj_batch, advantages, returns)
                train_state = train_state.apply_gradients(grads=grads)

                # compute average grad norm
                grad_l2_norms = jax.tree.map(lambda g: jnp.linalg.norm(g.astype(jnp.float32)), grads)
                sum_of_grad_norms = jax.tree.reduce(lambda x, y: x + y, grad_l2_norms)
                n_elements = len(jax.tree.leaves(grad_l2_norms))
                avg_grad_norm = sum_of_grad_norms / n_elements

                return train_state, (loss_val, aux_vals, avg_grad_norm)

            def _update_epoch(update_state, unused):
                train_state, init_hstate, traj_batch, advantages, targets, current_ent_coef, rng = update_state
                rng, perm_rng = jax.random.split(rng)
                minibatches = _create_minibatches(traj_batch, advantages, targets, init_hstate, config["NUM_ACTORS"], config["NUM_MINIBATCHES"], perm_rng)
                ent_coef_minibatches = jnp.broadcast_to(current_ent_coef, (config["NUM_MINIBATCHES"],))
                minibatches = (*minibatches, ent_coef_minibatches)
                train_state, losses_and_grads = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, init_hstate, traj_batch, advantages, targets, current_ent_coef, rng)
                return update_state, losses_and_grads

            def _update_step(update_runner_state, unused):
                """
                1. Collect rollouts
                2. Compute advantage
                3. PPO updates
                """
                (train_state, rng, rng_eval, update_steps, curriculum_state) = update_runner_state
                # Init envs & partner indices
                rng, reset_rng = jax.random.split(rng, 2)
                reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
                law_reset_out = jax.vmap(env.reset, in_axes=(0,))(reset_rngs)
                no_law_reset_out = jax.vmap(no_law_env.reset, in_axes=(0,))(reset_rngs)
                (init_obs, init_full_obs), init_env_state = _select_env_data(
                    curriculum_state.use_no_law_env,
                    law_reset_out,
                    no_law_reset_out,
                )
                init_done = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}

                # 1) rollout
                runner_state = (train_state, init_env_state, init_obs, init_done, init_hstate, curriculum_state.use_no_law_env, rng)

                runner_state, traj_batch = jax.lax.scan(
                    _env_step, runner_state, None, config["ROLLOUT_LENGTH"])
                (train_state, env_state, obs, done, hstate, _, rng) = runner_state

                # 2) advantage
                # Get available actions for agent 0 from environment state
                avail_actions = env.get_avail_actions(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(batchify(avail_actions,
                    env.agents, config["NUM_ACTORS"]).astype(jnp.float32))

                # Get final value estimate for completed trajectory
                _, last_val, _, _ = policy.get_action_value_policy(
                    params=train_state.params,
                    obs=batchify(obs, env.agents, config["NUM_ACTORS"]).reshape(1, config["NUM_ACTORS"], -1),
                    done=batchify(done, env.agents, config["NUM_ACTORS"]).reshape(1, config["NUM_ACTORS"]),
                    avail_actions=avail_actions,
                    hstate=hstate,
                    rng=jax.random.PRNGKey(0)  # Dummy key since we're just extracting the value
                )
                last_val = last_val.squeeze(axis=0)
                advantages, targets = _calculate_gae(traj_batch, last_val)

                current_ent_coef = jax.lax.select(
                    curriculum_state.use_no_law_env,
                    jnp.asarray(no_law_ent_coef, dtype=jnp.float32),
                    jnp.asarray(law_ent_coef, dtype=jnp.float32),
                )

                # 3) PPO update
                update_state = (
                    train_state,
                    init_hstate, # shape is (num_controlled_actors, gru_hidden_dim) with all-0s value
                    traj_batch, # obs has shape (rollout_len, num_controlled_actors, -1)
                    advantages,
                    targets,
                    current_ent_coef,
                    rng
                )
                update_state, losses_and_grads = jax.lax.scan(
                    _update_epoch, update_state, None, config["UPDATE_EPOCHS"])
                train_state = update_state[0]
                _, loss_terms, avg_grad_norm = losses_and_grads

                metric = traj_batch.info
                metric["update_steps"] = update_steps
                metric["actor_loss"] = loss_terms[1]
                metric["value_loss"] = loss_terms[0]
                metric["entropy_loss"] = loss_terms[2]
                metric["avg_grad_norm"] = avg_grad_norm
                metric["curriculum_entropy_coef"] = current_ent_coef
                metric["curriculum_use_no_law_env"] = curriculum_state.use_no_law_env.astype(jnp.float32)
                new_runner_state = (train_state, rng, rng_eval, update_steps + 1, curriculum_state)
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
                (update_state, checkpoint_array, ckpt_idx, init_ckpt_eval_last_info, init_eval_last_info) = state_with_ckpt

                # Single IPPO update
                new_update_state, metric = _update_step(
                    update_state,
                    None
                )
                (train_state, rng, rng_eval, update_steps, curriculum_state) = new_update_state

                # update steps is 1-indexed because it was incremented at the end of the update step
                to_store = jnp.logical_or(jnp.equal(jnp.mod(update_steps-1, ckpt_and_eval_interval), 0),
                                        jnp.equal(update_steps, config["NUM_UPDATES"]))


                def store_and_eval_ckpt(args):
                    ckpt_arr, cidx, rng, rng_eval, prev_ckpt_eval_ret_info, prev_eval_ret_info = args
                    new_ckpt_arr = jax.tree.map(
                        lambda c_arr, p: c_arr.at[cidx].set(p),
                        ckpt_arr, train_state.params
                    )

                    if config["FIXED_EVAL"]:
                        eval_rng = rng_eval
                    else:
                        rng_eval, eval_rng = jax.random.split(rng_eval, 2)
                    ckpt_eval_eps_last_infos = jax.lax.cond(
                        curriculum_state.use_no_law_env,
                        lambda key: run_episodes_vmap(
                            key, no_law_env, agent_param=train_state.params, agent_policy=policy,
                            max_episode_steps=max_episode_steps,
                            num_eps=config["NUM_EVAL_EPISODES"]),
                        lambda key: run_episodes_vmap(
                            key, env, agent_param=train_state.params, agent_policy=policy,
                            max_episode_steps=max_episode_steps,
                            num_eps=config["NUM_EVAL_EPISODES"]),
                        eval_rng,
                    )
                    return (new_ckpt_arr, cidx + 1, rng, rng_eval, ckpt_eval_eps_last_infos, ckpt_eval_eps_last_infos)

                def skip_ckpt_and_eval(args):
                    def do_eval(eval_args):
                        ckpt_arr, cidx, rng, rng_eval, prev_ckpt_eval_ret_info, prev_eval_ret_info = eval_args
                        if config["FIXED_EVAL"]:
                            eval_rng = rng_eval
                        else:
                            rng_eval, eval_rng = jax.random.split(rng_eval, 2)
                        eval_eps_last_infos = jax.lax.cond(
                            curriculum_state.use_no_law_env,
                            lambda key: run_episodes_vmap(
                                key, no_law_env, agent_param=train_state.params, agent_policy=policy,
                                max_episode_steps=max_episode_steps,
                                num_eps=config["NUM_EVAL_EPISODES"]),
                            lambda key: run_episodes_vmap(
                                key, env, agent_param=train_state.params, agent_policy=policy,
                                max_episode_steps=max_episode_steps,
                                num_eps=config["NUM_EVAL_EPISODES"]),
                            eval_rng,
                        )
                        return (ckpt_arr, cidx, rng, rng_eval, prev_ckpt_eval_ret_info, eval_eps_last_infos)

                    def skip_eval(eval_args):
                        return eval_args

                    (ckpt_arr, cidx, rng, rng_eval, prev_ckpt_eval_ret_info, eval_eps_last_infos) = jax.lax.cond(
                        config["TRAIN_EVAL"],
                        do_eval,
                        skip_eval,
                        args
                    )

                    return (ckpt_arr, cidx, rng, rng_eval, prev_ckpt_eval_ret_info, eval_eps_last_infos)

                (checkpoint_array, ckpt_idx, rng, rng_eval, ckpt_eval_eps_last_infos, eval_eps_last_infos) = jax.lax.cond(
                    to_store, store_and_eval_ckpt, skip_ckpt_and_eval, (checkpoint_array, ckpt_idx, rng, rng_eval, init_ckpt_eval_last_info, init_eval_last_info)
                )

                # Convergence check based on EMA of train rollout return from this update.
                current_train_return = jnp.mean(metric["returned_episode_returns"])
                new_ema_return = (1.0 - ema_alpha) * curriculum_state.ema_eval_return + ema_alpha * current_train_return
                ema_delta = jnp.abs(new_ema_return - curriculum_state.ema_eval_return)
                stable_update = jnp.logical_and(ema_delta <= convergence_tol, current_train_return >= target_eval_return)
                stable_updates = jax.lax.select(stable_update, curriculum_state.stable_updates + 1, jnp.array(0, dtype=jnp.int32))
                has_min_updates = update_steps >= min_updates_before_switch
                if switch_on_min_updates:
                    converged = has_min_updates
                else:
                    converged = jnp.logical_and(has_min_updates, stable_updates >= convergence_patience)
                use_no_law_env = jnp.logical_or(curriculum_state.use_no_law_env, converged)
                curriculum_state = CurriculumState(
                    use_no_law_env=use_no_law_env,
                    ema_eval_return=new_ema_return,
                    stable_updates=stable_updates,
                )

                metric["ckpt_eval_ep_last_info"] = ckpt_eval_eps_last_infos
                metric["eval_ep_last_info"] = eval_eps_last_infos
                metric["curriculum_ema_train_return"] = new_ema_return
                metric["curriculum_stable_updates"] = stable_updates.astype(jnp.float32)
                metric["curriculum_converged"] = converged.astype(jnp.float32)
                metric["curriculum_use_no_law_env"] = use_no_law_env.astype(jnp.float32)
                return ((train_state, rng, rng_eval, update_steps, curriculum_state),
                         checkpoint_array, ckpt_idx, ckpt_eval_eps_last_infos, eval_eps_last_infos), metric

            checkpoint_array = init_ckpt_array(train_state.params)
            ckpt_idx = 0

            rng, rng_train = jax.random.split(rng, 2)

            rng_eval = jax.random.PRNGKey(config["EVAL_SEED"])
            rng_eval, eval_rng = jax.random.split(rng_eval, 2)
            if config["FIXED_EVAL"]:
                eval_rng = rng_eval

            # Init eval return infos
            eval_eps_last_infos = run_episodes_vmap(eval_rng, env,
                                    agent_param=train_state.params, agent_policy=policy,
                                    max_episode_steps=max_episode_steps,
                                    num_eps=config["NUM_EVAL_EPISODES"])

            # initial runner state for scanning
            update_steps = 0

            curriculum_state = CurriculumState(
                use_no_law_env=jnp.array(False),
                ema_eval_return=jnp.array(0.0, dtype=jnp.float32),
                stable_updates=jnp.array(0, dtype=jnp.int32),
            )

            update_runner_state = (train_state, rng_train, rng_eval, update_steps, curriculum_state)
            state_with_ckpt = (update_runner_state, checkpoint_array, ckpt_idx, eval_eps_last_infos, eval_eps_last_infos)

            state_with_ckpt, metrics = jax.lax.scan(
                _update_step_with_ckpt,
                state_with_ckpt,
                xs=None,
                length=config["NUM_UPDATES"]
            )

            (final_runner_state, checkpoint_array, final_ckpt_idx, ckpt_eval_eps_last_infos, eval_eps_last_infos) = state_with_ckpt
            out = {
                "final_params": final_runner_state[0].params,
                "metrics": metrics,  # shape (NUM_UPDATES, ...)
                "checkpoints": checkpoint_array,
            }

            if env._render:
                # Collect final eval gifs for logging
                rng_eval = final_runner_state[2] # extract final rng_eval from the final runner state after training
                if config["FIXED_EVAL"]:
                    eval_rng = rng_eval
                else:
                    rng_eval, eval_rng = jax.random.split(rng_eval, 2)
                params = final_runner_state[0].params
                out["render_outs"] = run_episodes_vmap(eval_rng, env,
                                                    agent_param=params, agent_policy=policy,
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

    no_law_env_kwargs = dict(config["task"]["NO_LAW_ENV_KWARGS"])
    no_law_env = make_env(algorithm_config["ENV_NAME"], no_law_env_kwargs)
    no_law_env = LogWrapper(no_law_env)

    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"])
    _, init_rng, train_rng = jax.random.split(rng, 3)

    # Initialize agent
    policy, init_params = initialize_agent(algorithm_config["ACTOR_TYPE"], algorithm_config, env, init_rng)

    log.info(f"Starting IPPO training ...")
    start_time = time.perf_counter()

    # Run the training
    out = train_ippo_agent(
        config=algorithm_config,
        env=env,
        no_law_env=no_law_env,
        train_rng=train_rng,
        policy=policy,
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

    return out["final_params"], policy, init_params

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

    # Process eval return metrics - average across train seeds, eval episodes, and num_agents per game for each checkpoint
    all_ckpt_collisions = np.asarray(train_metrics["ckpt_eval_ep_last_info"]["returned_episode_collisions"]) # shape (n_train_seeds, num_updates, num_eval_episodes, num_agents_per_game)
    all_collisions = np.asarray(train_metrics["eval_ep_last_info"]["returned_episode_collisions"]) # shape (n_train_seeds, num_updates, num_eval_episodes, num_agents_per_game)
    average_ckpt_collisions_per_iter = np.mean(np.sum(all_ckpt_collisions, axis=(3)), axis=(0, 2)) # shape (num_updates,)
    average_collisions_per_iter = np.mean(np.sum(all_collisions, axis=(3)), axis=(0, 2)) # shape (num_updates,)
    average_ckpt_agent_collisions_per_iter = np.mean(all_ckpt_collisions, axis=(0, 2)) # shape (num_updates,)
    average_agent_collisions_per_iter = np.mean(all_collisions, axis=(0, 2)) # shape (num_updates,)


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
             logger.log_item(f"Eval/Agent_{agent_id}/Collisions", average_agent_collisions_per_iter[step, agent_id], train_step=step, commit=True)
             logger.log_item(f"Eval/Agent_{agent_id}/CheckpointCollisions", average_ckpt_agent_collisions_per_iter[step, agent_id], train_step=step, commit=True)

        logger.log_item(f"Eval/Return", average_rets_per_iter[step], train_step=step, commit=True)
        logger.log_item(f"Eval/CheckpointReturn", average_ckpt_rets_per_iter[step], train_step=step, commit=True)
        logger.log_item(f"Eval/Collisions", average_collisions_per_iter[step], train_step=step, commit=True)
        logger.log_item(f"Eval/CheckpointCollisions", average_ckpt_collisions_per_iter[step], train_step=step, commit=True)

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

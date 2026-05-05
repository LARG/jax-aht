'''
Based on the MAPPO implementation from JaxMarl. Trains a parameter-shared, MLP MAPPO agent on a
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

from social_laws.common.run_episodes_mappo import run_episodes_vmap
from common.plot_utils import get_stats, get_metric_names
from common.save_load_utils import save_train_run
from envs import make_env
from envs.log_wrapper import LogWrapper
from marl.ppo_utils import batchify, unbatchify

from agents.initialize_agents import initialize_ma_s5_agent, initialize_ma_mlp_agent, initialize_ma_rnn_agent

class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    world_state: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray

def swap_batchify(x: dict, agent_list, num_actors, swap_axes):
    x = jnp.stack([x[a] for a in agent_list]).swapaxes(swap_axes[0], swap_axes[1])
    return x.reshape((num_actors, -1))

def _create_minibatches(traj_batch, advantages, targets, init_actor_hstate, init_critic_hstate, num_actors, num_minibatches, perm_rng):
    """Create minibatches for PPO updates, where each leaf has shape
        (num_minibatches, rollout_len, num_actors / num_minibatches, ...)
    This function ensures that the rollout (time) dimension is kept separate from the minibatch and num_actors
    dimensions, so that the minibatches are compatible with recurrent ActorCritics.
    """
    # Create batch containing trajectory, advantages, and targets
    batch = (
        init_actor_hstate, # shape (1, num_actors, hidden_dim)
        init_critic_hstate, # shape (1, num_actors, hidden_dim)
        traj_batch, # pytree: obs is shape (rollout_len, num_actors, feat_shape)
        advantages, # shape (rollout_len, num_actors)
        targets # shape (rollout_len, num_actors)
    )

    permutation = jax.random.permutation(perm_rng, num_actors)

    # each leaf of shuffled batch has shape (rollout_len, num_actors, feat_shape)
    # except for init_hstate which has shape (1, num_actors, hidden_dim)
    shuffled_batch = jax.tree.map(
        lambda x: jnp.take(x, permutation, axis=1), batch
    )
    # each leaf has shape (num_minibatches, rollout_len, num_actors/num_minibatches, feat_shape)
    # except for init_hstate which has shape (num_minibatches, 1, num_actors/num_minibatches, hidden_dim)
    minibatches = jax.tree_util.tree_map(
        lambda x: jnp.swapaxes(
            jnp.reshape(
                x,
                [x.shape[0], num_minibatches, -1]
                + list(x.shape[2:]),
        ), 1, 0,),
        shuffled_batch,
    )

    return minibatches

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def initialize_ma_agent(actor_type, algorithm_config, env, init_rng):
    if actor_type == "s5":
        policy, init_params = initialize_ma_s5_agent(algorithm_config, env, init_rng)
    elif actor_type == "mlp":
        policy, init_params = initialize_ma_mlp_agent(algorithm_config, env, init_rng)
    elif actor_type == "rnn":
        policy, init_params = initialize_ma_rnn_agent(algorithm_config, env, init_rng)
    return policy, init_params

def train_mappo_agent(config, env, train_rng,
                    policy, init_params):
    '''
    Train MAPPO the given initial parameters.

    Args:
        config: dict, config for the training
        env: gymnasium environment
        train_rng: jax.random.PRNGKey, random key for training
        policy: AgentPolicy, policy for the agent
        init_params: dict, initial parameters for the agent
    '''
    # ------------------------------
    # Build the MAPPO training function
    # ------------------------------
    def make_mappo_train(config):
        num_agents = env.num_agents
        config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
        config["NUM_UPDATES"] = (
            config["TOTAL_TIMESTEPS"] // config["ROLLOUT_LENGTH"] // config["NUM_ENVS"]
        )
        config["MINIBATCH_SIZE"] = (
            config["NUM_ACTORS"] * config["ROLLOUT_LENGTH"] // config["NUM_MINIBATCHES"]
        )

        config["CLIP_EPS"] = config["CLIP_EPS"] / env.num_agents if config["SCALE_CLIP_EPS"] else config["CLIP_EPS"]

        def linear_schedule(count):
            frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
            return config["LR"] * frac

        def train(rng):
            if config["ANNEAL_LR"]:
                actor_tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(learning_rate=linear_schedule, eps=1e-5),
                )
                critic_tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(learning_rate=linear_schedule, eps=1e-5),
                )
            else:
                actor_tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(config["LR"], eps=1e-5),
                )
                critic_tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(config["LR"], eps=1e-5),
                )

            actor_train_state = TrainState.create(
                apply_fn=policy.actor.apply,
                params=init_params['actor'],
                tx=actor_tx,
            )
            critic_train_state = TrainState.create(
                apply_fn=policy.critic.apply,
                params=init_params['critic'],
                tx=critic_tx,
            )

            #  Init hstates
            init_actor_hstate = policy.init_hstate(config["NUM_ACTORS"])
            init_critic_hstate = policy.init_critic_hstate(config["NUM_ACTORS"])

            def _env_step(runner_state, unused):
                """
                One step of the environment:
                1. Get observations, sample actions from all agents
                2. Step environment using sampled actions
                3. Return state, reward, ...
                """
                actor_train_state, critic_train_state, env_state, prev_world_state, prev_obs, prev_done, actor_hstate, critic_hstate, rng = runner_state
                rng, actor_rng, step_rng = jax.random.split(rng, 3)

                 # Get available actions for the agent from environment state
                avail_actions = env.get_avail_actions(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(batchify(avail_actions,
                    env.agents, config["NUM_ACTORS"]).astype(jnp.float32))

                prev_obs_batch = batchify(prev_obs, env.agents, config["NUM_ACTORS"])
                prev_done_batch = batchify(prev_done, env.agents, config["NUM_ACTORS"])

                # output of wrapper is (num_envs, num_agents, world_state_size)
                # swap axes to (num_agents, num_envs, world_state_size) before reshaping to (num_actors, world_state_size)
                prev_world_state_batch = swap_batchify(prev_world_state, env.agents, config["NUM_ACTORS"], swap_axes=(0,1))

                # Note that we do not need to reset the hidden states for the agents
                # as the recurrent states are automatically reset when done is True.

                # Controlled Agent action, value, log_prob
                act, val, pi, (new_actor_hstate, new_critic_hstate) = policy.get_action_value_policy(
                    params=(actor_train_state.params, critic_train_state.params),
                    state=prev_world_state_batch.reshape(1, config["NUM_ACTORS"], -1),
                    obs=prev_obs_batch.reshape(1, config["NUM_ACTORS"], -1),
                    done=prev_done_batch.reshape(1, config["NUM_ACTORS"]),
                    avail_actions=avail_actions,
                    hstate=(actor_hstate, critic_hstate),
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
                (obs_next, obs_full_next), world_state_next, env_state_next, reward, done_next, info = jax.vmap(env.step, in_axes=(0,0,0))(
                    step_rngs, env_state, env_act
                )

                # Filter out rendering-only fields (pre_reset_state, pre_reset_obs) which
                # have incompatible shapes for the per-actor reshape
                info = {k: v for k, v in info.items() if k not in ('pre_reset_state', 'pre_reset_obs')}
                info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)

                # Store agent_idx data in transition
                transition = Transition(
                    global_done=jnp.tile(done_next["__all__"], env.num_agents),
                    # done=batchify(done_next, env.agents, config["NUM_ACTORS"]).squeeze(), # shape (num_envs,)
                    done=prev_done_batch, # JaxMARL MAPPO uses prev done here
                    action=act, # shape (num_envs,)
                    value=val, # shape (num_envs,)
                    reward=batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(), # shape (num_envs,)
                    log_prob=logp, # shape (num_envs,)
                    obs=prev_obs_batch, # shape (num_envs, obs_dim)
                    # Keep world_state as (num_actors, state_dim) for minibatch reshaping.
                    world_state=prev_world_state_batch,
                    info=info,
                    avail_actions=avail_actions # shape (num_envs, num_actions)
                )

                new_runner_state = (actor_train_state, critic_train_state, env_state_next, world_state_next, obs_next, done_next, new_actor_hstate, new_critic_hstate, rng)
                return new_runner_state, transition

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.global_done,
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

            def _update_minbatch(train_states, batch_info):
                actor_train_state, critic_train_state = train_states
                init_actor_hstate, init_critic_hstate, traj_batch, advantages, targets = batch_info

                def _actor_loss_fn(actor_params, init_hstate, traj_batch, gae):
                    _, pi, _ = policy.get_action_policy(
                        params=actor_params,
                        obs=traj_batch.obs,
                        done=traj_batch.done,
                        avail_actions=traj_batch.avail_actions,
                        hstate=init_hstate,
                        rng=jax.random.PRNGKey(0) # only used for action sampling, which is unused here
                    )
                    log_prob = pi.log_prob(traj_batch.action)

                    logratio = log_prob - traj_batch.log_prob
                    ratio = jnp.exp(logratio)
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    loss_actor1 = ratio * gae
                    loss_actor2 = (
                        jnp.clip(
                            ratio,
                            1.0 - config["CLIP_EPS"],
                            1.0 + config["CLIP_EPS"],
                        )
                        * gae
                    )
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                    loss_actor = loss_actor.mean()
                    entropy = pi.entropy().mean()

                    approx_kl = ((ratio - 1) - logratio).mean()
                    clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])

                    actor_loss = (
                        loss_actor
                        - config["ENT_COEF"] * entropy
                    )
                    return actor_loss, (loss_actor, entropy, ratio, approx_kl, clip_frac)

                def _critic_loss_fn(critic_params, init_hstate, traj_batch, targets):
                    value, _ = policy.get_value(
                        params=critic_params,
                        state=traj_batch.world_state,
                        done=traj_batch.done,
                        hstate=init_hstate,
                        rng=jax.random.PRNGKey(0) # only used for action sampling, which is unused here
                    )

                    # if value.ndim == traj_batch.value.ndim + 1 and value.shape[-1] == 1:
                    #     value = value.squeeze(-1)

                    value_pred_clipped = traj_batch.value + (
                        value - traj_batch.value
                    ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = (
                        0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                    )
                    critic_loss = config["VF_COEF"] * value_loss
                    return critic_loss, value_loss

                actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                (actor_loss_val, actor_aux_vals), actor_grads = actor_grad_fn(
                    actor_train_state.params, init_actor_hstate, traj_batch, advantages
                )
                critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                (critic_loss_val, critic_aux_vals), critic_grads = critic_grad_fn(
                    critic_train_state.params, init_critic_hstate, traj_batch, targets
                )

                actor_train_state = actor_train_state.apply_gradients(grads=actor_grads)
                critic_train_state = critic_train_state.apply_gradients(grads=critic_grads)

                # compute average grad norm
                actor_grad_l2_norms = jax.tree.map(lambda g: jnp.linalg.norm(g.astype(jnp.float32)), actor_grads)
                sum_of_actor_grad_norms = jax.tree.reduce(lambda x, y: x + y, actor_grad_l2_norms)
                n_elements = len(jax.tree.leaves(actor_grad_l2_norms))
                avg_actor_grad_norm = sum_of_actor_grad_norms / n_elements

                critic_grad_l2_norms = jax.tree.map(lambda g: jnp.linalg.norm(g.astype(jnp.float32)), critic_grads)
                sum_of_critic_grad_norms = jax.tree.reduce(lambda x, y: x + y, critic_grad_l2_norms)
                n_elements = len(jax.tree.leaves(critic_grad_l2_norms))
                avg_critic_grad_norm = sum_of_critic_grad_norms / n_elements

                return (actor_train_state, critic_train_state), (actor_loss_val + critic_loss_val, actor_loss_val, critic_loss_val, actor_aux_vals, critic_aux_vals, avg_actor_grad_norm, avg_critic_grad_norm)

            def _update_epoch(update_state, unused):
                actor_train_state, critic_train_state, init_actor_hstate, init_critic_hstate, traj_batch, advantages, targets, rng = update_state
                rng, perm_rng = jax.random.split(rng)
                minibatches = _create_minibatches(traj_batch, advantages, targets, init_actor_hstate, init_critic_hstate, config["NUM_ACTORS"], config["NUM_MINIBATCHES"], perm_rng)
                (actor_train_state, critic_train_state), losses_and_grads = jax.lax.scan(
                    _update_minbatch, (actor_train_state, critic_train_state), minibatches
                )
                update_state = (actor_train_state, critic_train_state, init_actor_hstate, init_critic_hstate, traj_batch, advantages, targets, rng)
                return update_state, losses_and_grads

            def _update_step(update_runner_state, unused):
                """
                1. Collect rollouts
                2. Compute advantage
                3. PPO updates
                """
                (actor_train_state, critic_train_state, rng, rng_eval, update_steps) = update_runner_state
                # Init envs & partner indices
                rng, reset_rng = jax.random.split(rng, 2)
                reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
                (init_obs, init_full_obs), init_world_state, init_env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rngs)
                init_done = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}

                # 1) rollout
                runner_state = (actor_train_state, critic_train_state, init_env_state, init_world_state, init_obs, init_done, init_actor_hstate, init_critic_hstate, rng)

                runner_state, traj_batch = jax.lax.scan(
                    _env_step, runner_state, None, config["ROLLOUT_LENGTH"])
                (actor_train_state, critic_train_state, env_state, world_state, obs, done, actor_hstate, critic_hstate, rng) = runner_state

                # 2) advantage
                # Get final value estimate for completed trajectory
                world_state_batch = swap_batchify(world_state, env.agents, config["NUM_ACTORS"], swap_axes=(0,1))
                last_val, _ = policy.get_value(
                    params=critic_train_state.params,
                    state=world_state_batch.reshape((1, config["NUM_ACTORS"], -1)),
                    done=batchify(done, env.agents, config["NUM_ACTORS"]).reshape(1, config["NUM_ACTORS"]),
                    hstate=critic_hstate,
                    rng=jax.random.PRNGKey(0)  # Dummy key since we're just extracting the value
                )
                last_val = last_val.squeeze(axis=0)
                advantages, targets = _calculate_gae(traj_batch, last_val)

                # 3) MAPPO update
                update_state = (
                    actor_train_state, critic_train_state,
                    init_actor_hstate, init_critic_hstate, # shape is (num_controlled_actors, gru_hidden_dim) with all-0s value
                    traj_batch, # obs has shape (rollout_len, num_controlled_actors, -1)
                    advantages,
                    targets,
                    rng
                )
                update_state, losses_and_grads = jax.lax.scan(
                    _update_epoch, update_state, None, config["UPDATE_EPOCHS"])
                actor_train_state = update_state[0]
                critic_train_state = update_state[1]
                total_loss, actor_loss, critic_loss, actor_loss_terms, critic_loss_terms, avg_actor_grad_norm, avg_critic_grad_norm = losses_and_grads

                metric = traj_batch.info
                metric["update_steps"] = update_steps
                metric["actor_loss"] = actor_loss
                metric["critic_loss"] = critic_loss
                metric["entropy_loss"] = actor_loss_terms[1]
                metric["avg_actor_grad_norm"] = avg_actor_grad_norm
                metric["avg_critic_grad_norm"] = avg_critic_grad_norm
                new_runner_state = (actor_train_state, critic_train_state, rng, rng_eval, update_steps + 1)
                return (new_runner_state, metric)

            # MAPPO Update and Checkpoint saving
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

                # Single MAPPO update
                new_update_state, metric = _update_step(
                    update_state,
                    None
                )
                (actor_train_state, critic_train_state, rng, rng_eval, update_steps) = new_update_state

                # update steps is 1-indexed because it was incremented at the end of the update step
                to_store = jnp.logical_or(jnp.equal(jnp.mod(update_steps-1, ckpt_and_eval_interval), 0),
                                        jnp.equal(update_steps, config["NUM_UPDATES"]))


                def store_and_eval_ckpt(args):
                    ckpt_arr, cidx, rng, rng_eval, prev_ckpt_eval_ret_info, prev_eval_ret_info = args
                    new_ckpt_arr = jax.tree.map(
                        lambda c_arr, p: c_arr.at[cidx].set(p),
                        ckpt_arr, {'actor': actor_train_state.params, 'critic': critic_train_state.params}
                    )

                    if config["FIXED_EVAL"]:
                        eval_rng = rng_eval
                    else:
                        rng_eval, eval_rng = jax.random.split(rng_eval, 2)
                    ckpt_eval_eps_last_infos = run_episodes_vmap(
                        eval_rng, env, agent_param=actor_train_state.params, agent_policy=policy,
                        max_episode_steps=max_episode_steps,
                        num_eps=config["NUM_EVAL_EPISODES"])
                    return (new_ckpt_arr, cidx + 1, rng, rng_eval, ckpt_eval_eps_last_infos, ckpt_eval_eps_last_infos)

                def skip_ckpt_and_eval(args):
                    def do_eval(eval_args):
                        ckpt_arr, cidx, rng, rng_eval, prev_ckpt_eval_ret_info, prev_eval_ret_info = eval_args
                        if config["FIXED_EVAL"]:
                            eval_rng = rng_eval
                        else:
                            rng_eval, eval_rng = jax.random.split(rng_eval, 2)
                        eval_eps_last_infos = run_episodes_vmap(
                            eval_rng, env, agent_param=actor_train_state.params, agent_policy=policy,
                            max_episode_steps=max_episode_steps,
                            num_eps=config["NUM_EVAL_EPISODES"])
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

                metric["ckpt_eval_ep_last_info"] = ckpt_eval_eps_last_infos
                metric["eval_ep_last_info"] = eval_eps_last_infos
                return ((actor_train_state, critic_train_state, rng, rng_eval, update_steps),
                         checkpoint_array, ckpt_idx, ckpt_eval_eps_last_infos, eval_eps_last_infos), metric

            checkpoint_array = init_ckpt_array({'actor': actor_train_state.params, 'critic': critic_train_state.params})
            ckpt_idx = 0

            rng, rng_train = jax.random.split(rng, 2)

            rng_eval = jax.random.PRNGKey(config["EVAL_SEED"])
            rng_eval, eval_rng = jax.random.split(rng_eval, 2)
            if config["FIXED_EVAL"]:
                eval_rng = rng_eval

            # Init eval return infos
            eval_eps_last_infos = run_episodes_vmap(eval_rng, env,
                                    agent_param=actor_train_state.params, agent_policy=policy,
                                    max_episode_steps=max_episode_steps,
                                    num_eps=config["NUM_EVAL_EPISODES"])

            # initial runner state for scanning
            update_steps = 0

            update_runner_state = (actor_train_state, critic_train_state, rng_train, rng_eval, update_steps)
            state_with_ckpt = (update_runner_state, checkpoint_array, ckpt_idx, eval_eps_last_infos, eval_eps_last_infos)

            state_with_ckpt, metrics = jax.lax.scan(
                _update_step_with_ckpt,
                state_with_ckpt,
                xs=None,
                length=config["NUM_UPDATES"]
            )

            (final_runner_state, checkpoint_array, final_ckpt_idx, ckpt_eval_eps_last_infos, eval_eps_last_infos) = state_with_ckpt
            out = {
                "final_params": {'actor': final_runner_state[0].params, 'critic': final_runner_state[1].params},
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
                actor_params = final_runner_state[0].params
                out["render_outs"] = run_episodes_vmap(eval_rng, env,
                                                    agent_param=actor_params, agent_policy=policy,
                                                    max_episode_steps=env.horizon,
                                                    num_eps=5, render=True)

            return out
        return train

    # ------------------------------
    # Actually run the MAPPO training
    # ------------------------------
    rngs = jax.random.split(train_rng, config["NUM_TRAIN_SEEDS"])

    # Run training seeds in parallel using vmap
    train_fn = make_mappo_train(config)
    out = jax.vmap(train_fn, in_axes=(0))(rngs)
    return out

def run_training(config, wandb_logger):
    '''Run MAPPO training.

    Args:
        config: dict, config for the training
    '''
    algorithm_config = dict(config["algorithm"])

    # Create only one environment instance
    env_kwargs = dict(algorithm_config["ENV_KWARGS"])
    env_kwargs["world_state"] = True
    env = make_env(algorithm_config["ENV_NAME"], env_kwargs)
    env = LogWrapper(env)

    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"])
    _, init_rng, train_rng = jax.random.split(rng, 3)

    # Initialize agent
    policy, init_params = initialize_ma_agent(algorithm_config["ACTOR_TYPE"], algorithm_config, env, init_rng)

    log.info(f"Starting MAPPO training ...")
    start_time = time.perf_counter()

    # Run the training
    out = train_mappo_agent(
        config=algorithm_config,
        env=env,
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
    log.info(f"MAPPO Training completed in {elapsed_time:.2f}s")
    log.info(f"MAPPO Training completed in {int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s {milliseconds:03d}ms {microseconds:03d}µs")

    # process and log metrics
    log.info(f"Starting MAPPO logging ...")
    start_time = time.perf_counter()
    metric_names = get_metric_names(config["ENV_NAME"])
    log_metrics(env, config, out, wandb_logger, metric_names)
    elapsed_time = time.perf_counter() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, rem = divmod(rem, 60)
    seconds, rem = divmod(rem, 1)
    milliseconds = int(rem * 1000)
    microseconds = int((rem * 1_000_000) % 1000)
    log.info(f"MAPPO Logging completed in {elapsed_time:.2f}s")
    log.info(f"MAPPO Logging completed in {int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s {milliseconds:03d}ms {microseconds:03d}µs")

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

    all_agent_critic_losses = np.asarray(train_metrics["critic_loss"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    all_agent_actor_losses = np.asarray(train_metrics["actor_loss"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    all_agent_entropy_losses = np.asarray(train_metrics["entropy_loss"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    all_agent_actor_grad_norms = np.asarray(train_metrics["avg_actor_grad_norm"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    all_agent_critic_grad_norms = np.asarray(train_metrics["avg_critic_grad_norm"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)

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
    average_agent_critic_losses = np.mean(all_agent_critic_losses, axis=(0, 2, 3))  # shape (num_updates,)
    average_agent_actor_losses = np.mean(all_agent_actor_losses, axis=(0, 2, 3)) # shape (num_updates,)
    average_agent_entropy_losses = np.mean(all_agent_entropy_losses, axis=(0, 2, 3)) # shape (num_updates,)
    average_agent_actor_grad_norms = np.mean(all_agent_actor_grad_norms, axis=(0, 2, 3)) # shape (num_updates,)
    average_agent_critic_grad_norms = np.mean(all_agent_critic_grad_norms, axis=(0, 2, 3)) # shape (num_updates,)

    # Log metrics for each update step
    num_agents = env.num_agents
    num_updates = len(average_agent_critic_losses)
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

        logger.log_item(f"Train/CriticLoss", average_agent_critic_losses[step], train_step=step, commit=True)
        logger.log_item(f"Train/ActorLoss", average_agent_actor_losses[step], train_step=step, commit=True)
        logger.log_item(f"Train/EntropyLoss", average_agent_entropy_losses[step], train_step=step, commit=True)
        logger.log_item(f"Train/ActorGradNorm", average_agent_actor_grad_norms[step], train_step=step, commit=True)
        logger.log_item(f"Train/CriticGradNorm", average_agent_critic_grad_norms[step], train_step=step, commit=True)
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

    out_savepath = save_train_run(train_out, savedir, savename=f"MAPPO_Agent_Train_Run")
    if config["logger"]["log_train_out"]:
        logger.log_artifact(name=f"MAPPO_Agent_Train_Run", path=out_savepath, type_name="train_run")
        # Cleanup locally logged out file
    if not config["local_logger"]["save_train_out"]:
        shutil.rmtree(out_savepath)

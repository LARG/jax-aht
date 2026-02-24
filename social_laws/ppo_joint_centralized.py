'''
Script for training a PPO agent for social laws single agent projection.
Does not support training against heuristic partner agents.

Command to run PPO single agent projection training:
python social_laws/run.py algorithm=ppo/lbf task=lbf label=test_ppo_single_agent_projection

Suggested debug command:
python social_laws/run.py algorithm=ppo/lbf task=lbf logger.mode=disabled label=debug algorithm.TOTAL_TIMESTEPS=1e5
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
from typing import NamedTuple

from social_laws.common.initialize_agents import initialize_agent
from social_laws.common.run_episodes_w_robustness_centralized import run_episodes_vmap
from common.plot_utils import get_stats, get_metric_names
from common.save_load_utils import save_train_run
from envs import make_env
from envs.log_wrapper import LogWrapper
from marl.ppo_utils import _create_minibatches, batchify, unbatchify

class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray
    world_state: jnp.ndarray # Full state or all agents' observations, depending env

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def train_ppo_joint_agents(config, env, optimal_env, train_rng,
                           joint_policy, init_joint_params,
                           ppo_policies, ppo_params,
                           vf_policies, vf_params,
                           agent_idx):
    '''
    Train PPO joint agents using the given initial parameters.

    Args:
        config: dict, config for the training
        env: gymnasium environment
        train_rng: jax.random.PRNGKey, random key for training
        joint_policies: tuple of AgentPolicy, policies for the agents
        init_joint_params: tuple of dict, initial parameters for the agents
        ppo_policies: tuple of AgentPolicy, PPO policies for the agents
        ppo_params: tuple of dict, PPO parameters for the agents
        vf_policies: tuple of AgentPolicy, value function policies for the agents
        vf_params: tuple of dict, value function parameters for the agents
        agent_idx: int, index of the agent to optimize
    '''
    # ------------------------------
    # Build the PPO joint training function
    # ------------------------------
    agent_0_ppo_policy, agent_1_ppo_policy = ppo_policies
    agent_0_ppo_params, agent_1_ppo_params = ppo_params
    agent_0_vf_policy, agent_1_vf_policy = vf_policies
    agent_0_vf_params, agent_1_vf_params = vf_params

    def make_ppo_joint_train(config):
        '''The controlled agent is based on the agent_idx parameter'''
        num_agents = env.num_agents
        assert num_agents == 2, "This snippet assumes exactly 2 agents."

        config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
        config["NUM_UNCONTROLLED_ACTORS"] = config["NUM_ENVS"] # assumption: we control 1 agent
        config["NUM_CONTROLLED_ACTORS"] = config["NUM_ENVS"] # assumption: we control 1 agent
        config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["ROLLOUT_LENGTH"] // config["NUM_ENVS"]

        config["NUM_ACTIONS"] = env.action_space(f"agent_{agent_idx}").n
        assert config["NUM_CONTROLLED_ACTORS"] % config["NUM_MINIBATCHES"] == 0, "NUM_CONTROLLED_ACTORS must be divisible by NUM_MINIBATCHES"
        assert config["NUM_CONTROLLED_ACTORS"] >= config["NUM_MINIBATCHES"], "NUM_CONTROLLED_ACTORS must be >= NUM_MINIBATCHES"

        def linear_schedule(count):
            frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
            return config["LR"] * frac

        def train(rng, agent_idx, agent_0_ppo_params, agent_1_ppo_params, agent_0_vf_params, agent_1_vf_params):
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

            joint_train_state = TrainState.create(
                apply_fn=joint_policy.network.apply,
                params=init_joint_params,
                tx=tx,
            )

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
            joint_init_hstate = joint_policy.init_hstate(config["NUM_ACTORS"])
            agent_0_vf_init_hstate = agent_0_vf_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])
            agent_1_vf_init_hstate = agent_1_vf_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])

            def _get_vf_restricted_avail_actions(obs, done, avail_actions, hstate, vf_params, vf_policy):
                """
                Get restricted available actions based on value function Q-values.
                Only allows actions within epsilon of the maximum Q-value.

                Args:
                    obs: observations (shape: num_envs, obs_dim)
                    done: done flags (shape: num_envs,)
                    avail_actions: available actions mask (shape: num_envs, num_actions)
                    hstate: hidden state for the value function policy (shape depends on policy)
                    vf_params: value function parameters
                    vf_policy: value function policy

                Returns:
                    restricted_avail_actions: mask with only near-optimal actions (shape: num_envs, num_actions)
                """
                _, q_values, _, next_hstate = vf_policy.get_action_value_policy(
                    vf_params, obs, done, avail_actions,
                    hstate, jax.random.PRNGKey(0) # Use dummy since we only need the qvals
                ) # (num_envs, num_actions)

                # Mask Q-values with available actions (set unavailable to -inf)
                q_values_masked = jnp.where(avail_actions, q_values, -jnp.inf)

                # Find max Q-value for each environment
                max_q = jnp.max(q_values_masked, axis=-1, keepdims=True)  # (num_envs, 1)

                # Create mask for actions within epsilon of max
                optimal_action_mask = (q_values_masked >= (max_q - config['EPSILON_OPTIMAL'])).astype(jnp.float32)

                # Combine with original available actions
                restricted_avail_actions = optimal_action_mask * avail_actions

                # remove extra batch dim
                return restricted_avail_actions.squeeze(axis=0), next_hstate

            def _env_step(runner_state, unused):
                """
                One step of the environment:
                1. Get observations, sample actions from all agents
                2. Step environment using sampled actions
                3. Return state, reward, ...
                """
                joint_train_state, env_state, prev_obs, prev_done, joint_hstate, vf_hstate_0, vf_hstate_1, rng = runner_state
                prev_obs, prev_full_obs = prev_obs
                rng, actor_rng, step_rng = jax.random.split(rng, 3)

                 # Get available actions for the agent from environment state
                avail_actions = env.get_avail_actions(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(avail_actions)

                # Restrict available actions based on value function
                vf_restricted_avail_actions_0, new_vf_hstate_0 = _get_vf_restricted_avail_actions(
                    obs=get_agent_data(prev_obs, 0).reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                    done=get_agent_data(prev_done, 0).reshape(1, config["NUM_CONTROLLED_ACTORS"]),
                    avail_actions=get_agent_data(avail_actions, 0).astype(jnp.float32),
                    hstate=vf_hstate_0,
                    vf_params=agent_0_vf_params,
                    vf_policy=agent_0_vf_policy
                )
                vf_restricted_avail_actions_1, new_vf_hstate_1 = _get_vf_restricted_avail_actions(
                    obs=get_agent_data(prev_obs, 1).reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                    done=get_agent_data(prev_done, 1).reshape(1, config["NUM_CONTROLLED_ACTORS"]),
                    avail_actions=get_agent_data(avail_actions, 1).astype(jnp.float32),
                    hstate=vf_hstate_1,
                    vf_params=agent_1_vf_params,
                    vf_policy=agent_1_vf_policy
                )

                # Note that we do not need to reset the hidden states for the agents
                # as the recurrent states are automatically reset when done is True.

                # Centralized joint observation - concatenate both agents' full observations
                joint_prev_obs = batchify(prev_obs, env.agents, config["NUM_ACTORS"]) # shape (num_agents * num_envs, obs_dim)
                joint_prev_full_obs = batchify(prev_full_obs, env.agents, config["NUM_ACTORS"]) # shape (num_agents * num_envs)
                joint_prev_done = batchify(prev_done, env.agents, config["NUM_ACTORS"])

                # Stack restricted available actions for both agents
                joint_vf_restricted_avail_actions = jnp.stack([vf_restricted_avail_actions_0, vf_restricted_avail_actions_1], axis=0).reshape(config["NUM_ACTORS"], -1)

                # Joint policy outputs actions for both agents
                joint_act, joint_val, joint_pi, new_joint_hstate = joint_policy.get_action_value_policy(
                    params=joint_train_state.params,
                    obs=jnp.expand_dims(joint_prev_full_obs, axis=0) if config["JOINT_USE_FULL_OBS"] else jnp.expand_dims(joint_prev_obs, axis=0),
                    done=jnp.expand_dims(joint_prev_done, axis=0),
                    avail_actions=joint_vf_restricted_avail_actions,
                    hstate=joint_hstate,
                    rng=actor_rng
                )
                joint_logp = joint_pi.log_prob(joint_act)

                joint_act = joint_act.squeeze(axis=0)
                joint_logp = joint_logp.squeeze(axis=0)
                joint_val = joint_val.squeeze(axis=0)

                env_act = unbatchify(joint_act, env.agents, config["NUM_ENVS"], env.num_agents)

                # Step env
                step_rngs = jax.random.split(step_rng, config["NUM_ENVS"])
                obs_next, env_state_next, reward, done_next, info = jax.vmap(env.step, in_axes=(0,0,0))(
                    step_rngs, env_state, env_act
                )
                # note that num_actors = num_envs * num_agents
                # Get agent_idx info from info dict, excluding certain keys
                # Primarily, concerned with episode returns
                keys_to_exclude = {'pre_reset_state', 'pre_reset_obs'}  # Add any keys that shouldn't be indexed

                def filter_agent_info(path, x, idx):
                    # Get the key name from the path
                    key_name = path[-1].key if hasattr(path[-1], 'key') else None
                    if key_name in keys_to_exclude or x.ndim <= 1:
                        return x
                    else:
                        return x[:, idx]

                info = jax.tree_util.tree_map_with_path(partial(filter_agent_info, idx=agent_idx), info)

                reward = get_agent_data(reward, agent_idx) * -1

                # Store joint transition data
                joint_transition = Transition(
                    global_done=jnp.tile(done_next["__all__"], env.num_agents),
                    done=batchify(prev_done, env.agents, config["NUM_ACTORS"]),
                    action=joint_act,
                    value=joint_val,
                    reward=jnp.stack([reward, reward], axis=0).reshape(config["NUM_ACTORS"]),
                    log_prob=joint_logp,
                    obs=joint_prev_full_obs if config["JOINT_USE_FULL_OBS"] else joint_prev_obs,
                    info=info,
                    avail_actions=joint_vf_restricted_avail_actions,
                    world_state=joint_prev_full_obs
                )

                new_runner_state = (joint_train_state, env_state_next, obs_next, done_next,
                                    new_joint_hstate, new_vf_hstate_0, new_vf_hstate_1, rng)
                return new_runner_state, joint_transition

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

            # TODO: Might need to change network and loss to match JAXAMRL MAPPO
            def _update_minbatch(train_state, batch_info):
                init_hstate, traj_batch, advantages, returns = batch_info
                def _loss_fn(params, init_hstate, traj_batch, gae, target_v):
                    _, value, pi, _ = joint_policy.get_action_value_policy(
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

                    total_loss = pg_loss + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
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
                train_state, init_hstate, traj_batch, advantages, targets, rng = update_state
                rng, perm_rng = jax.random.split(rng)
                minibatches = _create_minibatches(traj_batch, advantages, targets, init_hstate, config["NUM_CONTROLLED_ACTORS"], config["NUM_MINIBATCHES"], perm_rng)
                train_state, losses_and_grads = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
                return update_state, losses_and_grads

            def _update_step(update_runner_state, unused):
                """
                1. Collect rollouts
                2. Compute advantage
                3. PPO updates
                """
                (joint_train_state, rng, rng_eval, update_steps) = update_runner_state
                # Init envs & partner indices
                rng, reset_rng = jax.random.split(rng, 2)
                reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
                init_obs, init_env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rngs)
                init_done = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}

                # 1) rollout
                runner_state = (joint_train_state, init_env_state, init_obs, init_done, joint_init_hstate, agent_0_vf_init_hstate, agent_1_vf_init_hstate, rng)

                runner_state, joint_traj_batch = jax.lax.scan(
                    _env_step, runner_state, None, config["ROLLOUT_LENGTH"])
                (joint_train_state, env_state, obs, done, joint_hstate, agent_0_vf_hstate, agent_1_vf_hstate, rng) = runner_state
                obs, obs_full = obs

                # 2) advantage
                # Get available actions from environment state
                avail_actions = env.get_avail_actions(env_state.env_state)

                # Restrict available actions based on value function
                vf_restricted_avail_actions_0, _ = _get_vf_restricted_avail_actions(
                    obs=get_agent_data(obs, 0).reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                    done=get_agent_data(done, 0).reshape(1, config["NUM_CONTROLLED_ACTORS"]),
                    avail_actions=get_agent_data(avail_actions, 0).astype(jnp.float32),
                    hstate=agent_0_vf_hstate,
                    vf_params=agent_0_vf_params,
                    vf_policy=agent_0_vf_policy
                )
                vf_restricted_avail_actions_1, _ = _get_vf_restricted_avail_actions(
                    obs=get_agent_data(obs, 1).reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                    done=get_agent_data(done, 1).reshape(1, config["NUM_CONTROLLED_ACTORS"]),
                    avail_actions=get_agent_data(avail_actions, 1).astype(jnp.float32),
                    hstate=agent_1_vf_hstate,
                    vf_params=agent_1_vf_params,
                    vf_policy=agent_1_vf_policy
                )

                # Centralized joint observation - concatenate both agents' full observations
                joint_obs = batchify(obs, env.agents, config["NUM_ACTORS"]) # shape (num_agents * num_envs, obs_dim)
                joint_obs_full = batchify(obs_full, env.agents, config["NUM_ACTORS"]) # shape (num_agents * num_envs)
                joint_done = batchify(done, env.agents, config["NUM_ACTORS"])

                # Stack restricted available actions for both agents
                joint_vf_restricted_avail_actions = jnp.stack([vf_restricted_avail_actions_0, vf_restricted_avail_actions_1], axis=0).reshape(config["NUM_ACTORS"], -1)

                # Get final value estimate for completed trajectory using joint policy
                _, last_joint_val, _, _ = joint_policy.get_action_value_policy(
                    params=joint_train_state.params,
                    obs=jnp.expand_dims(joint_obs_full, axis=0) if config["JOINT_USE_FULL_OBS"] else jnp.expand_dims(joint_obs, axis=0),
                    done=jnp.expand_dims(joint_done, axis=0),
                    avail_actions=jax.lax.stop_gradient(joint_vf_restricted_avail_actions),
                    hstate=joint_hstate,
                    rng=jax.random.PRNGKey(0)  # Dummy key since we're just extracting the value
                )
                last_joint_val = last_joint_val.squeeze(axis=0)

                # Calculate GAE for joint policy
                advantages, targets = _calculate_gae(joint_traj_batch, last_joint_val)

                # 3) PPO update for joint policy
                update_state = (
                    joint_train_state,
                    joint_init_hstate,
                    joint_traj_batch,
                    advantages,
                    targets,
                    rng
                )

                update_state, losses_and_grads = jax.lax.scan(
                    _update_epoch, update_state, None, config["UPDATE_EPOCHS"])
                joint_train_state = update_state[0]
                _, loss_terms, avg_grad_norm = losses_and_grads

                # Metrics
                metric = joint_traj_batch.info
                metric["update_steps"] = update_steps
                metric["actor_loss"] = loss_terms[1]
                metric["value_loss"] = loss_terms[0]
                metric["entropy_loss"] = loss_terms[2]
                metric["avg_grad_norm"] = avg_grad_norm
                new_runner_state = (joint_train_state, rng, rng_eval, update_steps + 1)
                return (new_runner_state, metric)

            # PPO Update and Checkpoint saving
            ckpt_and_eval_interval = config["NUM_UPDATES"] // max(1, config["NUM_CHECKPOINTS"] - 1)  # -1 because we store a ckpt at the last update
            num_ckpts = config["NUM_CHECKPOINTS"]

            # Build a PyTree that holds parameters for all FCP checkpoints
            def init_ckpt_array(params_pytree):
                return jax.tree.map(
                    lambda x: jnp.zeros((num_ckpts,) + x.shape, x.dtype),
                    params_pytree)

            max_episode_steps = config["ROLLOUT_LENGTH"]

            def _update_step_with_ckpt(state_with_ckpt, unused):
                (update_state, checkpoint_array, ckpt_idx, init_ckpt_eval_last_info, init_eval_last_info) = state_with_ckpt

                # Single PPO update
                new_update_state, metric = _update_step(
                    update_state,
                    None
                )
                (joint_train_state, rng, rng_eval, update_steps) = new_update_state

                # update steps is 1-indexed because it was incremented at the end of the update step
                to_store = jnp.logical_or(jnp.equal(jnp.mod(update_steps-1, ckpt_and_eval_interval), 0),
                                        jnp.equal(update_steps, config["NUM_UPDATES"]))


                def store_and_eval_ckpt(args):
                    ckpt_arr, cidx, rng, rng_eval, prev_ckpt_eval_ret_info, prev_eval_ret_info = args
                    new_ckpt_arr = jax.tree.map(
                        lambda c_arr, p: c_arr.at[cidx].set(p),
                        ckpt_arr, joint_train_state.params
                    )

                    if config["FIXED_EVAL"]:
                        eval_rng = rng_eval
                    else:
                        rng_eval, eval_rng = jax.random.split(rng_eval, 2)
                    ckpt_eval_eps_last_infos = run_episodes_vmap(eval_rng, env, optimal_env, agent_idx,
                        agent_params=joint_train_state.params,
                        agent_policy=joint_policy,
                        ppo_params=(agent_0_ppo_params, agent_1_ppo_params),
                        ppo_policies=(agent_0_ppo_policy, agent_1_ppo_policy),
                        vf_params=(agent_0_vf_params, agent_1_vf_params),
                        vf_policies=(agent_0_vf_policy, agent_1_vf_policy),
                        max_episode_steps=max_episode_steps,
                        num_eps=config["NUM_EVAL_EPISODES"],
                        epsilon_optimal=config["EPSILON_OPTIMAL"],
                        use_full_obs=config["JOINT_USE_FULL_OBS"])

                    return (new_ckpt_arr, cidx + 1, rng, rng_eval, ckpt_eval_eps_last_infos, ckpt_eval_eps_last_infos)

                def skip_ckpt_and_eval(args):
                    def do_eval(eval_args):
                        ckpt_arr, cidx, rng, rng_eval, prev_ckpt_eval_ret_info, prev_eval_ret_info = eval_args
                        if config["FIXED_EVAL"]:
                            eval_rng = rng_eval
                        else:
                            rng_eval, eval_rng = jax.random.split(rng_eval, 2)
                        eval_eps_last_infos = run_episodes_vmap(eval_rng, env, optimal_env, agent_idx,
                            agent_params=joint_train_state.params,
                            agent_policy=joint_policy,
                            ppo_params=(agent_0_ppo_params, agent_1_ppo_params),
                            ppo_policies=(agent_0_ppo_policy, agent_1_ppo_policy),
                            vf_params=(agent_0_vf_params, agent_1_vf_params),
                            vf_policies=(agent_0_vf_policy, agent_1_vf_policy),
                            max_episode_steps=max_episode_steps,
                            num_eps=config["NUM_EVAL_EPISODES"],
                            epsilon_optimal=config["EPSILON_OPTIMAL"],
                            use_full_obs=config["JOINT_USE_FULL_OBS"])

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
                return ((joint_train_state, rng, rng_eval, update_steps),
                         checkpoint_array, ckpt_idx, ckpt_eval_eps_last_infos, eval_eps_last_infos), metric

            checkpoint_array = init_ckpt_array(joint_train_state.params)
            ckpt_idx = 0

            rng, rng_train = jax.random.split(rng, 2)

            rng_eval = jax.random.PRNGKey(config["EVAL_SEED"])# + agent_idx)# + 42)
            rng_eval, eval_rng = jax.random.split(rng_eval, 2)

            # Init eval return infos
            eval_eps_last_infos = run_episodes_vmap(eval_rng, env, optimal_env,agent_idx,
                                    agent_params=joint_train_state.params,
                                    agent_policy=joint_policy,
                                    ppo_params=(agent_0_ppo_params, agent_1_ppo_params),
                                    ppo_policies=(agent_0_ppo_policy, agent_1_ppo_policy),
                                    vf_params=(agent_0_vf_params, agent_1_vf_params),
                                    vf_policies=(agent_0_vf_policy, agent_1_vf_policy),
                                    max_episode_steps=max_episode_steps,
                                    num_eps=config["NUM_EVAL_EPISODES"],
                                    epsilon_optimal=config["EPSILON_OPTIMAL"],
                                    use_full_obs=config["JOINT_USE_FULL_OBS"])

            # initial runner state for scanning
            update_steps = 0

            update_runner_state = (joint_train_state, rng_train, rng_eval, update_steps)
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
                joint_params = final_runner_state[0].params
                out["render_outs"] = run_episodes_vmap(eval_rng, env, optimal_env,agent_idx,
                                        agent_params=joint_params,
                                        agent_policy=joint_policy,
                                        ppo_params=(agent_0_ppo_params, agent_1_ppo_params),
                                        ppo_policies=(agent_0_ppo_policy, agent_1_ppo_policy),
                                        vf_params=(agent_0_vf_params, agent_1_vf_params),
                                        vf_policies=(agent_0_vf_policy, agent_1_vf_policy),
                                        max_episode_steps=env.horizon,
                                        num_eps=5, epsilon_optimal=config["EPSILON_OPTIMAL"],
                                        use_full_obs=config["JOINT_USE_FULL_OBS"],
                                        render=True)

            return out
        return train

    # ------------------------------
    # Actually run the PPO training
    # ------------------------------
    rngs = jax.random.split(train_rng, config["NUM_TRAIN_SEEDS"])

    # Run training seeds in parallel using vmap
    train_fn = make_ppo_joint_train(config)
    out = jax.vmap(train_fn, in_axes=(0, None, 0, 0, 0, 0))(rngs, agent_idx, agent_0_ppo_params, agent_1_ppo_params, agent_0_vf_params, agent_1_vf_params)
    return out

def run_training(config, wandb_logger, ppo_params, ppo_policies,
                 vf_params, vf_policies, agent_idx=0):
    '''Run joint training.

    Args:
        config: dict, config for the training
        wandb_logger: Logger, logger for logging metrics
        ppo_params: tuple, PPO single agent projection parameters for all agents
        ppo_policies: tuple, PPO single agent projection policies for all agents
        vf_params: tuple, value function estimation parameters for all agents
        vf_policies: tuple, value function estimation policies for all agents
        agent_idx: int, index of the agent to optimize
    '''
    algorithm_config = dict(config["algorithm"])

    # Create only one environment instance
    env_kwargs = algorithm_config["ENV_KWARGS"].copy()
    env_kwargs["render_dir"] = os.path.join("render", "joint", f"agent_{agent_idx + 1}_optimize")

    env = make_env(algorithm_config["ENV_NAME"], env_kwargs)
    env = LogWrapper(env)

    env_kwargs = algorithm_config["ENV_KWARGS"].copy()
    env_kwargs["render_dir"] = os.path.join("render", "joint", f"agent_{agent_idx + 1}_optimize")
    env_kwargs["instance"] = config['task'][f"SINGLE_AGENT_{agent_idx + 1}_PROJECTION"]
    optimal_env = make_env(algorithm_config["ENV_NAME"], env_kwargs)
    optimal_env = LogWrapper(optimal_env)

    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"])# + agent_idx)# + 35)
    _, init_rng, train_rng = jax.random.split(rng, 3)

    # Initialize agent
    agent_policy, agent_init_params = initialize_agent(algorithm_config, env, init_rng, agent_index=0, observation_type="full" if algorithm_config["JOINT_USE_FULL_OBS"] else "agent")

    # Squeeze PPO params to remove leading dimension for compatibility with single-agent training
    agent_0_ppo_params, agent_1_ppo_params = ppo_params
    agent_0_ppo_policy, agent_1_ppo_policy = ppo_policies
    # agent_0_ppo_params = jax.tree.map(lambda x: x.squeeze(axis=0), agent_0_ppo_params)
    # agent_1_ppo_params = jax.tree.map(lambda x: x.squeeze(axis=0), agent_1_ppo_params)

    agent_0_vf_params, agent_1_vf_params = vf_params
    agent_0_vf_policy, agent_1_vf_policy = vf_policies
    # agent_0_vf_params = jax.tree.map(lambda x: x.squeeze(axis=0), agent_0_vf_params)
    # agent_1_vf_params = jax.tree.map(lambda x: x.squeeze(axis=0), agent_1_vf_params)

    log.info(f"Starting PPO joint training optimizing for agent {agent_idx}...")
    start_time = time.perf_counter()

    # Run the training
    out = train_ppo_joint_agents(
        config=algorithm_config,
        env=env,
        optimal_env=optimal_env,
        train_rng=train_rng,
        joint_policy=agent_policy,
        init_joint_params=agent_init_params,
        ppo_policies=(agent_0_ppo_policy, agent_1_ppo_policy),
        ppo_params=(agent_0_ppo_params, agent_1_ppo_params),
        vf_policies=(agent_0_vf_policy, agent_1_vf_policy),
        vf_params=(agent_0_vf_params, agent_1_vf_params),
        agent_idx=agent_idx
    )

    elapsed_time = time.perf_counter() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, rem = divmod(rem, 60)
    seconds, rem = divmod(rem, 1)
    milliseconds = int(rem * 1000)
    microseconds = int((rem * 1_000_000) % 1000)
    log.info(f"PPO Joint Training completed optimizing for agent {agent_idx} in {elapsed_time:.2f}s")
    log.info(f"PPO Joint Training completed optimizing for agent {agent_idx} in {int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s {milliseconds:03d}ms {microseconds:03d}µs")

    # process and log metrics
    log.info(f"Starting PPO joint logging optimizing for agent {agent_idx}...")
    start_time = time.perf_counter()
    # metric_names = get_metric_names(config["ENV_NAME"])
    metric_names = get_metric_names("social_laws_joint")
    log_metrics(env, optimal_env, config, out, wandb_logger, metric_names, agent_idx)
    elapsed_time = time.perf_counter() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, rem = divmod(rem, 60)
    seconds, rem = divmod(rem, 1)
    milliseconds = int(rem * 1000)
    microseconds = int((rem * 1_000_000) % 1000)
    log.info(f"PPO Joint Logging completed optimizing for agent {agent_idx} in {elapsed_time:.2f}s")
    log.info(f"PPO Joint Logging completed optimizing for agent {agent_idx} in {int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s {milliseconds:03d}ms {microseconds:03d}µs")

    return out["final_params"], agent_policy, agent_init_params

def log_metrics(env, optimal_env, config, train_out, logger, metric_names: tuple, agent_idx: int):
    """Process training metrics and log them using the provided logger.

    Args:
        env: the environment used for training, needed for logging videos
        optimal_env: the optimal environment used for training, needed for logging videos
        config: dict, the training configuration
        train_out: dict, the logs from training
        logger: Logger, instance to log metrics
        metric_names: tuple, names of metrics to extract from training logs
        agent_idx: int, index of the trained agent
    """
    train_metrics = train_out["metrics"]

    # Add additional metrics for logging
    train_metrics["returned_episode_minimized_returns"] = train_metrics["returned_episode_returns"].clone() * -1

    #### Extract train metrics ####
    train_stats = get_stats(train_metrics, metric_names)
    # each key in train_stats is a metric name, and the value is an array of shape (num_seeds, num_updates, 2)
    # where the last dimension contains the mean and std of the metric
    train_stats = {k: np.mean(np.array(v), axis=0) for k, v in train_stats.items()}

    # Train metrics include loss values and gradient norms, which we can average across seeds and minibatches for each update step.
    all_value_losses = np.asarray(train_metrics["value_loss"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    all_actor_losses = np.asarray(train_metrics["actor_loss"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    all_entropy_losses = np.asarray(train_metrics["entropy_loss"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    all_grad_norms = np.asarray(train_metrics["avg_grad_norm"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)

    # Process loss metrics - average across train seeds and minibatches dims
    average_value_losses = np.mean(all_value_losses, axis=(0, 2, 3))  # shape (num_updates,)
    average_actor_losses = np.mean(all_actor_losses, axis=(0, 2, 3)) # shape (num_updates,)
    average_entropy_losses = np.mean(all_entropy_losses, axis=(0, 2, 3)) # shape (num_updates,)
    average_grad_norms = np.mean(all_grad_norms, axis=(0, 2, 3)) # shape (num_updates,)

    # Eval metrics include worst-case returns and optimal returns, which we can use to compute alpha-returns.
    # Process these metrics by averaging across train seeds and eval episodes for each checkpoint, then
    # compute alpha-returns using the formula: alpha_return = worst_case_return / optimal_return.
    all_ckpt_worst_case_returns = np.asarray(train_metrics["ckpt_eval_ep_last_info"][0]["returned_episode_returns"]) # shape (n_train_seeds, num_updates, num_eval_episodes, num_agents_per_game)
    all_worst_case_returns = np.asarray(train_metrics["eval_ep_last_info"][0]["returned_episode_returns"]) # shape (n_train_seeds, num_updates, num_eval_episodes, num_agents_per_game)
    all_ckpt_worst_case_returns = all_ckpt_worst_case_returns[:, :, :, agent_idx] # shape (n_train_seeds, num_updates, num_eval_episodes)
    all_worst_case_returns = all_worst_case_returns[:, :, :, agent_idx] # shape (n_train_seeds, num_updates, num_eval_episodes)
    all_ckpt_optimal_returns = np.asarray(train_metrics["ckpt_eval_ep_last_info"][1]["returned_episode_returns"]) # shape (n_train_seeds, num_updates, num_eval_episodes, num_agents_per_game)
    all_optimal_returns = np.asarray(train_metrics["eval_ep_last_info"][1]["returned_episode_returns"]) # shape (n_train_seeds, num_updates, num_eval_episodes, num_agents_per_game)
    all_ckpt_optimal_returns = all_ckpt_optimal_returns[:, :, :, agent_idx] # shape (n_train_seeds, num_updates, num_eval_episodes)
    all_optimal_returns = all_optimal_returns[:, :, :, agent_idx] # shape (n_train_seeds, num_updates, num_eval_episodes)
    all_ckpt_alpha_returns = all_ckpt_worst_case_returns / all_ckpt_optimal_returns # shape (n_train_seeds, num_updates, num_eval_episodes)
    all_alpha_returns = all_worst_case_returns / all_optimal_returns # shape (n_train_seeds, num_updates, num_eval_episodes)
    average_ckpt_worst_case_rets_per_iter = np.mean(all_ckpt_worst_case_returns, axis=(0, 2)) # shape (num_updates,)
    average_agent_worst_case_rets_per_iter = np.mean(all_worst_case_returns, axis=(0, 2)) # shape (num_updates,)
    average_ckpt_optimal_rets_per_iter = np.mean(all_ckpt_optimal_returns, axis=(0, 2)) # shape (num_updates,)
    average_agent_optimal_rets_per_iter = np.mean(all_optimal_returns, axis=(0, 2)) # shape (num_updates,)
    average_ckpt_alpha_rets_per_iter = np.mean(all_ckpt_alpha_returns, axis=(0, 2)) # shape (num_updates,)
    average_agent_alpha_rets_per_iter = np.mean(all_alpha_returns, axis=(0, 2)) # shape (num_updates,)

    # Log metrics for each update step
    num_updates = len(average_value_losses)
    for step in range(num_updates):
        for stat_name, stat_data in train_stats.items():
            # second dimension contains the mean and std of the metric
            stat_mean = stat_data[step, 0]
            logger.log_item(f"Train/Joint/Agent_{agent_idx + 1}_Optimize/{stat_name}", stat_mean, train_step=step, commit=True)

        logger.log_item(f"Eval/Joint/Agent_{agent_idx + 1}_Optimize/WorstCaseReturn", average_agent_worst_case_rets_per_iter[step], train_step=step, commit=True)
        logger.log_item(f"Eval/Joint/Agent_{agent_idx + 1}_Optimize/CheckpointWorstCaseReturn", average_ckpt_worst_case_rets_per_iter[step], train_step=step, commit=True)
        logger.log_item(f"Eval/Joint/Agent_{agent_idx + 1}_Optimize/OptimalReturn", average_agent_optimal_rets_per_iter[step], train_step=step, commit=True)
        logger.log_item(f"Eval/Joint/Agent_{agent_idx + 1}_Optimize/CheckpointOptimalReturn", average_ckpt_optimal_rets_per_iter[step], train_step=step, commit=True)
        logger.log_item(f"Eval/Joint/Agent_{agent_idx + 1}_Optimize/AlphaReturn", average_agent_alpha_rets_per_iter[step], train_step=step, commit=True)
        logger.log_item(f"Eval/Joint/Agent_{agent_idx + 1}_Optimize/CheckpointAlphaReturn", average_ckpt_alpha_rets_per_iter[step], train_step=step, commit=True)

        logger.log_item(f"Train/Joint/Agent_{agent_idx + 1}_Optimize/ValueLoss", average_value_losses[step], train_step=step, commit=True)
        logger.log_item(f"Train/Joint/Agent_{agent_idx + 1}_Optimize/ActorLoss", average_actor_losses[step], train_step=step, commit=True)
        logger.log_item(f"Train/Joint/Agent_{agent_idx + 1}_Optimize/EntropyLoss", average_entropy_losses[step], train_step=step, commit=True)
        logger.log_item(f"Train/Joint/Agent_{agent_idx + 1}_Optimize/GradNorm", average_grad_norms[step], train_step=step, commit=True)
        logger.commit()

    # Saving artifacts
    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    if env._render:
        # shape of render_outs should be (num_train_seeds, num_eps, max_episode_steps, ...)
        eval_render_init_env_state = train_out['render_outs'][2].env_state.env_state # LogEnvState
        eval_render_optimal_env_state = train_out['render_outs'][1][-1]['pre_reset_state'].env_state # WrappedEnvState
        eval_render_optimal_dones = train_out['render_outs'][1][4]['__all__']
        eval_render_worst_case_env_state = train_out['render_outs'][0][-1]['pre_reset_state'].env_state # WrappedEnvState
        eval_render_worst_case_dones = train_out['render_outs'][0][4]['__all__']
        num_episodes = eval_render_worst_case_env_state.state['agent-at'].shape[1] # (num_train_seeds, num_eval_episodes, num_max_timesteps, num_agents_per_game, ...)
        optimal_env.animate((eval_render_init_env_state, eval_render_optimal_env_state), eval_render_optimal_dones, num_episodes, extra_dir="Optimal", debug=True)
        env.animate((eval_render_init_env_state, eval_render_worst_case_env_state), eval_render_worst_case_dones, num_episodes, extra_dir="WorstCase", debug=True)

        for eval_ep in range(num_episodes):
            logger.log_video(
                tag=f"Videos/Joint/Agent_{agent_idx + 1}_Optimize/WorstCase/Episode_{eval_ep}",
                path=os.path.join(env._render_dir, "WorstCase", f"{env._render_name}_ep_{eval_ep}.gif")
            )
            logger.log_video(
                tag=f"Videos/Joint/Agent_{agent_idx + 1}_Optimize/Optimal/Episode_{eval_ep}",
                path=os.path.join(optimal_env._render_dir, "Optimal", f"{optimal_env._render_name}_ep_{eval_ep}.gif")
            )

    out_savepath = save_train_run(train_out, savedir, savename=f"PPO_Joint_Agent_{agent_idx + 1}_Optimize_Train_Run")
    if config["logger"]["log_train_out"]:
        logger.log_artifact(name=f"PPO_Joint_Agent_{agent_idx + 1}_Optimize_Train_Run", path=out_savepath, type_name="joint_train_run")
        # Cleanup locally logged out file
    if not config["local_logger"]["save_train_out"]:
        shutil.rmtree(out_savepath)

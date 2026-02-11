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

from social_laws.common.initialize_agents import initialize_agent
from social_laws.common.run_episodes_w_robustness import run_episodes, run_render_episodes
from common.plot_utils import get_stats, get_metric_names
from common.save_load_utils import save_train_run
from envs import make_env
from envs.log_wrapper import LogWrapper
from marl.ppo_utils import _create_minibatches, Transition, unbatchify

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def train_ppo_joint_agents(config, env, train_rng,
                           joint_policies, init_joint_params,
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
    agent_0_policy, agent_1_policy = joint_policies
    agent_0_init_params, agent_1_init_params = init_joint_params
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

        def train(rng, agent_idx):
            if config["ANNEAL_LR"]:
                tx_0 = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(learning_rate=linear_schedule, eps=1e-5),
                )
                tx_1 = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(learning_rate=linear_schedule, eps=1e-5),
                )
            else:
                tx_0 = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(config["LR"], eps=1e-5),
                )
                tx_1 = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(config["LR"], eps=1e-5),
                )

            agent_0_train_state = TrainState.create(
                apply_fn=agent_0_policy.network.apply,
                params=agent_0_init_params,
                tx=tx_0,
            )
            agent_1_train_state = TrainState.create(
                apply_fn=agent_1_policy.network.apply,
                params=agent_1_init_params,
                tx=tx_1,
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
            agent_0_init_hstate = agent_0_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])
            agent_1_init_hstate = agent_1_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])

            def _get_vf_restricted_avail_actions(obs, avail_actions, vf_params, vf_policy):
                """
                Get restricted available actions based on value function Q-values.
                Only allows actions within epsilon of the maximum Q-value.

                Args:
                    obs: observations (shape: num_envs, obs_dim)
                    avail_actions: available actions mask (shape: num_envs, num_actions)
                    vf_params: value function parameters
                    vf_policy: value function policy

                Returns:
                    restricted_avail_actions: mask with only near-optimal actions (shape: num_envs, num_actions)
                """
                # Get Q-values from value function
                q_values = vf_policy.network.apply(vf_params, obs)  # (num_envs, num_actions)

                # Mask Q-values with available actions (set unavailable to -inf)
                q_values_masked = jnp.where(avail_actions, q_values, -jnp.inf)

                # Find max Q-value for each environment
                max_q = jnp.max(q_values_masked, axis=-1, keepdims=True)  # (num_envs, 1)

                # Create mask for actions within epsilon of max
                optimal_action_mask = (q_values_masked >= (max_q - config['EPSILON_OPTIMAL'])).astype(jnp.float32)

                # Combine with original available actions
                restricted_avail_actions = optimal_action_mask * avail_actions

                return restricted_avail_actions.squeeze(axis=0)  # remove extra batch dim

            def _env_step(runner_state, unused):
                """
                One step of the environment:
                1. Get observations, sample actions from all agents
                2. Step environment using sampled actions
                3. Return state, reward, ...
                """
                agent_0_train_state, agent_1_train_state, env_state, prev_obs, prev_done, hstate_0, hstate_1, rng = runner_state
                prev_obs, prev_full_obs = prev_obs
                rng, actor_0_rng, actor_1_rng, step_rng = jax.random.split(rng, 4)

                 # Get available actions for the agent from environment state
                avail_actions = env.get_avail_actions(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(avail_actions)
                avail_actions_0 = get_agent_data(avail_actions, 0).astype(jnp.float32)
                avail_actions_1 = get_agent_data(avail_actions, 1).astype(jnp.float32)

                prev_obs_0 = get_agent_data(prev_obs, 0).reshape(1, config["NUM_CONTROLLED_ACTORS"], -1)
                prev_obs_1 = get_agent_data(prev_obs, 1).reshape(1, config["NUM_CONTROLLED_ACTORS"], -1)
                prev_full_obs_0 = get_agent_data(prev_full_obs, 0).reshape(1, config["NUM_CONTROLLED_ACTORS"], -1)
                prev_full_obs_1 = get_agent_data(prev_full_obs, 1).reshape(1, config["NUM_CONTROLLED_ACTORS"], -1)

                # Restrict available actions based on value function
                vf_restricted_avail_actions_0 = _get_vf_restricted_avail_actions(
                    obs=prev_obs_0,
                    avail_actions=avail_actions_0,
                    vf_params=agent_0_vf_params,
                    vf_policy=agent_0_vf_policy
                )
                vf_restricted_avail_actions_1 = _get_vf_restricted_avail_actions(
                    obs=prev_obs_1,
                    avail_actions=avail_actions_1,
                    vf_params=agent_1_vf_params,
                    vf_policy=agent_1_vf_policy
                )

                # Note that we do not need to reset the hidden states for the agents
                # as the recurrent states are automatically reset when done is True.

                # Agent 0 action, value, log_prob (using VF-restricted available actions)
                act_0, val_0, pi_0, new_hstate_0 = agent_0_policy.get_action_value_policy(
                    params=agent_0_train_state.params,
                    obs=prev_full_obs_0,
                    done=get_agent_data(prev_done, 0).reshape(1, config["NUM_CONTROLLED_ACTORS"]),
                    avail_actions=vf_restricted_avail_actions_0,
                    hstate=hstate_0,
                    rng=actor_0_rng
                )
                logp_0 = pi_0.log_prob(act_0)

                act_0 = act_0.squeeze(axis=0)
                logp_0 = logp_0.squeeze(axis=0)
                val_0 = val_0.squeeze(axis=0)

                # Agent 1 action, value, log_prob (using VF-restricted available actions)
                act_1, val_1, pi_1, new_hstate_1 = agent_1_policy.get_action_value_policy(
                    params=agent_1_train_state.params,
                    obs=prev_full_obs_1,
                    done=get_agent_data(prev_done, 1).reshape(1, config["NUM_CONTROLLED_ACTORS"]),
                    avail_actions=vf_restricted_avail_actions_1,
                    hstate=hstate_1,
                    rng=actor_1_rng
                )
                logp_1 = pi_1.log_prob(act_1)

                act_1 = act_1.squeeze(axis=0)
                logp_1 = logp_1.squeeze(axis=0)
                val_1 = val_1.squeeze(axis=0)

                # Combine actions into the env format
                combined_actions = jnp.concatenate([act_0, act_1], axis=0)  # shape (2*num_envs,)
                env_act = unbatchify(combined_actions, env.agents, config["NUM_ENVS"], num_agents)
                env_act = {k: v.flatten() for k, v in env_act.items()}

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

                # Store agent_idx data in transition (using VF-restricted available actions)
                transition_0 = Transition(
                    done=get_agent_data(done_next, 0), # shape (num_envs,)
                    action=act_0, # shape (num_envs,)
                    value=val_0, # shape (num_envs,)
                    reward=get_agent_data(reward, agent_idx) * -1, # shape (num_envs,)
                    log_prob=logp_0, # shape (num_envs,)
                    obs=get_agent_data(prev_full_obs, 0), # shape (num_envs, obs_dim)
                    info=info,
                    avail_actions=vf_restricted_avail_actions_0 # shape (num_envs, num_actions)
                )

                transition_1 = Transition(
                    done=get_agent_data(done_next, 1), # shape (num_envs,)
                    action=act_1, # shape (num_envs,)
                    value=val_1, # shape (num_envs,)
                    reward=get_agent_data(reward, agent_idx) * -1, # shape (num_envs,)
                    log_prob=logp_1, # shape (num_envs,)
                    obs=get_agent_data(prev_full_obs, 1), # shape (num_envs, obs_dim)
                    info=info,
                    avail_actions=vf_restricted_avail_actions_1, # shape (num_envs, num_actions)
                )

                new_runner_state = (agent_0_train_state, agent_1_train_state, env_state_next, obs_next, done_next,
                                    new_hstate_0, new_hstate_1, rng)
                return new_runner_state, (transition_0, transition_1)

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

            def _update_minbatch_0(train_state, batch_info):
                init_hstate, traj_batch, advantages, returns = batch_info
                def _loss_fn(params, init_hstate, traj_batch, gae, target_v):
                    _, value, pi, _ = agent_0_policy.get_action_value_policy(
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

            def _update_minbatch_1(train_state, batch_info):
                init_hstate, traj_batch, advantages, returns = batch_info
                def _loss_fn(params, init_hstate, traj_batch, gae, target_v):
                    _, value, pi, _ = agent_1_policy.get_action_value_policy(
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

            def _update_epoch_0(update_state, unused):
                train_state, init_hstate, traj_batch, advantages, targets, rng = update_state
                rng, perm_rng = jax.random.split(rng)
                minibatches = _create_minibatches(traj_batch, advantages, targets, init_hstate, config["NUM_CONTROLLED_ACTORS"], config["NUM_MINIBATCHES"], perm_rng)
                train_state, losses_and_grads = jax.lax.scan(
                    _update_minbatch_0, train_state, minibatches
                )
                update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
                return update_state, losses_and_grads

            def _update_epoch_1(update_state, unused):
                train_state, init_hstate, traj_batch, advantages, targets, rng = update_state
                rng, perm_rng = jax.random.split(rng)
                minibatches = _create_minibatches(traj_batch, advantages, targets, init_hstate, config["NUM_CONTROLLED_ACTORS"], config["NUM_MINIBATCHES"], perm_rng)
                train_state, losses_and_grads = jax.lax.scan(
                    _update_minbatch_1, train_state, minibatches
                )
                update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
                return update_state, losses_and_grads

            def _update_step(update_runner_state, unused):
                """
                1. Collect rollouts
                2. Compute advantage
                3. PPO updates
                """
                (agent_0_train_state, agent_1_train_state, rng, rng_eval, update_steps) = update_runner_state
                # Init envs & partner indices
                rng, reset_rng = jax.random.split(rng, 2)
                reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
                init_obs, init_env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rngs)
                init_done = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}

                # 1) rollout
                runner_state = (agent_0_train_state, agent_1_train_state, init_env_state, init_obs, init_done, agent_0_init_hstate, agent_1_init_hstate, rng)

                runner_state, traj_batch = jax.lax.scan(
                    _env_step, runner_state, None, config["ROLLOUT_LENGTH"])
                (agent_0_train_state, agent_1_train_state, env_state, obs, done, agent_0_hstate, agent_1_hstate, rng) = runner_state
                agent_0_traj_batch, agent_1_traj_batch = traj_batch
                obs, obs_full = obs

                # 2) advantage
                # Get available actions for agent 0 from environment state
                avail_actions = env.get_avail_actions(env_state.env_state)
                avail_actions_0 = get_agent_data(avail_actions, 0).astype(jnp.float32)
                avail_actions_1 = get_agent_data(avail_actions, 1).astype(jnp.float32)

                obs_0 = get_agent_data(obs, 0).reshape(1, config["NUM_CONTROLLED_ACTORS"], -1)
                obs_1 = get_agent_data(obs, 1).reshape(1, config["NUM_CONTROLLED_ACTORS"], -1)
                obs_full_0 = get_agent_data(obs_full, 0).reshape(1, config["NUM_CONTROLLED_ACTORS"], -1)
                obs_full_1 = get_agent_data(obs_full, 1).reshape(1, config["NUM_CONTROLLED_ACTORS"], -1)

                # Restrict available actions based on value function
                vf_restricted_avail_actions_0 = _get_vf_restricted_avail_actions(
                    obs=obs_0,
                    avail_actions=avail_actions_0,
                    vf_params=agent_0_vf_params,
                    vf_policy=agent_0_vf_policy
                )
                vf_restricted_avail_actions_1 = _get_vf_restricted_avail_actions(
                    obs=obs_1,
                    avail_actions=avail_actions_1,
                    vf_params=agent_1_vf_params,
                    vf_policy=agent_1_vf_policy
                )

                # Get agent 0 final value estimate for completed trajectory
                _, last_val_0, _, _ = agent_0_policy.get_action_value_policy(
                    params=agent_0_train_state.params,
                    obs=obs_full_0,
                    done=get_agent_data(done, 0).reshape(1, config["NUM_CONTROLLED_ACTORS"]),
                    avail_actions=jax.lax.stop_gradient(vf_restricted_avail_actions_0),
                    hstate=agent_0_hstate,
                    rng=jax.random.PRNGKey(0)  # Dummy key since we're just extracting the value
                )
                last_val_0 = last_val_0.squeeze(axis=0)

                # Get agent 1 final value estimate for completed trajectory
                _, last_val_1, _, _ = agent_1_policy.get_action_value_policy(
                    params=agent_1_train_state.params,
                    obs=obs_full_1,
                    done=get_agent_data(done, 1).reshape(1, config["NUM_CONTROLLED_ACTORS"]),
                    avail_actions=jax.lax.stop_gradient(vf_restricted_avail_actions_1),
                    hstate=agent_1_hstate,
                    rng=jax.random.PRNGKey(0)  # Dummy key since we're just extracting the value
                )
                last_val_1 = last_val_1.squeeze(axis=0)

                # Calculate GAE for each agent separately
                advantages_0, targets_0 = _calculate_gae(agent_0_traj_batch, last_val_0)
                advantages_1, targets_1 = _calculate_gae(agent_1_traj_batch, last_val_1)

                # 3) PPO update for each agent separately
                rng, rng_0, rng_1 = jax.random.split(rng, 3)

                # Update agent 0
                update_state_0 = (
                    agent_0_train_state,
                    agent_0_init_hstate,
                    agent_0_traj_batch,
                    advantages_0,
                    targets_0,
                    rng_0
                )
                update_state_0, losses_and_grads_0 = jax.lax.scan(
                    _update_epoch_0, update_state_0, None, config["UPDATE_EPOCHS"])
                agent_0_train_state = update_state_0[0]
                _, loss_terms_0, avg_grad_norm_0 = losses_and_grads_0

                # Update agent 1
                update_state_1 = (
                    agent_1_train_state,
                    agent_1_init_hstate,
                    agent_1_traj_batch,
                    advantages_1,
                    targets_1,
                    rng_1
                )
                update_state_1, losses_and_grads_1 = jax.lax.scan(
                    _update_epoch_1, update_state_1, None, config["UPDATE_EPOCHS"])
                agent_1_train_state = update_state_1[0]
                _, loss_terms_1, avg_grad_norm_1 = losses_and_grads_1

                # Average metrics across both agents
                metric = agent_0_traj_batch.info
                metric["update_steps"] = update_steps
                metric["agent_0/actor_loss"] = loss_terms_0[1]
                metric["agent_1/actor_loss"] = loss_terms_1[1]
                metric["agent_0/value_loss"] = loss_terms_0[0]
                metric["agent_1/value_loss"] = loss_terms_1[0]
                metric["agent_0/entropy_loss"] = loss_terms_0[2]
                metric["agent_1/entropy_loss"] = loss_terms_1[2]
                metric["agent_0/avg_grad_norm"] = avg_grad_norm_0
                metric["agent_1/avg_grad_norm"] = avg_grad_norm_1
                new_runner_state = (agent_0_train_state, agent_1_train_state, rng, rng_eval, update_steps + 1)
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
                (update_state, checkpoint_array_0, ckpt_idx_0, checkpoint_array_1, ckpt_idx_1, init_eval_last_info) = state_with_ckpt

                # Single PPO update
                new_update_state, metric = _update_step(
                    update_state,
                    None
                )
                (agent_0_train_state, agent_1_train_state, rng, rng_eval, update_steps) = new_update_state

                # update steps is 1-indexed because it was incremented at the end of the update step
                to_store = jnp.logical_or(jnp.equal(jnp.mod(update_steps-1, ckpt_and_eval_interval), 0),
                                        jnp.equal(update_steps, config["NUM_UPDATES"]))


                def store_and_eval_ckpt(args):
                    ckpt_arr_0, cidx_0, ckpt_arr_1, cidx_1, rng, rng_eval, prev_eval_ret_info = args
                    new_ckpt_arr_0 = jax.tree.map(
                        lambda c_arr, p: c_arr.at[cidx_0].set(p),
                        ckpt_arr_0, agent_0_train_state.params
                    )
                    new_ckpt_arr_1 = jax.tree.map(
                        lambda c_arr, p: c_arr.at[cidx_1].set(p),
                        ckpt_arr_1, agent_1_train_state.params
                    )

                    rng_eval, eval_rng = jax.random.split(rng_eval, 2)
                    eval_eps_last_infos = run_episodes(eval_rng, env, agent_idx,
                        agent_params=(agent_0_train_state.params, agent_1_train_state.params),
                        agent_policies=(agent_0_policy, agent_1_policy),
                        ppo_params=(agent_0_ppo_params, agent_1_ppo_params),
                        ppo_policies=(agent_0_ppo_policy, agent_1_ppo_policy),
                        vf_params=(agent_0_vf_params, agent_1_vf_params),
                        vf_policies=(agent_0_vf_policy, agent_1_vf_policy),
                        max_episode_steps=max_episode_steps,
                        num_eps=config["NUM_EVAL_EPISODES"],
                        epsilon_optimal=config["EPSILON_OPTIMAL"])

                    return (new_ckpt_arr_0, cidx_0 + 1, new_ckpt_arr_1, cidx_1 + 1, rng, rng_eval, eval_eps_last_infos)

                def skip_ckpt(args):
                    return args

                (checkpoint_array_0, ckpt_idx_0, checkpoint_array_1, ckpt_idx_1, rng, rng_eval, eval_last_infos) = jax.lax.cond(
                    to_store, store_and_eval_ckpt, skip_ckpt, (checkpoint_array_0, ckpt_idx_0, checkpoint_array_1, ckpt_idx_1, rng, rng_eval, init_eval_last_info)
                )

                metric["eval_ep_last_info"] = eval_last_infos
                return ((agent_0_train_state, agent_1_train_state, rng, rng_eval, update_steps),
                         checkpoint_array_0, ckpt_idx_0, checkpoint_array_1, ckpt_idx_1, eval_last_infos), metric

            checkpoint_array_0 = init_ckpt_array(agent_0_train_state.params)
            checkpoint_array_1 = init_ckpt_array(agent_1_train_state.params)
            ckpt_idx_0 = 0
            ckpt_idx_1 = 0

            rng, rng_train = jax.random.split(rng, 2)

            rng_eval = jax.random.PRNGKey(config["EVAL_SEED"] + agent_idx + 42)
            rng_eval, eval_rng = jax.random.split(rng_eval, 2)

            # Init eval return infos
            eval_eps_last_infos = run_episodes(eval_rng, env, agent_idx,
                                    agent_params=(agent_0_train_state.params, agent_1_train_state.params),
                                    agent_policies=(agent_0_policy, agent_1_policy),
                                    ppo_params=(agent_0_ppo_params, agent_1_ppo_params),
                                    ppo_policies=(agent_0_ppo_policy, agent_1_ppo_policy),
                                    vf_params=(agent_0_vf_params, agent_1_vf_params),
                                    vf_policies=(agent_0_vf_policy, agent_1_vf_policy),
                                    max_episode_steps=max_episode_steps,
                                    num_eps=config["NUM_EVAL_EPISODES"],
                                    epsilon_optimal=config["EPSILON_OPTIMAL"])

            # initial runner state for scanning
            update_steps = 0

            update_runner_state = (agent_0_train_state, agent_1_train_state, rng_train, rng_eval, update_steps)
            state_with_ckpt = (update_runner_state, checkpoint_array_0, ckpt_idx_0, checkpoint_array_1, ckpt_idx_1, eval_eps_last_infos)

            state_with_ckpt, metrics = jax.lax.scan(
                _update_step_with_ckpt,
                state_with_ckpt,
                xs=None,
                length=config["NUM_UPDATES"]
            )

            (final_runner_state, checkpoint_array_0, final_ckpt_idx_0, checkpoint_array_1, final_ckpt_idx_1, eval_eps_last_infos) = state_with_ckpt
            out = {
                "final_params_agent_0": final_runner_state[0].params,
                "final_params_agent_1": final_runner_state[1].params,
                "metrics": metrics,  # shape (NUM_UPDATES, ...)
                "checkpoints_agent_0": checkpoint_array_0,
                "checkpoints_agent_1": checkpoint_array_1,
            }

            # Collect final eval gifs for logging
            rng_eval = final_runner_state[3] # extract final rng_eval from the final runner state after training
            rng_eval, eval_rng = jax.random.split(rng_eval, 2)
            agent_0_params = final_runner_state[0].params
            agent_1_params = final_runner_state[1].params
            out["render_outs"] = run_render_episodes(eval_rng, env, agent_idx,
                                    agent_params=(agent_0_params, agent_1_params),
                                    agent_policies=(agent_0_policy, agent_1_policy),
                                    ppo_params=(agent_0_ppo_params, agent_1_ppo_params),
                                    ppo_policies=(agent_0_ppo_policy, agent_1_ppo_policy),
                                    vf_params=(agent_0_vf_params, agent_1_vf_params),
                                    vf_policies=(agent_0_vf_policy, agent_1_vf_policy),
                                    max_episode_steps=env.env.horizon,
                                    num_eps=5, epsilon_optimal=config["EPSILON_OPTIMAL"])

            return out
        return train

    # ------------------------------
    # Actually run the PPO training
    # ------------------------------
    rngs = jax.random.split(train_rng, 1)
    agent_idx_arr = jnp.array([agent_idx] * 1)

    # Define scan function to run training seeds sequentially
    train_fn = make_ppo_joint_train(config)
    def scan_train(carry, inputs):
        rng, agent_idx = inputs
        result = train_fn(rng, agent_idx)
        return carry, result

    # Run training seeds sequentially using scan
    _, out = jax.lax.scan(scan_train, None, (rngs, agent_idx_arr))
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

    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"] + agent_idx + 35)
    _, init_rng, train_rng = jax.random.split(rng, 3)

    # Initialize agent
    agent_0_policy, agent_0_init_params = initialize_agent(algorithm_config, env, init_rng, agent_index=0, observation_type="full")
    agent_1_policy, agent_1_init_params = initialize_agent(algorithm_config, env, init_rng, agent_index=1, observation_type="full")

    # Squeeze PPO params to remove leading dimension for compatibility with single-agent training
    agent_0_ppo_params, agent_1_ppo_params = ppo_params
    agent_0_ppo_policy, agent_1_ppo_policy = ppo_policies
    agent_0_ppo_params = jax.tree.map(lambda x: x.squeeze(axis=0), agent_0_ppo_params)
    agent_1_ppo_params = jax.tree.map(lambda x: x.squeeze(axis=0), agent_1_ppo_params)

    agent_0_vf_params, agent_1_vf_params = vf_params
    agent_0_vf_policy, agent_1_vf_policy = vf_policies
    agent_0_vf_params = jax.tree.map(lambda x: x.squeeze(axis=0), agent_0_vf_params)
    agent_1_vf_params = jax.tree.map(lambda x: x.squeeze(axis=0), agent_1_vf_params)

    log.info(f"Starting PPO joint training optimizing for agent {agent_idx}...")
    start_time = time.perf_counter()

    # Run the training
    out = train_ppo_joint_agents(
        config=algorithm_config,
        env=env,
        train_rng=train_rng,
        joint_policies=(agent_0_policy, agent_1_policy),
        init_joint_params=(agent_0_init_params, agent_1_init_params),
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
    # metric_names = get_metric_names(config["ENV_NAME"])
    metric_names = get_metric_names("social_laws_joint")
    log_metrics(env, config, out, wandb_logger, metric_names, agent_idx)

    return (out["final_params_agent_0"], out["final_params_agent_1"]), (agent_0_policy, agent_1_policy), (agent_0_init_params, agent_1_init_params)

def log_metrics(env, config, train_out, logger, metric_names: tuple, agent_idx: int):
    """Process training metrics and log them using the provided logger.

    Args:
        env: the environment used for training, needed for logging videos
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

    # Train metrics include loss values and gradient norms, which we can average across seeds, partners and minibatches for each update step.
    all_agent_0_value_losses = np.asarray(train_metrics["agent_0/value_loss"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    all_agent_0_actor_losses = np.asarray(train_metrics["agent_0/actor_loss"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    all_agent_0_entropy_losses = np.asarray(train_metrics["agent_0/entropy_loss"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    all_agent_0_grad_norms = np.asarray(train_metrics["agent_0/avg_grad_norm"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    all_agent_1_value_losses = np.asarray(train_metrics["agent_1/value_loss"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    all_agent_1_actor_losses = np.asarray(train_metrics["agent_1/actor_loss"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    all_agent_1_entropy_losses = np.asarray(train_metrics["agent_1/entropy_loss"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    all_agent_1_grad_norms = np.asarray(train_metrics["agent_1/avg_grad_norm"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)

    # Process loss metrics - average across train seeds, partners and minibatches dims
    average_agent_0_value_losses = np.mean(all_agent_0_value_losses, axis=(0, 2, 3))  # shape (num_updates,)
    average_agent_0_actor_losses = np.mean(all_agent_0_actor_losses, axis=(0, 2, 3)) # shape (num_updates,)
    average_agent_0_entropy_losses = np.mean(all_agent_0_entropy_losses, axis=(0, 2, 3)) # shape (num_updates,)
    average_agent_0_grad_norms = np.mean(all_agent_0_grad_norms, axis=(0, 2, 3)) # shape (num_updates,)
    average_agent_1_value_losses = np.mean(all_agent_1_value_losses, axis=(0, 2, 3))  # shape (num_updates,)
    average_agent_1_actor_losses = np.mean(all_agent_1_actor_losses, axis=(0, 2, 3)) # shape (num_updates,)
    average_agent_1_entropy_losses = np.mean(all_agent_1_entropy_losses, axis=(0, 2, 3)) # shape (num_updates,)
    average_agent_1_grad_norms = np.mean(all_agent_1_grad_norms, axis=(0, 2, 3)) # shape (num_updates,)

    # Eval metrics include worst-case returns and optimal returns, which we can use to compute alpha-returns.
    # Process these metrics by averaging across train seeds and eval episodes for each checkpoint, then
    # compute alpha-returns using the formula: alpha_return = worst_case_return / optimal_return.
    all_worst_case_returns = np.asarray(train_metrics["eval_ep_last_info"][0]["returned_episode_returns"]) # shape (n_train_seeds, num_updates, num_eval_episodes, num_agents_per_game)
    all_worst_case_returns = all_worst_case_returns[:, :, :, agent_idx] # shape (n_train_seeds, num_updates, num_eval_episodes)
    all_optimal_returns = np.asarray(train_metrics["eval_ep_last_info"][1]["returned_episode_returns"]) # shape (n_train_seeds, num_updates, num_eval_episodes, num_agents_per_game)
    all_optimal_returns = all_optimal_returns[:, :, :, agent_idx] # shape (n_train_seeds, num_updates, num_eval_episodes)
    all_alpha_returns = all_worst_case_returns / all_optimal_returns # shape (n_train_seeds, num_updates, num_eval_episodes)
    average_agent_worst_case_rets_per_iter = np.mean(all_worst_case_returns, axis=(0, 2)) # shape (num_updates,)
    average_agent_optimal_rets_per_iter = np.mean(all_optimal_returns, axis=(0, 2)) # shape (num_updates,)
    average_agent_alpha_rets_per_iter = np.mean(all_alpha_returns, axis=(0, 2)) # shape (num_updates,)

    # Log metrics for each update step
    num_updates = len(average_agent_0_value_losses)
    for step in range(num_updates):
        for stat_name, stat_data in train_stats.items():
            # second dimension contains the mean and std of the metric
            stat_mean = stat_data[step, 0]
            logger.log_item(f"Train/Joint/Agent_{agent_idx + 1}_Optimize/{stat_name}", stat_mean, train_step=step, commit=True)

        logger.log_item(f"Eval/Joint/Agent_{agent_idx + 1}_Optimize/WorstCaseReturn", average_agent_worst_case_rets_per_iter[step], train_step=step, commit=True)
        logger.log_item(f"Eval/Joint/Agent_{agent_idx + 1}_Optimize/OptimalReturn", average_agent_optimal_rets_per_iter[step], train_step=step, commit=True)
        logger.log_item(f"Eval/Joint/Agent_{agent_idx + 1}_Optimize/AlphaReturn", average_agent_alpha_rets_per_iter[step], train_step=step, commit=True)

        logger.log_item(f"Train/Joint/Agent_{agent_idx + 1}_Optimize/Agent_1/ValueLoss", average_agent_0_value_losses[step], train_step=step, commit=True)
        logger.log_item(f"Train/Joint/Agent_{agent_idx + 1}_Optimize/Agent_1/ActorLoss", average_agent_0_actor_losses[step], train_step=step, commit=True)
        logger.log_item(f"Train/Joint/Agent_{agent_idx + 1}_Optimize/Agent_1/EntropyLoss", average_agent_0_entropy_losses[step], train_step=step, commit=True)
        logger.log_item(f"Train/Joint/Agent_{agent_idx + 1}_Optimize/Agent_1/GradNorm", average_agent_0_grad_norms[step], train_step=step, commit=True)

        logger.log_item(f"Train/Joint/Agent_{agent_idx + 1}_Optimize/Agent_2/ValueLoss", average_agent_1_value_losses[step], train_step=step, commit=True)
        logger.log_item(f"Train/Joint/Agent_{agent_idx + 1}_Optimize/Agent_2/ActorLoss", average_agent_1_actor_losses[step], train_step=step, commit=True)
        logger.log_item(f"Train/Joint/Agent_{agent_idx + 1}_Optimize/Agent_2/EntropyLoss", average_agent_1_entropy_losses[step], train_step=step, commit=True)
        logger.log_item(f"Train/Joint/Agent_{agent_idx + 1}_Optimize/Agent_2/GradNorm", average_agent_1_grad_norms[step], train_step=step, commit=True)
        logger.commit()

    # Saving artifacts
    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # shape of render_outs should be (num_train_seeds, num_eps, max_episode_steps, ...)
    eval_render_init_env_state = train_out['render_outs'][2].env_state.env_state # LogEnvState
    eval_render_optimal_env_state = train_out['render_outs'][1][-1]['pre_reset_state'].env_state # WrappedEnvState
    eval_render_optimal_dones = train_out['render_outs'][1][4]['__all__']
    eval_render_worst_case_env_state = train_out['render_outs'][0][-1]['pre_reset_state'].env_state # WrappedEnvState
    eval_render_worst_case_dones = train_out['render_outs'][0][4]['__all__']
    num_episodes = eval_render_worst_case_env_state.state['agent-at'].shape[1] # (num_train_seeds, num_eval_episodes, num_max_timesteps, num_agents_per_game, ...)
    env.animate((eval_render_init_env_state, eval_render_optimal_env_state), eval_render_optimal_dones, num_episodes, extra_dir="Optimal", debug=True)
    env.animate((eval_render_init_env_state, eval_render_worst_case_env_state), eval_render_worst_case_dones, num_episodes, extra_dir="WorstCase", debug=True)

    for eval_ep in range(num_episodes):
        logger.log_video(
            tag=f"Videos/Joint/Agent_{agent_idx + 1}_Optimize/WorstCase/Episode_{eval_ep}",
            path=os.path.join(env._render_dir, "WorstCase", f"{env._render_name}_ep_{eval_ep}.gif")
        )
        logger.log_video(
            tag=f"Videos/Joint/Agent_{agent_idx + 1}_Optimize/Optimal/Episode_{eval_ep}",
            path=os.path.join(env._render_dir, "Optimal", f"{env._render_name}_ep_{eval_ep}.gif")
        )

    train_out_agent_0 = {
        "final_params": train_out["final_params_agent_0"],
        "metrics": train_out["metrics"],
        "checkpoints": train_out["checkpoints_agent_0"],
    }
    train_out_agent_1 = {
        "final_params": train_out["final_params_agent_1"],
        "metrics": train_out["metrics"],
        "checkpoints": train_out["checkpoints_agent_1"],
    }

    agent_0_out_savepath = save_train_run(train_out_agent_0, savedir, savename=f"PPO_Agent_{agent_idx + 1}_Optimize_Train_Run-Agent_1")
    agent_1_out_savepath = save_train_run(train_out_agent_1, savedir, savename=f"PPO_Agent_{agent_idx + 1}_Optimize_Train_Run-Agent_2")
    if config["logger"]["log_train_out"]:
        logger.log_artifact(name=f"PPO_Agent_{agent_idx + 1}_Optimize_Train_Run-Agent_1", path=agent_0_out_savepath, type_name="joint_train_run")
        logger.log_artifact(name=f"PPO_Agent_{agent_idx + 1}_Optimize_Train_Run-Agent_2", path=agent_1_out_savepath, type_name="joint_train_run")
        # Cleanup locally logged out file
    if not config["local_logger"]["save_train_out"]:
        shutil.rmtree(agent_0_out_savepath)
        shutil.rmtree(agent_1_out_savepath)

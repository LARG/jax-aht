'''
Script for training a REPPO agent for social laws single agent projection.
Does not surepport training against heuristic partner agents.

Command to run REPPO single agent projection training:
python social_laws/run.py algorithm=reppo/lbf task=lbf label=test_reppo_single_agent_projection

Suggested debug command:
python social_laws/run.py algorithm=reppo/lbf task=lbf logger.mode=disabled label=debug algorithm.TOTAL_TIMESTEPS=1e5
'''
import os
import shutil
import time
import logging

from copy import deepcopy
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import hydra

from social_laws.common.initialize_agents import initialize_reppo_agent
from social_laws.common.run_episodes_reppo_w_robustness import run_episodes_vmap
from common.plot_utils import get_stats, get_metric_names
from common.save_load_utils import save_train_run
from envs import make_env
from envs.log_wrapper import LogWrapper
from marl.ppo_utils import unbatchify

from agents.mlp_reppo import hl_gauss
from agents.mlp_reppo_agent import ReppoTrainState, CustomTrainState, Transition

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def train_reppo_joint_agents(config, env, optimal_env, train_rng,
                           joint_policies, init_joint_params,
                           optimal_policies, optimal_params,
                           agent_idx):
    '''
    Train REPPO joint agents using the given initial parameters.

    Args:
        config: dict, config for the training
        env: gymnasium environment
        optimal_env: gymnasium environment for evaluating optimal returns
        train_rng: jax.random.PRNGKey, random key for training
        joint_policies: tuple of AgentPolicy, policies for the agents
        init_joint_params: tuple of dict, initial parameters for the agents
        optimal_policies: tuple of AgentPolicy, optimal policies for the agents
        optimal_params: tuple of dict, optimal parameters for the agents
        agent_idx: int, index of the agent to optimize
    '''
    # ------------------------------
    # Build the REPPO joint training function
    # ------------------------------
    agent_0_policy, agent_1_policy = joint_policies
    agent_0_init_params, agent_1_init_params = init_joint_params
    agent_0_optimal_policy, agent_1_optimal_policy = optimal_policies
    agent_0_optimal_params, agent_1_optimal_params = optimal_params

    def make_reppo_joint_train(config):
        num_agents = env.num_agents
        assert num_agents == 2, "This snippet assumes exactly 2 agents."

        config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
        config["NUM_UNCONTROLLED_ACTORS"] = config["NUM_ENVS"] # assumption: we control 1 agent
        config["NUM_CONTROLLED_ACTORS"] = config["NUM_ENVS"] # assumption: we control 1 agent
        config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["ROLLOUT_LENGTH"] // config["NUM_ENVS"]
        config["NUM_UPDATES_DECAY"] = (config["TOTAL_TIMESTEPS_DECAY"] // config["ROLLOUT_LENGTH"] // config["NUM_ENVS"])
        assert (config["ROLLOUT_LENGTH"] * config["NUM_ENVS"]) % config["NUM_MINIBATCHES"] == 0, "NUM_MINIBATCHES must divide ROLLOUT_LENGTH * NUM_ENVS"

        config["MAX_EPISODE_STEPS"] = env.horizon
        config["NUM_ACTIONS"] = env.action_space(f"agent_{agent_idx}").n

        lr_scheduler = optax.linear_schedule(
            init_value=config["LR"],
            end_value=1e-20,
            transition_steps=(config["NUM_UPDATES_DECAY"])
            * config["NUM_MINIBATCHES"]
            * config["NUM_EPOCHS"],
        )
        temp_scheduler = optax.linear_schedule(
            init_value=config["TEMP_START"],
            end_value=config["TEMP_FINISH"],
            transition_steps=(config["NUM_UPDATES_DECAY"])
            * config["NUM_MINIBATCHES"]
            * config["NUM_EPOCHS"],
        )

        def train(rng, agent_idx, agent_0_optimal_params, agent_1_optimal_params):
            original_seed = rng

            if config["ANNEAL_LR"]: # config['LR_LINEAR_DECAY']
                tx_0 = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(learning_rate=lr_scheduler),
                )

                tx_1 = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(learning_rate=lr_scheduler),
                )
            else:
                tx_0 = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(learning_rate=config["LR"]),
                )
                tx_1 = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(learning_rate=config["LR"]),
                )

            q_network_train_state_0 = CustomTrainState.create(
                apply_fn=agent_0_policy.q_network.apply,
                params=agent_0_init_params['q_network']["params"],
                batch_stats=agent_0_init_params['q_network']["batch_stats"],
                tx=tx_0,
            )
            actor_train_state_0 = CustomTrainState.create(
                apply_fn=agent_0_policy.actor.apply,
                params=agent_0_init_params['actor']["params"],
                batch_stats=agent_0_init_params['actor']["batch_stats"],
                tx=tx_0,
            )
            target_actor_train_state_0 = CustomTrainState.create(
                apply_fn=agent_0_policy.actor.apply,
                params=deepcopy(agent_0_init_params['actor']["params"]),
                batch_stats=deepcopy(agent_0_init_params['actor']["batch_stats"]),
                tx=optax.set_to_zero(),
            )

            q_network_train_state_1 = CustomTrainState.create(
                apply_fn=agent_1_policy.q_network.apply,
                params=agent_1_init_params['q_network']["params"],
                batch_stats=agent_1_init_params['q_network']["batch_stats"],
                tx=tx_1,
            )
            actor_train_state_1 = CustomTrainState.create(
                apply_fn=agent_1_policy.actor.apply,
                params=agent_1_init_params['actor']["params"],
                batch_stats=agent_1_init_params['actor']["batch_stats"],
                tx=tx_1,
            )
            target_actor_train_state_1 = CustomTrainState.create(
                apply_fn=agent_1_policy.actor.apply,
                params=deepcopy(agent_1_init_params['actor']["params"]),
                batch_stats=deepcopy(agent_1_init_params['actor']["batch_stats"]),
                tx=optax.set_to_zero(),
            )

            agent_0_train_state = ReppoTrainState(
                timesteps=0,
                n_updates=0,
                grad_steps=0,
                actor_train_state=actor_train_state_0,
                q_network_train_state=q_network_train_state_0,
                target_actor_train_state=target_actor_train_state_0,
            )

            agent_1_train_state = ReppoTrainState(
                timesteps=0,
                n_updates=0,
                grad_steps=0,
                actor_train_state=actor_train_state_1,
                q_network_train_state=q_network_train_state_1,
                target_actor_train_state=target_actor_train_state_1,
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
            agent_0_optimal_init_hstate = agent_0_optimal_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])
            agent_1_optimal_init_hstate = agent_1_optimal_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])

            def _get_optimal_restricted_avail_actions(obs, done, avail_actions, hstate, optimal_params, optimal_policy):
                """
                Get restricted available actions based on optimal Q-values.
                Only allows actions within epsilon of the maximum Q-value.

                Args:
                    obs: observations (shape: num_envs, obs_dim)
                    done: done flags (shape: num_envs,)
                    avail_actions: available actions mask (shape: num_envs, num_actions)
                    hstate: hidden state for the optimal policy (shape depends on policy)
                    optimal_params: optimal value function parameters
                    optimal_policy: optimal value function policy

                Returns:
                    restricted_avail_actions: mask with only near-optimal actions (shape: num_envs, num_actions)
                """
                _, _, q_vals, new_critic_hstate  = optimal_policy.get_critic_logits_probs_values(
                    params=(optimal_params['q_network']["params"], optimal_params['q_network']["batch_stats"]),
                    obs=obs,
                    done=done,
                    avail_actions=avail_actions,
                    hstate=hstate,
                    rng=jax.random.PRNGKey(0)
                ) # (num_envs, num_actions)

                # Mask Q-values with available actions (set unavailable to -inf)
                q_values_masked = jnp.where(avail_actions, q_vals, -jnp.inf)

                # Find max Q-value for each environment
                max_q = jnp.max(q_values_masked, axis=-1, keepdims=True)  # (num_envs, 1)

                # Create mask for actions within epsilon of max
                optimal_action_mask = (q_values_masked >= (max_q - config['EPSILON_OPTIMAL'])).astype(jnp.float32)

                # Combine with original available actions
                restricted_avail_actions = optimal_action_mask * avail_actions

                # remove extra batch dim
                return restricted_avail_actions, new_critic_hstate

            def _env_step(runner_state, unused):
                """
                One step of the environment:
                1. Get observations, sample actions from all agents
                2. Step environment using sampled actions
                3. Return state, reward, ...
                """
                agent_0_train_state, agent_1_train_state, env_state, prev_obs, prev_done, hstate_0, hstate_1, optimal_hstate_0, optimal_hstate_1, rng = runner_state
                prev_obs, prev_full_obs = prev_obs
                rng, actor_0_rng, actor_1_rng, critic_0_rng, critic_1_rng, step_rng = jax.random.split(rng, 6)

                actor_hstate_0, critic_hstate_0, target_hstate_0 = hstate_0
                actor_hstate_1, critic_hstate_1, target_hstate_1 = hstate_1
                optimal_actor_hstate_0, optimal_critic_hstate_0, optimal_target_hstate_0 = optimal_hstate_0
                optimal_actor_hstate_1, optimal_critic_hstate_1, optimal_target_hstate_1 = optimal_hstate_1

                 # Get available actions for the agent from environment state
                avail_actions = env.get_avail_actions(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(avail_actions)
                avail_actions_0 = get_agent_data(avail_actions, 0).astype(jnp.float32)
                avail_actions_1 = get_agent_data(avail_actions, 1).astype(jnp.float32)

                prev_obs_0 = get_agent_data(prev_obs, 0).reshape(1, config["NUM_CONTROLLED_ACTORS"], -1)
                prev_obs_1 = get_agent_data(prev_obs, 1).reshape(1, config["NUM_CONTROLLED_ACTORS"], -1)
                prev_full_obs_0 = get_agent_data(prev_full_obs, 0).reshape(1, config["NUM_CONTROLLED_ACTORS"], -1)
                prev_full_obs_1 = get_agent_data(prev_full_obs, 1).reshape(1, config["NUM_CONTROLLED_ACTORS"], -1)

                prev_done_0 = get_agent_data(prev_done, 0).reshape(1, config["NUM_CONTROLLED_ACTORS"])
                prev_done_1 = get_agent_data(prev_done, 1).reshape(1, config["NUM_CONTROLLED_ACTORS"])

                # Restrict available actions based on value function
                optimal_restricted_avail_actions_0, new_optimal_critic_hstate_0 = _get_optimal_restricted_avail_actions(
                    obs=prev_obs_0,
                    done=prev_done_0,
                    avail_actions=avail_actions_0,
                    hstate=optimal_critic_hstate_0,
                    optimal_params=agent_0_optimal_params,
                    optimal_policy=agent_0_optimal_policy
                )
                optimal_restricted_avail_actions_1, new_optimal_critic_hstate_1 = _get_optimal_restricted_avail_actions(
                    obs=prev_obs_1,
                    done=prev_done_1,
                    avail_actions=avail_actions_1,
                    hstate=optimal_critic_hstate_1,
                    optimal_params=agent_1_optimal_params,
                    optimal_policy=agent_1_optimal_policy
                )

                # Note that we do not need to reset the hidden states for the agents
                # as the recurrent states are automatically reset when done is True.

                # Agent 0 action, value, log_prob (using optimal-restricted available actions)
                act_0, importance_weight_0, pi_0, new_actor_hstate_0 = agent_0_policy.get_action_importance_policy(
                    params=(agent_0_train_state.actor_train_state.params, agent_0_train_state.actor_train_state.batch_stats),
                    obs=prev_full_obs_0 if config["JOINT_USE_FULL_OBS"] else prev_obs_0,
                    done=prev_done_0,
                    avail_actions=optimal_restricted_avail_actions_0,
                    hstate=actor_hstate_0,
                    rng=actor_0_rng,
                    temp_schedule=temp_scheduler(agent_0_train_state.n_updates)
                )
                logp_0 = pi_0.log_prob(act_0)

                act_0 = act_0.squeeze(axis=0)
                logp_0 = logp_0.squeeze(axis=0)
                importance_weight_0 = importance_weight_0.squeeze(axis=0)  # (1, NUM_ENVS) -> (NUM_ENVS,)

                # Agent 1 action, value, log_prob (using optimal-restricted available actions)
                act_1, importance_weight_1, pi_1, new_actor_hstate_1 = agent_1_policy.get_action_importance_policy(
                    params=(agent_1_train_state.actor_train_state.params, agent_1_train_state.actor_train_state.batch_stats),
                    obs=prev_full_obs_1 if config["JOINT_USE_FULL_OBS"] else prev_obs_1,
                    done=prev_done_1,
                    avail_actions=optimal_restricted_avail_actions_1,
                    hstate=actor_hstate_1,
                    rng=actor_1_rng,
                    temp_schedule=temp_scheduler(agent_1_train_state.n_updates)
                )
                logp_1 = pi_1.log_prob(act_1)

                act_1 = act_1.squeeze(axis=0)
                logp_1 = logp_1.squeeze(axis=0)
                importance_weight_1 = importance_weight_1.squeeze(axis=0)  # (1, NUM_ENVS) -> (NUM_ENVS,)

                # Combine actions into the env format
                combined_actions = jnp.concatenate([act_0, act_1], axis=0)  # shape (2*num_envs,)
                env_act = unbatchify(combined_actions, env.agents, config["NUM_ENVS"], num_agents)
                env_act = {k: v.flatten() for k, v in env_act.items()}

                # Step env
                step_rngs = jax.random.split(step_rng, config["NUM_ENVS"])
                (obs_next, obs_full_next), env_state_next, reward, done_next, info = jax.vmap(env.step, in_axes=(0,0,0))(
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

                # Next values
                next_avail_actions = env.get_avail_actions(env_state_next.env_state)
                next_avail_actions = jax.lax.stop_gradient(next_avail_actions)
                next_avail_actions_0 = get_agent_data(next_avail_actions, 0).astype(jnp.float32)
                next_avail_actions_1 = get_agent_data(next_avail_actions, 1).astype(jnp.float32)

                obs_next_0 = get_agent_data(obs_next, 0).reshape(1, config["NUM_CONTROLLED_ACTORS"], -1)
                obs_next_1 = get_agent_data(obs_next, 1).reshape(1, config["NUM_CONTROLLED_ACTORS"], -1)
                obs_full_next_0 = get_agent_data(obs_full_next, 0).reshape(1, config["NUM_CONTROLLED_ACTORS"], -1)
                obs_full_next_1 = get_agent_data(obs_full_next, 1).reshape(1, config["NUM_CONTROLLED_ACTORS"], -1)

                done_next_0 = get_agent_data(done_next, 0).reshape(1, config["NUM_CONTROLLED_ACTORS"])
                done_next_1 = get_agent_data(done_next, 1).reshape(1, config["NUM_CONTROLLED_ACTORS"])

                # Restrict available actions based on value function
                next_optimal_restricted_avail_actions_0, _ = _get_optimal_restricted_avail_actions(
                    obs=obs_next_0,
                    done=done_next_0,
                    avail_actions=next_avail_actions_0,
                    hstate=new_optimal_critic_hstate_0,
                    optimal_params=agent_0_optimal_params,
                    optimal_policy=agent_0_optimal_policy
                )
                next_optimal_restricted_avail_actions_1, _ = _get_optimal_restricted_avail_actions(
                    obs=obs_next_1,
                    done=done_next_1,
                    avail_actions=next_avail_actions_1,
                    hstate=new_optimal_critic_hstate_1,
                    optimal_params=agent_1_optimal_params,
                    optimal_policy=agent_1_optimal_policy
                )

                # Agent 0
                next_pi_0 = agent_0_train_state.target_actor_train_state.apply_fn(
                    {
                        "params": agent_0_train_state.target_actor_train_state.params,
                        "batch_stats": agent_0_train_state.target_actor_train_state.batch_stats,
                    },
                    (obs_full_next_0 if config["JOINT_USE_FULL_OBS"] else obs_next_0,
                     next_optimal_restricted_avail_actions_0),
                    train=False,
                )
                # TODO: Deal with recurrent target
                new_target_hstate_0 = None

                _, _, q_vals_0, new_critic_hstate_0  = agent_0_policy.get_critic_logits_probs_values(
                    params=(agent_0_train_state.q_network_train_state.params, agent_0_train_state.q_network_train_state.batch_stats),
                    obs=obs_full_next_0 if config["JOINT_USE_FULL_OBS"] else obs_next_0,
                    done=done_next_0,
                    avail_actions=next_optimal_restricted_avail_actions_0,
                    hstate=critic_hstate_0,
                    rng=critic_0_rng,
                )
                next_values_0 = jnp.sum(next_pi_0.probs * q_vals_0, axis=-1).squeeze(axis=0)  # (1, NUM_ENVS) -> (NUM_ENVS,)

                # Agent 1
                next_pi_1 = agent_1_train_state.target_actor_train_state.apply_fn(
                    {
                        "params": agent_1_train_state.target_actor_train_state.params,
                        "batch_stats": agent_1_train_state.target_actor_train_state.batch_stats,
                    },
                    (obs_full_next_1 if config["JOINT_USE_FULL_OBS"] else obs_next_1,
                     next_optimal_restricted_avail_actions_1),
                    train=False,
                )
                # TODO: Deal with recurrent target
                new_target_hstate_1 = None

                _, _, q_vals_1, new_critic_hstate_1  = agent_1_policy.get_critic_logits_probs_values(
                    params=(agent_1_train_state.q_network_train_state.params, agent_1_train_state.q_network_train_state.batch_stats),
                    obs=obs_full_next_1 if config["JOINT_USE_FULL_OBS"] else obs_next_1,
                    done=done_next_1,
                    avail_actions=next_optimal_restricted_avail_actions_1,
                    hstate=critic_hstate_1,
                    rng=critic_1_rng,
                )
                next_values_1 = jnp.sum(next_pi_1.probs * q_vals_1, axis=-1).squeeze(axis=0)  # (1, NUM_ENVS) -> (NUM_ENVS,)


                # Store agent_idx data in transition (using VF-restricted available actions)
                transition_0 = Transition(
                    obs=get_agent_data(prev_full_obs, 0) if config["JOINT_USE_FULL_OBS"] else get_agent_data(prev_obs, 0),
                    action=act_0,
                    action_logp=logp_0,
                    reward=get_agent_data(reward, agent_idx) * -1,
                    done=get_agent_data(done_next, 0),
                    avail_actions=optimal_restricted_avail_actions_0,
                    next_obs=get_agent_data(obs_full_next, 0) if config["JOINT_USE_FULL_OBS"] else get_agent_data(obs_next, 0),
                    next_avail_actions=next_optimal_restricted_avail_actions_0,
                    next_val=next_values_0,
                    importance_weight=importance_weight_0,
                    info=info
                )
                transition_1 = Transition(
                    obs=get_agent_data(prev_full_obs, 1) if config["JOINT_USE_FULL_OBS"] else get_agent_data(prev_obs, 1),
                    action=act_1,
                    action_logp=logp_1,
                    reward=get_agent_data(reward, agent_idx) * -1,
                    done=get_agent_data(done_next, 1),
                    avail_actions=optimal_restricted_avail_actions_1,
                    next_obs=get_agent_data(obs_full_next, 1) if config["JOINT_USE_FULL_OBS"] else get_agent_data(obs_next, 1),
                    next_avail_actions=next_optimal_restricted_avail_actions_1,
                    next_val=next_values_1,
                    importance_weight=importance_weight_1,
                    info=info
                )

                new_runner_state = (agent_0_train_state, agent_1_train_state, env_state_next, (obs_next, obs_full_next), done_next,
                                    (new_actor_hstate_0, new_critic_hstate_0, new_target_hstate_0),
                                    (new_actor_hstate_1, new_critic_hstate_1, new_target_hstate_1),
                                    (optimal_actor_hstate_0, new_optimal_critic_hstate_0, optimal_target_hstate_0),
                                    (optimal_actor_hstate_1, new_optimal_critic_hstate_1, optimal_target_hstate_1),
                                    rng)
                return new_runner_state, (transition_0, transition_1)

            def compute_nstep_lambda(carry, transition):
                lambda_return, truncated, importance_weight = carry

                # Combine importance_weights with TD lambda
                done = transition.done
                reward = transition.soft_reward
                value = transition.next_val

                lambda_sum = (
                    jnp.exp(importance_weight) * config["LAMBDA"] * lambda_return
                    + (1 - jnp.exp(importance_weight) * config["LAMBDA"]) * value
                )

                delta = config["GAMMA"] * jnp.where(
                    truncated, value, (1.0 - done) * lambda_sum
                )

                lambda_return = reward + delta

                return (
                    lambda_return,
                    jnp.zeros_like(truncated),
                    transition.importance_weight,
                ), lambda_return

            def _update_step(update_runner_state, unused):
                """
                1. Collect rollouts
                2. Compute advantage
                3. REPPO updates
                """
                (agent_0_train_state, agent_1_train_state, rng, rng_eval, update_steps) = update_runner_state
                # Init envs & partner indices
                rng, reset_rng = jax.random.split(rng, 2)
                reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
                init_obs, init_env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rngs)
                init_done = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}

                # 1) rollout
                runner_state = (agent_0_train_state, agent_1_train_state, init_env_state, init_obs, init_done, agent_0_init_hstate, agent_1_init_hstate, agent_0_optimal_init_hstate, agent_1_optimal_init_hstate, rng)

                runner_state, traj_batch = jax.lax.scan(
                    _env_step, runner_state, None, config["ROLLOUT_LENGTH"])
                (agent_0_train_state, agent_1_train_state, env_state, obs, done, agent_0_hstate, agent_1_hstate, agent_0_optimal_hstate, agent_1_optimal_hstate, rng) = runner_state
                agent_0_traj_batch, agent_1_traj_batch = traj_batch

                agent_0_train_state = agent_0_train_state.replace(
                    timesteps=agent_0_train_state.timesteps
                    + config["ROLLOUT_LENGTH"] * config["NUM_ENVS"]
                )  # update timesteps count

                agent_1_train_state = agent_1_train_state.replace(
                    timesteps=agent_1_train_state.timesteps
                    + config["ROLLOUT_LENGTH"] * config["NUM_ENVS"]
                )  # update timesteps count

                # 2) Compute soft rewards and n-step lambda returns
                # Agent 0
                next_pi_0 = agent_0_train_state.target_actor_train_state.apply_fn(
                    {
                        "params": agent_0_train_state.target_actor_train_state.params,
                        "batch_stats": agent_0_train_state.target_actor_train_state.batch_stats,
                    },
                    (agent_0_traj_batch.next_obs[-1].reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                     agent_0_traj_batch.next_avail_actions[-1]),
                    train=False,
                )
                # TODO: Deal with recurrent target

                rng, sample_rng = jax.random.split(rng)
                _, last_action_logp_0 = next_pi_0.sample_and_log_prob(seed=sample_rng)
                last_action_logp_0 = last_action_logp_0.squeeze(axis=0)
                next_logp_0 = jnp.concatenate(
                    [agent_0_traj_batch.action_logp[1:], last_action_logp_0[None]], axis=0
                )
                soft_reward_0 = agent_0_traj_batch.reward - config["GAMMA"] * next_logp_0 * jnp.exp(
                    agent_0_train_state.actor_train_state.params["log_alpha"]
                )
                agent_0_traj_batch = agent_0_traj_batch.replace(soft_reward=soft_reward_0)

                _, target_values_0 = jax.lax.scan(
                    compute_nstep_lambda,
                    (
                        agent_0_traj_batch.next_val[-1],
                        jnp.ones_like(agent_0_traj_batch.done[0]),
                        jnp.zeros_like(agent_0_traj_batch.importance_weight[0]),
                    ),
                    agent_0_traj_batch,
                    reverse=True,
                )

                # Agent 1
                next_pi_1 = agent_1_train_state.target_actor_train_state.apply_fn(
                    {
                        "params": agent_1_train_state.target_actor_train_state.params,
                        "batch_stats": agent_1_train_state.target_actor_train_state.batch_stats,
                    },
                    (agent_1_traj_batch.next_obs[-1].reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                     agent_1_traj_batch.next_avail_actions[-1]),
                    train=False,
                )
                # TODO: Deal with recurrent target

                rng, sample_rng = jax.random.split(rng)
                _, last_action_logp_1 = next_pi_1.sample_and_log_prob(seed=sample_rng)
                last_action_logp_1 = last_action_logp_1.squeeze(axis=0)
                next_logp_1 = jnp.concatenate(
                    [agent_1_traj_batch.action_logp[1:], last_action_logp_1[None]], axis=0
                )
                soft_reward_1 = agent_1_traj_batch.reward - config["GAMMA"] * next_logp_1 * jnp.exp(
                    agent_1_train_state.actor_train_state.params["log_alpha"]
                )
                agent_1_traj_batch = agent_1_traj_batch.replace(soft_reward=soft_reward_1)

                _, target_values_1 = jax.lax.scan(
                    compute_nstep_lambda,
                    (
                        agent_1_traj_batch.next_val[-1],
                        jnp.ones_like(agent_1_traj_batch.done[0]),
                        jnp.zeros_like(agent_1_traj_batch.importance_weight[0]),
                    ),
                    agent_1_traj_batch,
                    reverse=True,
                )

                # 3) REPPO update for each agent separately
                def _learn_epoch_0(carry, _):
                    train_state, rng = carry

                    def _learn_phase(carry, minibatch_and_target):
                        train_state, rng = carry
                        minibatch, target = minibatch_and_target

                        def _critic_loss_fn(params, train_state):
                            critic_out, updates = agent_0_policy.q_network.apply(
                                {
                                    "params": params,
                                    "batch_stats": train_state.q_network_train_state.batch_stats,
                                },
                                minibatch.obs,
                                train=True,
                                mutable=["batch_stats"],
                            )  # (batch_size*2, num_actions)
                            # TODO: Deal with recurrent critic

                            logits = jnp.take_along_axis(
                                critic_out["logits"],
                                minibatch.action[..., None, None],
                                axis=-2,
                            ).squeeze(axis=-2)
                            q_vals = jnp.take_along_axis(
                                critic_out["q_values"],
                                jnp.expand_dims(minibatch.action, axis=-1),
                                axis=-1,
                            ).squeeze(axis=-1)


                            target_cat = jax.vmap(hl_gauss, in_axes=(0, None, None, None))(
                                target,
                                config["NUM_BINS"],
                                config["VMIN"],
                                config["VMAX"],
                            )

                            loss = optax.softmax_cross_entropy(logits, target_cat).mean()
                            metrics = {
                                "agent_0/critic_loss": loss,
                                "agent_0/q_values": q_vals.mean(),
                                "agent_0/q_error": optax.l2_loss(target - q_vals).mean(),
                            }
                            return loss, dict(updates=updates, metrics=metrics)

                        def _actor_loss_fn(params, train_state):
                            pi, updates = train_state.actor_train_state.apply_fn(
                                {
                                    "params": params,
                                    "batch_stats": train_state.actor_train_state.batch_stats,
                                },
                                (minibatch.obs, minibatch.avail_actions),
                                train=True,
                                mutable=["batch_stats"],
                            )
                            # TODO: Deal with recurrent actor

                            old_pi = train_state.target_actor_train_state.apply_fn(
                                {
                                    "params": train_state.target_actor_train_state.params,
                                    "batch_stats": train_state.target_actor_train_state.batch_stats,
                                },
                                (minibatch.obs, minibatch.avail_actions),
                                train=False,
                            )
                            # TODO: Deal with recurrent target

                            critic_output = agent_0_policy.q_network.apply(
                                {
                                    "params": train_state.q_network_train_state.params,
                                    "batch_stats": train_state.q_network_train_state.batch_stats,
                                },
                                minibatch.obs,
                                train=False,
                            )
                            q_vals = critic_output["q_values"]
                            alpha = jax.lax.stop_gradient(jnp.exp(params["log_alpha"]))
                            lagrangian = jax.lax.stop_gradient(
                                jnp.exp(params["log_lagrangian"])
                            )
                            actor_loss = jnp.sum(
                                pi.probs * (alpha * pi.logits - q_vals), axis=-1
                            )
                            kl = old_pi.kl_divergence(pi)

                            loss = jnp.mean(
                                jnp.where(
                                    kl < config["KL_BOUND"],
                                    actor_loss,
                                    kl * lagrangian,
                                )
                            )

                            pi = jax.lax.stop_gradient(pi)
                            target_entropy = config["TARGET_ENTROPY_MULT"] * jnp.log(config["NUM_ACTIONS"])
                            alpha = jnp.exp(params["log_alpha"])
                            alpha_loss = jnp.sum(
                                pi.probs * (-alpha * (pi.logits + target_entropy)),
                                axis=-1,
                            )

                            lagrangian_loss = -jnp.exp(
                                params["log_lagrangian"]
                            ) * jax.lax.stop_gradient(kl - config["KL_BOUND"])

                            total_loss = (
                                loss + jnp.mean(alpha_loss) + jnp.mean(lagrangian_loss)
                            )
                            metrics = {
                                "agent_0/actor_loss": loss,
                                "agent_0/target_entropy_loss": jnp.mean(alpha_loss),
                                "agent_0/lagrangian_loss": jnp.mean(lagrangian_loss),
                                "agent_0/alpha": alpha,
                                "agent_0/lagrangian": lagrangian,
                                "agent_0/kl": kl.mean(),
                                "agent_0/entropy": pi.entropy().mean(),
                            }
                            return total_loss, dict(updates=updates, metrics=metrics)

                        (_, critic_update_output), grads = jax.value_and_grad(
                            _critic_loss_fn, has_aux=True
                        )(train_state.q_network_train_state.params, train_state)
                        q_network_train_state = (
                            train_state.q_network_train_state.apply_gradients(grads=grads)
                        )

                        (_, actor_update_output), grads = jax.value_and_grad(
                            _actor_loss_fn, has_aux=True
                        )(train_state.actor_train_state.params, train_state)
                        actor_train_state = train_state.actor_train_state.apply_gradients(
                            grads=grads
                        )
                        train_state = train_state.replace(
                            q_network_train_state=q_network_train_state.replace(
                                batch_stats=critic_update_output["updates"]["batch_stats"]
                            ),
                            actor_train_state=actor_train_state.replace(
                                batch_stats=actor_update_output["updates"]["batch_stats"]
                            ),
                            grad_steps=train_state.grad_steps + 1,
                        )
                        return (train_state, rng), {
                            **critic_update_output["metrics"],
                            **actor_update_output["metrics"],
                        }

                    def preprocess_transition(x, rng):
                        x = x.reshape(
                            -1, *x.shape[2:]
                        )  # num_steps*num_envs (batch_size), ...
                        x = jax.random.permutation(rng, x)  # shuffle the transitions
                        x = x.reshape(
                            config["NUM_MINIBATCHES"], -1, *x.shape[1:]
                        )  # num_mini_updates, batch_size/num_mini_updates, ...
                        return x

                    rng, _rng = jax.random.split(rng)
                    minibatches = jax.tree_util.tree_map(
                        lambda x: preprocess_transition(x, _rng), agent_0_traj_batch
                    )  # num_actors*num_envs (batch_size), ...
                    targets = jax.tree_util.tree_map(
                        lambda x: preprocess_transition(x, _rng), target_values_0
                    )

                    rng, _rng = jax.random.split(rng)
                    (train_state, rng), metrics = jax.lax.scan(
                        _learn_phase, init=(train_state, rng), xs=(minibatches, targets)
                    )

                    return (train_state, rng), metrics

                def _learn_epoch_1(carry, _):
                    train_state, rng = carry

                    def _learn_phase(carry, minibatch_and_target):
                        train_state, rng = carry
                        minibatch, target = minibatch_and_target

                        def _critic_loss_fn(params, train_state):
                            critic_out, updates = agent_1_policy.q_network.apply(
                                {
                                    "params": params,
                                    "batch_stats": train_state.q_network_train_state.batch_stats,
                                },
                                minibatch.obs,
                                train=True,
                                mutable=["batch_stats"],
                            )  # (batch_size*2, num_actions)
                            # TODO: Deal with recurrent critic

                            logits = jnp.take_along_axis(
                                critic_out["logits"],
                                minibatch.action[..., None, None],
                                axis=-2,
                            ).squeeze(axis=-2)
                            q_vals = jnp.take_along_axis(
                                critic_out["q_values"],
                                jnp.expand_dims(minibatch.action, axis=-1),
                                axis=-1,
                            ).squeeze(axis=-1)


                            target_cat = jax.vmap(hl_gauss, in_axes=(0, None, None, None))(
                                target,
                                config["NUM_BINS"],
                                config["VMIN"],
                                config["VMAX"],
                            )

                            loss = optax.softmax_cross_entropy(logits, target_cat).mean()
                            metrics = {
                                "agent_1/critic_loss": loss,
                                "agent_1/q_values": q_vals.mean(),
                                "agent_1/q_error": optax.l2_loss(target - q_vals).mean(),
                            }
                            return loss, dict(updates=updates, metrics=metrics)

                        def _actor_loss_fn(params, train_state):
                            pi, updates = train_state.actor_train_state.apply_fn(
                                {
                                    "params": params,
                                    "batch_stats": train_state.actor_train_state.batch_stats,
                                },
                                (minibatch.obs, minibatch.avail_actions),
                                train=True,
                                mutable=["batch_stats"],
                            )
                            # TODO: Deal with recurrent actor

                            old_pi = train_state.target_actor_train_state.apply_fn(
                                {
                                    "params": train_state.target_actor_train_state.params,
                                    "batch_stats": train_state.target_actor_train_state.batch_stats,
                                },
                                (minibatch.obs, minibatch.avail_actions),
                                train=False,
                            )
                            # TODO: Deal with recurrent target

                            critic_output = agent_1_policy.q_network.apply(
                                {
                                    "params": train_state.q_network_train_state.params,
                                    "batch_stats": train_state.q_network_train_state.batch_stats,
                                },
                                minibatch.obs,
                                train=False,
                            )
                            q_vals = critic_output["q_values"]
                            alpha = jax.lax.stop_gradient(jnp.exp(params["log_alpha"]))
                            lagrangian = jax.lax.stop_gradient(
                                jnp.exp(params["log_lagrangian"])
                            )
                            actor_loss = jnp.sum(
                                pi.probs * (alpha * pi.logits - q_vals), axis=-1
                            )
                            kl = old_pi.kl_divergence(pi)

                            loss = jnp.mean(
                                jnp.where(
                                    kl < config["KL_BOUND"],
                                    actor_loss,
                                    kl * lagrangian,
                                )
                            )

                            pi = jax.lax.stop_gradient(pi)
                            target_entropy = config["TARGET_ENTROPY_MULT"] * jnp.log(config["NUM_ACTIONS"])
                            alpha = jnp.exp(params["log_alpha"])
                            alpha_loss = jnp.sum(
                                pi.probs * (-alpha * (pi.logits + target_entropy)),
                                axis=-1,
                            )

                            lagrangian_loss = -jnp.exp(
                                params["log_lagrangian"]
                            ) * jax.lax.stop_gradient(kl - config["KL_BOUND"])

                            total_loss = (
                                loss + jnp.mean(alpha_loss) + jnp.mean(lagrangian_loss)
                            )
                            metrics = {
                                "agent_1/actor_loss": loss,
                                "agent_1/target_entropy_loss": jnp.mean(alpha_loss),
                                "agent_1/lagrangian_loss": jnp.mean(lagrangian_loss),
                                "agent_1/alpha": alpha,
                                "agent_1/lagrangian": lagrangian,
                                "agent_1/kl": kl.mean(),
                                "agent_1/entropy": pi.entropy().mean(),
                            }
                            return total_loss, dict(updates=updates, metrics=metrics)

                        (_, critic_update_output), grads = jax.value_and_grad(
                            _critic_loss_fn, has_aux=True
                        )(train_state.q_network_train_state.params, train_state)
                        q_network_train_state = (
                            train_state.q_network_train_state.apply_gradients(grads=grads)
                        )

                        (_, actor_update_output), grads = jax.value_and_grad(
                            _actor_loss_fn, has_aux=True
                        )(train_state.actor_train_state.params, train_state)
                        actor_train_state = train_state.actor_train_state.apply_gradients(
                            grads=grads
                        )
                        train_state = train_state.replace(
                            q_network_train_state=q_network_train_state.replace(
                                batch_stats=critic_update_output["updates"]["batch_stats"]
                            ),
                            actor_train_state=actor_train_state.replace(
                                batch_stats=actor_update_output["updates"]["batch_stats"]
                            ),
                            grad_steps=train_state.grad_steps + 1,
                        )
                        return (train_state, rng), {
                            **critic_update_output["metrics"],
                            **actor_update_output["metrics"],
                        }

                    def preprocess_transition(x, rng):
                        x = x.reshape(
                            -1, *x.shape[2:]
                        )  # num_steps*num_envs (batch_size), ...
                        x = jax.random.permutation(rng, x)  # shuffle the transitions
                        x = x.reshape(
                            config["NUM_MINIBATCHES"], -1, *x.shape[1:]
                        )  # num_mini_updates, batch_size/num_mini_updates, ...
                        return x

                    rng, _rng = jax.random.split(rng)
                    minibatches = jax.tree_util.tree_map(
                        lambda x: preprocess_transition(x, _rng), agent_1_traj_batch
                    )  # num_actors*num_envs (batch_size), ...
                    targets = jax.tree_util.tree_map(
                        lambda x: preprocess_transition(x, _rng), target_values_1
                    )

                    rng, _rng = jax.random.split(rng)
                    (train_state, rng), metrics = jax.lax.scan(
                        _learn_phase, init=(train_state, rng), xs=(minibatches, targets)
                    )

                    return (train_state, rng), metrics

                # Agent 0 update
                rng, learn_rng = jax.random.split(rng)
                (agent_0_train_state, rng), learn_metrics_0 = jax.lax.scan(
                    f=_learn_epoch_0,
                    init=(agent_0_train_state, learn_rng),
                    xs=None,
                    length=config["NUM_EPOCHS"],
                )
                target_actor_train_state = agent_0_train_state.target_actor_train_state.replace(
                    params=deepcopy(agent_0_train_state.actor_train_state.params),
                    batch_stats=deepcopy(agent_0_train_state.actor_train_state.batch_stats),
                )
                agent_0_train_state = agent_0_train_state.replace(
                    target_actor_train_state=target_actor_train_state,
                    n_updates=agent_0_train_state.n_updates + 1,
                )

                # Agent 1 update
                rng, learn_rng = jax.random.split(rng)
                (agent_1_train_state, rng), learn_metrics_1 = jax.lax.scan(
                    f=_learn_epoch_1,
                    init=(agent_1_train_state, learn_rng),
                    xs=None,
                    length=config["NUM_EPOCHS"],
                )
                target_actor_train_state = agent_1_train_state.target_actor_train_state.replace(
                    params=deepcopy(agent_1_train_state.actor_train_state.params),
                    batch_stats=deepcopy(agent_1_train_state.actor_train_state.batch_stats),
                )
                agent_1_train_state = agent_1_train_state.replace(
                    target_actor_train_state=target_actor_train_state,
                    n_updates=agent_1_train_state.n_updates + 1,
                )

                # Average metrics across both agents
                metric = agent_0_traj_batch.info
                metric["update_steps"] = update_steps
                metric.update(learn_metrics_0)
                metric.update(learn_metrics_1)
                new_runner_state = (agent_0_train_state, agent_1_train_state, rng, rng_eval, update_steps + 1)
                return (new_runner_state, metric)

            # REPPO Update and Checkpoint saving
            ckpt_and_eval_interval = config["NUM_UPDATES"] // max(1, config["NUM_CHECKPOINTS"] - 1)  # -1 because we store a ckpt at the last update
            num_ckpts = config["NUM_CHECKPOINTS"]

            # Build a PyTree that holds parameters for all FCP checkpoints
            def init_ckpt_array(params_pytree):
                return jax.tree.map(
                    lambda x: jnp.zeros((num_ckpts,) + x.shape, x.dtype),
                    params_pytree)

            max_episode_steps = config["MAX_EPISODE_STEPS"]

            def _update_step_with_ckpt(state_with_ckpt, unused):
                (update_state, checkpoint_array_0, ckpt_idx_0, checkpoint_array_1, ckpt_idx_1, init_ckpt_eval_last_info, init_eval_last_info) = state_with_ckpt

                # Single REPPO update
                new_update_state, metric = _update_step(
                    update_state,
                    None
                )
                (agent_0_train_state, agent_1_train_state, rng, rng_eval, update_steps) = new_update_state

                # update steps is 1-indexed because it was incremented at the end of the update step
                to_store = jnp.logical_or(jnp.equal(jnp.mod(update_steps-1, ckpt_and_eval_interval), 0),
                                        jnp.equal(update_steps, config["NUM_UPDATES"]))


                def store_and_eval_ckpt(args):
                    ckpt_arr_0, cidx_0, ckpt_arr_1, cidx_1, rng, rng_eval, prev_ckpt_eval_ret_info, prev_eval_ret_info = args
                    ckpt_params_0 = {
                        "actor": {
                            "params": agent_0_train_state.actor_train_state.params,
                            "batch_stats": agent_0_train_state.actor_train_state.batch_stats
                        },
                        "q_network": {
                            "params": agent_0_train_state.q_network_train_state.params,
                            "batch_stats": agent_0_train_state.q_network_train_state.batch_stats
                        }
                    }
                    new_ckpt_arr_0 = jax.tree.map(
                        lambda c_arr, p: c_arr.at[cidx_0].set(p),
                        ckpt_arr_0, ckpt_params_0
                    )
                    ckpt_params_1 = {
                        "actor": {
                            "params": agent_1_train_state.actor_train_state.params,
                            "batch_stats": agent_1_train_state.actor_train_state.batch_stats
                        },
                        "q_network": {
                            "params": agent_1_train_state.q_network_train_state.params,
                            "batch_stats": agent_1_train_state.q_network_train_state.batch_stats
                        }
                    }
                    new_ckpt_arr_1 = jax.tree.map(
                        lambda c_arr, p: c_arr.at[cidx_1].set(p),
                        ckpt_arr_1, ckpt_params_1
                    )

                    if config["FIXED_EVAL"]:
                        eval_rng = rng_eval
                    else:
                        rng_eval, eval_rng = jax.random.split(rng_eval, 2)
                    ckpt_eval_eps_last_infos = run_episodes_vmap(eval_rng, env, optimal_env, agent_idx,
                        agent_params=(ckpt_params_0, ckpt_params_1),
                        agent_policies=(agent_0_policy, agent_1_policy),
                        optimal_params=(agent_0_optimal_params, agent_1_optimal_params),
                        optimal_policies=(agent_0_optimal_policy, agent_1_optimal_policy),
                        max_episode_steps=max_episode_steps,
                        num_eps=config["NUM_EVAL_EPISODES"],
                        epsilon_optimal=config["EPSILON_OPTIMAL"],
                        use_full_obs=config["JOINT_USE_FULL_OBS"],
                        agent_test_mode=True)

                    return (new_ckpt_arr_0, cidx_0 + 1, new_ckpt_arr_1, cidx_1 + 1, rng, rng_eval, ckpt_eval_eps_last_infos, ckpt_eval_eps_last_infos)

                def skip_ckpt_and_eval(args):
                    def do_eval(eval_args):
                        ckpt_arr_0, cidx_0, ckpt_arr_1, cidx_1, rng, rng_eval, prev_ckpt_eval_ret_info, prev_eval_ret_info = eval_args
                        eval_params_0 = {
                            "actor": {
                                "params": agent_0_train_state.actor_train_state.params,
                                "batch_stats": agent_0_train_state.actor_train_state.batch_stats
                            },
                            "q_network": {
                                "params": agent_0_train_state.q_network_train_state.params,
                                "batch_stats": agent_0_train_state.q_network_train_state.batch_stats
                            }
                        }
                        eval_params_1 = {
                            "actor": {
                                "params": agent_1_train_state.actor_train_state.params,
                                "batch_stats": agent_1_train_state.actor_train_state.batch_stats
                            },
                            "q_network": {
                                "params": agent_1_train_state.q_network_train_state.params,
                                "batch_stats": agent_1_train_state.q_network_train_state.batch_stats
                            }
                        }

                        if config["FIXED_EVAL"]:
                            eval_rng = rng_eval
                        else:
                            rng_eval, eval_rng = jax.random.split(rng_eval, 2)
                        eval_eps_last_infos = run_episodes_vmap(eval_rng, env, optimal_env, agent_idx,
                            agent_params=(eval_params_0, eval_params_1),
                            agent_policies=(agent_0_policy, agent_1_policy),
                            optimal_params=(agent_0_optimal_params, agent_1_optimal_params),
                            optimal_policies=(agent_0_optimal_policy, agent_1_optimal_policy),
                            max_episode_steps=max_episode_steps,
                            num_eps=config["NUM_EVAL_EPISODES"],
                            epsilon_optimal=config["EPSILON_OPTIMAL"],
                            use_full_obs=config["JOINT_USE_FULL_OBS"],
                            agent_test_mode=True)

                        return (ckpt_arr_0, cidx_0, ckpt_arr_1, cidx_1, rng, rng_eval, prev_ckpt_eval_ret_info, eval_eps_last_infos)
                    def skip_eval(eval_args):
                        return eval_args

                    (ckpt_arr_0, cidx_0, ckpt_arr_1, cidx_1, rng, rng_eval, prev_ckpt_eval_ret_info, eval_eps_last_infos) = jax.lax.cond(
                        config["TRAIN_EVAL"],
                        do_eval,
                        skip_eval,
                        args
                    )

                    return (ckpt_arr_0, cidx_0, ckpt_arr_1, cidx_1, rng, rng_eval, prev_ckpt_eval_ret_info, eval_eps_last_infos)

                (checkpoint_array_0, ckpt_idx_0, checkpoint_array_1, ckpt_idx_1, rng, rng_eval, ckpt_eval_eps_last_infos, eval_eps_last_infos) = jax.lax.cond(
                    to_store, store_and_eval_ckpt, skip_ckpt_and_eval, (checkpoint_array_0, ckpt_idx_0, checkpoint_array_1, ckpt_idx_1, rng, rng_eval, init_ckpt_eval_last_info, init_eval_last_info)
                )

                metric["ckpt_eval_ep_last_info"] = ckpt_eval_eps_last_infos
                metric["eval_ep_last_info"] = eval_eps_last_infos
                return ((agent_0_train_state, agent_1_train_state, rng, rng_eval, update_steps),
                         checkpoint_array_0, ckpt_idx_0, checkpoint_array_1, ckpt_idx_1, ckpt_eval_eps_last_infos, eval_eps_last_infos), metric

            init_ckpt_params_0 = {
                "actor": {
                    "params": agent_0_train_state.actor_train_state.params,
                    "batch_stats": agent_0_train_state.actor_train_state.batch_stats
                },
                "q_network": {
                    "params": agent_0_train_state.q_network_train_state.params,
                    "batch_stats": agent_0_train_state.q_network_train_state.batch_stats
                }
            }
            init_ckpt_params_1 = {
                "actor": {
                    "params": agent_1_train_state.actor_train_state.params,
                    "batch_stats": agent_1_train_state.actor_train_state.batch_stats
                },
                "q_network": {
                    "params": agent_1_train_state.q_network_train_state.params,
                    "batch_stats": agent_1_train_state.q_network_train_state.batch_stats
                }
            }
            checkpoint_array_0 = init_ckpt_array(init_ckpt_params_0)
            checkpoint_array_1 = init_ckpt_array(init_ckpt_params_1)
            ckpt_idx_0 = 0
            ckpt_idx_1 = 0

            rng, rng_train = jax.random.split(rng, 2)

            rng_eval = jax.random.PRNGKey(config["EVAL_SEED"] + agent_idx)# + 42)
            rng_eval, eval_rng = jax.random.split(rng_eval, 2)

            # Init eval return infos
            eval_eps_last_infos = run_episodes_vmap(eval_rng, env, optimal_env, agent_idx,
                                    agent_params=(init_ckpt_params_0, init_ckpt_params_1),
                                    agent_policies=(agent_0_policy, agent_1_policy),
                                    optimal_params=(agent_0_optimal_params, agent_1_optimal_params),
                                    optimal_policies=(agent_0_optimal_policy, agent_1_optimal_policy),
                                    max_episode_steps=max_episode_steps,
                                    num_eps=config["NUM_EVAL_EPISODES"],
                                    epsilon_optimal=config["EPSILON_OPTIMAL"],
                                    use_full_obs=config["JOINT_USE_FULL_OBS"],
                                    agent_test_mode=True)

            # initial runner state for scanning
            update_steps = 0

            update_runner_state = (agent_0_train_state, agent_1_train_state, rng_train, rng_eval, update_steps)
            state_with_ckpt = (update_runner_state, checkpoint_array_0, ckpt_idx_0, checkpoint_array_1, ckpt_idx_1, eval_eps_last_infos, eval_eps_last_infos)

            state_with_ckpt, metrics = jax.lax.scan(
                _update_step_with_ckpt,
                state_with_ckpt,
                xs=None,
                length=config["NUM_UPDATES"]
            )

            (final_runner_state, checkpoint_array_0, final_ckpt_idx_0, checkpoint_array_1, final_ckpt_idx_1, ckpt_eval_eps_last_infos, eval_eps_last_infos) = state_with_ckpt
            final_train_state_0 = final_runner_state[0]
            final_train_state_1 = final_runner_state[1]
            final_params_0 = {
                "actor": {
                    "params": final_train_state_0.actor_train_state.params,
                    "batch_stats": final_train_state_0.actor_train_state.batch_stats
                },
                "q_network": {
                    "params": final_train_state_0.q_network_train_state.params,
                    "batch_stats": final_train_state_0.q_network_train_state.batch_stats
                }
            }
            final_params_1 = {
                "actor": {
                    "params": final_train_state_1.actor_train_state.params,
                    "batch_stats": final_train_state_1.actor_train_state.batch_stats
                },
                "q_network": {
                    "params": final_train_state_1.q_network_train_state.params,
                    "batch_stats": final_train_state_1.q_network_train_state.batch_stats
                }
            }
            out = {
                "final_params_agent_0": final_params_0,
                "final_params_agent_1": final_params_1,
                "metrics": metrics,  # shape (NUM_UPDATES, ...)
                "checkpoints_agent_0": checkpoint_array_0,
                "checkpoints_agent_1": checkpoint_array_1,
            }

            if env._render:
                # Collect final eval gifs for logging
                rng_eval = final_runner_state[3] # extract final rng_eval from the final runner state after training
                if config["FIXED_EVAL"]:
                    eval_rng = rng_eval
                else:
                    rng_eval, eval_rng = jax.random.split(rng_eval, 2)
                out["render_outs"] = run_episodes_vmap(eval_rng, env, optimal_env, agent_idx,
                                        agent_params=(final_params_0, final_params_1),
                                        agent_policies=(agent_0_policy, agent_1_policy),
                                        optimal_params=(agent_0_optimal_params, agent_1_optimal_params),
                                        optimal_policies=(agent_0_optimal_policy, agent_1_optimal_policy),
                                        max_episode_steps=env.horizon,
                                        num_eps=5, epsilon_optimal=config["EPSILON_OPTIMAL"],
                                        use_full_obs=config["JOINT_USE_FULL_OBS"],
                                        render=True, agent_test_mode=True)

            return out
        return train

    # ------------------------------
    # Actually run the REPPO training
    # ------------------------------
    rngs = jax.random.split(train_rng, config["NUM_TRAIN_SEEDS"])

    # Run training seeds in parallel using vmap
    train_fn = make_reppo_joint_train(config)
    out = jax.vmap(train_fn, in_axes=(0, None, 0, 0))(rngs, agent_idx, agent_0_optimal_params, agent_1_optimal_params)
    return out

def run_training(config, wandb_logger, optimal_params, optimal_policies,
                agent_idx=0):
    '''Run joint training.

    Args:
        config: dict, config for the training
        wandb_logger: Logger, logger for logging metrics
        optimal_params: tuple, optimal parameters for all agents
        optimal_policies: tuple, optimal policies for all agents
        agent_idx: int, index of the agent to optimize
    '''
    algorithm_config = dict(config["algorithm"])

    VMIN = algorithm_config["VMIN"]
    VMAX = algorithm_config["VMAX"]

    algorithm_config = algorithm_config.copy()
    algorithm_config["VMIN"] = VMAX * -1
    algorithm_config["VMAX"] = VMIN * -1

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

    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"] + agent_idx)# + 35)
    _, init_rng, train_rng = jax.random.split(rng, 3)

    # Initialize agent
    agent_0_policy, agent_0_init_params = initialize_reppo_agent(algorithm_config, env, init_rng, agent_index=0, observation_type="full" if algorithm_config["JOINT_USE_FULL_OBS"] else "agent")
    agent_1_policy, agent_1_init_params = initialize_reppo_agent(algorithm_config, env, init_rng, agent_index=1, observation_type="full" if algorithm_config["JOINT_USE_FULL_OBS"] else "agent")

    # Squeeze REPPO params to remove leading dimension for compatibility with single-agent training
    agent_0_optimal_params, agent_1_optimal_params = optimal_params
    agent_0_optimal_policy, agent_1_optimal_policy = optimal_policies
    # agent_0_reppo_params = jax.tree.map(lambda x: x.squeeze(axis=0), agent_0_reppo_params)
    # agent_1_reppo_params = jax.tree.map(lambda x: x.squeeze(axis=0), agent_1_reppo_params)

    log.info(f"Starting REPPO joint training optimizing for agent {agent_idx}...")
    start_time = time.perf_counter()

    # Run the training
    out = train_reppo_joint_agents(
        config=algorithm_config,
        env=env,
        optimal_env=optimal_env,
        train_rng=train_rng,
        joint_policies=(agent_0_policy, agent_1_policy),
        init_joint_params=(agent_0_init_params, agent_1_init_params),
        optimal_policies=(agent_0_optimal_policy, agent_1_optimal_policy),
        optimal_params=(agent_0_optimal_params, agent_1_optimal_params),
        agent_idx=agent_idx
    )

    elapsed_time = time.perf_counter() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, rem = divmod(rem, 60)
    seconds, rem = divmod(rem, 1)
    milliseconds = int(rem * 1000)
    microseconds = int((rem * 1_000_000) % 1000)
    log.info(f"REPPO Joint Training completed optimizing for agent {agent_idx} in {elapsed_time:.2f}s")
    log.info(f"REPPO Joint Training completed optimizing for agent {agent_idx} in {int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s {milliseconds:03d}ms {microseconds:03d}µs")

    # process and log metrics
    log.info(f"Starting REPPO joint logging optimizing for agent {agent_idx}...")
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
    log.info(f"REPPO Joint Logging completed optimizing for agent {agent_idx} in {elapsed_time:.2f}s")
    log.info(f"REPPO Joint Logging completed optimizing for agent {agent_idx} in {int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s {milliseconds:03d}ms {microseconds:03d}µs")

    return (out["final_params_agent_0"], out["final_params_agent_1"]), (agent_0_policy, agent_1_policy), (agent_0_init_params, agent_1_init_params)

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

    # Train metrics include loss values and gradient norms, which we can average across seeds, partners and minibatches for each update step.
    all_agent_0_actor_loss = np.asarray(train_metrics["agent_0/actor_loss"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    all_agent_0_target_entropy_loss = np.asarray(train_metrics["agent_0/target_entropy_loss"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    all_agent_0_lagrangian_loss = np.asarray(train_metrics["agent_0/lagrangian_loss"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    all_agent_0_alpha = np.asarray(train_metrics["agent_0/alpha"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    all_agent_0_lagrangian = np.asarray(train_metrics["agent_0/lagrangian"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    all_agent_0_kl = np.asarray(train_metrics["agent_0/kl"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    all_agent_0_entropy = np.asarray(train_metrics["agent_0/entropy"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    all_agent_0_critic_loss = np.asarray(train_metrics["agent_0/critic_loss"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    all_agent_0_q_values = np.asarray(train_metrics["agent_0/q_values"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    all_agent_0_q_error = np.asarray(train_metrics["agent_0/q_error"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)

    all_agent_1_actor_loss = np.asarray(train_metrics["agent_1/actor_loss"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    all_agent_1_target_entropy_loss = np.asarray(train_metrics["agent_1/target_entropy_loss"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    all_agent_1_lagrangian_loss = np.asarray(train_metrics["agent_1/lagrangian_loss"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    all_agent_1_alpha = np.asarray(train_metrics["agent_1/alpha"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    all_agent_1_lagrangian = np.asarray(train_metrics["agent_1/lagrangian"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    all_agent_1_kl = np.asarray(train_metrics["agent_1/kl"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    all_agent_1_entropy = np.asarray(train_metrics["agent_1/entropy"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    all_agent_1_critic_loss = np.asarray(train_metrics["agent_1/critic_loss"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    all_agent_1_q_values = np.asarray(train_metrics["agent_1/q_values"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    all_agent_1_q_error = np.asarray(train_metrics["agent_1/q_error"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)

    # Process loss metrics - average across train seeds, partners and minibatches dims
    average_agent_0_actor_losses = np.mean(all_agent_0_actor_loss, axis=(0, 2, 3))  # shape (num_updates,)
    average_agent_0_target_entropy_losses = np.mean(all_agent_0_target_entropy_loss, axis=(0, 2, 3))  # shape (num_updates,)
    average_agent_0_lagrangian_losses = np.mean(all_agent_0_lagrangian_loss, axis=(0, 2, 3))  # shape (num_updates,)
    average_agent_0_alphas = np.mean(all_agent_0_alpha, axis=(0, 2, 3))  # shape (num_updates,)
    average_agent_0_lagrangians = np.mean(all_agent_0_lagrangian, axis=(0, 2, 3))  # shape (num_updates,)
    average_agent_0_kls = np.mean(all_agent_0_kl, axis=(0, 2, 3))  # shape (num_updates,)
    average_agent_0_entropies = np.mean(all_agent_0_entropy, axis=(0, 2, 3))  # shape (num_updates,)
    average_agent_0_critic_losses = np.mean(all_agent_0_critic_loss, axis=(0, 2, 3))  # shape (num_updates,)
    average_agent_0_q_values = np.mean(all_agent_0_q_values, axis=(0, 2, 3))  # shape (num_updates,)
    average_agent_0_q_errors = np.mean(all_agent_0_q_error, axis=(0, 2, 3))  # shape (num_updates,)

    average_agent_1_actor_losses = np.mean(all_agent_1_actor_loss, axis=(0, 2, 3))  # shape (num_updates,)
    average_agent_1_target_entropy_losses = np.mean(all_agent_1_target_entropy_loss, axis=(0, 2, 3))  # shape (num_updates,)
    average_agent_1_lagrangian_losses = np.mean(all_agent_1_lagrangian_loss, axis=(0, 2, 3))  # shape (num_updates,)
    average_agent_1_alphas = np.mean(all_agent_1_alpha, axis=(0, 2, 3))  # shape (num_updates,)
    average_agent_1_lagrangians = np.mean(all_agent_1_lagrangian, axis=(0, 2, 3))  # shape (num_updates,)
    average_agent_1_kls = np.mean(all_agent_1_kl, axis=(0, 2, 3))  # shape (num_updates,)
    average_agent_1_entropies = np.mean(all_agent_1_entropy, axis=(0, 2, 3))  # shape (num_updates,)
    average_agent_1_critic_losses = np.mean(all_agent_1_critic_loss, axis=(0, 2, 3))  # shape (num_updates,)
    average_agent_1_q_values = np.mean(all_agent_1_q_values, axis=(0, 2, 3))  # shape (num_updates,)
    average_agent_1_q_errors = np.mean(all_agent_1_q_error, axis=(0, 2, 3))  # shape (num_updates,)

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
    num_updates = len(average_agent_1_actor_losses)
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

        logger.log_item(f"Train/Joint/Agent_{agent_idx + 1}_Optimize/Agent_1/ActorLoss", average_agent_0_actor_losses[step], train_step=step, commit=True)
        logger.log_item(f"Train/Joint/Agent_{agent_idx + 1}_Optimize/Agent_1/TargetEntropyLoss", average_agent_0_target_entropy_losses[step], train_step=step, commit=True)
        logger.log_item(f"Train/Joint/Agent_{agent_idx + 1}_Optimize/Agent_1/LagrangianLoss", average_agent_0_lagrangian_losses[step], train_step=step, commit=True)
        logger.log_item(f"Train/Joint/Agent_{agent_idx + 1}_Optimize/Agent_1/Alpha", average_agent_0_alphas[step], train_step=step, commit=True)
        logger.log_item(f"Train/Joint/Agent_{agent_idx + 1}_Optimize/Agent_1/Lagrangian", average_agent_0_lagrangians[step], train_step=step, commit=True)
        logger.log_item(f"Train/Joint/Agent_{agent_idx + 1}_Optimize/Agent_1/KL", average_agent_0_kls[step], train_step=step, commit=True)
        logger.log_item(f"Train/Joint/Agent_{agent_idx + 1}_Optimize/Agent_1/Entropy", average_agent_0_entropies[step], train_step=step, commit=True)
        logger.log_item(f"Train/Joint/Agent_{agent_idx + 1}_Optimize/Agent_1/CriticLoss", average_agent_0_critic_losses[step], train_step=step, commit=True)
        logger.log_item(f"Train/Joint/Agent_{agent_idx + 1}_Optimize/Agent_1/QValueMean", average_agent_0_q_values[step], train_step=step, commit=True)
        logger.log_item(f"Train/Joint/Agent_{agent_idx + 1}_Optimize/Agent_1/QValueError", average_agent_0_q_errors[step], train_step=step, commit=True)

        logger.log_item(f"Train/Joint/Agent_{agent_idx + 1}_Optimize/Agent_2/ActorLoss", average_agent_1_actor_losses[step], train_step=step, commit=True)
        logger.log_item(f"Train/Joint/Agent_{agent_idx + 1}_Optimize/Agent_2/TargetEntropyLoss", average_agent_1_target_entropy_losses[step], train_step=step, commit=True)
        logger.log_item(f"Train/Joint/Agent_{agent_idx + 1}_Optimize/Agent_2/LagrangianLoss", average_agent_1_lagrangian_losses[step], train_step=step, commit=True)
        logger.log_item(f"Train/Joint/Agent_{agent_idx + 1}_Optimize/Agent_2/Alpha", average_agent_1_alphas[step], train_step=step, commit=True)
        logger.log_item(f"Train/Joint/Agent_{agent_idx + 1}_Optimize/Agent_2/Lagrangian", average_agent_1_lagrangians[step], train_step=step, commit=True)
        logger.log_item(f"Train/Joint/Agent_{agent_idx + 1}_Optimize/Agent_2/KL", average_agent_1_kls[step], train_step=step, commit=True)
        logger.log_item(f"Train/Joint/Agent_{agent_idx + 1}_Optimize/Agent_2/Entropy", average_agent_1_entropies[step], train_step=step, commit=True)
        logger.log_item(f"Train/Joint/Agent_{agent_idx + 1}_Optimize/Agent_2/CriticLoss", average_agent_1_critic_losses[step], train_step=step, commit=True)
        logger.log_item(f"Train/Joint/Agent_{agent_idx + 1}_Optimize/Agent_2/QValueMean", average_agent_1_q_values[step], train_step=step, commit=True)
        logger.log_item(f"Train/Joint/Agent_{agent_idx + 1}_Optimize/Agent_2/QValueError", average_agent_1_q_errors[step], train_step=step, commit=True)
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

    agent_0_out_savepath = save_train_run(train_out_agent_0, savedir, savename=f"REPPO_Agent_{agent_idx + 1}_Optimize_Train_Run-Agent_1")
    agent_1_out_savepath = save_train_run(train_out_agent_1, savedir, savename=f"REPPO_Agent_{agent_idx + 1}_Optimize_Train_Run-Agent_2")
    if config["logger"]["log_train_out"]:
        logger.log_artifact(name=f"REPPO_Agent_{agent_idx + 1}_Optimize_Train_Run-Agent_1", path=agent_0_out_savepath, type_name="joint_train_run")
        logger.log_artifact(name=f"REPPO_Agent_{agent_idx + 1}_Optimize_Train_Run-Agent_2", path=agent_1_out_savepath, type_name="joint_train_run")
        # Cleanup locally logged out file
    if not config["local_logger"]["save_train_out"]:
        shutil.rmtree(agent_0_out_savepath)
        shutil.rmtree(agent_1_out_savepath)

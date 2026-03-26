'''
Script for training a CREPPO agent for social laws single agent projection.
Does not sucrepport training against heuristic partner agents.

Command to run CREPPO single agent projection training:
python social_laws/run.py algorithm=creppo/lbf task=lbf label=test_creppo_single_agent_projection

Suggested debug command:
python social_laws/run.py algorithm=creppo/lbf task=lbf logger.mode=disabled label=debug algorithm.TOTAL_TIMESTEPS=1e5
'''
from email import policy
import os
import shutil
import time
import logging

from copy import deepcopy
from functools import partial

import distrax
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import open_dict
import optax
import hydra

from flax.linen import softmax, log_softmax

from social_laws.common.initialize_agents import initialize_creppo_agent
from social_laws.common.run_episodes_creppo_w_robustness_centralized import run_episodes_vmap
from common.plot_utils import get_stats, get_metric_names
from common.save_load_utils import save_train_run
from envs import make_env
from envs.log_wrapper import LogWrapper
from marl.ppo_utils import unbatchify, batchify

from agents.mlp_creppo import hl_gauss
from agents.mlp_creppo_agent import CReppoTrainState, CustomTrainState, Transition

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def train_creppo_joint_agents(config, env, optimal_env, train_rng,
                           joint_policy, init_joint_params,
                           optimal_policies, optimal_params,
                           agent_idx):
    '''
    Train CREPPO joint agents using the given initial parameters with centralized learning.

    Args:
        config: dict, config for the training
        env: gymnasium environment
        optimal_env: gymnasium environment for evaluating optimal returns
        train_rng: jax.random.PRNGKey, random key for training
        joint_policy: AgentPolicy, single centralized policy for all agents
        init_joint_params: dict, initial parameters for the centralized joint policy
        optimal_policies: tuple of AgentPolicy, optimal policies for the agents
        optimal_params: tuple of dict, optimal parameters for the agents
        agent_idx: int, index of the agent to optimize
    '''
    # ------------------------------
    # Build the CREPPO joint training function
    # ------------------------------

    def make_creppo_joint_train(config):
        num_agents = env.num_agents
        # assert num_agents == 2, "This snippet assumes exactly 2 agents."

        config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
        # config["NUM_UNCONTROLLED_ACTORS"] = config["NUM_ENVS"] # assumption: we control 1 agent
        config["NUM_CONTROLLED_ACTORS"] = config["NUM_ENVS"]
        config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["ROLLOUT_LENGTH"] // config["NUM_ENVS"]
        config["NUM_UPDATES_DECAY"] = (config["TOTAL_TIMESTEPS_DECAY"] // config["ROLLOUT_LENGTH"] // config["NUM_ENVS"])
        assert config["NUM_CONTROLLED_ACTORS"] % config["NUM_MINIBATCHES"] == 0, "NUM_CONTROLLED_ACTORS must be divisible by NUM_MINIBATCHES"
        assert config["NUM_CONTROLLED_ACTORS"] >= config["NUM_MINIBATCHES"], "NUM_CONTROLLED_ACTORS must be >= NUM_MINIBATCHES"

        config["MAX_EPISODE_STEPS"] = env.horizon
        config["NUM_ACTIONS"] = env.action_space(f"agent_{agent_idx}").n

        lr_scheduler = optax.linear_schedule(
            init_value=config["LR"],
            end_value=1e-20,
            transition_steps=(config["NUM_UPDATES_DECAY"])
            * config["NUM_MINIBATCHES"]
            * config["NUM_EPOCHS"],
        )

        def train(rng, agent_idx, *optimal_params):
            original_seed = rng

            if config["ANNEAL_LR"]: # config['LR_LINEAR_DECAY']
                tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(learning_rate=lr_scheduler),
                )
            else:
                tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(learning_rate=config["LR"]),
                )

            q_network_ts = CustomTrainState.create(
                apply_fn=joint_policy.q_network.apply,
                params=init_joint_params["params"],
                batch_stats=init_joint_params["batch_stats"],
                tx=tx,
            )
            target_ts = CustomTrainState.create(
                apply_fn=joint_policy.q_network.apply,
                params=deepcopy(init_joint_params["params"]),
                batch_stats=deepcopy(init_joint_params["batch_stats"]),
                tx=optax.set_to_zero(),
            )
            joint_train_state = CReppoTrainState(
                timesteps=0,
                n_updates=0,
                grad_steps=0,
                q_network_train_state=q_network_ts,
                target_train_state=target_ts,
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
            optimal_init_hstates = [optimal_policies[i].init_hstate(config["NUM_CONTROLLED_ACTORS"]) for i in range(num_agents)]

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
                critic_outs, new_hstate = optimal_policy.get_critic_out(
                    params=(optimal_params["params"], optimal_params["batch_stats"]),
                    obs=obs,
                    done=done,
                    avail_actions=avail_actions,
                    hstate=hstate,
                    rng=jax.random.PRNGKey(0)
                ) # (num_envs, num_actions)

                # Find max Q-value for each environment
                max_q = jnp.max(critic_outs["q_values"], axis=-1, keepdims=True)  # (num_envs, 1)

                # Create mask for actions within epsilon of max
                optimal_action_mask = (critic_outs["q_values"] >= (max_q - config['EPSILON_OPTIMAL'])).astype(jnp.float32)

                # Combine with original available actions
                restricted_avail_actions = optimal_action_mask * avail_actions

                # remove extra batch dim
                return restricted_avail_actions, new_hstate

            def _env_step(runner_state, unused):
                """
                One step of the environment:
                1. Get observations, sample actions from centralized joint policy
                2. Step environment using sampled actions
                3. Return state, reward, ...
                """
                joint_train_state, env_state, prev_obs, prev_done, joint_hstate, optimal_hstates, rng = runner_state
                prev_obs, prev_full_obs = prev_obs
                rng, actor_rng, target_rng, step_rng = jax.random.split(rng, 4)

                # Get available actions for the agent from environment state
                avail_actions = env.get_avail_actions(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(avail_actions)

                prev_obs_per_agent = [get_agent_data(prev_obs, i).reshape(1, config["NUM_CONTROLLED_ACTORS"], -1) for i in range(num_agents)]
                prev_full_obs_per_agent = [get_agent_data(prev_full_obs, i).reshape(1, config["NUM_CONTROLLED_ACTORS"], -1) for i in range(num_agents)]
                prev_done_per_agent = [get_agent_data(prev_done, i).reshape(1, config["NUM_CONTROLLED_ACTORS"]) for i in range(num_agents)]
                avail_actions_per_agent = [get_agent_data(avail_actions, i).astype(jnp.float32) for i in range(num_agents)]

                # Restrict available actions based on optimal value function
                optimal_restricted_avail_actions = []
                new_optimal_hstates = []
                for i in range(num_agents):
                    opt_restricted_i, new_opt_hstate_i = _get_optimal_restricted_avail_actions(
                        obs=prev_obs_per_agent[i],
                        done=prev_done_per_agent[i],
                        avail_actions=avail_actions_per_agent[i],
                        hstate=optimal_hstates[i],
                        optimal_params=optimal_params[i],
                        optimal_policy=optimal_policies[i]
                    )
                    optimal_restricted_avail_actions.append(opt_restricted_i)
                    new_optimal_hstates.append(new_opt_hstate_i)

                # Centralized joint observation - concatenate all agents' observations
                joint_prev_obs = batchify(prev_obs, env.agents, config["NUM_ACTORS"])
                joint_prev_full_obs = batchify(prev_full_obs, env.agents, config["NUM_ACTORS"])
                joint_prev_done = batchify(prev_done, env.agents, config["NUM_ACTORS"]).squeeze(axis=-1)

                # Stack restricted available actions for all agents
                joint_optimal_restricted_avail_actions = jnp.stack(optimal_restricted_avail_actions, axis=0).reshape(config["NUM_ACTORS"], -1)

                # Joint policy outputs actions for all agents at once
                joint_act, _, _, new_joint_hstate = joint_policy.get_action_value_policy(
                    params=(joint_train_state.q_network_train_state.params, joint_train_state.q_network_train_state.batch_stats),
                    obs=jnp.expand_dims(joint_prev_full_obs, axis=0) if config["JOINT_USE_FULL_OBS"] else jnp.expand_dims(joint_prev_obs, axis=0),
                    done=jnp.expand_dims(joint_prev_done, axis=0),
                    avail_actions=joint_optimal_restricted_avail_actions,
                    hstate=joint_hstate,
                    rng=actor_rng,
                )

                # Convert to env format
                env_act = unbatchify(joint_act, env.agents, config["NUM_ENVS"], num_agents)

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

                negative_reward = get_agent_data(reward, agent_idx) * -1

                # Next values - for computing targets
                next_avail_actions = env.get_avail_actions(env_state_next.env_state)
                next_avail_actions = jax.lax.stop_gradient(next_avail_actions)
                next_avail_per_agent = [get_agent_data(next_avail_actions, i).astype(jnp.float32) for i in range(num_agents)]

                obs_next_per_agent = [get_agent_data(obs_next, i).reshape(1, config["NUM_CONTROLLED_ACTORS"], -1) for i in range(num_agents)]
                obs_full_next_per_agent = [get_agent_data(obs_full_next, i).reshape(1, config["NUM_CONTROLLED_ACTORS"], -1) for i in range(num_agents)]
                done_next_per_agent = [get_agent_data(done_next, i).reshape(1, config["NUM_CONTROLLED_ACTORS"]) for i in range(num_agents)]

                # Restrict next available actions for computing next values
                next_optimal_restricted_avail_actions = []
                for i in range(num_agents):
                    next_opt_restricted_i, _ = _get_optimal_restricted_avail_actions(
                        obs=obs_next_per_agent[i],
                        done=done_next_per_agent[i],
                        avail_actions=next_avail_per_agent[i],
                        hstate=new_optimal_hstates[i],
                        optimal_params=optimal_params[i],
                        optimal_policy=optimal_policies[i]
                    )
                    next_optimal_restricted_avail_actions.append(next_opt_restricted_i)

                # Centralized next observations
                joint_obs_next = batchify(obs_next, env.agents, config["NUM_ACTORS"])
                joint_obs_full_next = batchify(obs_full_next, env.agents, config["NUM_ACTORS"])
                joint_done_next = batchify(done_next, env.agents, config["NUM_ACTORS"]).squeeze(axis=-1)

                # Stack restricted next available actions
                joint_next_optimal_restricted_avail_actions = jnp.stack(next_optimal_restricted_avail_actions, axis=0).reshape(config["NUM_ACTORS"], -1)

                # Get next Q-values from joint policy for target computation
                next_q, _ = joint_policy.get_critic_out(
                    params=(joint_train_state.q_network_train_state.params, joint_train_state.q_network_train_state.batch_stats),
                    obs=jnp.expand_dims(joint_obs_full_next, axis=0) if config["JOINT_USE_FULL_OBS"] else jnp.expand_dims(joint_obs_next, axis=0),
                    done=jnp.expand_dims(joint_done_next, axis=0),
                    avail_actions=joint_next_optimal_restricted_avail_actions,
                    hstate=new_joint_hstate,
                    rng=target_rng,
                )

                q_probs = softmax(
                    next_q["policy_logits"],
                    axis=-1,
                )
                next_values = jnp.sum(
                    q_probs * next_q["q_values"],
                    axis=-1,
                )

                # Compute entropy for the joint policy
                entropy = -jnp.sum(jnp.where(joint_next_optimal_restricted_avail_actions, q_probs * jnp.log(q_probs + 1e-8), 0), axis=-1) * config["NUM_ACTIONS"] / joint_next_optimal_restricted_avail_actions.sum(-1)

                # Store centralized joint transition data
                joint_transition = Transition(
                    obs=joint_prev_full_obs if config["JOINT_USE_FULL_OBS"] else joint_prev_obs,
                    action=joint_act,
                    action_logp=entropy,
                    reward=jnp.tile(negative_reward, num_agents),
                    done=batchify(done_next, env.agents, config["NUM_ACTORS"]).squeeze(axis=-1),
                    avail_actions=joint_optimal_restricted_avail_actions,
                    next_obs=joint_obs_full_next if config["JOINT_USE_FULL_OBS"] else joint_obs_next,
                    next_avail_actions=joint_next_optimal_restricted_avail_actions,
                    next_val=next_values,
                    info=info
                )

                new_runner_state = (joint_train_state, env_state_next, (obs_next, obs_full_next), done_next,
                                    new_joint_hstate, new_optimal_hstates, rng)
                return new_runner_state, joint_transition

            def compute_nstep_lambda(carry, transition):
                lambda_return, truncated = carry

                # Combine importance_weights with TD lambda
                done = transition.done
                reward = transition.soft_reward
                value = transition.next_val

                lambda_sum = (
                    config["LAMBDA"] * lambda_return + (1 - config["LAMBDA"]) * value
                )

                delta = config["GAMMA"] * jnp.where(
                    truncated, value, (1.0 - done) * lambda_sum
                )

                lambda_return = reward + delta

                return (
                    lambda_return,
                    jnp.zeros_like(truncated),
                ), lambda_return

            def _update_step(update_runner_state, unused):
                """
                1. Collect rollouts
                2. Compute soft rewards and advantage
                3. CREPPO updates for centralized joint policy
                """
                (joint_train_state, rng, rng_eval, update_steps) = update_runner_state
                # Init envs & partner indices
                rng, reset_rng = jax.random.split(rng, 2)
                reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
                init_obs, init_env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rngs)
                init_done = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}

                # 1) rollout
                runner_state = (joint_train_state, init_env_state, init_obs, init_done, joint_init_hstate, optimal_init_hstates, rng)

                runner_state, joint_traj_batch = jax.lax.scan(
                    _env_step, runner_state, None, config["ROLLOUT_LENGTH"])
                (joint_train_state, env_state, obs, done, joint_hstate, optimal_hstates, rng) = runner_state

                joint_train_state = joint_train_state.replace(
                    timesteps=joint_train_state.timesteps + config["ROLLOUT_LENGTH"] * config["NUM_ACTORS"]
                )

                # 2) Compute soft rewards and n-step lambda returns for centralized joint policy
                rng, sample_rng = jax.random.split(rng)
                next_pi, _ = joint_policy.get_critic_out(
                    params=(joint_train_state.q_network_train_state.params, joint_train_state.q_network_train_state.batch_stats),
                    obs=jnp.expand_dims(joint_traj_batch.next_obs[-1], axis=0),
                    done=jnp.expand_dims(joint_traj_batch.done[-1], axis=0),
                    avail_actions=joint_traj_batch.next_avail_actions[-1],
                    hstate=joint_hstate,
                    rng=sample_rng
                )

                next_pi_probs = softmax(next_pi["policy_logits"], axis=-1)

                next_pi_ent = -jnp.sum(
                    jnp.where(joint_traj_batch.next_avail_actions[-1], next_pi_probs * jnp.log(next_pi_probs + 1e-8), 0), axis=-1
                ) * config["NUM_ACTIONS"] / joint_traj_batch.next_avail_actions[-1].sum(-1)
                next_logp = jnp.concatenate(
                    [joint_traj_batch.action_logp[1:], next_pi_ent[None]], axis=0
                )

                soft_reward = joint_traj_batch.reward + config["GAMMA"] * next_logp * jnp.exp(
                    joint_train_state.q_network_train_state.params["log_alpha"]
                )

                joint_traj_batch = joint_traj_batch.replace(soft_reward=soft_reward)
                _, target_values = jax.lax.scan(
                    compute_nstep_lambda,
                    (
                        joint_traj_batch.next_val[-1],
                        jnp.ones_like(joint_traj_batch.done[0])
                    ),
                    joint_traj_batch,
                    reverse=True,
                )

                # 3) CREPPO update for centralized joint policy
                def _learn_epoch(carry, _):
                    train_state, rng = carry

                    def _learn_phase(carry, minibatch_and_target):
                        train_state, rng = carry
                        minibatch, target = minibatch_and_target

                        def _critic_loss_fn(params, train_state):
                            critic_out, updates = joint_policy.q_network.apply(
                                {
                                    "params": params,
                                    "batch_stats": train_state.q_network_train_state.batch_stats,
                                },
                                (minibatch.obs, minibatch.avail_actions),
                                train=True,
                                mutable=["batch_stats"],
                            ) # TODO: Deal with recurrent network

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
                            pi = jax.lax.stop_gradient(
                                distrax.Categorical(
                                    logits=critic_out["policy_logits"]
                                )
                            )

                            alpha = jnp.exp(params["log_alpha"])
                            target_entropy = config["TARGET_ENTROPY_MULT"] * jnp.log(minibatch.avail_actions.sum(axis=-1)).mean()

                            pi_entropy = -jnp.sum(
                                jnp.where(minibatch.avail_actions, pi.probs * jnp.log(pi.probs + 1e-8), 0), axis=-1
                            ) * config["NUM_ACTIONS"] / minibatch.avail_actions.sum(-1)
                            alpha_loss = -alpha * jax.lax.stop_gradient(target_entropy - pi_entropy)
                            loss = optax.softmax_cross_entropy(logits, target_cat)
                            old_critic_out = joint_policy.q_network.apply(
                                {
                                    "params": train_state.target_train_state.params,
                                    "batch_stats": train_state.target_train_state.batch_stats,
                                },
                                (minibatch.obs, minibatch.avail_actions),
                                train=False,
                            ) # TODO: Deal with recurrent network
                            old_q_probs = old_critic_out["policy_logits"]
                            kl = jnp.sum(
                                jnp.where(minibatch.avail_actions, softmax(old_q_probs, axis=-1) * (log_softmax(old_q_probs + 1e-8, axis=-1) - log_softmax(critic_out["policy_logits"] + 1e-8, axis=-1)), 0), axis=-1
                            ) * config["NUM_ACTIONS"] / minibatch.avail_actions.sum(-1)
                            if config["KL_BOUND"] is not None:
                                loss = jnp.where(
                                    jax.lax.stop_gradient(kl) < config["KL_BOUND"], loss, jnp.zeros_like(loss)
                                )
                            loss = loss.mean()

                            metrics = {
                                "critic_loss": loss,
                                "q_values": q_vals.mean(),
                                "q_error": optax.l2_loss(target - q_vals).mean(),
                                "alpha_loss": jnp.mean(alpha_loss),
                                "alpha": alpha,
                                "kl": kl.mean(),
                                "entropy": pi.entropy().mean(),
                            }
                            return loss + jnp.mean(alpha_loss), dict(
                                updates=updates, metrics=metrics
                            )

                        (_, critic_update_output), grads = jax.value_and_grad(
                            _critic_loss_fn, has_aux=True
                        )(train_state.q_network_train_state.params, train_state)
                        q_network_train_state = (
                            train_state.q_network_train_state.apply_gradients(grads=grads)
                        )

                        train_state = train_state.replace(
                            q_network_train_state=q_network_train_state.replace(
                                batch_stats=critic_update_output["updates"]["batch_stats"]
                            ),
                            grad_steps=train_state.grad_steps + 1,
                        )
                        return (train_state, rng), {
                            **critic_update_output["metrics"],
                        }

                    def preprocess_transition(x, rng):
                        x = x.reshape(
                            -1, *x.shape[2:]
                        )  # num_steps*num_actors (batch_size), ...
                        x = jax.random.permutation(rng, x)  # shuffle the transitions
                        x = x.reshape(
                            config["NUM_MINIBATCHES"], -1, *x.shape[1:]
                        )  # num_mini_updates, batch_size/num_mini_updates, ...
                        return x

                    rng, _rng = jax.random.split(rng)
                    minibatches = jax.tree_util.tree_map(
                        lambda x: preprocess_transition(x, _rng), joint_traj_batch
                    )
                    targets = jax.tree_util.tree_map(
                        lambda x: preprocess_transition(x, _rng), target_values
                    )

                    rng, _rng = jax.random.split(rng)
                    (train_state, rng), metrics = jax.lax.scan(
                        _learn_phase, init=(train_state, rng), xs=(minibatches, targets)
                    )

                    return (train_state, rng), metrics

                # Update target network and run learning epochs
                joint_train_state = joint_train_state.replace(
                    target_train_state=joint_train_state.target_train_state.replace(
                        params=joint_train_state.q_network_train_state.params,
                        batch_stats=joint_train_state.q_network_train_state.batch_stats,
                    )
                )

                rng, learn_rng = jax.random.split(rng)
                (joint_train_state, rng), learn_metrics = jax.lax.scan(
                    f=_learn_epoch,
                    init=(joint_train_state, learn_rng),
                    xs=None,
                    length=config["NUM_EPOCHS"],
                )

                joint_train_state = joint_train_state.replace(
                    n_updates=joint_train_state.n_updates + 1,
                )

                metric = joint_traj_batch.info
                metric["update_steps"] = update_steps
                metric.update(learn_metrics)
                new_runner_state = (joint_train_state, rng, rng_eval, update_steps + 1)
                return (new_runner_state, metric)

            # CREPPO Update and Checkpoint saving
            ckpt_and_eval_interval = config["NUM_UPDATES"] // max(1, config["NUM_CHECKPOINTS"] - 1)  # -1 because we store a ckpt at the last update
            num_ckpts = config["NUM_CHECKPOINTS"]

            # Build a PyTree that holds parameters for all FCP checkpoints
            def init_ckpt_array(params_pytree):
                return jax.tree.map(
                    lambda x: jnp.zeros((num_ckpts,) + x.shape, x.dtype),
                    params_pytree)

            max_episode_steps = config["MAX_EPISODE_STEPS"]

            def _update_step_with_ckpt(state_with_ckpt, unused):
                (update_state, checkpoint_array, ckpt_idx, init_ckpt_eval_last_info, init_eval_last_info) = state_with_ckpt

                # Single CREPPO update
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
                    ckpt_params = {
                        "params": joint_train_state.q_network_train_state.params,
                        "batch_stats": joint_train_state.q_network_train_state.batch_stats
                    }
                    new_ckpt_arr = jax.tree.map(
                        lambda c_arr, p: c_arr.at[cidx].set(p),
                        ckpt_arr, ckpt_params
                    )

                    if config["FIXED_EVAL"]:
                        eval_rng = rng_eval
                    else:
                        rng_eval, eval_rng = jax.random.split(rng_eval, 2)
                    ckpt_eval_eps_last_infos = run_episodes_vmap(eval_rng, env, optimal_env, agent_idx,
                        agent_params=ckpt_params,
                        agent_policy=joint_policy,
                        optimal_params=optimal_params,
                        optimal_policies=optimal_policies,
                        max_episode_steps=max_episode_steps,
                        num_eps=config["NUM_EVAL_EPISODES"],
                        epsilon_optimal=config["EPSILON_OPTIMAL"],
                        use_full_obs=config["JOINT_USE_FULL_OBS"],
                        agent_test_mode=True)

                    return (new_ckpt_arr, cidx + 1, rng, rng_eval, ckpt_eval_eps_last_infos, ckpt_eval_eps_last_infos)

                def skip_ckpt_and_eval(args):
                    def do_eval(eval_args):
                        ckpt_arr, cidx, rng, rng_eval, prev_ckpt_eval_ret_info, prev_eval_ret_info = eval_args
                        eval_params = {
                            "params": joint_train_state.q_network_train_state.params,
                            "batch_stats": joint_train_state.q_network_train_state.batch_stats
                        }

                        if config["FIXED_EVAL"]:
                            eval_rng = rng_eval
                        else:
                            rng_eval, eval_rng = jax.random.split(rng_eval, 2)
                        eval_eps_last_infos = run_episodes_vmap(eval_rng, env, optimal_env, agent_idx,
                            agent_params=eval_params,
                            agent_policy=joint_policy,
                            optimal_params=optimal_params,
                            optimal_policies=optimal_policies,
                            max_episode_steps=max_episode_steps,
                            num_eps=config["NUM_EVAL_EPISODES"],
                            epsilon_optimal=config["EPSILON_OPTIMAL"],
                            use_full_obs=config["JOINT_USE_FULL_OBS"],
                            agent_test_mode=True)

                        return (ckpt_arr, cidx, rng, rng_eval, prev_ckpt_eval_ret_info, eval_eps_last_infos)

                    def skip_eval(eval_args):
                        return eval_args

                    return jax.lax.cond(config["TRAIN_EVAL"], do_eval, skip_eval, args)

                (checkpoint_array, ckpt_idx, rng, rng_eval, ckpt_eval_eps_last_infos, eval_eps_last_infos) = jax.lax.cond(
                    to_store, store_and_eval_ckpt, skip_ckpt_and_eval,
                    (checkpoint_array, ckpt_idx, rng, rng_eval, init_ckpt_eval_last_info, init_eval_last_info)
                )

                metric["ckpt_eval_ep_last_info"] = ckpt_eval_eps_last_infos
                metric["eval_ep_last_info"] = eval_eps_last_infos
                return ((joint_train_state, rng, rng_eval, update_steps),
                         checkpoint_array, ckpt_idx, ckpt_eval_eps_last_infos, eval_eps_last_infos), metric

            init_ckpt_params = {
                "params": joint_train_state.q_network_train_state.params,
                "batch_stats": joint_train_state.q_network_train_state.batch_stats
            }
            checkpoint_array = init_ckpt_array(init_ckpt_params)
            ckpt_idx = 0

            rng, rng_train = jax.random.split(rng, 2)

            rng_eval = jax.random.PRNGKey(config["EVAL_SEED"])# + agent_idx)# + 42)
            rng_eval, eval_rng = jax.random.split(rng_eval, 2)
            if config["FIXED_EVAL"]:
                eval_rng = rng_eval

            # Init eval return infos
            eval_eps_last_infos = run_episodes_vmap(eval_rng, env, optimal_env, agent_idx,
                                    agent_params=init_ckpt_params,
                                    agent_policy=joint_policy,
                                    optimal_params=optimal_params,
                                    optimal_policies=optimal_policies,
                                    max_episode_steps=max_episode_steps,
                                    num_eps=config["NUM_EVAL_EPISODES"],
                                    epsilon_optimal=config["EPSILON_OPTIMAL"],
                                    use_full_obs=config["JOINT_USE_FULL_OBS"],
                                    agent_test_mode=True)

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
            final_joint_ts = final_runner_state[0]

            final_params = {
                "params": final_joint_ts.q_network_train_state.params,
                "batch_stats": final_joint_ts.q_network_train_state.batch_stats
            }
            out = {
                "final_params": final_params,
                "metrics": metrics,
                "checkpoints": checkpoint_array,
            }

            if env._render:
                # Collect final eval gifs for logging
                rng_eval = final_runner_state[2]
                if config["FIXED_EVAL"]:
                    eval_rng = rng_eval
                else:
                    rng_eval, eval_rng = jax.random.split(rng_eval, 2)
                out["render_outs"] = run_episodes_vmap(eval_rng, env, optimal_env, agent_idx,
                                        agent_params=final_params,
                                        agent_policy=joint_policy,
                                        optimal_params=optimal_params,
                                        optimal_policies=optimal_policies,
                                        max_episode_steps=env.horizon,
                                        num_eps=5, epsilon_optimal=config["EPSILON_OPTIMAL"],
                                        use_full_obs=config["JOINT_USE_FULL_OBS"],
                                        render=True, agent_test_mode=True)

            return out
        return train

    # ------------------------------
    # Actually run the CREPPO training
    # ------------------------------
    rngs = jax.random.split(train_rng, config["NUM_TRAIN_SEEDS"])

    # Run training seeds in parallel using vmap
    train_fn = make_creppo_joint_train(config)
    num_agents = env.num_agents
    in_axes = (0, None) + (0,) * num_agents
    out = jax.vmap(train_fn, in_axes=in_axes)(rngs, agent_idx, *optimal_params)
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
    algorithm_config["VMIN"] = algorithm_config.get("JOINT_VMIN", VMAX * -1)
    algorithm_config["VMAX"] = algorithm_config.get("JOINT_VMAX", VMIN * -1)

    # algorithm_config["TARGET_ENTROPY_MULT"] = algorithm_config["TARGET_ENTROPY_MULT"] * -1

    # Create only one environment instance
    env_kwargs = algorithm_config["ENV_KWARGS"].copy()
    env_kwargs["render_dir"] = os.path.join("render", "joint", f"agent_{agent_idx + 1}_optimize")
    with open_dict(env_kwargs):
        env_kwargs["done_condition"] = f"agent_{agent_idx}"  # End only when focal agent finishes; adversary cannot exploit early termination

    env = make_env(algorithm_config["ENV_NAME"], env_kwargs)
    env = LogWrapper(env)

    env_kwargs = algorithm_config["ENV_KWARGS"].copy()
    env_kwargs["render_dir"] = os.path.join("render", "joint", f"agent_{agent_idx + 1}_optimize")
    env_kwargs["instance"] = config['task'][f"SINGLE_AGENT_{agent_idx + 1}_PROJECTION"]
    with open_dict(env_kwargs):
        env_kwargs["done_condition"] = "any"  # SAP: terminate as soon as agent i takes its picture
    optimal_env = make_env(algorithm_config["ENV_NAME"], env_kwargs)
    optimal_env = LogWrapper(optimal_env)

    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"])# + agent_idx)# + 35)
    _, init_rng, train_rng = jax.random.split(rng, 3)

    num_agents = env.num_agents
    obs_type = "full" if algorithm_config["JOINT_USE_FULL_OBS"] else "agent"

    # Initialize single centralized joint policy for all agents
    joint_policy, init_joint_params = initialize_creppo_agent(
        algorithm_config, env, init_rng, agent_index=0, observation_type=obs_type
    )

    log.info(f"Starting CREPPO joint training optimizing for agent {agent_idx}...")
    start_time = time.perf_counter()

    # Run the training
    out = train_creppo_joint_agents(
        config=algorithm_config,
        env=env,
        optimal_env=optimal_env,
        train_rng=train_rng,
        joint_policy=joint_policy,
        init_joint_params=init_joint_params,
        optimal_policies=optimal_policies,
        optimal_params=optimal_params,
        agent_idx=agent_idx
    )

    elapsed_time = time.perf_counter() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, rem = divmod(rem, 60)
    seconds, rem = divmod(rem, 1)
    milliseconds = int(rem * 1000)
    microseconds = int((rem * 1_000_000) % 1000)
    log.info(f"CREPPO Joint Training completed optimizing for agent {agent_idx} in {elapsed_time:.2f}s")
    log.info(f"CREPPO Joint Training completed optimizing for agent {agent_idx} in {int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s {milliseconds:03d}ms {microseconds:03d}µs")

    # process and log metrics
    log.info(f"Starting CREPPO joint logging optimizing for agent {agent_idx}...")
    start_time = time.perf_counter()
    # metric_names = get_metric_names(config["ENV_NAME"])
    metric_names = get_metric_names(f"social_laws_joint-{config['ENV_NAME']}")
    log_metrics(env, optimal_env, config, out, wandb_logger, metric_names, agent_idx)
    elapsed_time = time.perf_counter() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, rem = divmod(rem, 60)
    seconds, rem = divmod(rem, 1)
    milliseconds = int(rem * 1000)
    microseconds = int((rem * 1_000_000) % 1000)
    log.info(f"CREPPO Joint Logging completed optimizing for agent {agent_idx} in {elapsed_time:.2f}s")
    log.info(f"CREPPO Joint Logging completed optimizing for agent {agent_idx} in {int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s {milliseconds:03d}ms {microseconds:03d}µs")

    final_params = out["final_params"]
    return final_params, joint_policy, init_joint_params

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

    # Extract centralized joint policy metrics
    avg_alpha_loss = np.mean(np.asarray(train_metrics["alpha_loss"]), axis=(0, 2, 3)) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    avg_alpha = np.mean(np.asarray(train_metrics["alpha"]), axis=(0, 2, 3))
    avg_kl = np.mean(np.asarray(train_metrics["kl"]), axis=(0, 2, 3))
    avg_entropy = np.mean(np.asarray(train_metrics["entropy"]), axis=(0, 2, 3))
    avg_critic_loss = np.mean(np.asarray(train_metrics["critic_loss"]), axis=(0, 2, 3))
    avg_q_values = np.mean(np.asarray(train_metrics["q_values"]), axis=(0, 2, 3))
    avg_q_error = np.mean(np.asarray(train_metrics["q_error"]), axis=(0, 2, 3))

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

    if config["algorithm"].get("ALPHA_COST", False):
        all_ckpt_alpha_returns = all_ckpt_optimal_returns / all_ckpt_worst_case_returns # shape (n_train_seeds, num_updates, num_eval_episodes)
        all_alpha_returns = all_optimal_returns / all_worst_case_returns # shape (n_train_seeds, num_updates, num_eval_episodes)
    else:
        all_ckpt_alpha_returns = all_ckpt_worst_case_returns / all_ckpt_optimal_returns # shape (n_train_seeds, num_updates, num_eval_episodes)
        all_alpha_returns = all_worst_case_returns / all_optimal_returns # shape (n_train_seeds, num_updates, num_eval_episodes)

    average_ckpt_worst_case_rets_per_iter = np.mean(all_ckpt_worst_case_returns, axis=(0, 2)) # shape (num_updates,)
    average_agent_worst_case_rets_per_iter = np.mean(all_worst_case_returns, axis=(0, 2)) # shape (num_updates,)
    average_ckpt_optimal_rets_per_iter = np.mean(all_ckpt_optimal_returns, axis=(0, 2)) # shape (num_updates,)
    average_agent_optimal_rets_per_iter = np.mean(all_optimal_returns, axis=(0, 2)) # shape (num_updates,)
    average_ckpt_alpha_rets_per_iter = np.mean(all_ckpt_alpha_returns, axis=(0, 2)) # shape (num_updates,)
    average_agent_alpha_rets_per_iter = np.mean(all_alpha_returns, axis=(0, 2)) # shape (num_updates,)

    all_ckpt_worst_case_collisions = np.asarray(train_metrics["ckpt_eval_ep_last_info"][0]["returned_episode_collisions"]) # shape (n_train_seeds, num_updates, num_eval_episodes, num_agents_per_game)
    all_worst_case_collisions = np.asarray(train_metrics["eval_ep_last_info"][0]["returned_episode_collisions"]) # shape (n_train_seeds, num_updates, num_eval_episodes, num_agents_per_game)
    all_ckpt_worst_case_collisions = np.sum(all_ckpt_worst_case_collisions, axis=3) # shape (n_train_seeds, num_updates, num_eval_episodes)
    all_worst_case_collisions = np.sum(all_worst_case_collisions, axis=3) # shape (n_train_seeds, num_updates, num_eval_episodes)
    all_ckpt_optimal_collisions = np.asarray(train_metrics["ckpt_eval_ep_last_info"][1]["returned_episode_collisions"]) # shape (n_train_seeds, num_updates, num_eval_episodes, num_agents_per_game)
    all_optimal_collisions = np.asarray(train_metrics["eval_ep_last_info"][1]["returned_episode_collisions"]) # shape (n_train_seeds, num_updates, num_eval_episodes, num_agents_per_game)
    all_ckpt_optimal_collisions = np.sum(all_ckpt_optimal_collisions, axis=3) # shape (n_train_seeds, num_updates, num_eval_episodes)
    all_optimal_collisions = np.sum(all_optimal_collisions, axis=3) # shape (n_train_seeds, num_updates, num_eval_episodes)

    average_ckpt_worst_case_collisions_per_iter = np.mean(all_ckpt_worst_case_collisions, axis=(0, 2)) # shape (num_updates,)
    average_agent_worst_case_collisions_per_iter = np.mean(all_worst_case_collisions, axis=(0, 2)) # shape (num_updates,)
    average_ckpt_optimal_collisions_per_iter = np.mean(all_ckpt_optimal_collisions, axis=(0, 2)) # shape (num_updates,)
    average_agent_optimal_collisions_per_iter = np.mean(all_optimal_collisions, axis=(0, 2)) # shape (num_updates,)

    # Log metrics for each update step
    num_updates = len(avg_alpha_loss)
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

        logger.log_item(f"Eval/Joint/Agent_{agent_idx + 1}_Optimize/WorstCaseCollisions", average_agent_worst_case_collisions_per_iter[step], train_step=step, commit=True)
        logger.log_item(f"Eval/Joint/Agent_{agent_idx + 1}_Optimize/CheckpointWorstCaseCollisions", average_ckpt_worst_case_collisions_per_iter[step], train_step=step, commit=True)
        logger.log_item(f"Eval/Joint/Agent_{agent_idx + 1}_Optimize/OptimalCollisions", average_agent_optimal_collisions_per_iter[step], train_step=step, commit=True)
        logger.log_item(f"Eval/Joint/Agent_{agent_idx + 1}_Optimize/CheckpointOptimalCollisions", average_ckpt_optimal_collisions_per_iter[step], train_step=step, commit=True)

        # Log centralized joint policy metrics
        logger.log_item(f"Train/Joint/Agent_{agent_idx + 1}_Optimize/AlphaLoss", avg_alpha_loss[step], train_step=step, commit=True)
        logger.log_item(f"Train/Joint/Agent_{agent_idx + 1}_Optimize/Alpha", avg_alpha[step], train_step=step, commit=True)
        logger.log_item(f"Train/Joint/Agent_{agent_idx + 1}_Optimize/KL", avg_kl[step], train_step=step, commit=True)
        logger.log_item(f"Train/Joint/Agent_{agent_idx + 1}_Optimize/Entropy", avg_entropy[step], train_step=step, commit=True)
        logger.log_item(f"Train/Joint/Agent_{agent_idx + 1}_Optimize/CriticLoss", avg_critic_loss[step], train_step=step, commit=True)
        logger.log_item(f"Train/Joint/Agent_{agent_idx + 1}_Optimize/QValueMean", avg_q_values[step], train_step=step, commit=True)
        logger.log_item(f"Train/Joint/Agent_{agent_idx + 1}_Optimize/QValueError", avg_q_error[step], train_step=step, commit=True)
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
        num_episodes = eval_render_worst_case_dones.shape[1]
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

    out_savepath = save_train_run(train_out, savedir, savename=f"CREPPO_Joint_Agent_{agent_idx + 1}_Optimize_Train_Run")
    if config["logger"]["log_train_out"]:
        logger.log_artifact(name=f"CREPPO_Joint_Agent_{agent_idx + 1}_Optimize_Train_Run", path=out_savepath, type_name="joint_train_run")

    if not config["local_logger"]["save_train_out"]:
        shutil.rmtree(out_savepath)

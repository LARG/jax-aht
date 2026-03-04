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
import optax
import hydra

from flax.linen import softmax, log_softmax

from social_laws.common.initialize_agents import initialize_creppo_agent
from social_laws.common.run_episodes_creppo_w_robustness import run_episodes_vmap
from common.plot_utils import get_stats, get_metric_names
from common.save_load_utils import save_train_run
from envs import make_env
from envs.log_wrapper import LogWrapper
from marl.ppo_utils import unbatchify

from agents.mlp_creppo import hl_gauss
from agents.mlp_creppo_agent import CReppoTrainState, CustomTrainState, Transition

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def train_creppo_joint_agents(config, env, optimal_env, train_rng,
                           joint_policies, init_joint_params,
                           optimal_policies, optimal_params,
                           agent_idx):
    '''
    Train CREPPO joint agents using the given initial parameters.

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
    # Build the CREPPO joint training function
    # ------------------------------

    def make_creppo_joint_train(config):
        num_agents = env.num_agents
        # assert num_agents == 2, "This snippet assumes exactly 2 agents."

        # config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
        # config["NUM_UNCONTROLLED_ACTORS"] = config["NUM_ENVS"] # assumption: we control 1 agent
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

            train_states = []
            for i in range(num_agents):
                q_network_ts = CustomTrainState.create(
                    apply_fn=joint_policies[i].q_network.apply,
                    params=init_joint_params[i]["params"],
                    batch_stats=init_joint_params[i]["batch_stats"],
                    tx=tx,
                )
                target_ts = CustomTrainState.create(
                    apply_fn=joint_policies[i].q_network.apply,
                    params=deepcopy(init_joint_params[i]["params"]),
                    batch_stats=deepcopy(init_joint_params[i]["batch_stats"]),
                    tx=optax.set_to_zero(),
                )
                train_states.append(CReppoTrainState(
                    timesteps=0,
                    n_updates=0,
                    grad_steps=0,
                    q_network_train_state=q_network_ts,
                    target_train_state=target_ts,
                ))

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
            init_hstates = [joint_policies[i].init_hstate(config["NUM_CONTROLLED_ACTORS"]) for i in range(num_agents)]
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
                1. Get observations, sample actions from all agents
                2. Step environment using sampled actions
                3. Return state, reward, ...
                """
                train_states, env_state, prev_obs, prev_done, hstates, optimal_hstates, rng = runner_state
                prev_obs, prev_full_obs = prev_obs
                rng, *actor_rngs, step_rng = jax.random.split(rng, num_agents * 2 + 2)
                target_rngs = actor_rngs[num_agents:]
                actor_rngs = actor_rngs[:num_agents]

                # Get available actions for the agent from environment state
                avail_actions = env.get_avail_actions(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(avail_actions)

                prev_obs_per_agent = [get_agent_data(prev_obs, i).reshape(1, config["NUM_CONTROLLED_ACTORS"], -1) for i in range(num_agents)]
                prev_full_obs_per_agent = [get_agent_data(prev_full_obs, i).reshape(1, config["NUM_CONTROLLED_ACTORS"], -1) for i in range(num_agents)]
                prev_done_per_agent = [get_agent_data(prev_done, i).reshape(1, config["NUM_CONTROLLED_ACTORS"]) for i in range(num_agents)]
                avail_actions_per_agent = [get_agent_data(avail_actions, i).astype(jnp.float32) for i in range(num_agents)]

                # Restrict available actions based on value function
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

                # Note that we do not need to reset the hidden states for the agents
                # as the recurrent states are automatically reset when done is True.

                # Per-agent action sampling
                acts = []
                new_hstates = []
                for i in range(num_agents):
                    act_i, _, _, new_hstate_i = joint_policies[i].get_action_value_policy(
                        params=(train_states[i].q_network_train_state.params, train_states[i].q_network_train_state.batch_stats),
                        obs=prev_full_obs_per_agent[i] if config["JOINT_USE_FULL_OBS"] else prev_obs_per_agent[i],
                        done=prev_done_per_agent[i],
                        avail_actions=optimal_restricted_avail_actions[i],
                        hstate=hstates[i],
                        rng=actor_rngs[i],
                    )
                    # acts.append(act_i.squeeze(axis=0))
                    acts.append(act_i)
                    new_hstates.append(new_hstate_i)

                # Combine actions into the env format
                combined_actions = jnp.concatenate(acts, axis=0)  # shape (num_agents*num_envs,)
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

                negative_reward = get_agent_data(reward, agent_idx) * -1

                # Next values - per agent
                next_avail_actions = env.get_avail_actions(env_state_next.env_state)
                next_avail_actions = jax.lax.stop_gradient(next_avail_actions)
                next_avail_per_agent = [get_agent_data(next_avail_actions, i).astype(jnp.float32) for i in range(num_agents)]

                obs_next_per_agent = [get_agent_data(obs_next, i).reshape(1, config["NUM_CONTROLLED_ACTORS"], -1) for i in range(num_agents)]
                obs_full_next_per_agent = [get_agent_data(obs_full_next, i).reshape(1, config["NUM_CONTROLLED_ACTORS"], -1) for i in range(num_agents)]
                done_next_per_agent = [get_agent_data(done_next, i).reshape(1, config["NUM_CONTROLLED_ACTORS"]) for i in range(num_agents)]

                # Restrict next available actions and get next values per agent
                transitions = []
                for i in range(num_agents):
                    next_opt_restricted_i, _ = _get_optimal_restricted_avail_actions(
                        obs=obs_next_per_agent[i],
                        done=done_next_per_agent[i],
                        avail_actions=next_avail_per_agent[i],
                        hstate=new_optimal_hstates[i],
                        optimal_params=optimal_params[i],
                        optimal_policy=optimal_policies[i]
                    )

                    next_q, _ = joint_policies[i].get_critic_out(
                        params=(train_states[i].q_network_train_state.params, train_states[i].q_network_train_state.batch_stats),
                        obs=obs_full_next_per_agent[i] if config["JOINT_USE_FULL_OBS"] else obs_next_per_agent[i],
                        done=done_next_per_agent[i],
                        avail_actions=next_opt_restricted_i,
                        hstate=new_hstates[i],
                        rng=target_rngs[i],
                    )

                    q_probs = softmax(
                        next_q["policy_logits"],
                        axis=-1,
                    )
                    next_values = jnp.sum(
                        q_probs * next_q["q_values"],
                        axis=-1,
                    )

                    # TODO: Prev or next avail actions?
                    entropy = -jnp.sum(jnp.where(next_opt_restricted_i, q_probs * jnp.log(q_probs + 1e-8), 0), axis=-1) * config["NUM_ACTIONS"] / next_opt_restricted_i.sum(-1)

                    # Build per-agent transitions
                    transitions.append(
                        Transition(
                            obs=get_agent_data(prev_full_obs, i) if config["JOINT_USE_FULL_OBS"] else get_agent_data(prev_obs, i),
                            action=acts[i],
                            action_logp=entropy,
                            reward=negative_reward,
                            done=get_agent_data(done_next, i),
                            avail_actions=optimal_restricted_avail_actions[i],
                            next_obs=get_agent_data(obs_full_next, i) if config["JOINT_USE_FULL_OBS"] else get_agent_data(obs_next, i),
                            next_avail_actions=next_opt_restricted_i,
                            next_val=next_values,
                            info=info
                        )
                    )

                new_runner_state = (train_states, env_state_next, (obs_next, obs_full_next), done_next,
                                    new_hstates, new_optimal_hstates, rng)
                return new_runner_state, transitions

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
                2. Compute advantage
                3. CREPPO updates
                """
                (train_states, rng, rng_eval, update_steps) = update_runner_state
                # Init envs & partner indices
                rng, reset_rng = jax.random.split(rng, 2)
                reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
                init_obs, init_env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rngs)
                init_done = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}

                # 1) rollout
                runner_state = (train_states, init_env_state, init_obs, init_done, init_hstates, optimal_init_hstates, rng)

                runner_state, traj_batches = jax.lax.scan(
                    _env_step, runner_state, None, config["ROLLOUT_LENGTH"])
                (train_states, env_state, obs, done, hstates, optimal_hstates, rng) = runner_state

                train_states = [
                    train_states[i].replace(
                        timesteps=train_states[i].timesteps
                        + config["ROLLOUT_LENGTH"] * config["NUM_ENVS"]
                    )
                    for i in range(num_agents)
                ]  # update timesteps count

                # 2) Compute soft rewards and n-step lambda returns for each agent
                target_values = []
                for i in range(num_agents):
                    rng, sample_rng = jax.random.split(rng)
                    next_pi_i, _ = joint_policies[i].get_critic_out(
                        params=(train_states[i].q_network_train_state.params, train_states[i].q_network_train_state.batch_stats),
                        obs=traj_batches[i].next_obs[-1].reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                        done=traj_batches[i].done[-1].reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                        avail_actions=traj_batches[i].next_avail_actions[-1],
                        hstate=hstates[i],
                        rng=sample_rng
                    )

                    next_pi_probs_i = softmax(next_pi_i["policy_logits"], axis=-1)

                    next_pi_ent_i: jax.Array = -jnp.sum(
                        jnp.where(traj_batches[i].next_avail_actions[-1], next_pi_probs_i * jnp.log(next_pi_probs_i + 1e-8), 0), axis=-1
                    ) * config["NUM_ACTIONS"] / traj_batches[i].next_avail_actions[-1].sum(-1)
                    next_logp_i = jnp.concatenate(
                        [traj_batches[i].action_logp[1:], next_pi_ent_i[None]], axis=0
                    )

                    soft_reward = traj_batches[i].reward + config["GAMMA"] * next_logp_i * jnp.exp(
                        train_states[i].q_network_train_state.params["log_alpha"]
                    )

                    traj_batches[i] = traj_batches[i].replace(soft_reward=soft_reward)
                    _, target_values_i = jax.lax.scan(
                        compute_nstep_lambda,
                        (
                            traj_batches[i].next_val[-1],
                            jnp.ones_like(traj_batches[i].done[0])
                        ),
                        traj_batches[i],
                        reverse=True,
                    )
                    target_values.append(target_values_i)

                # 3) CREPPO update for each agent separately using factory
                def _make_learn_epoch(agent_i):
                    def _learn_epoch(carry, _):
                        train_state, rng = carry

                        def _learn_phase(carry, minibatch_and_target):
                            train_state, rng = carry
                            minibatch, target = minibatch_and_target

                            def _critic_loss_fn(params, train_state):
                                critic_out, updates = joint_policies[agent_i].q_network.apply(
                                    {
                                        "params": params,
                                        "batch_stats": train_state.q_network_train_state.batch_stats,
                                    },
                                    (minibatch.obs, minibatch.avail_actions),
                                    train=True,
                                    mutable=["batch_stats"],
                                )  # (batch_size*2, num_actions)
                                # TODO: Deal with recurrent network

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
                                old_critic_out = joint_policies[agent_i].q_network.apply(
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
                                    f"agent_{agent_i}/critic_loss": loss,
                                    f"agent_{agent_i}/q_values": q_vals.mean(),
                                    f"agent_{agent_i}/q_error": optax.l2_loss(target - q_vals).mean(),
                                    f"agent_{agent_i}/alpha_loss": jnp.mean(alpha_loss),
                                    f"agent_{agent_i}/alpha": alpha,
                                    f"agent_{agent_i}/kl": kl.mean(),
                                    f"agent_{agent_i}/entropy": pi.entropy().mean(),
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
                            )  # num_steps*num_envs (batch_size), ...
                            x = jax.random.permutation(rng, x)  # shuffle the transitions
                            x = x.reshape(
                                config["NUM_MINIBATCHES"], -1, *x.shape[1:]
                            )  # num_mini_updates, batch_size/num_mini_updates, ...
                            return x

                        rng, _rng = jax.random.split(rng)
                        minibatches = jax.tree_util.tree_map(
                            lambda x: preprocess_transition(x, _rng), traj_batches[agent_i]
                        )  # num_actors*num_envs (batch_size), ...
                        targets = jax.tree_util.tree_map(
                            lambda x: preprocess_transition(x, _rng), target_values[agent_i]
                        )

                        rng, _rng = jax.random.split(rng)
                        (train_state, rng), metrics = jax.lax.scan(
                            _learn_phase, init=(train_state, rng), xs=(minibatches, targets)
                        )

                        return (train_state, rng), metrics
                    return _learn_epoch

                # Run update epoch for each agent and collect metrics
                learn_epoch_fns = [_make_learn_epoch(i) for i in range(num_agents)]
                learn_metrics = {}
                for i in range(num_agents):
                    train_states[i] = train_states[i].replace(
                        target_train_state=train_states[i].target_train_state.replace(
                            params=train_states[i].q_network_train_state.params,
                            batch_stats=train_states[i].q_network_train_state.batch_stats,
                        )
                    )

                    rng, learn_rng = jax.random.split(rng)
                    (train_states[i], rng), learn_metrics_i = jax.lax.scan(
                        f=learn_epoch_fns[i],
                        init=(train_states[i], learn_rng),
                        xs=None,
                        length=config["NUM_EPOCHS"],
                    )

                    train_states[i] = train_states[i].replace(
                        n_updates=train_states[i].n_updates + 1,
                    )
                    learn_metrics.update(learn_metrics_i)

                metric = traj_batches[0].info
                metric["update_steps"] = update_steps
                metric.update(learn_metrics)
                new_runner_state = (train_states, rng, rng_eval, update_steps + 1)
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
                (update_state, checkpoint_arrays, ckpt_idx, init_ckpt_eval_last_info, init_eval_last_info) = state_with_ckpt

                # Single CREPPO update
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
                    ckpt_params = []
                    new_ckpt_arrs = []
                    for i in range(num_agents):
                        ckpt_params_i = {
                            "params": train_states[i].q_network_train_state.params,
                            "batch_stats": train_states[i].q_network_train_state.batch_stats
                        }
                        new_ckpt_array_i = jax.tree.map(
                            lambda c_arr, p: c_arr.at[cidx].set(p),
                            ckpt_arrs[i], ckpt_params_i
                        )
                        ckpt_params.append(ckpt_params_i)
                        new_ckpt_arrs.append(new_ckpt_array_i)

                    if config["FIXED_EVAL"]:
                        eval_rng = rng_eval
                    else:
                        rng_eval, eval_rng = jax.random.split(rng_eval, 2)
                    ckpt_eval_eps_last_infos = run_episodes_vmap(eval_rng, env, optimal_env, agent_idx,
                        agent_params=ckpt_params,
                        agent_policies=joint_policies,
                        optimal_params=optimal_params,
                        optimal_policies=optimal_policies,
                        max_episode_steps=max_episode_steps,
                        num_eps=config["NUM_EVAL_EPISODES"],
                        epsilon_optimal=config["EPSILON_OPTIMAL"],
                        use_full_obs=config["JOINT_USE_FULL_OBS"],
                        agent_test_mode=True)

                    return (new_ckpt_arrs, cidx + 1, rng, rng_eval, ckpt_eval_eps_last_infos, ckpt_eval_eps_last_infos)

                def skip_ckpt_and_eval(args):
                    def do_eval(eval_args):
                        ckpt_arrs, cidx, rng, rng_eval, prev_ckpt_eval_ret_info, prev_eval_ret_info = eval_args
                        eval_params = [
                            {
                                "params": train_states[i].q_network_train_state.params,
                                "batch_stats": train_states[i].q_network_train_state.batch_stats
                            }
                            for i in range(num_agents)
                        ]

                        if config["FIXED_EVAL"]:
                            eval_rng = rng_eval
                        else:
                            rng_eval, eval_rng = jax.random.split(rng_eval, 2)
                        eval_eps_last_infos = run_episodes_vmap(eval_rng, env, optimal_env, agent_idx,
                            agent_params=eval_params,
                            agent_policies=joint_policies,
                            optimal_params=optimal_params,
                            optimal_policies=optimal_policies,
                            max_episode_steps=max_episode_steps,
                            num_eps=config["NUM_EVAL_EPISODES"],
                            epsilon_optimal=config["EPSILON_OPTIMAL"],
                            use_full_obs=config["JOINT_USE_FULL_OBS"],
                            agent_test_mode=True)

                        return (ckpt_arrs, cidx, rng, rng_eval, prev_ckpt_eval_ret_info, eval_eps_last_infos)

                    def skip_eval(eval_args):
                        return eval_args

                    return jax.lax.cond(config["TRAIN_EVAL"], do_eval, skip_eval, args)

                (checkpoint_arrays, ckpt_idx, rng, rng_eval, ckpt_eval_eps_last_infos, eval_eps_last_infos) = jax.lax.cond(
                    to_store, store_and_eval_ckpt, skip_ckpt_and_eval,
                    (checkpoint_arrays, ckpt_idx, rng, rng_eval, init_ckpt_eval_last_info, init_eval_last_info)
                )

                metric["ckpt_eval_ep_last_info"] = ckpt_eval_eps_last_infos
                metric["eval_ep_last_info"] = eval_eps_last_infos
                return ((train_states, rng, rng_eval, update_steps),
                         checkpoint_arrays, ckpt_idx, ckpt_eval_eps_last_infos, eval_eps_last_infos), metric

            init_ckpt_params = [
                {
                    "params": train_states[i].q_network_train_state.params,
                    "batch_stats": train_states[i].q_network_train_state.batch_stats
                }
                for i in range(num_agents)
            ]
            checkpoint_arrays = [init_ckpt_array(init_ckpt_params[i]) for i in range(num_agents)]
            ckpt_idx = 0

            rng, rng_train = jax.random.split(rng, 2)

            rng_eval = jax.random.PRNGKey(config["EVAL_SEED"])# + agent_idx)# + 42)
            rng_eval, eval_rng = jax.random.split(rng_eval, 2)

            # Init eval return infos
            eval_eps_last_infos = run_episodes_vmap(eval_rng, env, optimal_env, agent_idx,
                                    agent_params=init_ckpt_params,
                                    agent_policies=joint_policies,
                                    optimal_params=optimal_params,
                                    optimal_policies=optimal_policies,
                                    max_episode_steps=max_episode_steps,
                                    num_eps=config["NUM_EVAL_EPISODES"],
                                    epsilon_optimal=config["EPSILON_OPTIMAL"],
                                    use_full_obs=config["JOINT_USE_FULL_OBS"],
                                    agent_test_mode=True)

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

            final_params = []
            out = {"metrics": metrics}
            for i in range(num_agents):
                final_params_i = {
                    "params": final_agent_ts[i].q_network_train_state.params,
                    "batch_stats": final_agent_ts[i].q_network_train_state.batch_stats
                }
                final_params.append(final_params_i)
                out[f"final_params_agent_{i}"] = final_params_i
                out[f"checkpoints_agent_{i}"] = final_checkpoint_arrays[i]

            if env._render:
                # Collect final eval gifs for logging
                rng_eval = final_runner_state[2]
                if config["FIXED_EVAL"]:
                    eval_rng = rng_eval
                else:
                    rng_eval, eval_rng = jax.random.split(rng_eval, 2)
                out["render_outs"] = run_episodes_vmap(eval_rng, env, optimal_env, agent_idx,
                                        agent_params=final_params,
                                        agent_policies=joint_policies,
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

    env = make_env(algorithm_config["ENV_NAME"], env_kwargs)
    env = LogWrapper(env)

    env_kwargs = algorithm_config["ENV_KWARGS"].copy()
    env_kwargs["render_dir"] = os.path.join("render", "joint", f"agent_{agent_idx + 1}_optimize")
    env_kwargs["instance"] = config['task'][f"SINGLE_AGENT_{agent_idx + 1}_PROJECTION"]
    optimal_env = make_env(algorithm_config["ENV_NAME"], env_kwargs)
    optimal_env = LogWrapper(optimal_env)

    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"])# + agent_idx)# + 35)
    _, init_rng, train_rng = jax.random.split(rng, 3)

    num_agents = env.num_agents
    obs_type = "full" if algorithm_config["JOINT_USE_FULL_OBS"] else "agent"

    # Initialize one joint policy per agent
    joint_policies, init_joint_params = map(list, zip(*[
        initialize_creppo_agent(algorithm_config, env, init_rng, agent_index=i, observation_type=obs_type)
        for i in range(num_agents)
    ]))

    log.info(f"Starting CREPPO joint training optimizing for agent {agent_idx}...")
    start_time = time.perf_counter()

    # Run the training
    out = train_creppo_joint_agents(
        config=algorithm_config,
        env=env,
        optimal_env=optimal_env,
        train_rng=train_rng,
        joint_policies=joint_policies,
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
    metric_names = get_metric_names("social_laws_joint")
    log_metrics(env, optimal_env, config, out, wandb_logger, metric_names, agent_idx)
    elapsed_time = time.perf_counter() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, rem = divmod(rem, 60)
    seconds, rem = divmod(rem, 1)
    milliseconds = int(rem * 1000)
    microseconds = int((rem * 1_000_000) % 1000)
    log.info(f"CREPPO Joint Logging completed optimizing for agent {agent_idx} in {elapsed_time:.2f}s")
    log.info(f"CREPPO Joint Logging completed optimizing for agent {agent_idx} in {int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s {milliseconds:03d}ms {microseconds:03d}µs")

    final_params = [out[f"final_params_agent_{i}"] for i in range(num_agents)]
    return final_params, joint_policies, init_joint_params

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

    avg_per_agent_alpha_losses = []
    avg_per_agent_alpha = []
    avg_per_agent_kl = []
    avg_per_agent_entropy = []
    avg_per_agent_critic_losses = []
    avg_per_agent_q_values = []
    avg_per_agent_q_error = []
    num_agents_log = sum(1 for k in train_metrics if k.endswith("/alpha_loss") and k.startswith("agent_"))
    for i in range(num_agents_log):
        avg_per_agent_alpha_losses.append(np.mean(np.asarray(train_metrics[f"agent_{i}/alpha_loss"]), axis=(0, 2, 3))) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
        avg_per_agent_alpha.append(np.mean(np.asarray(train_metrics[f"agent_{i}/alpha"]), axis=(0, 2, 3))) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
        avg_per_agent_kl.append(np.mean(np.asarray(train_metrics[f"agent_{i}/kl"]), axis=(0, 2, 3)))
        avg_per_agent_entropy.append(np.mean(np.asarray(train_metrics[f"agent_{i}/entropy"]), axis=(0, 2, 3)))
        avg_per_agent_critic_losses.append(np.mean(np.asarray(train_metrics[f"agent_{i}/critic_loss"]), axis=(0, 2, 3)))
        avg_per_agent_q_values.append(np.mean(np.asarray(train_metrics[f"agent_{i}/q_values"]), axis=(0, 2, 3)))
        avg_per_agent_q_error.append(np.mean(np.asarray(train_metrics[f"agent_{i}/q_error"]), axis=(0, 2, 3)))

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

    if config.get("ALPHA_COST", False):
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

    # Log metrics for each update step
    num_updates = len(avg_per_agent_alpha_losses[0])
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

        for i in range(num_agents_log):
            prefix = f"Train/Joint/Agent_{agent_idx + 1}_Optimize/Agent_{i + 1}"
            logger.log_item(f"{prefix}/AlphaLoss", avg_per_agent_alpha_losses[i][step], train_step=step, commit=True)
            logger.log_item(f"{prefix}/Alpha", avg_per_agent_alpha[i][step], train_step=step, commit=True)
            logger.log_item(f"{prefix}/KL", avg_per_agent_kl[i][step], train_step=step, commit=True)
            logger.log_item(f"{prefix}/Entropy", avg_per_agent_entropy[i][step], train_step=step, commit=True)
            logger.log_item(f"{prefix}/CriticLoss", avg_per_agent_critic_losses[i][step], train_step=step, commit=True)
            logger.log_item(f"{prefix}/QValueMean", avg_per_agent_q_values[i][step], train_step=step, commit=True)
            logger.log_item(f"{prefix}/QValueError", avg_per_agent_q_error[i][step], train_step=step, commit=True)
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

    out_savepaths = []
    for i in range(num_agents_log):
        train_out_i = {
            "final_params": train_out[f"final_params_agent_{i}"],
            "metrics": train_out["metrics"],
            "checkpoints": train_out[f"checkpoints_agent_{i}"],
        }
        savepath = save_train_run(train_out_i, savedir, savename=f"CREPPO_Agent_{agent_idx + 1}_Optimize_Train_Run-Agent_{i + 1}")
        out_savepaths.append(savepath)
        if config["logger"]["log_train_out"]:
            logger.log_artifact(name=f"CREPPO_Agent_{agent_idx + 1}_Optimize_Train_Run-Agent_{i + 1}", path=savepath, type_name="joint_train_run")

    if not config["local_logger"]["save_train_out"]:
        for savepath in out_savepaths:
            shutil.rmtree(savepath)

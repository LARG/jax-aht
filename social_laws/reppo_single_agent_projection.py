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

from agents import q_network
from social_laws.common.initialize_agents import initialize_reppo_agent
from social_laws.common.run_episodes_reppo_w_q_eval import run_episodes_vmap
from common.plot_utils import get_stats, get_metric_names
from common.save_load_utils import save_train_run
from envs import make_env
from envs.log_wrapper import LogWrapper
from marl.ppo_utils import unbatchify

from agents.mlp_reppo import hl_gauss
from agents.mlp_reppo_agent import ReppoTrainState, CustomTrainState, Transition

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def train_reppo_agent(config, env, q_env, train_rng,
                    policy, init_params, agent_idx):
    '''
    Train REPPO single agent projection using the given initial parameters.

    Args:
        config: dict, config for the training
        env: gymnasium environment
        train_rng: jax.random.PRNGKey, random key for training
        policy: AgentPolicy, policy for the agent
        init_params: dict, initial parameters for the agent
    '''
    # ------------------------------
    # Build the REPPO single agent projection training function
    # ------------------------------
    def make_reppo_train(config):
        '''The controlled agent is based on the agent_idx parameter'''
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
        temp_scheduler = optax.linear_schedule(
            init_value=config["TEMP_START"],
            end_value=config["TEMP_FINISH"],
            transition_steps=(config["NUM_UPDATES_DECAY"])
            * config["NUM_MINIBATCHES"]
            * config["NUM_EPOCHS"],
        )

        def train(rng, agent_idx):
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

            q_network_train_state = CustomTrainState.create(
                apply_fn=policy.q_network.apply,
                params=init_params['q_network']["params"],
                batch_stats=init_params['q_network']["batch_stats"],
                tx=tx,
            )
            actor_train_state = CustomTrainState.create(
                apply_fn=policy.actor.apply,
                params=init_params['actor']["params"],
                batch_stats=init_params['actor']["batch_stats"],
                tx=tx,
            )
            target_actor_train_state = CustomTrainState.create(
                apply_fn=policy.actor.apply,
                params=deepcopy(init_params['actor']["params"]),
                batch_stats=deepcopy(init_params['actor']["batch_stats"]),
                tx=optax.set_to_zero(),
            )

            train_state = ReppoTrainState(
                timesteps=0,
                n_updates=0,
                grad_steps=0,
                actor_train_state=actor_train_state,
                q_network_train_state=q_network_train_state,
                target_actor_train_state=target_actor_train_state,
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
            init_hstate = policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])

            def _env_step(runner_state, unused):
                """
                One step of the environment:
                1. Get observations, sample actions from all agents
                2. Step environment using sampled actions
                3. Return state, reward, ...
                """
                train_state, env_state, prev_obs, prev_done, hstate, rng = runner_state
                rng, actor_rng, target_rng, critic_rng, step_rng = jax.random.split(rng, 5)

                actor_hstate, critic_hstate, target_hstate = hstate

                 # Get available actions for the agent from environment state
                avail_actions = env.get_avail_actions(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(avail_actions)
                avail_actions = get_agent_data(avail_actions, agent_idx).astype(jnp.float32)

                # Note that we do not need to reset the hidden states for the agents
                # as the recurrent states are automatically reset when done is True.

                # Controlled Agent action, value, log_prob
                act, importance_weight, pi, new_actor_hstate = policy.get_action_importance_policy(
                    params=(train_state.actor_train_state.params, train_state.actor_train_state.batch_stats),
                    obs=get_agent_data(prev_obs, agent_idx).reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                    done=get_agent_data(prev_done, agent_idx).reshape(1, config["NUM_CONTROLLED_ACTORS"]),
                    avail_actions=avail_actions,
                    hstate=actor_hstate,
                    rng=actor_rng,
                    temp_schedule=temp_scheduler(train_state.n_updates)
                )
                logp = pi.log_prob(act)

                act = act.squeeze(axis=0)
                logp = logp.squeeze(axis=0)
                importance_weight = importance_weight.squeeze(axis=0)  # (1, NUM_ENVS) -> (NUM_ENVS,)

                # Combine actions into the env format
                # No-op for uncontrolled agents - use JAX conditional for traced agent_idx
                def make_combined_actions(idx, actions):
                    """Create combined actions with controlled agent at position idx"""
                    branches = []
                    for i in range(num_agents):
                        def make_branch(pos, a):
                            # Create list with actions at position pos, zeros elsewhere
                            parts = [jnp.zeros_like(a) if j != pos else a for j in range(num_agents)]
                            return jnp.concatenate(parts, axis=0)
                        branches.append(partial(make_branch, i))
                    return jax.lax.switch(idx, branches, actions)

                combined_actions = make_combined_actions(agent_idx, act)  # shape (num_agents*num_envs,)
                env_act = unbatchify(combined_actions, env.agents, config["NUM_ENVS"], num_agents)
                env_act = {k: v.flatten() for k, v in env_act.items()}

                # Step env
                step_rngs = jax.random.split(step_rng, config["NUM_ENVS"])
                (obs_next, obs_full_next), env_state_next, reward, done_next, info = jax.vmap(env.step, in_axes=(0,0,0))(
                    step_rngs, env_state, env_act
                )
                # note that num_actors = num_envs * num_agents
                # Get agent_idx info from info dict, excluding certain keys
                keys_to_exclude = {'pre_reset_state', 'pre_reset_obs'}  # Add any keys that shouldn't be indexed

                def filter_agent_info(path, x):
                    # Get the key name from the path
                    key_name = path[-1].key if hasattr(path[-1], 'key') else None
                    if key_name in keys_to_exclude or x.ndim <= 1:
                        return x
                    else:
                        return x[:, agent_idx]

                info = jax.tree_util.tree_map_with_path(filter_agent_info, info)

                # Next values
                next_avail_actions = env.get_avail_actions(env_state_next.env_state)
                next_avail_actions = jax.lax.stop_gradient(next_avail_actions)
                next_avail_actions = get_agent_data(next_avail_actions, agent_idx).astype(jnp.float32)

                next_pi = train_state.target_actor_train_state.apply_fn(
                    {
                        "params": train_state.target_actor_train_state.params,
                        "batch_stats": train_state.target_actor_train_state.batch_stats,
                    },
                    (get_agent_data(obs_next, agent_idx).reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                     next_avail_actions),
                    train=False,
                )
                # TODO: Deal with recurrent target
                new_target_hstate = None

                _, _, q_vals, new_critic_hstate  = policy.get_critic_logits_probs_values(
                    params=(train_state.q_network_train_state.params, train_state.q_network_train_state.batch_stats),
                    obs=get_agent_data(obs_next, agent_idx).reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                    done=get_agent_data(done_next, agent_idx).reshape(1, config["NUM_CONTROLLED_ACTORS"]),
                    avail_actions=next_avail_actions,
                    hstate=critic_hstate,
                    rng=critic_rng,
                )
                next_values = jnp.sum(next_pi.probs * q_vals, axis=-1).squeeze(axis=0)  # (1, NUM_ENVS) -> (NUM_ENVS,)

                # Store agent_idx data in transition
                transition = Transition(
                    obs=get_agent_data(prev_obs, agent_idx),
                    action=act,
                    action_logp=logp,
                    reward=get_agent_data(reward, agent_idx),
                    done=get_agent_data(done_next, agent_idx),
                    avail_actions=avail_actions,
                    next_obs=get_agent_data(obs_next, agent_idx),
                    next_avail_actions=next_avail_actions,
                    next_val=next_values,
                    importance_weight=importance_weight,
                    info=info
                )

                new_runner_state = (train_state, env_state_next, obs_next, done_next,
                                    (new_actor_hstate, new_critic_hstate, new_target_hstate), rng)
                return new_runner_state, transition

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
                (train_state, rng, rng_eval, update_steps) = update_runner_state
                # Init envs & partner indices
                rng, reset_rng = jax.random.split(rng, 2)
                reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
                (init_obs, init_full_obs), init_env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rngs)
                init_done = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}

                # 1) rollout
                runner_state = (train_state, init_env_state, init_obs, init_done, init_hstate, rng)

                runner_state, traj_batch = jax.lax.scan(
                    _env_step, runner_state, None, config["ROLLOUT_LENGTH"])
                (train_state, env_state, obs, done, hstate, rng) = runner_state

                train_state = train_state.replace(
                    timesteps=train_state.timesteps
                    + config["ROLLOUT_LENGTH"] * config["NUM_ENVS"]
                )  # update timesteps count

                # 2) Compute soft rewards and n-step lambda returns
                next_pi = train_state.target_actor_train_state.apply_fn(
                    {
                        "params": train_state.target_actor_train_state.params,
                        "batch_stats": train_state.target_actor_train_state.batch_stats,
                    },
                    (traj_batch.next_obs[-1].reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                     traj_batch.next_avail_actions[-1]),
                    train=False,
                )
                # TODO: Deal with recurrent target

                rng, sample_rng = jax.random.split(rng)
                _, last_action_logp = next_pi.sample_and_log_prob(seed=sample_rng)
                last_action_logp = last_action_logp.squeeze(axis=0)
                next_logp = jnp.concatenate(
                    [traj_batch.action_logp[1:], last_action_logp[None]], axis=0
                )
                soft_reward = traj_batch.reward - config["GAMMA"] * next_logp * jnp.exp(
                    train_state.actor_train_state.params["log_alpha"]
                )
                traj_batch = traj_batch.replace(soft_reward=soft_reward)

                _, target_values = jax.lax.scan(
                    compute_nstep_lambda,
                    (
                        traj_batch.next_val[-1],
                        jnp.ones_like(traj_batch.done[0]),
                        jnp.zeros_like(traj_batch.importance_weight[0]),
                    ),
                    traj_batch,
                    reverse=True,
                )

                # 3) REPPO update
                def _learn_epoch(carry, _):
                    train_state, rng = carry

                    def _learn_phase(carry, minibatch_and_target):
                        train_state, rng = carry
                        minibatch, target = minibatch_and_target

                        def _critic_loss_fn(params, train_state):
                            critic_out, updates = policy.q_network.apply(
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
                                "critic_loss": loss,
                                "q_values": q_vals.mean(),
                                "q_error": optax.l2_loss(target - q_vals).mean(),
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

                            critic_output = policy.q_network.apply(
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
                            # target_entropy = config["TARGET_ENTROPY_MULT"] * jnp.log(config["NUM_ACTIONS"])
                            # Normalize based on available actions
                            target_entropy = config["TARGET_ENTROPY_MULT"] * jnp.log(minibatch.avail_actions.sum(axis=-1)).mean()
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
                                "actor_loss": loss,
                                "target_entropy_loss": jnp.mean(alpha_loss),
                                "lagrangian_loss": jnp.mean(lagrangian_loss),
                                "alpha": alpha,
                                "lagrangian": lagrangian,
                                "kl": kl.mean(),
                                "entropy": pi.entropy().mean(),
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
                        lambda x: preprocess_transition(x, _rng), traj_batch
                    )  # num_actors*num_envs (batch_size), ...
                    targets = jax.tree_util.tree_map(
                        lambda x: preprocess_transition(x, _rng), target_values
                    )

                    rng, _rng = jax.random.split(rng)
                    (train_state, rng), metrics = jax.lax.scan(
                        _learn_phase, init=(train_state, rng), xs=(minibatches, targets)
                    )

                    return (train_state, rng), metrics


                rng, learn_rng = jax.random.split(rng)
                (train_state, rng), learn_metrics = jax.lax.scan(
                    f=_learn_epoch,
                    init=(train_state, learn_rng),
                    xs=None,
                    length=config["NUM_EPOCHS"],
                )
                target_actor_train_state = train_state.target_actor_train_state.replace(
                    params=deepcopy(train_state.actor_train_state.params),
                    batch_stats=deepcopy(train_state.actor_train_state.batch_stats),
                )
                train_state = train_state.replace(
                    target_actor_train_state=target_actor_train_state,
                    n_updates=train_state.n_updates + 1,
                )

                metric = traj_batch.info
                metric["update_steps"] = update_steps
                metric.update(learn_metrics)
                new_runner_state = (train_state, rng, rng_eval, update_steps + 1)
                return (new_runner_state, metric)

            # REPPO Update and Checkpoint saving
            ckpt_and_eval_interval = config["NUM_UPDATES"] // max(1, config["NUM_CHECKPOINTS"] - 1)  # -1 because we store a ckpt at the last update
            num_ckpts = config["NUM_CHECKPOINTS"]

            # Build a PyTree that holds parameters for all FCP checkpoints
            def init_ckpt_array(params_pytree):
                return jax.tree.map(
                    lambda x: jnp.zeros((num_ckpts,) + x.shape, x.dtype),
                    params_pytree)

            def init_eval_ckpt_array(params_pytree):
                return jax.tree.map(
                    lambda x: jnp.zeros((int(config["NUM_UPDATES"]) + 1,) + x.shape, x.dtype),
                    params_pytree)

            max_episode_steps = config["MAX_EPISODE_STEPS"]

            def _update_step_with_ckpt(state_with_ckpt, unused):
                (update_state, checkpoint_array, ckpt_idx, eval_checkpoint_array, eval_ckpt_idx, init_ckpt_eval_last_info, init_eval_last_info) = state_with_ckpt

                # Single REPPO update
                new_update_state, metric = _update_step(
                    update_state,
                    None
                )
                (train_state, rng, rng_eval, update_steps) = new_update_state

                # update steps is 1-indexed because it was incremented at the end of the update step
                to_store = jnp.logical_or(jnp.equal(jnp.mod(update_steps-1, ckpt_and_eval_interval), 0),
                                        jnp.equal(update_steps, config["NUM_UPDATES"]))


                def store_and_eval_ckpt(args):
                    ckpt_arr, cidx, eval_ckpt_arr, eval_cidx, rng, rng_eval, prev_ckpt_eval_ret_info, prev_eval_ret_info = args
                    ckpt_params = {
                        "actor": {
                            "params": train_state.actor_train_state.params,
                            "batch_stats": train_state.actor_train_state.batch_stats
                        },
                        "q_network": {
                            "params": train_state.q_network_train_state.params,
                            "batch_stats": train_state.q_network_train_state.batch_stats
                        }
                    }
                    new_ckpt_arr = jax.tree.map(
                        lambda c_arr, p: c_arr.at[cidx].set(p),
                        ckpt_arr, ckpt_params
                    )
                    new_eval_ckpt_arr = jax.tree.map(
                        lambda e_arr, p: e_arr.at[eval_cidx].set(p),
                        eval_ckpt_arr, ckpt_params
                    )

                    if config["FIXED_EVAL"]:
                        eval_rng = rng_eval
                    else:
                        rng_eval, eval_rng = jax.random.split(rng_eval, 2)
                    ckpt_eval_eps_last_infos = run_episodes_vmap(
                        eval_rng, env, q_env, agent_idx, agent_param=ckpt_params, agent_policy=policy,
                        max_episode_steps=max_episode_steps,
                        num_eps=config["NUM_EVAL_EPISODES"],
                        agent_test_mode=True)
                    return (new_ckpt_arr, cidx + 1, new_eval_ckpt_arr, eval_cidx + 1, rng, rng_eval, ckpt_eval_eps_last_infos, ckpt_eval_eps_last_infos)

                def skip_ckpt_and_eval(args):
                    ckpt_arr, cidx, eval_ckpt_arr, eval_cidx, rng, rng_eval, prev_ckpt_eval_ret_info, prev_eval_ret_info = args
                    eval_params = {
                        "actor": {
                            "params": train_state.actor_train_state.params,
                            "batch_stats": train_state.actor_train_state.batch_stats
                        },
                        "q_network": {
                            "params": train_state.q_network_train_state.params,
                            "batch_stats": train_state.q_network_train_state.batch_stats
                        }
                    }
                    new_eval_ckpt_arr = jax.tree.map(
                        lambda e_arr, p: e_arr.at[eval_cidx].set(p),
                        eval_ckpt_arr, eval_params
                    )

                    def do_eval(eval_args):
                        ckpt_arr, cidx, eval_ckpt_arr, eval_cidx, rng, rng_eval, prev_ckpt_eval_ret_info, prev_eval_ret_info = eval_args
                        if config["FIXED_EVAL"]:
                            eval_rng = rng_eval
                        else:
                            rng_eval, eval_rng = jax.random.split(rng_eval, 2)
                        eval_eps_last_infos = run_episodes_vmap(
                            eval_rng, env, q_env, agent_idx, agent_param=eval_params, agent_policy=policy,
                            max_episode_steps=max_episode_steps,
                            num_eps=config["NUM_EVAL_EPISODES"],
                            agent_test_mode=True)
                        return (ckpt_arr, cidx, eval_ckpt_arr, eval_cidx, rng, rng_eval, prev_ckpt_eval_ret_info, eval_eps_last_infos)

                    def skip_eval(eval_args):
                        return eval_args

                    (ckpt_arr, cidx, eval_ckpt_arr, eval_cidx, rng, rng_eval, prev_ckpt_eval_ret_info, eval_eps_last_infos) = jax.lax.cond(
                        config["TRAIN_EVAL"],
                        do_eval,
                        skip_eval,
                        (ckpt_arr, cidx, new_eval_ckpt_arr, eval_cidx + 1, rng, rng_eval, prev_ckpt_eval_ret_info, prev_eval_ret_info)
                    )

                    return (ckpt_arr, cidx, eval_ckpt_arr, eval_cidx, rng, rng_eval, prev_ckpt_eval_ret_info, eval_eps_last_infos)

                (checkpoint_array, ckpt_idx, eval_checkpoint_array, eval_ckpt_idx, rng, rng_eval, ckpt_eval_eps_last_infos, eval_eps_last_infos) = jax.lax.cond(
                    to_store, store_and_eval_ckpt, skip_ckpt_and_eval, (checkpoint_array, ckpt_idx, eval_checkpoint_array, eval_ckpt_idx, rng, rng_eval, init_ckpt_eval_last_info, init_eval_last_info)
                )

                metric["ckpt_eval_ep_last_info"] = ckpt_eval_eps_last_infos
                metric["eval_ep_last_info"] = eval_eps_last_infos
                return ((train_state, rng, rng_eval, update_steps),
                         checkpoint_array, ckpt_idx, eval_checkpoint_array, eval_ckpt_idx, ckpt_eval_eps_last_infos, eval_eps_last_infos), metric


            init_ckpt_params = {
                "actor": {
                    "params": train_state.actor_train_state.params,
                    "batch_stats": train_state.actor_train_state.batch_stats
                },
                "q_network": {
                    "params": train_state.q_network_train_state.params,
                    "batch_stats": train_state.q_network_train_state.batch_stats
                }
            }
            checkpoint_array = init_ckpt_array(init_ckpt_params)
            eval_checkpoint_array = init_eval_ckpt_array(init_ckpt_params)
            ckpt_idx = 0
            eval_ckpt_idx = 0

            # Store initial params in eval checkpoint array
            eval_checkpoint_array = jax.tree.map(
                lambda e_arr, p: e_arr.at[eval_ckpt_idx].set(p),
                eval_checkpoint_array, init_ckpt_params
            )
            eval_ckpt_idx = eval_ckpt_idx + 1

            rng, rng_train = jax.random.split(rng, 2)

            rng_eval = jax.random.PRNGKey(config["EVAL_SEED"])# + agent_idx)# + 14)
            rng_eval, eval_rng = jax.random.split(rng_eval, 2)
            if config["FIXED_EVAL"]:
                eval_rng = rng_eval

            # Init eval return infos
            eval_eps_last_infos = run_episodes_vmap(eval_rng, env, q_env, agent_idx,
                                    agent_param=init_ckpt_params, agent_policy=policy,
                                    max_episode_steps=max_episode_steps,
                                    num_eps=config["NUM_EVAL_EPISODES"],
                                    agent_test_mode=True)

            # initial runner state for scanning
            update_steps = 0

            update_runner_state = (train_state, rng_train, rng_eval, update_steps)
            state_with_ckpt = (update_runner_state, checkpoint_array, ckpt_idx, eval_checkpoint_array, eval_ckpt_idx, eval_eps_last_infos, eval_eps_last_infos)

            state_with_ckpt, metrics = jax.lax.scan(
                _update_step_with_ckpt,
                state_with_ckpt,
                xs=None,
                length=config["NUM_UPDATES"]
            )

            (final_runner_state, checkpoint_array, final_ckpt_idx, eval_checkpoint_array, final_eval_ckpt_idx, ckpt_eval_eps_last_infos, eval_eps_last_infos) = state_with_ckpt
            final_train_state = final_runner_state[0]
            final_params = {
                "actor": {
                    "params": final_train_state.actor_train_state.params,
                    "batch_stats": final_train_state.actor_train_state.batch_stats
                },
                "q_network": {
                    "params": final_train_state.q_network_train_state.params,
                    "batch_stats": final_train_state.q_network_train_state.batch_stats
                }
            }
            out = {
                "final_params": final_params,
                "metrics": metrics,  # shape (NUM_UPDATES, ...)
                "checkpoints": checkpoint_array,
                "eval_checkpoints": eval_checkpoint_array,
            }

            if env._render:
                # Collect final eval gifs for logging
                rng_eval = final_runner_state[2] # extract final rng_eval from the final runner state after training
                if config["FIXED_EVAL"]:
                    eval_rng = rng_eval
                else:
                    rng_eval, eval_rng = jax.random.split(rng_eval, 2)
                out["render_outs"] = run_episodes_vmap(eval_rng, env, q_env, agent_idx,
                                                    agent_param=final_params, agent_policy=policy,
                                                    max_episode_steps=env.horizon,
                                                    num_eps=5, render=True,
                                                    agent_test_mode=True)

            return out
        return train

    # ------------------------------
    # Actually run the REPPO training
    # ------------------------------
    rngs = jax.random.split(train_rng, config["NUM_TRAIN_SEEDS"])

    # Run training seeds in parallel using vmap
    train_fn = make_reppo_train(config)
    out = jax.vmap(train_fn, in_axes=(0, None))(rngs, agent_idx)
    return out

def run_training(config, wandb_logger, agent_idx=0):
    '''Run single agent projection training.

    Args:
        config: dict, config for the training
    '''
    algorithm_config = dict(config["algorithm"])

    # Create only one environment instance
    env_kwargs = dict(algorithm_config["ENV_KWARGS"])

    env_kwargs["instance"] = config['task'][f"SINGLE_AGENT_{agent_idx + 1}_PROJECTION"]
    env_kwargs["render_dir"] = os.path.join("render", "reppo", f"agent_{agent_idx + 1}")
    env_kwargs["done_condition"] = "any"  # SAP: terminate as soon as active agent takes their picture
    env = make_env(algorithm_config["ENV_NAME"], env_kwargs)
    env = LogWrapper(env)

    q_env = make_env(algorithm_config["ENV_NAME"], env_kwargs)
    q_env = LogWrapper(q_env)

    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"])# + agent_idx)# + 7)
    _, init_rng, train_rng = jax.random.split(rng, 3)

    # Initialize agent
    policy, init_params = initialize_reppo_agent(algorithm_config, env, init_rng, agent_index=agent_idx)

    log.info(f"Starting single agent projection training for agent {agent_idx}...")
    start_time = time.perf_counter()

    # Run the training
    out = train_reppo_agent(
        config=algorithm_config,
        env=env,
        q_env=q_env,
        train_rng=train_rng,
        policy=policy,
        init_params=init_params,
        agent_idx=agent_idx
    )

    elapsed_time = time.perf_counter() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, rem = divmod(rem, 60)
    seconds, rem = divmod(rem, 1)
    milliseconds = int(rem * 1000)
    microseconds = int((rem * 1_000_000) % 1000)
    log.info(f"Single Agent Projection Training completed for agent {agent_idx} in {elapsed_time:.2f}s")
    log.info(f"Single Agent Projection Training completed for agent {agent_idx} in {int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s {milliseconds:03d}ms {microseconds:03d}µs")

    # process and log metrics
    log.info(f"Starting single agent projection logging for agent {agent_idx}...")
    start_time = time.perf_counter()
    metric_names = get_metric_names(config["ENV_NAME"])
    log_metrics(env, q_env, config, out, wandb_logger, metric_names, agent_idx)
    elapsed_time = time.perf_counter() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, rem = divmod(rem, 60)
    seconds, rem = divmod(rem, 1)
    milliseconds = int(rem * 1000)
    microseconds = int((rem * 1_000_000) % 1000)
    log.info(f"Single Agent Projection Logging completed for agent {agent_idx} in {elapsed_time:.2f}s")
    log.info(f"Single Agent Projection Logging completed for agent {agent_idx} in {int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s {milliseconds:03d}ms {microseconds:03d}µs")

    return out["final_params"], policy, init_params, out["eval_checkpoints"]

def log_metrics(env, q_env, config, train_out, logger, metric_names: tuple, agent_idx: int):
    """Process training metrics and log them using the provided logger.

    Args:
        env: the environment used for training, needed for logging videos
        q_env: the q environment used for training, needed for logging videos
        config: dict, the training configuration
        train_out: dict, the logs from training
        logger: Logger, instance to log metrics
        metric_names: tuple, names of metrics to extract from training logs
        agent_idx: int, index of the trained agent
    """
    train_metrics = train_out["metrics"]

    #### Extract train metrics ####
    train_stats = get_stats(train_metrics, metric_names)
    # each key in train_stats is a metric name, and the value is an array of shape (num_seeds, num_updates, 2)
    # where the last dimension contains the mean and std of the metric
    train_stats = {k: np.mean(np.array(v), axis=0) for k, v in train_stats.items()}

    all_agent_actor_loss = np.asarray(train_metrics["actor_loss"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    all_agent_target_entropy_loss = np.asarray(train_metrics["target_entropy_loss"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    all_agent_lagrangian_loss = np.asarray(train_metrics["lagrangian_loss"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    all_agent_alpha = np.asarray(train_metrics["alpha"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    all_agent_lagrangian = np.asarray(train_metrics["lagrangian"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    all_agent_kl = np.asarray(train_metrics["kl"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    all_agent_entropy = np.asarray(train_metrics["entropy"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    all_agent_critic_loss = np.asarray(train_metrics["critic_loss"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    all_agent_q_values = np.asarray(train_metrics["q_values"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    all_agent_q_error = np.asarray(train_metrics["q_error"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)

    # Process eval return metrics - average across train seeds, eval episodes, and num_agents per game for each checkpoint
    all_ckpt_returns = np.asarray(train_metrics["ckpt_eval_ep_last_info"][0]["returned_episode_returns"]) # shape (n_train_seeds, num_updates, num_eval_episodes, num_agents_per_game)
    all_returns = np.asarray(train_metrics["eval_ep_last_info"][0]["returned_episode_returns"]) # shape (n_train_seeds, num_updates, num_eval_episodes, num_agents_per_game)
    all_ckpt_agent_returns = all_ckpt_returns[:, :, :, agent_idx] # shape (n_train_seeds, num_updates, num_eval_episodes)
    all_agent_returns = all_returns[:, :, :, agent_idx] # shape (n_train_seeds, num_updates, num_eval_episodes)
    average_ckpt_agent_rets_per_iter = np.mean(all_ckpt_agent_returns, axis=(0, 2)) # shape (num_updates,)
    average_agent_rets_per_iter = np.mean(all_agent_returns, axis=(0, 2)) # shape (num_updates,)

    all_ckpt_q_returns = np.asarray(train_metrics["ckpt_eval_ep_last_info"][1]["returned_episode_returns"]) # shape (n_train_seeds, num_updates, num_eval_episodes, num_agents_per_game)
    all_q_returns = np.asarray(train_metrics["eval_ep_last_info"][1]["returned_episode_returns"]) # shape (n_train_seeds, num_updates, num_eval_episodes, num_agents_per_game)
    all_ckpt_agent_q_returns = all_ckpt_q_returns[:, :, :, agent_idx] # shape (n_train_seeds, num_updates, num_eval_episodes)
    all_agent_q_returns = all_q_returns[:, :, :, agent_idx] # shape (n_train_seeds, num_updates, num_eval_episodes)
    average_ckpt_agent_q_rets_per_iter = np.mean(all_ckpt_agent_q_returns, axis=(0, 2)) # shape (num_updates,)
    average_agent_q_rets_per_iter = np.mean(all_agent_q_returns, axis=(0, 2)) # shape (num_updates,)

    # Process loss metrics - average across train seeds, partners and minibatches dims
    # Loss metrics shape should be (n_train_seeds, num_updates, ...)
    average_agent_actor_losses = np.mean(all_agent_actor_loss, axis=(0, 2, 3))  # shape (num_updates,)
    average_agent_target_entropy_losses = np.mean(all_agent_target_entropy_loss, axis=(0, 2, 3))  # shape (num_updates,)
    average_agent_lagrangian_losses = np.mean(all_agent_lagrangian_loss, axis=(0, 2, 3))  # shape (num_updates,)
    average_agent_alphas = np.mean(all_agent_alpha, axis=(0, 2, 3))  # shape (num_updates,)
    average_agent_lagrangians = np.mean(all_agent_lagrangian, axis=(0, 2, 3))  # shape (num_updates,)
    average_agent_kls = np.mean(all_agent_kl, axis=(0, 2, 3))  # shape (num_updates,)
    average_agent_entropies = np.mean(all_agent_entropy, axis=(0, 2, 3))  # shape (num_updates,)
    average_agent_critic_losses = np.mean(all_agent_critic_loss, axis=(0, 2, 3))  # shape (num_updates,)
    average_agent_q_values = np.mean(all_agent_q_values, axis=(0, 2, 3))  # shape (num_updates,)
    average_agent_q_errors = np.mean(all_agent_q_error, axis=(0, 2, 3))  # shape (num_updates,)

    # Log metrics for each update step
    num_updates = len(average_agent_actor_losses)
    for step in range(num_updates):
        for stat_name, stat_data in train_stats.items():
            # second dimension contains the mean and std of the metric
            stat_mean = stat_data[step, 0]
            logger.log_item(f"Train/Agent_{agent_idx + 1}_Proj/{stat_name}", stat_mean, train_step=step, commit=True)

        logger.log_item(f"Eval/Agent_{agent_idx + 1}_Proj/Return", average_agent_rets_per_iter[step], train_step=step, commit=True)
        logger.log_item(f"Eval/Agent_{agent_idx + 1}_Proj/CheckpointReturn", average_ckpt_agent_rets_per_iter[step], train_step=step, commit=True)
        logger.log_item(f"Eval/Agent_{agent_idx + 1}_Proj/QReturn", average_agent_q_rets_per_iter[step], train_step=step, commit=True)
        logger.log_item(f"Eval/Agent_{agent_idx + 1}_Proj/CheckpointQReturn", average_ckpt_agent_q_rets_per_iter[step], train_step=step, commit=True)

        logger.log_item(f"Train/Agent_{agent_idx + 1}_Proj/ActorLoss", average_agent_actor_losses[step], train_step=step, commit=True)
        logger.log_item(f"Train/Agent_{agent_idx + 1}_Proj/TargetEntropyLoss", average_agent_target_entropy_losses[step], train_step=step, commit=True)
        logger.log_item(f"Train/Agent_{agent_idx + 1}_Proj/LagrangianLoss", average_agent_lagrangian_losses[step], train_step=step, commit=True)
        logger.log_item(f"Train/Agent_{agent_idx + 1}_Proj/Alpha", average_agent_alphas[step], train_step=step, commit=True)
        logger.log_item(f"Train/Agent_{agent_idx + 1}_Proj/Lagrangian", average_agent_lagrangians[step], train_step=step, commit=True)
        logger.log_item(f"Train/Agent_{agent_idx + 1}_Proj/KL", average_agent_kls[step], train_step=step, commit=True)
        logger.log_item(f"Train/Agent_{agent_idx + 1}_Proj/Entropy", average_agent_entropies[step], train_step=step, commit=True)
        logger.log_item(f"Train/Agent_{agent_idx + 1}_Proj/CriticLoss", average_agent_critic_losses[step], train_step=step, commit=True)
        logger.log_item(f"Train/Agent_{agent_idx + 1}_Proj/QValueMean", average_agent_q_values[step], train_step=step, commit=True)
        logger.log_item(f"Train/Agent_{agent_idx + 1}_Proj/QValueError", average_agent_q_errors[step], train_step=step, commit=True)

        logger.commit()

    # Saving artifacts
    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    if env._render:
        # shape of render_outs should be (num_train_seeds, num_eps, max_episode_steps, ...)
        eval_render_init_env_state = train_out['render_outs'][2].env_state.env_state # LogEnvState
        eval_render_q_env_state = train_out['render_outs'][1][-1]['pre_reset_state'].env_state # WrappedEnvState
        eval_render_q_dones = train_out['render_outs'][1][4]['__all__']
        eval_render_env_state = train_out['render_outs'][0][-1]['pre_reset_state'].env_state # WrappedEnvState
        eval_render_dones = train_out['render_outs'][0][4]['__all__']
        num_episodes = eval_render_dones.shape[1]
        q_env.animate((eval_render_init_env_state, eval_render_q_env_state), eval_render_q_dones, num_episodes, extra_dir="Q_Values", debug=True)
        env.animate((eval_render_init_env_state, eval_render_env_state), eval_render_dones, num_episodes, extra_dir="Policy", debug=True)

        for eval_ep in range(num_episodes):
            logger.log_video(
                tag=f"Videos/Agent_{agent_idx + 1}_Proj/Policy/Agent_{agent_idx}_Episode_{eval_ep}",
                path=os.path.join(env._render_dir, "Policy", f"{env._render_name}_ep_{eval_ep}.gif")
            )

            logger.log_video(
                tag=f"Videos/Agent_{agent_idx + 1}_Proj/Q_Values/Agent_{agent_idx}_Episode_{eval_ep}",
                path=os.path.join(q_env._render_dir, "Q_Values", f"{q_env._render_name}_ep_{eval_ep}.gif")
            )

    out_savepath = save_train_run(train_out, savedir, savename=f"REPPO_Agent_{agent_idx + 1}_Proj_Train_Run")
    if config["logger"]["log_train_out"]:
        logger.log_artifact(name=f"REPPO_Agent_{agent_idx + 1}_Proj_Train_Run", path=out_savepath, type_name="single_agent_proj_train_run")
        # Cleanup locally logged out file
    if not config["local_logger"]["save_train_out"]:
        shutil.rmtree(out_savepath)

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
from social_laws.common.run_episodes_w_robustness_from_dqn import run_episodes_vmap
from common.plot_utils import get_stats, get_metric_names
from common.save_load_utils import save_train_run
from envs import make_env
from envs.log_wrapper import LogWrapper
from marl.ppo_utils import _create_minibatches, Transition, unbatchify

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def train_ppo_joint_agents(config, env, optimal_env, train_rng,
                           joint_policies, init_joint_params,
                           vf_policies, vf_params,
                           agent_idx):
    '''
    Train PPO joint agents using the given initial parameters.

    Args:
        config: dict, config for the training
        env: gymnasium environment
        optimal_env: gymnasium environment for evaluating optimal returns
        train_rng: jax.random.PRNGKey, random key for training
        joint_policies: tuple of AgentPolicy, policies for the agents
        init_joint_params: tuple of dict, initial parameters for the agents
        dqn_policies: tuple of AgentPolicy, DQN policies for the agents
        dqn_params: tuple of dict, DQN parameters for the agents
        agent_idx: int, index of the agent to optimize
    '''
    # ------------------------------
    # Build the PPO joint training function
    # ------------------------------

    def make_ppo_joint_train(config):
        '''The controlled agent is based on the agent_idx parameter'''
        num_agents = env.num_agents
        # assert num_agents == 2, "This snippet assumes exactly 2 agents."

        # config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
        # config["NUM_UNCONTROLLED_ACTORS"] = config["NUM_ENVS"] # assumption: we control 1 agent
        config["NUM_CONTROLLED_ACTORS"] = config["NUM_ENVS"] # assumption: we control 1 agent
        config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["ROLLOUT_LENGTH"] // config["NUM_ENVS"]

        config["NUM_ACTIONS"] = env.action_space(f"agent_{agent_idx}").n
        assert config["NUM_CONTROLLED_ACTORS"] % config["NUM_MINIBATCHES"] == 0, "NUM_CONTROLLED_ACTORS must be divisible by NUM_MINIBATCHES"
        assert config["NUM_CONTROLLED_ACTORS"] >= config["NUM_MINIBATCHES"], "NUM_CONTROLLED_ACTORS must be >= NUM_MINIBATCHES"

        def linear_schedule(count):
            frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
            return config["LR"] * frac

        def train(rng, agent_idx, *vf_params):

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
                    apply_fn=joint_policies[i].network.apply,
                    params=init_joint_params[i],
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
            init_hstates = [joint_policies[i].init_hstate(config["NUM_CONTROLLED_ACTORS"]) for i in range(num_agents)]
            vf_init_hstates = [vf_policies[i].init_hstate(config["NUM_CONTROLLED_ACTORS"]) for i in range(num_agents)]

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
                train_states, env_state, prev_obs, prev_done, hstates, vf_hstates, rng = runner_state
                prev_obs, prev_full_obs = prev_obs
                rng, *actor_rngs, step_rng = jax.random.split(rng, num_agents + 2)

                # Get available actions for the agent from environment state
                avail_actions = env.get_avail_actions(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(avail_actions)

                prev_obs_per = [get_agent_data(prev_obs, i).reshape(1, config["NUM_CONTROLLED_ACTORS"], -1) for i in range(num_agents)]
                prev_full_obs_per = [get_agent_data(prev_full_obs, i).reshape(1, config["NUM_CONTROLLED_ACTORS"], -1) for i in range(num_agents)]
                prev_done_per = [get_agent_data(prev_done, i).reshape(1, config["NUM_CONTROLLED_ACTORS"]) for i in range(num_agents)]
                avail_per = [get_agent_data(avail_actions, i).astype(jnp.float32) for i in range(num_agents)]

                # Restrict available actions based on value function
                vf_restricted_avail_actions = []
                new_vf_hstates = []
                for i in range(num_agents):
                    vf_aa, vf_hs = _get_vf_restricted_avail_actions(
                        obs=prev_obs_per[i],
                        done=prev_done_per[i],
                        avail_actions=avail_per[i],
                        hstate=vf_hstates[i],
                        vf_params=vf_params[i],
                        vf_policy=vf_policies[i]
                    )
                    vf_restricted_avail_actions.append(vf_aa)
                    new_vf_hstates.append(vf_hs)

                # Note that we do not need to reset the hidden states for the agents
                # as the recurrent states are automatically reset when done is True.

                # Per-agent action, value, log_prob (using VF-restricted available actions)
                acts, vals, logps, new_hstates = [], [], [], []
                for i in range(num_agents):
                    act_i, val_i, pi_i, new_hs_i = joint_policies[i].get_action_value_policy(
                        params=train_states[i].params,
                        obs=prev_full_obs_per[i] if config["JOINT_USE_FULL_OBS"] else prev_obs_per[i],
                        done=prev_done_per[i],
                        avail_actions=vf_restricted_avail_actions[i],
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

                # Store per-agent transitions (using VF-restricted available actions)
                negative_reward = get_agent_data(reward, agent_idx) * -1

                transitions = tuple(
                    Transition(
                        done=get_agent_data(done_next, i),
                        action=acts[i],
                        value=vals[i],
                        reward=negative_reward,
                        log_prob=logps[i],
                        obs=get_agent_data(prev_full_obs, i) if config["JOINT_USE_FULL_OBS"] else get_agent_data(prev_obs, i),
                        info=info,
                        avail_actions=vf_restricted_avail_actions[i]
                    )
                    for i in range(num_agents)
                )

                new_runner_state = (train_states, env_state_next, obs_next, done_next,
                                    new_hstates, new_vf_hstates, rng)
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

            def _make_update_minbatch(i):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, returns = batch_info
                    def _loss_fn(params, init_hstate, traj_batch, gae, target_v):
                        _, value, pi, _ = joint_policies[i].get_action_value_policy(
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
                return _update_minbatch

            def _make_update_epoch(i):
                def _update_epoch(update_state, unused):
                    train_state, init_hstate, traj_batch, advantages, targets, rng = update_state
                    rng, perm_rng = jax.random.split(rng)
                    minibatches = _create_minibatches(traj_batch, advantages, targets, init_hstate, config["NUM_CONTROLLED_ACTORS"], config["NUM_MINIBATCHES"], perm_rng)
                    train_state, losses_and_grads = jax.lax.scan(
                        _make_update_minbatch(i), train_state, minibatches
                    )
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
                init_obs, init_env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rngs)
                init_done = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}

                # 1) rollout
                runner_state = (train_states, init_env_state, init_obs, init_done, init_hstates, vf_init_hstates, rng)

                runner_state, traj_batch = jax.lax.scan(
                    _env_step, runner_state, None, config["ROLLOUT_LENGTH"])
                (train_states, env_state, obs, done, agent_hstates, agent_vf_hstates, rng) = runner_state
                obs, obs_full = obs

                # 2) advantage
                avail_actions = env.get_avail_actions(env_state.env_state)
                rng, *update_rngs = jax.random.split(rng, num_agents + 1)

                agent_metrics = {}
                for i in range(num_agents):
                    obs_i = get_agent_data(obs, i).reshape(1, config["NUM_CONTROLLED_ACTORS"], -1)
                    obs_full_i = get_agent_data(obs_full, i).reshape(1, config["NUM_CONTROLLED_ACTORS"], -1)
                    done_i = get_agent_data(done, i).reshape(1, config["NUM_CONTROLLED_ACTORS"])
                    avail_i = get_agent_data(avail_actions, i).astype(jnp.float32)

                    vf_restricted_i, _ = _get_vf_restricted_avail_actions(
                        obs=obs_i, done=done_i, avail_actions=avail_i,
                        hstate=agent_vf_hstates[i],
                        vf_params=vf_params[i], vf_policy=vf_policies[i]
                    )

                    _, last_val_i, _, _ = joint_policies[i].get_action_value_policy(
                        params=train_states[i].params,
                        obs=obs_full_i if config["JOINT_USE_FULL_OBS"] else obs_i,
                        done=done_i,
                        avail_actions=jax.lax.stop_gradient(vf_restricted_i),
                        hstate=agent_hstates[i],
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

            # PPO Update and Checkpoint saving
            ckpt_and_eval_interval = config["NUM_UPDATES"] // max(1, config["NUM_CHECKPOINTS"] - 1)  # -1 because we store a ckpt at the last update
            num_ckpts = config["NUM_CHECKPOINTS"]

            # Build a PyTree that holds parameters for all FCP checkpoints
            def init_ckpt_array(params_pytree):
                return jax.tree.map(
                    lambda x: jnp.zeros((num_ckpts,) + x.shape, x.dtype),
                    params_pytree)

            max_episode_steps = env.horizon # config["ROLLOUT_LENGTH"]

            def _run_eval(rng_eval):
                if config["FIXED_EVAL"]:
                    eval_rng = rng_eval
                else:
                    rng_eval, eval_rng = jax.random.split(rng_eval, 2)
                return eval_rng, run_episodes_vmap(
                    eval_rng, env, optimal_env, agent_idx,
                    agent_params=[train_state.params for train_state in train_states],
                    agent_policies=joint_policies,
                    vf_params=vf_params,
                    vf_policies=vf_policies,
                    max_episode_steps=max_episode_steps,
                    num_eps=config["NUM_EVAL_EPISODES"],
                    epsilon_optimal=config["EPSILON_OPTIMAL"],
                    use_full_obs=config["JOINT_USE_FULL_OBS"])

            def _update_step_with_ckpt(state_with_ckpt, unused):
                (update_state, checkpoint_arrays, ckpt_idx, init_ckpt_eval_last_info, init_eval_last_info) = state_with_ckpt

                # Single PPO update
                new_update_state, metric = _update_step(update_state, None)
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
                    rng_eval, ckpt_eval_eps_last_infos = _run_eval(rng_eval)
                    return (new_ckpt_arrs, cidx + 1, rng, rng_eval, ckpt_eval_eps_last_infos, ckpt_eval_eps_last_infos)

                def skip_ckpt_and_eval(args):
                    def do_eval(eval_args):
                        ckpt_arrs, cidx, rng, rng_eval, prev_ckpt_eval_ret_info, prev_eval_ret_info = eval_args
                        rng_eval, eval_eps_last_infos = _run_eval(rng_eval)
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

            def init_ckpt_array(params_pytree):
                return jax.tree.map(
                    lambda x: jnp.zeros((num_ckpts,) + x.shape, x.dtype),
                    params_pytree)

            checkpoint_arrays = [init_ckpt_array(train_states[i].params) for i in range(num_agents)]
            ckpt_idx = 0

            rng, rng_train = jax.random.split(rng, 2)

            rng_eval = jax.random.PRNGKey(config["EVAL_SEED"])# + agent_idx)# + 42)
            rng_eval, eval_rng = jax.random.split(rng_eval, 2)

            # Init eval return infos
            rng_eval, eval_eps_last_infos = _run_eval(eval_rng)

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
                rng_eval = final_runner_state[2]
                if config["FIXED_EVAL"]:
                    render_rng = rng_eval
                else:
                    _, render_rng = jax.random.split(rng_eval, 2)
                out["render_outs"] = run_episodes_vmap(
                    render_rng, env, optimal_env, agent_idx,
                    agent_params=tuple(final_agent_ts[i].params for i in range(num_agents)),
                    agent_policies=joint_policies,
                    vf_params=vf_params,
                    vf_policies=vf_policies,
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
    num_agents = env.num_agents
    train_fn = make_ppo_joint_train(config)
    in_axes = (0, None) + (0,) * num_agents
    out = jax.vmap(train_fn, in_axes=in_axes)(rngs, agent_idx, *vf_params)
    return out

def run_training(config, wandb_logger, dqn_params, dqn_policies, agent_idx=0):
    '''Run joint training.

    Args:
        config: dict, config for the training
        wandb_logger: Logger, logger for logging metrics
        dqn_params: tuple, DQN single agent projection parameters for all agents
        dqn_policies: tuple, DQN single agent projection policies for all agents
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

    num_agents = env.num_agents
    obs_type = "full" if algorithm_config["JOINT_USE_FULL_OBS"] else "agent"

    # Initialize one joint policy per agent
    joint_policies, init_joint_params = map(list, zip(*[
        initialize_agent(algorithm_config, env, init_rng, agent_index=i, observation_type=obs_type)
        for i in range(num_agents)
    ]))

    log.info(f"Starting PPO joint training optimizing for agent {agent_idx}...")
    start_time = time.perf_counter()

    # Run the training
    out = train_ppo_joint_agents(
        config=algorithm_config,
        env=env,
        optimal_env=optimal_env,
        train_rng=train_rng,
        joint_policies=joint_policies,
        init_joint_params=init_joint_params,
        vf_policies=dqn_policies,
        vf_params=dqn_params,
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

    # Train metrics include loss values and gradient norms, which we can average across seeds, partners and minibatches for each update step.
    num_agents_log = sum(1 for k in train_metrics if k.endswith("/value_loss") and k.startswith("agent_"))
    avg_per_agent_value_losses = []
    avg_per_agent_actor_losses = []
    avg_per_agent_entropy_losses = []
    avg_per_agent_grad_norms = []
    for i in range(num_agents_log):
        avg_per_agent_value_losses.append(np.mean(np.asarray(train_metrics[f"agent_{i}/value_loss"]), axis=(0, 2, 3)))
        avg_per_agent_actor_losses.append(np.mean(np.asarray(train_metrics[f"agent_{i}/actor_loss"]), axis=(0, 2, 3)))
        avg_per_agent_entropy_losses.append(np.mean(np.asarray(train_metrics[f"agent_{i}/entropy_loss"]), axis=(0, 2, 3)))
        avg_per_agent_grad_norms.append(np.mean(np.asarray(train_metrics[f"agent_{i}/avg_grad_norm"]), axis=(0, 2, 3)))


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
    num_updates = len(avg_per_agent_value_losses[0])
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
            logger.log_item(f"Train/Joint/Agent_{agent_idx + 1}_Optimize/Agent_{i + 1}/ValueLoss", avg_per_agent_value_losses[i][step], train_step=step, commit=True)
            logger.log_item(f"Train/Joint/Agent_{agent_idx + 1}_Optimize/Agent_{i + 1}/ActorLoss", avg_per_agent_actor_losses[i][step], train_step=step, commit=True)
            logger.log_item(f"Train/Joint/Agent_{agent_idx + 1}_Optimize/Agent_{i + 1}/EntropyLoss", avg_per_agent_entropy_losses[i][step], train_step=step, commit=True)
            logger.log_item(f"Train/Joint/Agent_{agent_idx + 1}_Optimize/Agent_{i + 1}/GradNorm", avg_per_agent_grad_norms[i][step], train_step=step, commit=True)
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

    savepaths = []
    for i in range(num_agents_log):
        agent_out = {
            "final_params": train_out[f"final_params_agent_{i}"],
            "metrics": train_out["metrics"],
            "checkpoints": train_out[f"checkpoints_agent_{i}"],
        }
        savepath = save_train_run(agent_out, savedir, savename=f"PPO_Agent_{agent_idx + 1}_Optimize_Train_Run-Agent_{i+1}")
        savepaths.append(savepath)
        if config["logger"]["log_train_out"]:
            logger.log_artifact(name=f"PPO_Agent_{agent_idx + 1}_Optimize_Train_Run-Agent_{i+1}", path=savepath, type_name="joint_train_run")
    if not config["local_logger"]["save_train_out"]:
        for savepath in savepaths:
            shutil.rmtree(savepath)

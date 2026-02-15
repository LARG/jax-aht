'''
Script for training a DQN value function estimation agent for social laws PPO single agent projection.
Does not support training against heuristic partner agents.

Command to run DQN value function estimation agent training:
python social_laws/run.py algorithm=ppo/lbf task=lbf label=test_dqn_value_function_estimation

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
import flashbax as fbx

from flax.struct import dataclass
from typing import NamedTuple

from social_laws.common.run_episodes import run_episodes_vmap
from social_laws.common.initialize_agents import initialize_dqn_actor_critic_fqe_agent
from agents.q_network import DQNTrainState
from common.plot_utils import get_stats, get_metric_names
from common.save_load_utils import save_train_run
from envs import make_env
from envs.log_wrapper import LogWrapper
from marl.ppo_utils import unbatchify

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray

@dataclass(frozen=True)
class TimeStep:
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    avail_actions: jnp.ndarray
    next_obs: jnp.ndarray
    next_avail_actions: jnp.ndarray

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def train_dqnppo_agent(config, env, train_rng,
                       policy, init_params,
                       ppo_policy, ppo_params,
                       agent_idx):
    '''
    Train DQN-PPO value function estimation using the given initial parameters.

    Args:
        config: dict, config for the training
        env: gymnasium environment
        train_rng: jax.random.PRNGKey, random key for training
        policy: AgentPolicy, policy for the agent
        ppo_policy: AgentPolicy, policy for the PPO agent used for value estimation
        ppo_params: dict, parameters for the PPO policy used for value estimation
        init_params: dict, initial parameters for the agent
    '''
    # ------------------------------
    # Build the DQN-PPO value function estimation training function
    # ------------------------------
    def make_dqnppo_train(config):
        '''The controlled agent is based on the agent_idx parameter'''
        num_agents = env.num_agents
        assert num_agents == 2, "This snippet assumes exactly 2 agents."

        config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
        config["NUM_UNCONTROLLED_ACTORS"] = config["NUM_ENVS"] # assumption: we control 1 agent
        config["NUM_CONTROLLED_ACTORS"] = config["NUM_ENVS"] # assumption: we control 1 agent
        config["ROLLOUT_LENGTH"] = env.horizon # assumption: rollout length is equal to episode length
        config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["ROLLOUT_LENGTH"] // config["NUM_ENVS"]
        config["NUM_ACTIONS"] = env.action_space(f"agent_{agent_idx}").n


        def linear_schedule(count):
            frac = 1.0 - (count / config["NUM_UPDATES"])
            return config["LR"] * frac

        def train(rng, agent_idx):
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

            train_state = DQNTrainState.create(
                apply_fn=policy.network.apply,
                params=init_params,
                target_network_params=jax.tree_map(lambda x: jnp.copy(x), init_params),
                tx=tx,
                timesteps=0,
                n_updates=0,
            )

            # INIT REPLAY BUFFER
            buffer = fbx.make_flat_buffer(
                max_length=config["BUFFER_SIZE"],
                min_length=config["BUFFER_BATCH_SIZE"],
                sample_batch_size=config["BUFFER_BATCH_SIZE"],
                add_sequences=False,
                add_batch_size=config["NUM_ENVS"],
            )
            buffer = buffer.replace(
                init=jax.jit(buffer.init),
                add=jax.jit(buffer.add, donate_argnums=0),
                sample=jax.jit(buffer.sample),
                can_sample=jax.jit(buffer.can_sample),
            )

            # buffer = fbx.make_flat_buffer(max_length=config["BUFFER_SIZE"], min_length=config["BUFFER_BATCH_SIZE"], sample_batch_size=config["BUFFER_BATCH_SIZE"], add_sequences=False, add_batch_size=config["NUM_ENVS"])
            # _timestep = TimeStep(obs=dummy_obs, action=dummy_action, reward=dummy_reward, done=dummy_done, avail_actions=dummy_avail_actions, next_obs=dummy_next_obs, next_avail_actions=dummy_next_avail_actions)

            # Initialize buffer with dummy timestep
            dummy_obs = jnp.zeros((policy.obs_dim,))
            dummy_action = jnp.zeros((1,), dtype=jnp.int32).squeeze()
            dummy_reward = jnp.zeros((1,)).squeeze()
            dummy_done = jnp.zeros((1,), dtype=bool).squeeze()
            dummy_avail_actions = jnp.ones((policy.action_dim,))
            dummy_next_obs = jnp.zeros((policy.obs_dim,))
            dummy_next_avail_actions = jnp.ones((policy.action_dim,))
            _timestep = TimeStep(obs=dummy_obs, action=dummy_action, reward=dummy_reward,
                                 done=dummy_done, avail_actions=dummy_avail_actions,
                                 next_obs=dummy_next_obs, next_avail_actions=dummy_next_avail_actions)
            buffer_state = buffer.init(_timestep)

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

            # Init hstates
            init_hstate = policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])
            init_ppo_hstate = policy.actor_critic.init_hstate(config["NUM_CONTROLLED_ACTORS"])

            # DQN Learning Phase
            def _dqn_learn_phase(train_state, ppo_params, buffer_state, rng):
                """Sample from buffer and update Q-network"""
                _, sample_rng = jax.random.split(rng, 2)
                learn_batch = buffer.sample(buffer_state, sample_rng).experience

                # Compute target Q-values using target network
                q_next_target = policy.network.apply(
                    train_state.target_network_params, learn_batch.second.obs
                )  # (batch_size, num_actions)
                q_next_target = jax.lax.stop_gradient(q_next_target)

                # Get PPO policy probabilities for next state
                ppo_pi = policy.actor_critic.get_action_value_policy(
                    ppo_params,
                    learn_batch.second.obs,
                    learn_batch.second.done,
                    learn_batch.second.avail_actions,
                    policy.actor_critic.init_hstate(config["BUFFER_BATCH_SIZE"]),
                    jax.random.PRNGKey(0) # Use dummy rng since with only need the policy
                )[2]
                ppo_probs = ppo_pi.probs  # (batch_size, num_actions)
                ppo_probs = jax.lax.stop_gradient(ppo_probs)

                # Compute expectation of Q-values under PPO policy
                target_sum = jnp.sum(q_next_target * ppo_probs, axis=-1, keepdims=True).squeeze(-1)  # (batch_size,)

                target = (
                    learn_batch.first.reward
                    + (1 - learn_batch.first.done) * config["GAMMA"] * target_sum
                )

                def _dqn_loss_fn(params):
                    q_vals = policy.network.apply(params, learn_batch.first.obs)  # (batch_size, num_actions)
                    chosen_action_qvals = jnp.take_along_axis(
                        q_vals,
                        jnp.expand_dims(learn_batch.first.action, axis=-1),
                        axis=-1,
                    ).squeeze(axis=-1)
                    return jnp.mean((chosen_action_qvals - target) ** 2), (chosen_action_qvals, q_vals)

                (dqn_loss, (chosen_action_qvals, q_vals)), grads = jax.value_and_grad(_dqn_loss_fn, has_aux=True)(train_state.params)
                train_state = train_state.apply_gradients(grads=grads)
                train_state = train_state.replace(n_updates=train_state.n_updates + 1)

                # Calculate metrics
                q_vals_mean = jnp.mean(chosen_action_qvals)
                td_target_mean = jnp.mean(target)

                return train_state, dqn_loss, q_vals_mean, td_target_mean

            def _env_step(runner_state, unused):
                """
                One step of the environment:
                1. Get observations, sample actions from DQN policy
                2. Step environment using sampled actions
                3. Add experience to replay buffer
                4. Optionally perform DQN update
                """
                train_state, buffer_state, env_state, prev_obs, prev_done, hstates, rng = runner_state
                hstate, ppo_hstate = hstates
                rng, actor_rng, step_rng, learn_rng = jax.random.split(rng, 4)

                 # Get available actions for the agent from environment state
                avail_actions = env.get_avail_actions(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(avail_actions)
                avail_actions = get_agent_data(avail_actions, agent_idx).astype(jnp.float32)

                # Note that we do not need to reset the hidden states for the agents
                # as the recurrent states are automatically reset when done is True.

                # Get actions from DQN policy (uses actor-critic under the hood for exploration)
                act, new_ppo_hstate = policy.get_actor_critic_action(
                    params=ppo_params,  # Use PPO params for action selection
                    obs=get_agent_data(prev_obs, agent_idx).reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                    done=get_agent_data(prev_done, agent_idx).reshape(1, config["NUM_CONTROLLED_ACTORS"]),
                    avail_actions=avail_actions,
                    hstate=ppo_hstate,
                    rng=actor_rng,
                    env_state=env_state,
                    test_mode=False
                )

                act = act.squeeze(axis=0)

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

                # Update timesteps counter
                train_state = train_state.replace(timesteps=train_state.timesteps + config["NUM_ENVS"])

                # ADD EXPERIENCE TO BUFFER
                agent_obs = get_agent_data(prev_obs, agent_idx)
                agent_next_obs = get_agent_data(obs_next, agent_idx)
                agent_reward = get_agent_data(reward, agent_idx)
                agent_done = get_agent_data(done_next, agent_idx)

                # Get next available actions for the agent from environment state
                next_avail_actions = env.get_avail_actions(env_state_next.env_state)
                next_avail_actions = jax.lax.stop_gradient(next_avail_actions)
                next_avail_actions = get_agent_data(next_avail_actions, agent_idx).astype(jnp.float32)

                timestep = TimeStep(
                    obs=agent_obs.squeeze(),
                    action=act,
                    reward=agent_reward,
                    done=agent_done,
                    avail_actions=avail_actions,
                    next_obs=agent_next_obs.squeeze(),
                    next_avail_actions=next_avail_actions
                )
                buffer_state = buffer.add(buffer_state, timestep)

                # PERFORM DQN UPDATE IF CONDITIONS MET
                is_learn_time = (
                    (buffer.can_sample(buffer_state))
                    & (train_state.timesteps > config["LEARNING_STARTS"])
                    & (train_state.timesteps % config["TRAINING_INTERVAL"] == 0)
                )
                train_state, dqn_loss, q_vals_mean, td_target_mean = jax.lax.cond(
                    is_learn_time,
                    lambda ts, pp, bs, rng: _dqn_learn_phase(ts, pp, bs, rng),
                    lambda ts, pp, bs, rng: (ts, jnp.nan, jnp.nan, jnp.nan),  # Use NaN when not learning
                    train_state,
                    ppo_params,
                    buffer_state,
                    learn_rng,
                )

                # UPDATE TARGET NETWORK
                train_state = jax.lax.cond(
                    train_state.timesteps % config["TARGET_UPDATE_INTERVAL"] == 0,
                    lambda ts: ts.replace(
                        target_network_params=optax.incremental_update(
                            ts.params,
                            ts.target_network_params,
                            config["TAU"],
                        )
                    ),
                    lambda ts: ts,
                    operand=train_state,
                )

                # Create transition for metrics (keeping for compatibility)
                transition = Transition(
                    done=agent_done,
                    action=act,
                    reward=agent_reward,
                    obs=agent_obs,
                    info=info,
                    avail_actions=avail_actions
                )

                # Add DQN metrics to info
                info["dqn_loss"] = dqn_loss
                info["q_vals_mean"] = q_vals_mean
                info["td_target_mean"] = td_target_mean

                new_runner_state = (train_state, buffer_state, env_state_next, obs_next, done_next,
                                    (hstate, new_ppo_hstate), rng)
                return new_runner_state, transition

            def _update_step(update_runner_state, unused):
                """
                DQN training loop:
                1. Collect rollouts and add to buffer
                2. DQN updates happen within _env_step
                """
                (train_state, buffer_state, rng, rng_eval, update_steps) = update_runner_state
                # Init envs
                rng, reset_rng = jax.random.split(rng, 2)
                reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
                (init_obs, init_obs_full), init_env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rngs)
                init_done = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}

                # Rollout (DQN updates happen inside _env_step)
                runner_state = (train_state, buffer_state, init_env_state, init_obs, init_done, (init_hstate, init_ppo_hstate), rng)

                runner_state, traj_batch = jax.lax.scan(
                    _env_step, runner_state, None, config["ROLLOUT_LENGTH"])
                (train_state, buffer_state, env_state, obs, done, hstates, rng) = runner_state

                metric = traj_batch.info
                metric["update_steps"] = update_steps
                metric["timesteps"] = train_state.timesteps
                metric["n_updates"] = train_state.n_updates
                new_runner_state = (train_state, buffer_state, rng, rng_eval, update_steps + 1)
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

                # Single DQN update step
                new_update_state, metric = _update_step(
                    update_state,
                    None
                )
                (train_state, buffer_state, rng, rng_eval, update_steps) = new_update_state

                # update steps is 1-indexed because it was incremented at the end of the update step
                to_store = jnp.logical_or(jnp.equal(jnp.mod(update_steps-1, ckpt_and_eval_interval), 0),
                                        jnp.equal(update_steps, config["NUM_UPDATES"]))


                def store_and_eval_ckpt(args):
                    ckpt_arr, cidx, rng, rng_eval, prev_ckpt_eval_ret_info, prev_eval_ret_info = args
                    new_ckpt_arr = jax.tree.map(
                        lambda c_arr, p: c_arr.at[cidx].set(p),
                        ckpt_arr, train_state.params
                    )

                    rng_eval, eval_rng = jax.random.split(rng_eval, 2)
                    ckpt_eval_eps_last_infos = run_episodes_vmap(
                        eval_rng, env, agent_idx, agent_param=train_state.params, agent_policy=policy,
                        max_episode_steps=max_episode_steps,
                        num_eps=config["NUM_EVAL_EPISODES"], agent_test_mode=True)  # Use test mode for eval
                    return (new_ckpt_arr, cidx + 1, rng, rng_eval, ckpt_eval_eps_last_infos, ckpt_eval_eps_last_infos)

                def skip_ckpt_and_eval(args):
                    def do_eval(eval_args):
                        ckpt_arr, cidx, rng, rng_eval, prev_ckpt_eval_ret_info, prev_eval_ret_info = eval_args
                        rng_eval, eval_rng = jax.random.split(rng_eval, 2)
                        eval_eps_last_infos = run_episodes_vmap(
                            eval_rng, env, agent_idx, agent_param=train_state.params, agent_policy=policy,
                            max_episode_steps=max_episode_steps,
                            num_eps=config["NUM_EVAL_EPISODES"], agent_test_mode=True)  # Use test mode for eval
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
                return ((train_state, buffer_state, rng, rng_eval, update_steps),
                         checkpoint_array, ckpt_idx, ckpt_eval_eps_last_infos, eval_eps_last_infos), metric

            checkpoint_array = init_ckpt_array(train_state.params)
            ckpt_idx = 0

            rng, rng_train = jax.random.split(rng, 2)

            rng_eval = jax.random.PRNGKey(config["EVAL_SEED"] + agent_idx + 14)
            rng_eval, eval_rng = jax.random.split(rng_eval, 2)


            # Init eval return infos
            eval_eps_last_infos = run_episodes_vmap(eval_rng, env, agent_idx,
                                    agent_param=train_state.params, agent_policy=policy,
                                    max_episode_steps=max_episode_steps,
                                    num_eps=config["NUM_EVAL_EPISODES"], agent_test_mode=True)

            # initial runner state for scanning
            update_steps = 0

            update_runner_state = (train_state, buffer_state, rng_train, rng_eval, update_steps)
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
                rng_eval = final_runner_state[3] # extract final rng_eval from the final runner state after training
                rng_eval, eval_rng = jax.random.split(rng_eval, 2)
                params = final_runner_state[0].params
                out["render_outs"] = run_episodes_vmap(eval_rng, env, agent_idx,
                                                    agent_param=params, agent_policy=policy,
                                                    max_episode_steps=env.horizon,
                                                    num_eps=5, render=True, agent_test_mode=True)
            return out
        return train

    # ------------------------------
    # Actually run the DQN training
    # ------------------------------
    rngs = jax.random.split(train_rng, 1)
    agent_idx_arr = jnp.array([agent_idx] * 1)

    # Define scan function to run training seeds sequentially
    train_fn = make_dqnppo_train(config)
    def scan_train(carry, inputs):
        rng, agent_idx = inputs
        result = train_fn(rng, agent_idx)
        return carry, result

    # Run training seeds sequentially using scan
    _, out = jax.lax.scan(scan_train, None, (rngs, agent_idx_arr))
    return out

def run_training(config, wandb_logger, ppo_params, ppo_policy, agent_idx=0):
    '''Run single agent projection training.

    Args:
        config: dict, config for the training
    '''
    algorithm_config = dict(config["value_function"])

    algorithm_config["EPSILON_ANNEAL_TIME"] = algorithm_config["TOTAL_TIMESTEPS"] * algorithm_config["EPSILON_EXPLORATION_FRACTION"]

    # Create only one environment instance
    env_kwargs = algorithm_config["ENV_KWARGS"].copy()

    env_kwargs["instance"] = config['task'][f"SINGLE_AGENT_{agent_idx + 1}_PROJECTION"]
    env_kwargs["render_dir"] = os.path.join("render", "dqnppo", f"agent_{agent_idx + 1}")
    env = make_env(algorithm_config["ENV_NAME"], env_kwargs)
    env = LogWrapper(env)

    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"] + agent_idx + 21)
    _, init_rng, train_rng = jax.random.split(rng, 3)

    # Initialize agent
    policy, init_params = initialize_dqn_actor_critic_fqe_agent(algorithm_config, env, init_rng, ppo_policy, agent_index=agent_idx)

    # Squeeze PPO params to remove leading dimension for compatibility with single-agent training
    ppo_params = jax.tree.map(lambda x: x.squeeze(axis=0), ppo_params)

    # [item.shape for item in jax.tree_leaves(ppo_params)] # debug print to check shapes of PPO params

    log.info(f"Starting value function estimation training for agent {agent_idx}...")
    start_time = time.perf_counter()

    # Run the training
    out = train_dqnppo_agent(
        config=algorithm_config,
        env=env,
        train_rng=train_rng,
        policy=policy,
        init_params=init_params,
        ppo_policy=ppo_policy,
        ppo_params=ppo_params,
        agent_idx=agent_idx
    )

    elapsed_time = time.perf_counter() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, rem = divmod(rem, 60)
    seconds, rem = divmod(rem, 1)
    milliseconds = int(rem * 1000)
    microseconds = int((rem * 1_000_000) % 1000)
    log.info(f"Value Function Estimation Training completed for agent {agent_idx} in {elapsed_time:.2f}s")
    log.info(f"Value Function Estimation Training completed for agent {agent_idx} in {int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s {milliseconds:03d}ms {microseconds:03d}µs")

    # process and log metrics
    metric_names = get_metric_names(config["ENV_NAME"])
    log_metrics(env, config, out, wandb_logger, metric_names, agent_idx)

    return out["final_params"], policy, init_params

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

    #### Extract train metrics ####
    train_stats = get_stats(train_metrics, metric_names)
    # each key in train_stats is a metric name, and the value is an array of shape (num_seeds, num_updates, 2)
    # where the last dimension contains the mean and std of the metric
    train_stats = {k: np.mean(np.array(v), axis=0) for k, v in train_stats.items()}

    # DQN-specific metrics
    all_dqn_losses = np.asarray(train_metrics["dqn_loss"]) # shape (n_train_seeds, num_updates, rollout_length)
    all_q_vals_mean = np.asarray(train_metrics["q_vals_mean"]) # shape (n_train_seeds, num_updates, rollout_length)
    all_td_target_mean = np.asarray(train_metrics["td_target_mean"]) # shape (n_train_seeds, num_updates, rollout_length)
    all_timesteps = np.asarray(train_metrics["timesteps"]) # shape (n_train_seeds, num_updates, rollout_length)
    all_n_updates = np.asarray(train_metrics["n_updates"]) # shape (n_train_seeds, num_updates, rollout_length)

    # Process eval return metrics - average across train seeds, eval episodes, and num_agents per game for each checkpoint
    all_ckpt_returns = np.asarray(train_metrics["ckpt_eval_ep_last_info"]["returned_episode_returns"]) # shape (n_train_seeds, num_updates, num_eval_episodes, num_agents_per_game)
    all_returns = np.asarray(train_metrics["eval_ep_last_info"]["returned_episode_returns"]) # shape (n_train_seeds, num_updates, num_eval_episodes, num_agents_per_game)
    all_ckpt_agent_returns = all_ckpt_returns[:, :, :, agent_idx] # shape (n_train_seeds, num_updates, num_eval_episodes)
    all_agent_returns = all_returns[:, :, :, agent_idx] # shape (n_train_seeds, num_updates, num_eval_episodes)
    average_ckpt_agent_rets_per_iter = np.mean(all_ckpt_agent_returns, axis=(0, 2)) # shape (num_updates,)
    average_agent_rets_per_iter = np.mean(all_agent_returns, axis=(0, 2)) # shape (num_updates,)

    # Process DQN loss metrics - average across train seeds and rollout steps, filtering out NaN values
    average_dqn_losses = np.nanmean(all_dqn_losses, axis=(0, 2))  # shape (num_updates,)
    average_q_vals_mean = np.nanmean(all_q_vals_mean, axis=(0, 2))  # shape (num_updates,)
    average_td_target_mean = np.nanmean(all_td_target_mean, axis=(0, 2))  # shape (num_updates,)
    average_timesteps = np.mean(all_timesteps, axis=(0))  # shape (num_updates,)
    average_n_updates = np.mean(all_n_updates, axis=(0))  # shape (num_updates,)

    # Log metrics for each update step
    num_updates = len(average_dqn_losses)
    for step in range(num_updates):
        for stat_name, stat_data in train_stats.items():
            # second dimension contains the mean and std of the metric
            stat_mean = stat_data[step, 0]
            logger.log_item(f"Train/ValueFunction/Agent_{agent_idx + 1}_Proj/{stat_name}", stat_mean, train_step=step, commit=True)

        logger.log_item(f"Eval/ValueFunction/Agent_{agent_idx + 1}_Proj/Return", average_agent_rets_per_iter[step], train_step=step, commit=True)
        logger.log_item(f"Eval/ValueFunction/Agent_{agent_idx + 1}_Proj/CheckpointReturn", average_ckpt_agent_rets_per_iter[step], train_step=step, commit=True)
        logger.log_item(f"Train/ValueFunction/Agent_{agent_idx + 1}_Proj/TD_Loss", average_dqn_losses[step], train_step=step, commit=True)
        logger.log_item(f"Train/ValueFunction/Agent_{agent_idx + 1}_Proj/Q_Values_Mean", average_q_vals_mean[step], train_step=step, commit=True)
        logger.log_item(f"Train/ValueFunction/Agent_{agent_idx + 1}_Proj/TD_Target_Mean", average_td_target_mean[step], train_step=step, commit=True)
        logger.log_item(f"Train/ValueFunction/Agent_{agent_idx + 1}_Proj/Timesteps", average_timesteps[step], train_step=step, commit=True)
        logger.log_item(f"Train/ValueFunction/Agent_{agent_idx + 1}_Proj/N_Updates", average_n_updates[step], train_step=step, commit=True)
        logger.commit()

    # Saving artifacts
    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    if env._render:
        # shape of render_outs should be (num_train_seeds, num_eps, max_episode_steps, ...)
        eval_render_init_env_state = train_out['render_outs'][1].env_state.env_state # LogEnvState
        eval_render_env_state = train_out['render_outs'][0][-1]['pre_reset_state'].env_state # WrappedEnvState
        eval_render_dones = train_out['render_outs'][0][4]['__all__']
        num_episodes = eval_render_env_state.state['agent-at'].shape[1] # (num_train_seeds, num_eval_episodes, num_max_timesteps, num_agents_per_game, ...)
        env.animate((eval_render_init_env_state, eval_render_env_state), eval_render_dones, num_episodes, debug=True)

        for eval_ep in range(num_episodes):
            logger.log_video(
                tag=f"Videos/ValueFunction/Agent_{agent_idx + 1}_Proj/Agent_{agent_idx}_Episode_{eval_ep}",
                path=os.path.join(env._render_dir, f"{env._render_name}_ep_{eval_ep}.gif")
            )

    out_savepath = save_train_run(train_out, savedir, savename=f"DQN_PPO_Agent_{agent_idx + 1}_Proj_Train_Run")
    if config["logger"]["log_train_out"]:
        logger.log_artifact(name=f"DQN_PPO_Agent_{agent_idx + 1}_Proj_Train_Run", path=out_savepath, type_name="single_agent_projection_value_function_train_run")
        # Cleanup locally logged out file
    if not config["local_logger"]["save_train_out"]:
        shutil.rmtree(out_savepath)

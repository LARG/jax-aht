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
from social_laws.common.run_episodes import run_episodes_vmap
from common.plot_utils import get_stats, get_metric_names
from common.save_load_utils import save_train_run
from envs import make_env
from envs.log_wrapper import LogWrapper
from marl.ppo_utils import _create_minibatches, Transition, unbatchify

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def train_ppo_agent(config, env, train_rng,
                    policy, init_params, agent_idx):
    '''
    Train PPO single agent projection using the given initial parameters.

    Args:
        config: dict, config for the training
        env: gymnasium environment
        train_rng: jax.random.PRNGKey, random key for training
        policy: AgentPolicy, policy for the agent
        init_params: dict, initial parameters for the agent
    '''
    # ------------------------------
    # Build the PPO single agent projection training function
    # ------------------------------
    def make_ppo_train(config):
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
                rng, actor_rng, step_rng = jax.random.split(rng, 3)

                 # Get available actions for the agent from environment state
                avail_actions = env.get_avail_actions(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(avail_actions)
                avail_actions = get_agent_data(avail_actions, agent_idx).astype(jnp.float32)

                # Note that we do not need to reset the hidden states for the agents
                # as the recurrent states are automatically reset when done is True.

                # Controlled Agent action, value, log_prob
                act, val, pi, new_hstate = policy.get_action_value_policy(
                    params=train_state.params,
                    obs=get_agent_data(prev_obs, agent_idx).reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                    done=get_agent_data(prev_done, agent_idx).reshape(1, config["NUM_CONTROLLED_ACTORS"]),
                    avail_actions=avail_actions,
                    hstate=hstate,
                    rng=actor_rng
                )
                logp = pi.log_prob(act)

                act = act.squeeze(axis=0)
                logp = logp.squeeze(axis=0)
                val = val.squeeze(axis=0)

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

                # Store agent_idx data in transition
                transition = Transition(
                    done=get_agent_data(done_next, agent_idx), # shape (num_envs,)
                    action=act, # shape (num_envs,)
                    value=val, # shape (num_envs,)
                    reward=get_agent_data(reward, agent_idx), # shape (num_envs,)
                    log_prob=logp, # shape (num_envs,)
                    obs=get_agent_data(prev_obs, agent_idx), # shape (num_envs, obs_dim)
                    info=info,
                    avail_actions=avail_actions # shape (num_envs, num_actions)
                )

                new_runner_state = (train_state, env_state_next, obs_next, done_next,
                                    new_hstate, rng)
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
                init_hstate, traj_batch, advantages, returns = batch_info
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

                # 2) advantage
                # Get available actions for agent 0 from environment state
                avail_actions_0 = get_agent_data(env.get_avail_actions(env_state.env_state), agent_idx).astype(jnp.float32)

                # Get final value estimate for completed trajectory
                _, last_val, _, _ = policy.get_action_value_policy(
                    params=train_state.params,
                    obs=get_agent_data(obs, agent_idx).reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                    done=get_agent_data(done, agent_idx).reshape(1, config["NUM_CONTROLLED_ACTORS"]),
                    avail_actions=jax.lax.stop_gradient(avail_actions_0),
                    hstate=hstate,
                    rng=jax.random.PRNGKey(0)  # Dummy key since we're just extracting the value
                )
                last_val = last_val.squeeze(axis=0)
                advantages, targets = _calculate_gae(traj_batch, last_val)

                # 3) PPO update
                update_state = (
                    train_state,
                    init_hstate, # shape is (num_controlled_actors, gru_hidden_dim) with all-0s value
                    traj_batch, # obs has shape (rollout_len, num_controlled_actors, -1)
                    advantages,
                    targets,
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
                new_runner_state = (train_state, rng, rng_eval, update_steps + 1)
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
                (train_state, rng, rng_eval, update_steps) = new_update_state

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
                    ckpt_eval_eps_last_infos = run_episodes_vmap(
                        eval_rng, env, agent_idx, agent_param=train_state.params, agent_policy=policy,
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
                            eval_rng, env, agent_idx, agent_param=train_state.params, agent_policy=policy,
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
                return ((train_state, rng, rng_eval, update_steps),
                         checkpoint_array, ckpt_idx, ckpt_eval_eps_last_infos, eval_eps_last_infos), metric

            checkpoint_array = init_ckpt_array(train_state.params)
            ckpt_idx = 0

            rng, rng_train = jax.random.split(rng, 2)

            rng_eval = jax.random.PRNGKey(config["EVAL_SEED"] + agent_idx)# + 14)
            rng_eval, eval_rng = jax.random.split(rng_eval, 2)

            # Init eval return infos
            eval_eps_last_infos = run_episodes_vmap(eval_rng, env, agent_idx,
                                    agent_param=train_state.params, agent_policy=policy,
                                    max_episode_steps=max_episode_steps,
                                    num_eps=config["NUM_EVAL_EPISODES"])

            # initial runner state for scanning
            update_steps = 0

            update_runner_state = (train_state, rng_train, rng_eval, update_steps)
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
                out["render_outs"] = run_episodes_vmap(eval_rng, env, agent_idx,
                                                    agent_param=params, agent_policy=policy,
                                                    max_episode_steps=env.horizon,
                                                    num_eps=5, render=True)

            return out
        return train

    # ------------------------------
    # Actually run the PPO training
    # ------------------------------
    rngs = jax.random.split(train_rng, config["NUM_TRAIN_SEEDS"])

    # Run training seeds in parallel using vmap
    train_fn = make_ppo_train(config)
    out = jax.vmap(train_fn, in_axes=(0, None))(rngs, agent_idx)
    return out

def run_training(config, wandb_logger, agent_idx=0):
    '''Run single agent projection training.

    Args:
        config: dict, config for the training
    '''
    algorithm_config = dict(config["algorithm"])

    # Create only one environment instance
    env_kwargs = algorithm_config["ENV_KWARGS"].copy()

    env_kwargs["instance"] = config['task'][f"SINGLE_AGENT_{agent_idx + 1}_PROJECTION"]
    env_kwargs["render_dir"] = os.path.join("render", "ppo", f"agent_{agent_idx + 1}")
    env = make_env(algorithm_config["ENV_NAME"], env_kwargs)
    env = LogWrapper(env)

    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"] + agent_idx)# + 7)
    _, init_rng, train_rng = jax.random.split(rng, 3)

    # Initialize agent
    policy, init_params = initialize_agent(algorithm_config, env, init_rng, agent_index=agent_idx)

    log.info(f"Starting single agent projection training for agent {agent_idx}...")
    start_time = time.perf_counter()

    # Run the training
    out = train_ppo_agent(
        config=algorithm_config,
        env=env,
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
    log_metrics(env, config, out, wandb_logger, metric_names, agent_idx)
    elapsed_time = time.perf_counter() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, rem = divmod(rem, 60)
    seconds, rem = divmod(rem, 1)
    milliseconds = int(rem * 1000)
    microseconds = int((rem * 1_000_000) % 1000)
    log.info(f"Single Agent Projection Logging completed for agent {agent_idx} in {elapsed_time:.2f}s")
    log.info(f"Single Agent Projection Logging completed for agent {agent_idx} in {int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s {milliseconds:03d}ms {microseconds:03d}µs")

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

    all_agent_value_losses = np.asarray(train_metrics["value_loss"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    all_agent__actor_losses = np.asarray(train_metrics["actor_loss"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    all_agent_entropy_losses = np.asarray(train_metrics["entropy_loss"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    all_agent_grad_norms = np.asarray(train_metrics["avg_grad_norm"]) # shape (n_train_seeds, num_updates, num_update_epochs, num_minibatches)
    # Process eval return metrics - average across train seeds, eval episodes, and num_agents per game for each checkpoint
    all_ckpt_returns = np.asarray(train_metrics["ckpt_eval_ep_last_info"]["returned_episode_returns"]) # shape (n_train_seeds, num_updates, num_eval_episodes, num_agents_per_game)
    all_returns = np.asarray(train_metrics["eval_ep_last_info"]["returned_episode_returns"]) # shape (n_train_seeds, num_updates, num_eval_episodes, num_agents_per_game)
    all_ckpt_agent_returns = all_ckpt_returns[:, :, :, agent_idx] # shape (n_train_seeds, num_updates, num_eval_episodes)
    all_agent_returns = all_returns[:, :, :, agent_idx] # shape (n_train_seeds, num_updates, num_eval_episodes)
    average_ckpt_agent_rets_per_iter = np.mean(all_ckpt_agent_returns, axis=(0, 2)) # shape (num_updates,)
    average_agent_rets_per_iter = np.mean(all_agent_returns, axis=(0, 2)) # shape (num_updates,)

    # Process loss metrics - average across train seeds, partners and minibatches dims
    # Loss metrics shape should be (n_train_seeds, num_updates, ...)
    average_agent_value_losses = np.mean(all_agent_value_losses, axis=(0, 2, 3))  # shape (num_updates,)
    average_agent_actor_losses = np.mean(all_agent__actor_losses, axis=(0, 2, 3)) # shape (num_updates,)
    average_agent_entropy_losses = np.mean(all_agent_entropy_losses, axis=(0, 2, 3)) # shape (num_updates,)
    average_agent_grad_norms = np.mean(all_agent_grad_norms, axis=(0, 2, 3)) # shape (num_updates,)

    # Log metrics for each update step
    num_updates = len(average_agent_value_losses)
    for step in range(num_updates):
        for stat_name, stat_data in train_stats.items():
            # second dimension contains the mean and std of the metric
            stat_mean = stat_data[step, 0]
            logger.log_item(f"Train/Agent_{agent_idx + 1}_Proj/{stat_name}", stat_mean, train_step=step, commit=True)

        logger.log_item(f"Eval/Agent_{agent_idx + 1}_Proj/Return", average_agent_rets_per_iter[step], train_step=step, commit=True)
        logger.log_item(f"Eval/Agent_{agent_idx + 1}_Proj/CheckpointReturn", average_ckpt_agent_rets_per_iter[step], train_step=step, commit=True)
        logger.log_item(f"Train/Agent_{agent_idx + 1}_Proj/ValueLoss", average_agent_value_losses[step], train_step=step, commit=True)
        logger.log_item(f"Train/Agent_{agent_idx + 1}_Proj/ActorLoss", average_agent_actor_losses[step], train_step=step, commit=True)
        logger.log_item(f"Train/Agent_{agent_idx + 1}_Proj/EntropyLoss", average_agent_entropy_losses[step], train_step=step, commit=True)
        logger.log_item(f"Train/Agent_{agent_idx + 1}_Proj/GradNorm", average_agent_grad_norms[step], train_step=step, commit=True)
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
                tag=f"Videos/Agent_{agent_idx + 1}_Proj/Agent_{agent_idx}_Episode_{eval_ep}",
                path=os.path.join(env._render_dir, f"{env._render_name}_ep_{eval_ep}.gif")
            )

    out_savepath = save_train_run(train_out, savedir, savename=f"PPO_Agent_{agent_idx + 1}_Proj_Train_Run")
    if config["logger"]["log_train_out"]:
        logger.log_artifact(name=f"PPO_Agent_{agent_idx + 1}_Proj_Train_Run", path=out_savepath, type_name="single_agent_proj_train_run")
        # Cleanup locally logged out file
    if not config["local_logger"]["save_train_out"]:
        shutil.rmtree(out_savepath)

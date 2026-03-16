'''
This file contains the code for running evaluation episodes with an ego agent and a partner agent.
Does not currently support actors that require aux_obs.
'''
import jax
import jax.numpy as jnp
import logging
import numpy as np
import hydra
import os
import time
from functools import partial
from typing import NamedTuple

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def run_single_episode(rng, env, agent_params, agent_policies,
                       max_episode_steps, render=False, agent_test_mode=False):
    # Reset the env.
    rng, reset_rng = jax.random.split(rng)
    (init_obs, init_obs_full), init_env_state = env.reset(reset_rng)
    init_done = {k: jnp.zeros((1), dtype=bool) for k in env.agents + ["__all__"]}

    init_reward = {k: jnp.zeros((1)) for i, k in enumerate(env.agents)}

    num_agents = env.num_agents

    # Initialize hidden states. Agent id is passed as part of the hstate initialization to support heuristic agents.
    init_hstates = [agent_policies[i].init_hstate(1, aux_info={"agent_id": i}) for i in range(num_agents)]

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

    # Get available actions and per-agent observations for the worst-case env
    avail_actions = env.get_avail_actions(init_env_state.env_state)
    avail_actions = jax.lax.stop_gradient(avail_actions)
    init_avail_per_agent = [get_agent_data(avail_actions, i).astype(jnp.float32) for i in range(num_agents)]
    init_obs_per_agent = [get_agent_data(init_obs, i).reshape(1, 1, -1) for i in range(num_agents)]
    init_done_per_agent = [get_agent_data(init_done, i).reshape(1, 1) for i in range(num_agents)]

    # Do one step to get a dummy info structure
    rng, *act_rngs, step_rng = jax.random.split(rng, num_agents + 2)

    acts = []
    hstates = []
    for i in range(num_agents):
        act_i, hstate_i = agent_policies[i].get_action(
            params=(agent_params[i]['actor']["params"], agent_params[i]['actor']["batch_stats"]),
            obs=init_obs_per_agent[i],
            done=init_done_per_agent[i],
            avail_actions=init_avail_per_agent[i],
            hstate=init_hstates[i],
            rng=act_rngs[i],
            aux_obs=None,
            env_state=init_env_state,
            test_mode=False
        )
        acts.append(act_i.squeeze(axis=0))
        hstates.append(hstate_i)

    env_act = {k: acts[i] for i, k in enumerate(env.agents)}
    env_act_onehot = {k: jax.nn.one_hot(env_act[env.agents[i]], env.action_space(env.agents[i]).n) for i, k in enumerate(env.agents)}
    (obs, obs_full), env_state, reward, done, info = env.step(step_rng, init_env_state, env_act)

    ep_ts = 1
    init_carry = (ep_ts, env_state, obs, rng, done, reward, env_act_onehot, hstates, info)
    def scan_step(carry, _):
        def take_step(carry_step):
            ep_ts, env_state, obs, rng, done, reward, act_onehot, hstates, last_info = carry_step

            # Get available actions for all agents
            avail_actions = env.get_avail_actions(env_state.env_state)
            avail_actions = jax.lax.stop_gradient(avail_actions)

            obs_per_agent = [get_agent_data(obs, i).reshape(1, 1, -1) for i in range(num_agents)]
            done_per_agent = [get_agent_data(done, i).reshape(1, 1) for i in range(num_agents)]
            avail_per_agent = [get_agent_data(avail_actions, i).astype(jnp.float32) for i in range(num_agents)]

            rng, *act_rngs, step_rng = jax.random.split(rng, num_agents + 2)

            # Get actions for all agents
            acts = []
            next_hstates = []
            for i in range(num_agents):
                act_i, hs_next_i = agent_policies[i].get_action(
                    params=(agent_params[i]['actor']["params"], agent_params[i]['actor']["batch_stats"]),
                    obs=obs_per_agent[i],
                    done=done_per_agent[i],
                    avail_actions=avail_per_agent[i],
                    hstate=hstates[i],
                    rng=act_rngs[i],
                    aux_obs=None,
                    env_state=env_state,
                    test_mode=agent_test_mode
                )
                acts.append(act_i.squeeze(axis=0))
                next_hstates.append(hs_next_i)

            env_act = {k: acts[i] for i, k in enumerate(env.agents)}
            env_act_onehot = {k: jax.nn.one_hot(env_act[env.agents[i]], env.action_space(env.agents[i]).n) for i, k in enumerate(env.agents)}
            (obs_next, obs_full_next), env_state_next, reward, done_next, info_next = env.step(step_rng, env_state, env_act)

            return (ep_ts + 1, env_state_next, obs_next, rng, done_next, reward, env_act_onehot, next_hstates, info_next)

        ep_ts, env_state, obs, rng, done, reward, act_onehot, hstate, last_info = carry
        output = carry
        new_carry = jax.lax.cond(
            done["__all__"],
            lambda curr_carry: curr_carry,  # True fn
            take_step,  # False fn
            operand=carry
        )
        return new_carry, output

    final_carry, stacked_carry = jax.lax.scan(
        scan_step, init_carry, None, length=max_episode_steps)

    if render:
        # If rendering, we return all step data (stacked_carry contains data for each timestep)
        return (stacked_carry, init_env_state, final_carry[-1])
    else:
        # If not rendering, we just return the final info (which includes the episode return via LogWrapper)
        # Return the final info (which includes the episode return via LogWrapper).
        return final_carry[-1]

def run_episodes(rng, env, optimal_env, agent_idx, agent_params, agent_policies,
                 max_episode_steps, num_eps, render=False, agent_test_mode=False):
    '''Run num_eps episodes sequentially using scan.'''
    # Create episode-specific RNGs
    rngs = jax.random.split(rng, num_eps + 1)
    ep_rngs = rngs[1:]

    # Define scan function to run episodes sequentially
    def scan_episode(carry, ep_rng):
        all_out = run_single_episode(
            ep_rng, env, optimal_env, agent_idx, agent_params, agent_policies,
            max_episode_steps, render, agent_test_mode
        )
        return carry, all_out

    # Run episodes sequentially using scan
    _, all_outs = jax.lax.scan(scan_episode, None, ep_rngs)
    return all_outs  # each leaf has shape (num_eps, ...)

def run_episodes_vmap(rng, env, agent_params, agent_policies,
                      max_episode_steps, num_eps, render=False, agent_test_mode=False):
    '''Run num_eps episodes in parallel using vmap.'''
    # Create episode-specific RNGs
    rngs = jax.random.split(rng, num_eps + 1)
    ep_rngs = rngs[1:]

    # Vmap over episodes - all episodes run in parallel
    vmapped_run = jax.vmap(
        run_single_episode,
        in_axes=(0, None, None, None,
                None, None, None),  # only RNG varies
        out_axes=0  # stack outputs along first axis
    )

    all_outs = vmapped_run(
        ep_rngs, env, agent_params, agent_policies,
        max_episode_steps, render, agent_test_mode
    )

    return all_outs  # each leaf has shape (num_eps, ...)


def run_single_agent_joint_eval(logger, eval_seed, env, agent_params, agent_policies,
                                max_episode_steps, num_eps, fixed_eval=False, render=False, agent_test_mode=False):

    log.info(f"Starting single agent projection joint evaluation...")
    start_time = time.perf_counter()

    def make_joint_eval():
        def joint_eval_fn(agent_params):
            # seed_agent_params: list of pytrees, each leaf shape (num_checkpoints, *param_shape)

            rng_eval = jax.random.PRNGKey(eval_seed)
            if fixed_eval:
                rng_eval, eval_rng = jax.random.split(rng_eval)
                eval_rng = rng_eval

            def checkpoint_scan_fn(rng, ck_agent_params):
                if fixed_eval:
                    eval_rng = rng
                else:
                    rng, eval_rng = jax.random.split(rng)

                out = run_episodes_vmap(
                    eval_rng, env, ck_agent_params, agent_policies,
                    max_episode_steps, num_eps, render, agent_test_mode
                )
                return rng, out

            _, eval_outputs = jax.lax.scan(checkpoint_scan_fn, rng_eval, agent_params)
            return eval_outputs

        return joint_eval_fn

    # num_train_seeds = jax.tree.leaves(agent_params[0])[0].shape[0]
    # rngs = jax.random.split(rng, num_train_seeds)

    # Run eval of training seeds in parallel using vmap
    # agent_params: list of pytrees per agent, each leaf shape (num_seeds, num_checkpoints, *param_shape)
    eval_fn = make_joint_eval()
    out = jax.vmap(eval_fn, in_axes=(0,))(agent_params)

    if render:
        returned_episode_returns = out[2]["returned_episode_returns"]
    else:
        returned_episode_returns = out["returned_episode_returns"]

    # Process eval return metrics - average across train seeds, eval episodes, and num_agents per game for each checkpoint
    all_returns = np.asarray(returned_episode_returns) # shape (n_train_seeds, num_updates, num_eval_episodes, num_agents_per_game)
    average_rets_per_iter = np.mean(np.sum(all_returns, axis=(3)), axis=(0, 2)) # shape (num_updates,)
    average_agent_rets_per_iter = np.mean(all_returns, axis=(0, 2)) # shape (num_updates, num_agents_per_game)

    num_agents = env.num_agents
    num_updates = len(average_rets_per_iter)
    for step in range(num_updates):
        for agent_id in range(num_agents):
             logger.log_item(f"Eval/Single_Agent_Proj_Joint/Agent_{agent_id + 1}/Return", average_agent_rets_per_iter[step, agent_id], train_step=step, commit=True)

        logger.log_item(f"Eval/Single_Agent_Proj_Joint/Return", average_rets_per_iter[step], train_step=step, commit=True)

    # Saving artifacts
    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    if env._render and render:
        # shape of render_outs should be (num_train_seeds, num_eps, max_episode_steps, ...)
        eval_render_init_env_state = out[1].env_state.env_state # LogEnvState
        eval_render_env_state = out[0][-1]['pre_reset_state'].env_state # WrappedEnvState
        eval_render_dones = out[0][4]['__all__']

        final_ckpt_eval_render_init_env_state = jax.tree_map(lambda x: x[:, -1, ...], eval_render_init_env_state)
        final_ckpt_eval_render_env_state = jax.tree_map(lambda x: x[:, -1, ...], eval_render_env_state)
        final_ckpt_eval_render_dones = jax.tree_map(lambda x: x[:, -1, ...], eval_render_dones)

        # num_episodes = final_ckpt_eval_render_dones.shape[1]
        num_episodes = 5
        env.animate((final_ckpt_eval_render_init_env_state, final_ckpt_eval_render_env_state), final_ckpt_eval_render_dones, num_episodes, debug=True)

        for eval_ep in range(num_episodes):
            logger.log_video(
                tag=f"Videos/Single_Agent_Proj_Joint/Episode_{eval_ep}",
                path=os.path.join(env._render_dir, f"{env._render_name}_ep_{eval_ep}.gif")
            )

    elapsed_time = time.perf_counter() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, rem = divmod(rem, 60)
    seconds, rem = divmod(rem, 1)
    milliseconds = int(rem * 1000)
    microseconds = int((rem * 1_000_000) % 1000)
    log.info(f"Single Agent Projection Joint Evaluation completed in {elapsed_time:.2f}s")
    log.info(f"Single Agent Projection Joint Evaluation completed in {int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s {milliseconds:03d}ms {microseconds:03d}µs")

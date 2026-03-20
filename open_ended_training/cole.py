'''Implementation of the COLE algorithm.

https://proceedings.mlr.press/v202/li23au.html


Recommended debug command:
python open_ended_training/run.py algorithm=cole/lbf task=lbf label=test_cole run_heldout_eval=false algorithm.TOTAL_TIMESTEPS_PER_ITERATION=2e5 algorithm.PARTNER_POP_SIZE=2 algorithm.NUM_SEEDS=1 logger.log_train_out=false logger.log_eval_out=false local_logger.save_train_out=false local_logger.save_eval_out=false
'''
from functools import partial
import logging
import shutil
import time
from typing import NamedTuple

from flax.training.train_state import TrainState
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb

from agents.initialize_agents import (
    initialize_s5_agent,
    initialize_mlp_agent,
    initialize_rnn_agent,
)
from agents.population_interface import AgentPopulation
from agents.population_buffer import BufferedPopulation
from common.save_load_utils import save_train_run
from common.plot_utils import get_metric_names
from common.run_episodes import run_episodes
from envs import make_env
from envs.log_wrapper import LogWrapper, LogEnvState
from marl.ppo_utils import Transition, unbatchify, _create_minibatches
from open_ended_training.shapley_utils import shapley_values

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def initialize_agent(actor_type, config, env, rng):
    """Dispatch to the appropriate agent initialiser based on ACTOR_TYPE."""
    if actor_type == "s5":
        return initialize_s5_agent(config, env, rng)
    elif actor_type == "mlp":
        return initialize_mlp_agent(config, env, rng)
    elif actor_type == "rnn":
        return initialize_rnn_agent(config, env, rng)
    else:
        raise ValueError(f"Unsupported ACTOR_TYPE for COLE: {actor_type!r}. "
                         "Choose from 's5', 'mlp', 'rnn'.")


def train_cole_partners(train_rng, wandb_logger, env, config):
    num_agents = env.num_agents
    assert num_agents == 2, "This code assumes the environment has exactly 2 agents."

    # Define 2 types of rollouts: SP, XP
    config["NUM_GAME_AGENTS"] = num_agents

    config["NUM_ACTORS"] = num_agents * config["NUM_ENVS"]
    # Right now assume control of both agent and its BR
    config["NUM_CONTROLLED_ACTORS"] = config["NUM_ACTORS"]
    config["POP_SIZE"] = config["PARTNER_POP_SIZE"]

    # Compute number of updates PER outermost iteration
    # Training rollouts: 2 distinct rollout phases (SP, XP) each using NUM_ENVS
    # TODO: make sure we're including timesteps used to compute XP matrix
    training_steps = 2 * config["ROLLOUT_LENGTH"] * config["NUM_ENVS"]
    config["NUM_UPDATES"] = int(config["TOTAL_TIMESTEPS_PER_ITERATION"] // training_steps)

    def make_cole_agents(config):
        def linear_schedule(count):
            frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
            return config["LR"] * frac

        def train(rng):
            # Start by training a single PPO agent via self-play
            rng, init_ppo_rng, init_conf_rng, xp_init_rng = jax.random.split(rng, 4)

            # Initialize a population buffer
            dummy_policy, dummy_init_params = initialize_agent(config["ACTOR_TYPE"], config, env, init_conf_rng)
            partner_population = BufferedPopulation(
                max_pop_size=config["PARTNER_POP_SIZE"],
                policy_cls=dummy_policy,
                sampling_strategy="softmax",  # scores from metasolve_game_graph are used directly via softmax; staleness is irrelevant
            )

            population_buffer = partner_population.reset_buffer(dummy_init_params)
            # explicitly add a random agent to buffer rather than a PPO agent
            population_buffer = partner_population.add_agent(
                population_buffer, 
                dummy_init_params
            )

            # Initialize XP matrix: shape (POP_SIZE, POP_SIZE)
            # Entry (i, j) = mean return of agent_i when playing as agent_0 against agent_j.
            # Seed entry (0, 0) with the SP return of the initial random policy.
            sp_init_return = run_episodes(
                rng=xp_init_rng, env=env,
                agent_0_param=dummy_init_params, agent_0_policy=dummy_policy,
                agent_1_param=dummy_init_params, agent_1_policy=dummy_policy,
                max_episode_steps=config["ROLLOUT_LENGTH"],
                num_eps=config["XP_EVAL_ROLLOUT_EPS"]
            )
            init_sp_mean = sp_init_return["returned_episode_returns"][:, 0].mean()
            xp_matrix = jnp.zeros((config["POP_SIZE"], config["POP_SIZE"]))
            xp_matrix = xp_matrix.at[0, 0].set(init_sp_mean)

            def add_conf_policy(carry, func_input):
                pop_buffer, xp_matrix = carry
                num_existing_agents, rng = func_input
                rng, init_conf_rng = jax.random.split(rng)

                policy, init_params = initialize_agent(config["ACTOR_TYPE"], config, env, init_conf_rng)

                # Create a train_state and optimizer for the newly initialzied model
                if config["ANNEAL_LR"]:
                    tx = optax.chain(
                        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                        optax.adam(learning_rate=linear_schedule, eps=1e-5),
                    )
                else:
                    tx = optax.chain(
                        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                        optax.adam(config["LR"], eps=1e-5))

                train_state = TrainState.create(
                    apply_fn=policy.network.apply,
                    params=init_params,
                    tx=tx,
                )

                # Reset envs for SP and XP
                rng, reset_rng_eval, reset_rng_sp, reset_rng_xp = jax.random.split(rng, 4)

                reset_rngs_sps = jax.random.split(reset_rng_sp, config["NUM_ENVS"])
                reset_rngs_xps = jax.random.split(reset_rng_xp, config["NUM_ENVS"])

                obsv_xp, env_state_xp = jax.vmap(env.reset, in_axes=(0,))(reset_rngs_xps)
                obsv_sp, env_state_sp = jax.vmap(env.reset, in_axes=(0,))(reset_rngs_sps)

                # build a pytree that can hold the parameters for all checkpoints.
                ckpt_and_eval_interval = config["NUM_UPDATES"] // max(1, config["NUM_CHECKPOINTS"] - 1)
                num_ckpts = config["NUM_CHECKPOINTS"]
                def init_ckpt_array(params_pytree):
                    return jax.tree.map(
                        lambda x: jnp.zeros((num_ckpts,) + x.shape, x.dtype),
                        params_pytree
                    )

                # define evaluation function
                rng, eval_rng = jax.random.split(rng, 2)
                def per_id_run_episode_fixed_rng(agent0_param, agent1_id):
                    agent1_param = partner_population.gather_agent_params(pop_buffer,
                                        agent_indices=agent1_id * jnp.ones((1,), dtype=np.int32))
                    agent1_param = jax.tree_map(lambda y: jnp.squeeze(y, 0), agent1_param)
                    all_outs =  run_episodes(
                        rng=eval_rng, env=env,
                        agent_0_param=agent0_param, agent_0_policy=policy,
                        agent_1_param=agent1_param, agent_1_policy=policy,
                        max_episode_steps=config["ROLLOUT_LENGTH"],
                        num_eps=config["XP_EVAL_ROLLOUT_EPS"]
                    )
                    return all_outs

                def metasolve_game_graph(xp_matrix, agent_idx, num_prev_trained_conf, rng):
                    """Derive a sampling distribution over the trained population.

                    Supports two modes, selected by config["METASOLVE_MODE"]:

                    ``"returns"`` (default)
                        Inverse-return heuristic on the row for ``agent_idx``:
                        partners that ``agent_idx`` performs *worse* against
                        receive higher probability (harder partners prioritised).

                    ``"shapley"``
                        Compute Shapley values over the valid sub-matrix of the
                        XP matrix using coalition-aware weighted PageRank.  The
                        XP matrix is min-max normalised *locally* (only for this
                        computation) so that PageRank edge weights are in [0, 1].
                        The resulting phi scores are converted to a sampling
                        distribution via softmax, giving higher probability to
                        agents with higher Shapley value (i.e. those whose
                        presence most improves coalition performance against hard
                        opponents).

                    In both modes, agents {num_prev_trained_conf, ..., POP_SIZE-1}
                    receive probability zero (untrained agents are excluded).

                    Args:
                        xp_matrix: (POP_SIZE, POP_SIZE) — entry (i, j) is the
                            mean return of agent i as agent_0 vs agent j.
                        agent_idx: Scalar int — current training agent index.
                        num_prev_trained_conf: Number of already-trained agents
                            (valid indices are 0..num_prev_trained_conf-1).
                        rng: JAX PRNG key (only consumed by the ``"shapley"`` mode).

                    Returns:
                        sampling_dist: (POP_SIZE,) array summing to 1.
                    """
                    valid_mask = jnp.arange(config["POP_SIZE"]) < num_prev_trained_conf
                    uniform = valid_mask.astype(jnp.float32) / jnp.maximum(valid_mask.sum(), 1)

                    if config["METASOLVE_MODE"] == "shapley":
                        # Min-max normalise the XP matrix locally so all edge weights
                        # are in [0, 1] for PageRank
                        xp_min = jnp.min(xp_matrix)
                        xp_max = jnp.max(xp_matrix)
                        xp_range = jnp.maximum(xp_max - xp_min, 1e-8)
                        xp_norm = (xp_matrix - xp_min) / xp_range

                        N = config["POP_SIZE"]
                        phi = shapley_values(
                            rng, xp_norm,
                            N=N,
                            max_iter=config["SHAPLEY_MAX_ITER"],
                            damping=config["SHAPLEY_DAMPING"],
                            pagerank_max_iter=config["SHAPLEY_PAGERANK_ITER"],
                            sigma_temperature=config["SHAPLEY_SIGMA_TEMP"],
                        )  # (POP_SIZE,)

                        # Zero out untrained agents, then softmax over valid ones.
                        phi_valid = jnp.where(valid_mask, phi, -jnp.inf)
                        phi_valid = jnp.where(
                            jnp.all(jnp.isinf(phi_valid)), jnp.zeros_like(phi_valid), phi_valid
                        )
                        # Stable softmax: shift by max before exp
                        phi_shifted = phi_valid - jnp.where(valid_mask, phi_valid, -jnp.inf).max()
                        exp_phi = jnp.where(valid_mask, jnp.exp(phi_shifted), 0.0)
                        total = exp_phi.sum()
                        sampling_dist = jnp.where(total > 0, exp_phi / total, uniform)

                    elif config["METASOLVE_MODE"] == "returns":
                        # ── Returns-heuristic mode (default) ─────────────────────────
                        xp_row = xp_matrix[agent_idx]  # (POP_SIZE,)
                        # Negate: lower return → harder partner → higher weight.
                        negated = -xp_row
                        shifted = negated - negated.min() + 1e-8
                        weights = jnp.where(valid_mask, shifted, 0.0)
                        total = weights.sum()
                        sampling_dist = jnp.where(total > 0, weights / total, uniform)
                    else:
                        raise ValueError(f"Unknown METASOLVE_MODE: {config['METASOLVE_MODE']}")

                    return sampling_dist

                def _update_step(update_with_ckpt_runner_state, unused):
                    update_runner_state, checkpoint_array, ckpt_idx = update_with_ckpt_runner_state
                    (
                        train_state, pop_buffer,
                        env_state_sp, obsv_sp,
                        env_state_xp, obsv_xp,
                        last_dones_xp,
                        last_dones_sp,
                        partner_indices_xp,
                        ego_hstate_xp, partner_hstate,
                        ego_hstate_sp0, ego_hstate_sp1,
                        rng, update_steps,
                        num_prev_trained_conf, partner_visit_counts
                    ) = update_runner_state

                    def _env_step_xp(runner_state, unused):
                        """
                        XP (cross-play) rollout step.
                        agent_0 = ego agent being trained, agent_1 = partner sampled from the population buffer.
                        Partners are resampled per env at episode boundaries using the metasolve distribution
                        stored in pop_buffer.scores.
                        Returns updated runner_state and a Transition for agent_0.
                        """
                        train_state, pop_buffer, partner_indices, env_state, last_obs, last_dones, ego_hstate, partner_hstate, partner_visit_counts, rng = runner_state
                        rng, act_rng, partner_rng, step_rng, resample_rng = jax.random.split(rng, 5)

                        # Resample partner index for envs that just finished an episode
                        needs_resample = last_dones["__all__"]
                        new_sampled, pop_buffer = partner_population.sample_agent_indices(
                            pop_buffer, config["NUM_ENVS"], resample_rng,
                            needs_resample_mask=needs_resample
                        )
                        partner_indices = jnp.where(needs_resample, new_sampled, partner_indices)
                        partner_visit_counts = partner_visit_counts.at[new_sampled].add(needs_resample.astype(jnp.int32))

                        # Get available actions from environment state
                        avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                        avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
                        avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

                        # Agent_0 (trained ego agent) action
                        act_0, val_0, pi_0, new_ego_hstate = policy.get_action_value_policy(
                            params=train_state.params,
                            obs=last_obs["agent_0"].reshape(1, config["NUM_ENVS"], -1),
                            done=last_dones["agent_0"].reshape(1, config["NUM_ENVS"]),
                            avail_actions=jax.lax.stop_gradient(avail_actions_0),
                            hstate=ego_hstate,
                            rng=act_rng,
                        )
                        logp_0 = pi_0.log_prob(act_0)
                        act_0 = act_0.squeeze()
                        logp_0 = logp_0.squeeze()
                        val_0 = val_0.squeeze()

                        # Agent_1 (population partner) action - one partner per env via BufferedPopulation
                        act_1, new_partner_hstate = partner_population.get_actions(
                            buffer=pop_buffer,
                            agent_indices=partner_indices,
                            obs=last_obs["agent_1"],
                            done=last_dones["agent_1"],
                            avail_actions=avail_actions_1,
                            hstate=partner_hstate,
                            rng=partner_rng,
                        )
                        act_1 = act_1.squeeze()

                        # Combine actions into the env format
                        combined_actions = jnp.concatenate([act_0, act_1], axis=0)  # (2*num_envs,)
                        env_act = unbatchify(combined_actions, env.agents, config["NUM_ENVS"], num_agents)
                        env_act = {k: v.flatten() for k, v in env_act.items()}

                        # Step env
                        step_rngs = jax.random.split(step_rng, config["NUM_ENVS"])
                        obs_next, env_state_next, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                            step_rngs, env_state, env_act
                        )
                        info_0 = jax.tree.map(lambda x: x[:, 0], info)

                        # Store agent_0 data in transition
                        transition = Transition(
                            done=done["agent_0"],
                            action=act_0,
                            value=val_0,
                            reward=reward["agent_0"],
                            log_prob=logp_0,
                            obs=last_obs["agent_0"],
                            info=info_0,
                            avail_actions=avail_actions_0
                        )
                        new_runner_state = (train_state, pop_buffer, partner_indices,
                                            env_state_next, obs_next, done,
                                            new_ego_hstate, new_partner_hstate, partner_visit_counts, 
                                            rng)
                        return new_runner_state, transition

                    def _env_step_sp(runner_state, unused):
                        """
                        SP (self-play) rollout step.
                        Both agent slots (agent_0 and agent_1) use the same confederate parameters.
                        Separate hidden states are tracked per slot so recurrent agents are handled correctly.
                        Returns updated runner_state and Transitions for both agent slots.
                        """
                        train_state, env_state, last_obs, last_dones, ego_hstate_sp0, ego_hstate_sp1, rng = runner_state
                        rng, conf_rng, br_rng, step_rng = jax.random.split(rng, 4)

                        # Get available actions for agent 0 from environment state
                        avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                        avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
                        avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

                        # Agent_0 (agent slot 0 in SP) action
                        act_0, val_0, pi_0, new_ego_hstate_sp0 = policy.get_action_value_policy(
                            params=train_state.params,
                            obs=last_obs["agent_0"].reshape(1, config["NUM_ENVS"], -1),
                            done=last_dones["agent_0"].reshape(1, config["NUM_ENVS"]),
                            avail_actions=jax.lax.stop_gradient(avail_actions_0),
                            hstate=ego_hstate_sp0,
                            rng=conf_rng
                        )
                        logp_0 = pi_0.log_prob(act_0)
                        act_0 = act_0.squeeze()
                        logp_0 = logp_0.squeeze()
                        val_0 = val_0.squeeze()

                        # Agent_1 (agent slot 1 in SP) action — same params, different hstate
                        act_1, val_1, pi_1, new_ego_hstate_sp1 = policy.get_action_value_policy(
                            params=train_state.params,
                            obs=last_obs["agent_1"].reshape(1, config["NUM_ENVS"], -1),
                            done=last_dones["agent_1"].reshape(1, config["NUM_ENVS"]),
                            avail_actions=jax.lax.stop_gradient(avail_actions_1),
                            hstate=ego_hstate_sp1,
                            rng=br_rng
                        )
                        logp_1 = pi_1.log_prob(act_1)
                        act_1 = act_1.squeeze()
                        logp_1 = logp_1.squeeze()
                        val_1 = val_1.squeeze()

                        # Combine actions into the env format
                        combined_actions = jnp.concatenate([act_0, act_1], axis=0)  # shape (2*num_envs,)
                        env_act = unbatchify(combined_actions, env.agents, config["NUM_ENVS"], num_agents)
                        env_act = {k: v.flatten() for k, v in env_act.items()}

                        # Step env
                        step_rngs = jax.random.split(step_rng, config["NUM_ENVS"])
                        obs_next, env_state_next, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                            step_rngs, env_state, env_act
                        )
                        info_0 = jax.tree.map(lambda x: x[:, 0], info)
                        info_1 = jax.tree.map(lambda x: x[:, 1], info)

                        # Store agent_0 data in transition
                        transition_0 = Transition(
                            done=done["agent_0"],
                            action=act_0,
                            value=val_0,
                            reward=reward["agent_0"],
                            log_prob=logp_0,
                            obs=last_obs["agent_0"],
                            info=info_0,
                            avail_actions=avail_actions_0
                        )
                        # Store agent_1 data in transition
                        transition_1 = Transition(
                            done=done["agent_1"],
                            action=act_1,
                            value=val_1,
                            reward=reward["agent_1"],
                            log_prob=logp_1,
                            obs=last_obs["agent_1"],
                            info=info_1,
                            avail_actions=avail_actions_1
                        )
                        new_runner_state = (train_state, env_state_next, obs_next, done,
                                            new_ego_hstate_sp0, new_ego_hstate_sp1, rng)
                        return new_runner_state, (transition_0, transition_1)

                    # Sample per-env XP partner indices, resampling only for envs that finished an episode
                    rng, sample_rng, rng_xp, rng_sp = jax.random.split(rng, 4)
                    needs_resample_xp = last_dones_xp["__all__"]
                    sampled_indices_all, pop_buffer = partner_population.sample_agent_indices(
                        pop_buffer, config["NUM_ENVS"], sample_rng,
                        needs_resample_mask=needs_resample_xp
                    )
                    partner_indices_xp = jnp.where(
                        needs_resample_xp, sampled_indices_all, partner_indices_xp
                    )

                    # Update sampling distribution using UCB logic
                    sum_counts = partner_visit_counts.sum()
                    ucb_scores = init_sampling_dist + config["C_UCT"] * jnp.sqrt(sum_counts / (1 + partner_visit_counts))
                    pop_buffer = partner_population.update_scores(
                        pop_buffer, jnp.arange(config["POP_SIZE"]), ucb_scores
                    )

                    # Do XP rollout
                    runner_state_xp = (train_state, pop_buffer, partner_indices_xp,
                                       env_state_xp, obsv_xp, last_dones_xp,
                                       ego_hstate_xp, partner_hstate, partner_visit_counts, rng_xp)
                    runner_state_xp, traj_batch_xp = jax.lax.scan(
                        _env_step_xp, 
                        runner_state_xp, 
                        None, 
                        config["ROLLOUT_LENGTH"])
                    (train_state, pop_buffer, partner_indices_xp, env_state_xp,
                     last_obs_xp, last_dones_xp, ego_hstate_xp, partner_hstate, partner_visit_counts, rng_xp) = runner_state_xp

                    # Do self-play (based on train_state params) rollout
                    runner_state_sp = (train_state, env_state_sp, obsv_sp, last_dones_sp,
                                       ego_hstate_sp0, ego_hstate_sp1, rng_sp)
                    runner_state_sp, (traj_batch_sp_agent0, traj_batch_sp_agent1) = jax.lax.scan(
                        _env_step_sp, runner_state_sp, None, config["ROLLOUT_LENGTH"])
                    (train_state, env_state_sp, last_obs_sp, last_dones_sp,
                     ego_hstate_sp0, ego_hstate_sp1, rng_sp) = runner_state_sp

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

                    def _compute_advantages_and_targets(env_state, policy, policy_params, policy_hstate,
                                                    last_obs, last_dones, traj_batch, agent_name):
                        """Compute GAE advantages and value targets for a trajectory."""
                        avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)[agent_name].astype(jnp.float32)
                        _, vals, _, _ = policy.get_action_value_policy(
                            params=policy_params,
                            obs=last_obs[agent_name].reshape(1, last_obs[agent_name].shape[0], -1),
                            done=last_dones[agent_name].reshape(1, last_obs[agent_name].shape[0]),
                            avail_actions=jax.lax.stop_gradient(avail_actions),
                            hstate=policy_hstate,
                            rng=jax.random.PRNGKey(0),  # dummy key as we don't sample actions
                        )
                        last_val = vals.squeeze()
                        advantages, targets = _calculate_gae(traj_batch, last_val)
                        return advantages, targets

                    # 5a) Compute conf advantages for XP interaction
                    advantages_xp_conf, targets_xp_conf = _compute_advantages_and_targets(
                        env_state_xp, policy, train_state.params, ego_hstate_xp,
                        last_obs_xp, last_dones_xp, traj_batch_xp, "agent_0")

                    # 5b) Compute conf and br advantages for SP (conf-br) interaction
                    advantages_sp_conf, targets_sp_conf = _compute_advantages_and_targets(
                        env_state_sp, policy, train_state.params, ego_hstate_sp0,
                        last_obs_sp, last_dones_sp, traj_batch_sp_agent0, "agent_0")

                    advantages_sp_br, targets_sp_br = _compute_advantages_and_targets(
                        env_state_sp, policy, train_state.params, ego_hstate_sp1,
                        last_obs_sp, last_dones_sp, traj_batch_sp_agent1, "agent_1")

                    def _update_epoch(update_state, unused):
                        def _compute_ppo_value_loss(pred_value, traj_batch, target_v):
                            '''Value loss function for PPO'''
                            value_pred_clipped = traj_batch.value + (
                                pred_value - traj_batch.value
                                ).clip(
                                -config["CLIP_EPS"], config["CLIP_EPS"])
                            value_losses = jnp.square(pred_value - target_v)
                            value_losses_clipped = jnp.square(value_pred_clipped - target_v)
                            value_loss = (
                                jnp.maximum(value_losses, value_losses_clipped).mean()
                            )
                            return value_loss

                        def _compute_ppo_pg_loss(log_prob, traj_batch, gae):
                            '''Policy gradient loss function for PPO'''
                            ratio = jnp.exp(log_prob - traj_batch.log_prob)
                            gae_norm = (gae - gae.mean()) / (gae.std() + 1e-8)
                            pg_loss_1 = ratio * gae_norm
                            pg_loss_2 = jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"]) * gae_norm
                            pg_loss = -jnp.mean(jnp.minimum(pg_loss_1, pg_loss_2))
                            return pg_loss

                        def _update_minbatch_conf(train_state_conf, batch_infos):
                            minbatch_xp, minbatch_sp1, minbatch_sp2 = batch_infos
                            init_hstate_xp, traj_batch_xp, advantages_xp, returns_xp = minbatch_xp
                            init_hstate_sp1, traj_batch_sp1, advantages_sp1, returns_sp1 = minbatch_sp1
                            init_hstate_sp2, traj_batch_sp2, advantages_sp2, returns_sp2 = minbatch_sp2

                            def _loss_fn_conf(params, traj_batch_xp, gae_xp, target_v_xp,
                                            traj_batch_sp, gae_sp, target_v_sp,
                                            traj_batch_sp2, gae_sp2, target_v_sp2):
                                # XP: training agent interacting with population partner
                                _, value_xp, pi_xp, _ = policy.get_action_value_policy(
                                    params=params,
                                    obs=traj_batch_xp.obs,
                                    done=traj_batch_xp.done,
                                    avail_actions=traj_batch_xp.avail_actions,
                                    hstate=init_hstate_xp,
                                    rng=jax.random.PRNGKey(0),
                                )

                                # SP: agent slot 0 interacting with itself
                                _, value_sp, pi_sp, _ = policy.get_action_value_policy(
                                    params=params,
                                    obs=traj_batch_sp.obs,
                                    done=traj_batch_sp.done,
                                    avail_actions=traj_batch_sp.avail_actions,
                                    hstate=init_hstate_sp1,
                                    rng=jax.random.PRNGKey(0),
                                )

                                # SP: agent slot 1 interacting with itself
                                _, value_sp2, pi_sp2, _ = policy.get_action_value_policy(
                                    params=params,
                                    obs=traj_batch_sp2.obs,
                                    done=traj_batch_sp2.done,
                                    avail_actions=traj_batch_sp2.avail_actions,
                                    hstate=init_hstate_sp2,
                                    rng=jax.random.PRNGKey(0),
                                )

                                log_prob_xp = pi_xp.log_prob(traj_batch_xp.action)
                                log_prob_sp = pi_sp.log_prob(traj_batch_sp.action)
                                log_prob_sp2 = pi_sp2.log_prob(traj_batch_sp2.action)

                                value_loss_xp = _compute_ppo_value_loss(value_xp, traj_batch_xp, target_v_xp)
                                value_loss_sp = _compute_ppo_value_loss(value_sp, traj_batch_sp, target_v_sp)
                                value_loss_sp2 = _compute_ppo_value_loss(value_sp2, traj_batch_sp2, target_v_sp2)

                                pg_loss_xp = _compute_ppo_pg_loss(log_prob_xp, traj_batch_xp, gae_xp)
                                pg_loss_sp = _compute_ppo_pg_loss(log_prob_sp, traj_batch_sp, gae_sp)
                                pg_loss_sp2 = _compute_ppo_pg_loss(log_prob_sp2, traj_batch_sp2, gae_sp2)

                                entropy_xp = jnp.mean(pi_xp.entropy())
                                entropy_sp = jnp.mean(pi_sp.entropy())
                                entropy_sp2 = jnp.mean(pi_sp2.entropy())

                                # Loss = XP_loss + COLE_ALPHA * SP_loss (both maximized)
                                cole_alpha = config["COLE_ALPHA"]
                                xp_loss = pg_loss_xp + config["VF_COEF"] * value_loss_xp - config["ENT_COEF"] * entropy_xp
                                sp_loss = pg_loss_sp + config["VF_COEF"] * value_loss_sp - config["ENT_COEF"] * entropy_sp
                                sp2_loss = pg_loss_sp2 + config["VF_COEF"] * value_loss_sp2 - config["ENT_COEF"] * entropy_sp2

                                total_loss = xp_loss + cole_alpha * (sp_loss + sp2_loss)
                                return total_loss, (value_loss_xp, value_loss_sp + value_loss_sp2,
                                                    pg_loss_xp, pg_loss_sp + pg_loss_sp2,
                                                    entropy_xp, entropy_sp + entropy_sp2)

                            grad_fn = jax.value_and_grad(_loss_fn_conf, has_aux=True)
                            (loss_val, aux_vals), grads = grad_fn(
                                train_state_conf.params,
                                traj_batch_xp, advantages_xp, returns_xp,
                                traj_batch_sp1, advantages_sp1, returns_sp1,
                                traj_batch_sp2, advantages_sp2, returns_sp2)
                            train_state_conf = train_state_conf.apply_gradients(grads=grads)
                            return train_state_conf, (loss_val, aux_vals)

                        (
                            train_state_conf, traj_batch_xp,
                            traj_batch_sp_conf, traj_batch_sp_br,
                            advantages_xp_conf, advantages_sp_conf,
                            advantages_sp_br, targets_xp_conf,
                            targets_sp_conf, targets_sp_br,
                            init_hstate_xp, init_hstate_sp0, init_hstate_sp1, rng
                        ) = update_state

                        rng, perm_rng_xp, perm_rng_sp_conf, perm_rng_sp_br = jax.random.split(rng, 4)

                        # Create minibatches for XP and SP interaction types
                        minibatches_xp = _create_minibatches(
                            traj_batch_xp, advantages_xp_conf, targets_xp_conf, init_hstate_xp,
                            config["NUM_ENVS"], config["NUM_MINIBATCHES"], perm_rng_xp
                        )
                        minibatches_sp_conf = _create_minibatches(
                            traj_batch_sp_conf, advantages_sp_conf, targets_sp_conf, init_hstate_sp0,
                            config["NUM_ENVS"], config["NUM_MINIBATCHES"], perm_rng_sp_conf
                        )
                        minibatches_sp_br = _create_minibatches(
                            traj_batch_sp_br, advantages_sp_br, targets_sp_br, init_hstate_sp1,
                            config["NUM_ENVS"], config["NUM_MINIBATCHES"], perm_rng_sp_br
                        )

                        # Update confederate
                        train_state_conf, total_loss_conf = jax.lax.scan(
                            _update_minbatch_conf, train_state_conf,
                            (minibatches_xp, minibatches_sp_conf, minibatches_sp_br)
                        )

                        update_state = (train_state_conf,
                            traj_batch_xp, traj_batch_sp_conf, traj_batch_sp_br,
                            advantages_xp_conf, advantages_sp_conf, advantages_sp_br,
                            targets_xp_conf, targets_sp_conf, targets_sp_br,
                            init_hstate_xp, init_hstate_sp0, init_hstate_sp1, rng
                        )
                        return update_state, total_loss_conf

                    # 3) PPO update
                    # init_hstate_* are fresh zero hstates to replay trajectory from (like ippo.py)
                    init_hstate_xp = policy.init_hstate(config["NUM_ENVS"])
                    init_hstate_sp0 = policy.init_hstate(config["NUM_ENVS"])
                    init_hstate_sp1 = policy.init_hstate(config["NUM_ENVS"])
                    rng, sub_rng = jax.random.split(rng, 2)
                    update_state = (
                        train_state,
                        traj_batch_xp, traj_batch_sp_agent0,
                        traj_batch_sp_agent1,
                        advantages_xp_conf,
                        advantages_sp_conf, advantages_sp_br,
                        targets_xp_conf, targets_sp_conf,
                        targets_sp_br,
                        init_hstate_xp, init_hstate_sp0, init_hstate_sp1, sub_rng
                    )
                    update_state, conf_losses = jax.lax.scan(
                        _update_epoch, update_state, None, config["UPDATE_EPOCHS"])
                    train_state = update_state[0]

                    (
                        conf_value_loss_xp, conf_value_loss_sp,
                        conf_pg_loss_xp, conf_pg_loss_sp,
                        conf_entropy_xp, conf_entropy_sp
                    ) = conf_losses[1]

                    new_update_runner_state = (
                        train_state, pop_buffer,
                        env_state_sp, last_obs_sp,
                        env_state_xp, last_obs_xp,
                        last_dones_xp, last_dones_sp,
                        partner_indices_xp,
                        ego_hstate_xp, partner_hstate,
                        ego_hstate_sp0, ego_hstate_sp1,
                        rng, update_steps+1, num_prev_trained_conf,
                        partner_visit_counts
                    )

                    # Metrics
                    metric = traj_batch_xp.info
                    metric["update_steps"] = update_steps
                    metric["value_loss_conf_xp"] = conf_value_loss_xp
                    metric["value_loss_conf_sp"] = conf_value_loss_sp

                    metric["pg_loss_conf_xp"] = conf_pg_loss_xp
                    metric["pg_loss_conf_sp"] = conf_pg_loss_sp

                    metric["entropy_conf_xp"] = conf_entropy_xp
                    metric["entropy_conf_sp"] = conf_entropy_sp

                    metric["average_rewards_ego"] = jnp.mean(traj_batch_xp.reward)
                    metric["average_rewards_br_sp"] = jnp.mean(traj_batch_sp_agent1.reward)

                    return (new_update_runner_state, checkpoint_array, ckpt_idx+1), metric

                # XP eval current policy against all policies in the buffer
                # shape (pop_size, num_eval_episodes, num_agents_per_game)
                xp_eval_returns = jax.vmap(per_id_run_episode_fixed_rng, in_axes=(None, 0))(
                        train_state.params, jnp.arange(config["POP_SIZE"]))

                # SP performance against itself
                sp_eval_returns = run_episodes(
                    eval_rng, env,
                    agent_0_param=train_state.params, agent_0_policy=policy,
                    agent_1_param=train_state.params, agent_1_policy=policy,
                    max_episode_steps=config["ROLLOUT_LENGTH"],
                    num_eps=config["NUM_EVAL_EPISODES"]
                )


                # Pre-compute the metasolve sampling distribution for agent i and write it into
                # pop_buffer.scores so that sample_agent_indices uses it throughout the PPO loop.
                rng, metasolve_rng = jax.random.split(rng)
                init_sampling_dist = metasolve_game_graph(
                    xp_matrix, num_existing_agents, num_existing_agents, metasolve_rng
                )
                pop_buffer = partner_population.update_scores(
                    pop_buffer, jnp.arange(config["POP_SIZE"]), init_sampling_dist
                )

                update_steps = 0
                init_done_xp = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}
                init_done_sp = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}
                # Initialize XP partner indices: default to agent 0 until resampled
                init_partner_indices_xp = jnp.zeros((config["NUM_ENVS"],), dtype=jnp.int32)

                # Initialize hidden states for the training agent (xp + sp slots) and population partner
                init_ego_hstate_xp = policy.init_hstate(config["NUM_ENVS"])
                init_partner_hstate = partner_population.policy_cls.init_hstate(config["NUM_ENVS"])
                init_ego_hstate_sp0 = policy.init_hstate(config["NUM_ENVS"])
                init_ego_hstate_sp1 = policy.init_hstate(config["NUM_ENVS"])

                # Initialize visit counts for UCB style sampling of partners
                init_partner_visit_counts = jnp.zeros((config["POP_SIZE"],), dtype=jnp.int32)

                update_runner_state = (
                    train_state, pop_buffer,
                    env_state_sp, obsv_sp,
                    env_state_xp, obsv_xp,
                    init_done_xp, init_done_sp,
                    init_partner_indices_xp,
                    init_ego_hstate_xp, init_partner_hstate,
                    init_ego_hstate_sp0, init_ego_hstate_sp1,
                    rng, update_steps,
                    num_existing_agents,
                    init_partner_visit_counts,
                )


                checkpoint_array = init_ckpt_array(train_state.params)
                ckpt_idx = 0
                update_with_ckpt_runner_state = (update_runner_state, checkpoint_array, ckpt_idx, xp_eval_returns, sp_eval_returns)

                def _update_step_with_ckpt(state_with_ckpt, unused):
                    (update_runner_state, checkpoint_array, ckpt_idx, xp_eval_returns, sp_eval_returns) = state_with_ckpt
                    train_state = update_runner_state[0]

                    # Single PPO update
                    new_state_with_ckpt, metric = _update_step(
                        (update_runner_state, checkpoint_array, ckpt_idx),
                        None
                    )
                    new_update_runner_state = new_state_with_ckpt[0]
                    rng, update_steps = new_update_runner_state[13], new_update_runner_state[14]

                    # Decide if we store a checkpoint
                    # update steps is 1-indexed because it was incremented at the end of the update step
                    to_store = jnp.logical_or(jnp.equal(jnp.mod(update_steps-1, ckpt_and_eval_interval), 0),
                                            jnp.equal(update_steps, config["NUM_UPDATES"]))

                    def store_and_eval_ckpt(args):
                        ckpt_arr_conf, rng, cidx, _, _ = args
                        new_ckpt_arr_conf = jax.tree.map(
                            lambda c_arr, p: c_arr.at[cidx].set(p),
                            ckpt_arr_conf, train_state.params
                        )

                        # Eval trained agent against all params in the pool
                        xp_eval_returns = jax.vmap(per_id_run_episode_fixed_rng, in_axes=(None, 0))(
                            train_state.params, jnp.arange(config["POP_SIZE"]))
                        # Eval trained agent against itself
                        sp_eval_returns = run_episodes(
                            eval_rng, env,
                            agent_0_param=train_state.params, agent_0_policy=policy,
                            agent_1_param=train_state.params, agent_1_policy=policy,
                            max_episode_steps=config["ROLLOUT_LENGTH"],
                            num_eps=config["NUM_EVAL_EPISODES"]
                        )
                        return (new_ckpt_arr_conf, rng, cidx + 1, xp_eval_returns, sp_eval_returns)

                    def skip_ckpt(args):
                        return args

                    rng, store_and_eval_rng = jax.random.split(rng, 2)
                    (checkpoint_array, store_and_eval_rng, ckpt_idx, xp_eval_returns, sp_eval_returns) = jax.lax.cond(
                        to_store,
                        store_and_eval_ckpt,
                        skip_ckpt,
                        (checkpoint_array, store_and_eval_rng, ckpt_idx, xp_eval_returns, sp_eval_returns)
                    )

                    return (new_update_runner_state, checkpoint_array,
                            ckpt_idx, xp_eval_returns, sp_eval_returns), (metric, xp_eval_returns, sp_eval_returns)

                new_update_with_ckpt_runner_state, (metric, xp_eval_returns, sp_eval_returns) = jax.lax.scan(
                    _update_step_with_ckpt,
                    update_with_ckpt_runner_state,
                    xs=None,  # No per-step input data
                    length=config["NUM_UPDATES"],
                )
                new_update_runner_state, new_checkpoint_array, _, _, _ = new_update_with_ckpt_runner_state
                final_train_state = new_update_runner_state[0]
                # xp_matrix is unchanged during the inner PPO loop; use the local variable directly

                # Add newly trained agent i to the population buffer (default score, updated below)
                updated_pop_buffer = partner_population.add_agent(pop_buffer, final_train_state.params)

                # ------------------------------------------------------------------
                # Update XP matrix for agent i (= num_existing_agents).
                # Evaluate agent i vs every agent j in {0, ..., i} and vice-versa.
                # Entry (i, j): mean return of agent_i as agent_0 vs agent_j as agent_1.
                # Entry (j, i): mean return of agent_j as agent_0 vs agent_i as agent_1.
                # ------------------------------------------------------------------
                rng, xp_update_rng = jax.random.split(rng)

                def eval_pair_ij(agent_i_param, agent_j_param):
                    """Return mean reward of agent_i (as agent_0) vs agent_j."""
                    outs = run_episodes(
                        rng=xp_update_rng, env=env,
                        agent_0_param=agent_i_param, agent_0_policy=policy,
                        agent_1_param=agent_j_param, agent_1_policy=policy,
                        max_episode_steps=config["ROLLOUT_LENGTH"],
                        num_eps=config["XP_EVAL_ROLLOUT_EPS"]
                    )
                    return outs["returned_episode_returns"][:, 0].mean()

                new_agent_i = num_existing_agents  # index of newly added agent

                # Gather all current agent params (including the new agent at index new_agent_i)
                all_indices = jnp.arange(config["POP_SIZE"])

                # Row i  : agent_i (new) as agent_0 vs each existing agent j as agent_1
                def eval_i_vs_j(j_idx):
                    agent_j_param = jax.tree_map(
                        lambda x: jnp.squeeze(x, 0),
                        partner_population.gather_agent_params(
                            updated_pop_buffer,
                            agent_indices=j_idx * jnp.ones((1,), dtype=jnp.int32)
                        )
                    )
                    return eval_pair_ij(final_train_state.params, agent_j_param)

                # Column i: each existing agent j as agent_0 vs agent_i (new) as agent_1
                def eval_j_vs_i(j_idx):
                    agent_j_param = jax.tree_map(
                        lambda x: jnp.squeeze(x, 0),
                        partner_population.gather_agent_params(
                            updated_pop_buffer,
                            agent_indices=j_idx * jnp.ones((1,), dtype=jnp.int32)
                        )
                    )
                    return eval_pair_ij(agent_j_param, final_train_state.params)

                # Evaluate over all population slots (masking of invalid entries done via xp_matrix)
                row_i_returns = jax.vmap(eval_i_vs_j)(all_indices)   # shape (POP_SIZE,)
                col_i_returns = jax.vmap(eval_j_vs_i)(all_indices)   # shape (POP_SIZE,)

                # Write into xp_matrix (entries for untrained agents will be 0 and are masked by metasolve)
                xp_matrix = xp_matrix.at[new_agent_i, :].set(row_i_returns)
                xp_matrix = xp_matrix.at[:, new_agent_i].set(col_i_returns)

                # ------------------------------------------------------------------
                # Update pop_buffer scores: for each trained agent j in {0,...,i},
                # recompute its sampling distribution and use it as a score so that
                # future training iterations can be guided by the current XP matrix.
                # We summarise each agent j's distribution as its mean sampling weight
                # (a high weight means j is a hard partner and should be prioritised).
                # ------------------------------------------------------------------
                num_trained_so_far = new_agent_i + 1  # agents {0, ..., i} are now trained

                def compute_score_for_agent(j_idx, score_rng):
                    """Sampling weight of agent j under the distribution for agent j."""
                    dist_j = metasolve_game_graph(xp_matrix, j_idx, num_trained_so_far, score_rng)
                    return dist_j[j_idx]  # agent j's own weight in its distribution

                rng, score_rng = jax.random.split(rng)
                score_rngs = jax.random.split(score_rng, config["POP_SIZE"])
                new_scores = jax.vmap(compute_score_for_agent)(all_indices, score_rngs)  # (POP_SIZE,)
                updated_pop_buffer = partner_population.update_scores(updated_pop_buffer, all_indices, new_scores)

                conf_checkpoints = new_checkpoint_array
                return (updated_pop_buffer, xp_matrix), (conf_checkpoints, metric, xp_eval_returns, sp_eval_returns)

            rngs = jax.random.split(rng, config["PARTNER_POP_SIZE"])
            rng, add_conf_iter_rngs = rngs[0], rngs[1:]

            # TODO: get rid of iter_ids
            iter_ids = jnp.arange(1, config["PARTNER_POP_SIZE"])
            (final_population_buffer, final_xp_matrix), (conf_checkpoints, metric, xp_eval_returns, sp_eval_returns) = jax.lax.scan(
                add_conf_policy, (population_buffer, xp_matrix), (iter_ids, add_conf_iter_rngs)
            )

            out = {
                "final_params_conf": final_population_buffer.params,
                "checkpoints_conf": conf_checkpoints,
                "metrics": metric,
                "last_ep_infos_xp": xp_eval_returns,
                "last_ep_infos_sp": sp_eval_returns
            }

            return out
        return train

    train_fn = make_cole_agents(config)
    out = train_fn(train_rng)
    return out

def get_cole_population(config, out, env):
    '''
    Get the partner params and partner population for ego training.
    '''
    cole_pop_size = config["algorithm"]["PARTNER_POP_SIZE"]

    # partner_params has shape (num_seeds, cole_pop_size, ...)
    partner_params = out['final_params_conf']

    rng = jax.random.PRNGKey(0)  # dummy key; only used for parameter shape
    partner_policy, _ = initialize_agent(
        config["algorithm"]["ACTOR_TYPE"], dict(config["algorithm"]), env, rng
    )

    # Create partner population
    partner_population = AgentPopulation(
        pop_size=cole_pop_size,
        policy_cls=partner_policy
    )

    return partner_params, partner_population

def run_cole(config, wandb_logger):
    algorithm_config = dict(config["algorithm"])

    env = make_env(algorithm_config["ENV_NAME"], algorithm_config["ENV_KWARGS"])
    env = LogWrapper(env)

    log.info("Starting COLE training...")
    start = time.time()

    # Generate multiple random seeds from the base seed
    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"])
    rng, eval_rng = jax.random.split(rng)
    rngs = jax.random.split(rng, algorithm_config["NUM_SEEDS"])

    # Create a vmapped version of train_cole_partners
    with jax.disable_jit(False):
        vmapped_train_fn = jax.jit(
            jax.vmap(
                partial(train_cole_partners, 
                        wandb_logger=wandb_logger,
                        env=env, 
                        config=algorithm_config)
            )
        )
        out = vmapped_train_fn(rngs)

    end = time.time()
    log.info(f"COLE training complete in {end - start} seconds")

    metric_names = get_metric_names(algorithm_config["ENV_NAME"])

    log_metrics(config, out, wandb_logger, metric_names)
    
    dummy_env = make_env(algorithm_config["ENV_NAME"], algorithm_config["ENV_KWARGS"])
    ego_policy, init_ego_params = initialize_agent(
        algorithm_config["ACTOR_TYPE"], algorithm_config, dummy_env, eval_rng
    )

    # TODO: we return the last trained ego agent as a place holder for now, while we don't have the
    # proper XP matrix.
    # ultimately, we want to compute the best ego params for each seed, selected from all checkpoints over the entire population,
    # i.e., select from out["checkpoints_conf"]
    best_ids = -1
    best_ego_params = jax.tree.map(lambda x: x[:, best_ids], out["final_params_conf"])

    return ego_policy, best_ego_params, init_ego_params

def log_metrics(config, outs, logger, metric_names: tuple):
    metrics = outs["metrics"]
    # trained_pop_size excludes the initial policy, so it's pop_size - 1
    num_seeds, trained_pop_size, num_updates, _, _ = metrics["pg_loss_conf_sp"].shape
    # TODO: add the eval_ep_last_info metrics

    ### Log evaluation metrics
    # we plot XP return curves separately from SP return curves
    # shape (num_seeds, pop_size - 1, num_updates, num_eval_episodes, num_agents_per_game)
    all_returns_sp = np.asarray(outs["last_ep_infos_sp"]["returned_episode_returns"])
    # shape (num_seeds, pop_size - 1, num_updates, pop_size, num_eval_episodes, num_agents_per_game)
    all_returns_xp = np.asarray(outs["last_ep_infos_xp"]["returned_episode_returns"])

    # Average over seeds, eval episodes and num_agents_per_game
    sp_return_curve = all_returns_sp.mean(axis=(0, 3, 4)) # shape (pop_size - 1, num_updates)
    xp_return_curve = all_returns_xp.mean(axis=(0, 4, 5)) #  shape (pop_size - 1, num_updates, pop_size)

    for num_add_policies in range(trained_pop_size):
        for update_step in range(num_updates):
            logger.log_item("Eval/AvgSPReturnCurve", sp_return_curve[num_add_policies, update_step], train_step=update_step)
            mean_xp_returns = xp_return_curve[num_add_policies, :, :(num_add_policies+1)].mean(axis=-1)
            logger.log_item("Eval/AvgXPReturnCurve", mean_xp_returns[update_step], train_step=update_step)
    logger.commit()

    ### Log population loss as multi-line plots, where each line is a different population member
    # both xp and sp metrics has shape (num_seeds, pop_size - 1, num_updates, update_epochs, num_minibatches)
    # Average over seeds
    processed_losses = {
        "ConfPGLossSP": np.asarray(metrics["pg_loss_conf_sp"]).mean(axis=(0, 3, 4)), # desired shape (pop_size - 1, num_updates)
        "ConfPGLossXP": np.asarray(metrics["pg_loss_conf_xp"]).mean(axis=(0, 3, 4)),
        "ConfValLossSP": np.asarray(metrics["value_loss_conf_sp"]).mean(axis=(0, 3, 4)),
        "ConfValLossXP": np.asarray(metrics["value_loss_conf_xp"]).mean(axis=(0, 3, 4)),
        "EntropySP": np.asarray(metrics["entropy_conf_sp"]).mean(axis=(0, 3, 4)),
        "EntropyXP": np.asarray(metrics["entropy_conf_xp"]).mean(axis=(0, 3, 4)),
    }

    xs = list(range(num_updates))
    keys = [f"pair {i}" for i in range(trained_pop_size)]

    for loss_name, loss_data in processed_losses.items():
        logger.log_item(f"Losses/{loss_name}",
            wandb.plot.line_series(xs=xs, ys=loss_data, keys=keys,
            title=loss_name, xname="train_step")
        )

    ### Log artifacts
    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    # Save train run output and log to wandb as artifact
    out_savepath = save_train_run(outs, savedir, savename="saved_train_run")
    if config["logger"]["log_train_out"]:
        logger.log_artifact(name="saved_train_run", path=out_savepath, type_name="train_run")

    # Cleanup locally logged out files
    if not config["local_logger"]["save_train_out"]:
        shutil.rmtree(out_savepath)

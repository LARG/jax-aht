'''Implementation of the CoMeDi teammate generation algorithm (Sarkar et al. NeurIPS 2023)
https://openreview.net/forum?id=MljeRycu9s

Command to run CoMeDi only on LBF:
python teammate_generation/run.py algorithm=comedi/lbf task=lbf label=test_comedi run_heldout_eval=false train_ego=false

Limitations: does not support recurrent actors.
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

from agents.mlp_actor_critic_agent import ActorWithConditionalCriticPolicy
from agents.initialize_agents import initialize_actor_with_conditional_critic
from agents.population_interface import AgentPopulation
from agents.population_buffer import BufferedPopulation
from common.save_load_utils import save_train_run
from common.plot_utils import get_metric_names
from common.run_episodes import run_episodes
from envs import make_env
from envs.log_wrapper import LogWrapper, LogEnvState
from marl.ippo import make_train as make_ppo_train
from marl.ppo_utils import Transition, unbatchify, _create_minibatches

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ResetTransition(NamedTuple):
    '''Stores extra information for resetting agents to a point in some trajectory.'''
    env_state: LogEnvState
    conf_obs: jnp.ndarray
    partner_obs: jnp.ndarray
    conf_done: jnp.ndarray
    partner_done: jnp.ndarray
    conf_hstate: jnp.ndarray
    partner_hstate: jnp.ndarray

def train_comedi_partners(train_rng, env, config):
    num_agents = env.num_agents
    assert num_agents == 2, "This code assumes the environment has exactly 2 agents."

    # Define 4 types of rollouts: SP, XP, MP, MP2
    config["NUM_GAME_AGENTS"] = num_agents

    config["NUM_ACTORS"] = num_agents * config["NUM_ENVS"]
    # Right now assume control of both agent and its BR
    config["NUM_CONTROLLED_ACTORS"] = config["NUM_ACTORS"]

    # Divide by 4 because we have 4 types of rollouts
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS_PER_ITERATION"] // ( 4 * num_agents * config["ROLLOUT_LENGTH"] * config["NUM_ENVS"])

    def make_comedi_agents(config):
        def linear_schedule(count):
            frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
            return config["LR"] * frac

        def train_init_ippo_partners(config, partner_rng, env):
            '''
            Train a pool IPPO agents w/parameter sharing.
            Returns out, a dictionary of the model checkpoints, final parameters, and metrics.
            '''
            config["TOTAL_TIMESTEPS"] = config["TOTAL_TIMESTEPS_PER_ITERATION"]
            config["ACTOR_TYPE"] = "pseudo_actor_with_conditional_critic"
            config["POP_SIZE"] = config["PARTNER_POP_SIZE"]
            out = make_ppo_train(config, env)(partner_rng) # train a single PPO agent
            return out

        def train(rng):
            # Start by training a single PPO agent via self-play
            rng, init_ppo_rng, init_conf_rng = jax.random.split(rng, 3)

            init_ppo_partner = train_init_ippo_partners(config, init_ppo_rng, env)

            # Initialize a population buffer
            dummy_policy, dummy_init_params = initialize_actor_with_conditional_critic(config, env, init_conf_rng)
            partner_population = BufferedPopulation(
                max_pop_size=config["PARTNER_POP_SIZE"],
                policy_cls=dummy_policy,
            )

            population_buffer = partner_population.reset_buffer(dummy_init_params)
            population_buffer = partner_population.add_agent(population_buffer, init_ppo_partner["final_params"])

            def add_conf_policy(pop_buffer, func_input):
                num_existing_agents, rng = func_input
                rng, init_conf_rng = jax.random.split(rng)

                # Create new confederate agent policy and critic
                policy, init_params = initialize_actor_with_conditional_critic(
                    config, env, init_conf_rng
                )

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

                # Reset envs for SP, XP, and MP
                rng, reset_rng_eval, reset_rng_sp, reset_rng_xp, reset_rng_mp, reset_rng_mp2 = jax.random.split(rng, 6)

                reset_rngs_sps = jax.random.split(reset_rng_sp, config["NUM_ENVS"])
                reset_rngs_xps = jax.random.split(reset_rng_xp, config["NUM_ENVS"])
                reset_rngs_mps = jax.random.split(reset_rng_mp, config["NUM_ENVS"])
                reset_rngs_mps2 = jax.random.split(reset_rng_mp2, config["NUM_ENVS"])

                obsv_xp, env_state_xp = jax.vmap(env.reset, in_axes=(0,))(reset_rngs_sps)
                obsv_sp, env_state_sp = jax.vmap(env.reset, in_axes=(0,))(reset_rngs_xps)
                obsv_mp, env_state_mp = jax.vmap(env.reset, in_axes=(0,))(reset_rngs_mps)
                obsv_mp2, env_state_mp2 = jax.vmap(env.reset, in_axes=(0,))(reset_rngs_mps2)

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
                        num_eps=config["NUM_ARGMAX_ROLLOUT_EPS"]
                    )
                    return all_outs

                def _update_step(update_with_ckpt_runner_state, unused):
                    update_runner_state, checkpoint_array, ckpt_idx = update_with_ckpt_runner_state
                    (
                        train_state, pop_buffer,
                        env_state_sp, obsv_sp,
                        env_state_xp, obsv_xp,
                        env_state_mp, obsv_mp,
                        env_state_mp2, obsv_mp2,
                        last_dones_xp,
                        last_dones_sp,
                        last_dones_mp,
                        last_dones_mp2,
                        rng, update_steps,
                        num_prev_trained_conf
                    ) = update_runner_state

                    # Identify the expected returns from the newly trained policy
                    # when interacting with the previously generated confederate
                    # policies
                    valid_sampling_indices = jnp.arange(config["POP_SIZE"])
                    run_all_rollouts = jax.vmap(per_id_run_episode_fixed_rng, in_axes=(None, 0))(
                        train_state.params,valid_sampling_indices)

                    # Mask out the XP returns against invalid policies
                    # resulting from IDs that are yet set to a specific
                    # confederate params
                    all_mean_returns = run_all_rollouts["returned_episode_returns"][:, :, 0].mean(axis=-1)
                    masked_mean_returns = jnp.where(
                        valid_sampling_indices >= num_prev_trained_conf, -jnp.inf, all_mean_returns
                    )

                    # Pick the right confederate params to act as the XP agent
                    max_means_id = masked_mean_returns.argmax()
                    xp_param = jax.tree_map(
                        lambda x: jnp.squeeze(x, 0),
                        partner_population.gather_agent_params(pop_buffer,
                                                               agent_indices=max_means_id * jnp.ones((1,), dtype=np.int32))
                    )

                    rng, rng_xp, rng_sp, rng_mp, rng_mp2 = jax.random.split(rng, 5)

                    def _env_step_conf_ego(runner_state, unused):
                        """
                        agent_0 = confederate, agent_1 = ego
                        Returns updated runner_state and a Transition for the confederate.
                        """
                        train_state, xp_param, xp_id, env_state, last_obs, last_dones, rng = runner_state
                        rng, act_rng, partner_rng, step_rng = jax.random.split(rng, 4)

                        obs_0 = last_obs["agent_0"]
                        obs_1 = last_obs["agent_1"]

                        # Get available actions for agent 0 from environment state
                        avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                        avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
                        avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

                        # Add one-hot ID of XP teammate
                        xp_one_hot_id = jnp.eye(config["POP_SIZE"])[xp_id]
                        xp_one_hot_id = jnp.expand_dims(
                            jnp.expand_dims(
                                xp_one_hot_id, 0
                            ), 0
                        )

                        # Agent_0 (confederate) action using policy interface
                        aux_obs = jnp.repeat(xp_one_hot_id, config["NUM_ENVS"], axis=1)
                        act_0, val_0, pi_0, _ = policy.get_action_value_policy(
                            params=train_state.params,
                            obs=obs_0.reshape(1, config["NUM_ENVS"], -1),
                            done=last_dones["agent_0"].reshape(1, config["NUM_ENVS"]),
                            avail_actions=jax.lax.stop_gradient(avail_actions_0),
                            hstate=None,
                            rng=act_rng,
                            aux_obs=aux_obs
                        )
                        logp_0 = pi_0.log_prob(act_0)

                        act_0 = act_0.squeeze()
                        logp_0 = logp_0.squeeze()
                        val_0 = val_0.squeeze()

                        # Agent_1 (ego) action using policy interface
                        act_1, _, _, _ = policy.get_action_value_policy(
                            params=xp_param,
                            obs=obs_1.reshape(1, config["NUM_ENVS"], -1),
                            done=last_dones["agent_1"].reshape(1, config["NUM_ENVS"]),
                            avail_actions=jax.lax.stop_gradient(avail_actions_1),
                            hstate=None,
                            rng=partner_rng,
                            aux_obs=aux_obs
                        )
                        act_1 = act_1.squeeze()

                        # Combine actions into the env format
                        combined_actions = jnp.concatenate([act_0, act_1], axis=0)  # shape (2*num_envs,)
                        env_act = unbatchify(combined_actions, env.agents, config["NUM_ENVS"], num_agents)
                        env_act = {k: v.flatten() for k, v in env_act.items()}

                        # Step env
                        step_rngs = jax.random.split(step_rng, config["NUM_ENVS"])
                        obs_next, env_state_next, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0))(
                            step_rngs, env_state, env_act
                        )
                        # note that num_actors = num_envs * num_agents
                        info_0 = jax.tree.map(lambda x: x[:, 0], info)

                        # Store agent_0 data in transition
                        transition = Transition(
                            done=done["agent_0"],
                            action=act_0,
                            value=val_0,
                            reward=reward["agent_1"],
                            log_prob=logp_0,
                            obs=obs_0,
                            info=info_0,
                            avail_actions=avail_actions_0
                        )
                        new_runner_state = (train_state, xp_param, xp_id, env_state_next, obs_next, done, rng)
                        return new_runner_state, transition

                    def _env_step_conf_br(runner_state, unused):
                        """
                        agent_0 = confederate, agent_1 = best response
                        Returns updated runner_state, and Transitions for the confederate and best response.
                        """
                        train_state, env_state, last_obs, last_dones, rng, current_trained_pop_id, reset_traj_batch = runner_state
                        rng, conf_rng, br_rng, step_rng = jax.random.split(rng, 4)

                        def gather_sampled(data_pytree, flat_indices, first_nonbatch_dim: int):
                            '''Will treat all dimensions up to the first_nonbatch_dim as batch dimensions. '''
                            batch_size = config["ROLLOUT_LENGTH"] * config["NUM_ENVS"]
                            flat_data = jax.tree.map(lambda x: x.reshape(batch_size, *x.shape[first_nonbatch_dim:]), data_pytree)
                            sampled_data = jax.tree.map(lambda x: x[flat_indices], flat_data) # Shape (N, ...)
                            return sampled_data

                        if reset_traj_batch is not None:
                            rng, sample_rng = jax.random.split(rng)
                            needs_resample = last_dones["__all__"] # shape (N,) bool

                            total_reset_states = config["ROLLOUT_LENGTH"] * config["NUM_ENVS"]
                            sampled_indices = jax.random.randint(sample_rng, shape=(config["NUM_ENVS"],), minval=0,
                                                                maxval=total_reset_states)

                            # Gather sampled leaves from each data pytree
                            sampled_env_state = gather_sampled(reset_traj_batch.env_state, sampled_indices, first_nonbatch_dim=2)
                            sampled_conf_obs = gather_sampled(reset_traj_batch.conf_obs, sampled_indices, first_nonbatch_dim=2)
                            sampled_br_obs = gather_sampled(reset_traj_batch.partner_obs, sampled_indices, first_nonbatch_dim=2)
                            sampled_conf_done = gather_sampled(reset_traj_batch.conf_done, sampled_indices, first_nonbatch_dim=2)
                            sampled_br_done = gather_sampled(reset_traj_batch.partner_done, sampled_indices, first_nonbatch_dim=2)

                            # for done environments, select data corresponding to the reset_traj_batch states
                            env_state = jax.tree.map(
                                lambda sampled, original: jnp.where(
                                    needs_resample.reshape((-1,) + (1,) * (original.ndim - 1)),
                                    sampled, original
                                ),
                                sampled_env_state,
                                env_state
                            )
                            obs_0 = jnp.where(needs_resample[:, jnp.newaxis], sampled_conf_obs, last_obs["agent_0"])
                            obs_1 = jnp.where(needs_resample[:, jnp.newaxis], sampled_br_obs, last_obs["agent_1"])

                            dones_0 = jnp.where(needs_resample, sampled_conf_done, last_dones["agent_0"])
                            dones_1 = jnp.where(needs_resample, sampled_br_done, last_dones["agent_1"])

                        else:

                            # Reset conf-br data collection from conf-ego states
                            obs_0, obs_1 = last_obs["agent_0"], last_obs["agent_1"]
                            dones_0, dones_1 = last_dones["agent_0"], last_dones["agent_1"]

                        # Get available actions for agent 0 from environment state
                        avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                        avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
                        avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

                        # Agent_0 (confederate) action
                        # Add one-hot ID of XP teammate
                        sp_one_hot_id = jnp.eye(config["POP_SIZE"])[current_trained_pop_id]
                        sp_one_hot_id = jnp.expand_dims(
                            jnp.expand_dims(
                                sp_one_hot_id, 0
                            ), 0
                        )

                        aux_obs = jnp.repeat(sp_one_hot_id, config["NUM_ENVS"], 1)
                        act_0, val_0, pi_0, _ = policy.get_action_value_policy(
                            params=train_state.params,
                            obs=obs_0.reshape(1, config["NUM_ENVS"], -1),
                            done=dones_0.reshape(1, config["NUM_ENVS"]),
                            avail_actions=jax.lax.stop_gradient(avail_actions_0),
                            hstate=None,
                            rng=conf_rng,
                            aux_obs=aux_obs
                        )
                        logp_0 = pi_0.log_prob(act_0)

                        act_0 = act_0.squeeze()
                        logp_0 = logp_0.squeeze()
                        val_0 = val_0.squeeze()

                        # Agent 1 (best response) action
                        act_1, val_1, pi_1, _ = policy.get_action_value_policy(
                            params=train_state.params,
                            obs=obs_1.reshape(1, config["NUM_ENVS"], -1),
                            done=dones_1.reshape(1, config["NUM_ENVS"]),
                            avail_actions=jax.lax.stop_gradient(avail_actions_1),
                            hstate=None,
                            rng=br_rng,
                            aux_obs=aux_obs
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
                        obs_next, env_state_next, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0))(
                            step_rngs, env_state, env_act
                        )
                        info_0 = jax.tree.map(lambda x: x[:, 0], info)
                        info_1 = jax.tree.map(lambda x: x[:, 1], info)

                        # Store agent_0 (confederate) data in transition
                        transition_0 = Transition(
                            done=done["agent_0"],
                            action=act_0,
                            value=val_0,
                            reward=reward["agent_0"],
                            log_prob=logp_0,
                            obs=obs_0,
                            info=info_0,
                            avail_actions=avail_actions_0
                        )
                        # Store agent_1 (best response) data in transition
                        transition_1 = Transition(
                            done=done["agent_1"],
                            action=act_1,
                            value=val_1,
                            reward=reward["agent_1"],
                            log_prob=logp_1,
                            obs=obs_1,
                            info=info_1,
                            avail_actions=avail_actions_1
                        )
                        # Pass reset_traj_batch and init_br_hstate through unchanged in the state tuple
                        new_runner_state = (train_state, env_state_next, obs_next, done, rng, current_trained_pop_id, reset_traj_batch)
                        return new_runner_state, (transition_0, transition_1)

                    def _env_step_mixed(runner_state, unused):
                        """
                        agent_0 = confederate, agent_1 = ego OR best response
                        Returns a ResetTransition for resetting to env states encountered here.
                        """
                        train_state_conf, ego_param, env_state, last_obs, last_dones, rng, current_trained_pop_id = runner_state
                        rng, act_rng, ego_act_rng, br_act_rng, partner_choice_rng, step_rng = jax.random.split(rng, 6)

                        obs_0 = last_obs["agent_0"]
                        obs_1 = last_obs["agent_1"]

                        # Get available actions for agent 0 from environment state
                        avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                        avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
                        avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

                        xp_one_hot_id = jnp.eye(config["POP_SIZE"])[current_trained_pop_id]
                        xp_one_hot_id = jnp.expand_dims(
                            jnp.expand_dims(
                                xp_one_hot_id, 0
                            ), 0
                        )

                        # Agent_0 (confederate) action using policy interface
                        aux_obs = jnp.repeat(xp_one_hot_id, config["NUM_ENVS"], axis=1)

                        # Agent_0 (confederate) action using policy interface
                        act_0, val_0, pi_0, _ = policy.get_action_value_policy(
                            params=train_state_conf.params,
                            obs=obs_0.reshape(1, config["NUM_ENVS"], -1),
                            done=last_dones["agent_0"].reshape(1, config["NUM_ENVS"]),
                            avail_actions=jax.lax.stop_gradient(avail_actions_0),
                            hstate=None,
                            rng=act_rng,
                            aux_obs=aux_obs
                        )
                        logp_0 = pi_0.log_prob(act_0)

                        act_0 = act_0.squeeze()
                        logp_0 = logp_0.squeeze()
                        val_0 = val_0.squeeze()

                        ### Compute both the ego action and the best response action
                        act_ego, _, _, _ = policy.get_action_value_policy(
                            params=ego_param,
                            obs=obs_1.reshape(1, config["NUM_ENVS"], -1),
                            done=last_dones["agent_1"].reshape(1, config["NUM_ENVS"]),
                            avail_actions=jax.lax.stop_gradient(avail_actions_1),
                            hstate=None,
                            rng=ego_act_rng,
                            aux_obs=aux_obs
                        )
                        act_br, _, _, _ = policy.get_action_value_policy(
                            params=train_state.params,
                            obs=obs_1.reshape(1, config["NUM_ENVS"], -1),
                            done=last_dones["agent_1"].reshape(1, config["NUM_ENVS"]),
                            avail_actions=jax.lax.stop_gradient(avail_actions_1),
                            hstate=None,
                            rng=br_act_rng,
                            aux_obs=aux_obs
                        )

                        act_ego = act_ego.squeeze()
                        act_br = act_br.squeeze()
                        # Agent 1 (ego or best response) action - choose between ego and best response
                        partner_choice = jax.random.randint(partner_choice_rng, shape=(config["NUM_ENVS"],), minval=0, maxval=2)
                        act_1 = jnp.where(partner_choice == 0, act_ego, act_br)

                        # Combine actions into the env format
                        combined_actions = jnp.concatenate([act_0, act_1], axis=0)
                        env_act = unbatchify(combined_actions, env.agents, config["NUM_ENVS"], num_agents)
                        env_act = {k: v.flatten() for k, v in env_act.items()}

                        # Step env
                        step_rngs = jax.random.split(step_rng, config["NUM_ENVS"])
                        obs_next, env_state_next, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0))(
                            step_rngs, env_state, env_act
                        )

                        reset_transition = ResetTransition(
                            # all of these are from before env step
                            env_state=env_state,
                            conf_obs=obs_0,
                            partner_obs=obs_1,
                            conf_done=last_dones["agent_0"],
                            partner_done=last_dones["agent_1"],
                            conf_hstate=None,
                            # we record the best response hstate because we use it to reset the best response
                            partner_hstate=None
                        )
                        new_runner_state = (train_state_conf, ego_param, env_state_next, obs_next, done, rng, current_trained_pop_id)
                        return new_runner_state, reset_transition

                    # Do XP rollout (based on train_state params and the param in pop_buffer identified in Step 1)
                    runner_state_xp = (train_state, xp_param, max_means_id, env_state_xp, obsv_xp, last_dones_xp, rng_xp)
                    runner_state_xp, traj_batch_xp = jax.lax.scan(
                        _env_step_conf_ego, runner_state_xp, None, config["ROLLOUT_LENGTH"])
                    (train_state, xp_param, max_means_id, env_state_xp, last_obs_xp, last_dones_xp, rng_xp) = runner_state_xp

                    # Do self-play (based on train_state params) rollout like in the IPPO code
                    runner_state_sp = (train_state, env_state_sp, obsv_sp, last_dones_sp, rng_sp, num_prev_trained_conf, None)
                    runner_state_sp, (traj_batch_sp_agent0, traj_batch_sp_agent1) = jax.lax.scan(
                        _env_step_conf_br, runner_state_sp, None, config["ROLLOUT_LENGTH"])
                    (train_state, env_state_sp, last_obs_sp, last_dones_sp, rng_sp, num_prev_trained_conf, mp_traj_batch) = runner_state_sp

                    # Step 4
                    # Do MP rollout (based on train_state params and the param in pop_buffer identified in Step 1)
                    runner_state_mp = (train_state, xp_param, env_state_mp, obsv_mp, last_dones_mp, rng_mp, num_prev_trained_conf)
                    runner_state_mp, traj_batch_mp = jax.lax.scan(
                        _env_step_mixed, runner_state_mp, None, config["ROLLOUT_LENGTH"])
                    (train_state, xp_param, env_state_mp, last_obs_mp, last_dones_mp, rng_mp, num_prev_trained_conf) = runner_state_mp

                    runner_state_smp = (train_state, env_state_mp2, obsv_mp2, last_dones_mp2, rng_mp2, num_prev_trained_conf, traj_batch_mp)
                    runner_state_smp, (traj_batch_smp0, traj_batch_smp1) = jax.lax.scan(
                        _env_step_conf_br, runner_state_smp, None, config["ROLLOUT_LENGTH"])
                    (train_state, env_state_mp2, last_obs_mp2, last_dones_mp2, rng_mp2, num_prev_trained_conf, mp2_traj_batch) = runner_state_smp

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
                                                    last_obs, last_dones, traj_batch, agent_name, value_idx=None):
                        '''Value_idx argument is to support the ActorWithDoubleCritic (confederate) policy, which
                        has two value heads. Value head 0 models the ego agent while value head 1 models the best response.'''
                        avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)[agent_name].astype(jnp.float32)

                        # Add one-hot ID of interaction teammate
                        xp_one_hot_id = jnp.eye(config["POP_SIZE"])[value_idx]
                        xp_one_hot_id = jnp.expand_dims(
                            jnp.expand_dims(
                                xp_one_hot_id, 0
                            ), 0
                        )

                        # Agent_0 (confederate) action using policy interface
                        aux_obs = jnp.repeat(xp_one_hot_id, last_obs[agent_name].shape[0], axis=1)

                        _, vals, _, _ = policy.get_action_value_policy(
                            params=policy_params,
                            obs=last_obs[agent_name].reshape(1, last_obs[agent_name].shape[0], -1),
                            done=last_dones[agent_name].reshape(1, last_obs[agent_name].shape[0]),
                            avail_actions=jax.lax.stop_gradient(avail_actions),
                            hstate=policy_hstate,
                            rng=jax.random.PRNGKey(0),  # dummy key as we don't sample actions
                            aux_obs=aux_obs
                        )
                        last_val = vals.squeeze()
                        advantages, targets = _calculate_gae(traj_batch, last_val)
                        return advantages, targets

                    # 5a) Compute conf advantages for XP (conf-ego) interaction
                    advantages_xp_conf, targets_xp_conf = _compute_advantages_and_targets(
                        env_state_xp, policy, train_state.params, None,
                        last_obs_xp, last_dones_xp, traj_batch_xp, "agent_0", value_idx=max_means_id)

                    # 5b) Compute conf and br advantages for SP (conf-br) interaction
                    advantages_sp_conf, targets_sp_conf = _compute_advantages_and_targets(
                        env_state_sp, policy, train_state.params, None,
                        last_obs_sp, last_dones_sp, traj_batch_sp_agent0, "agent_0", value_idx=num_prev_trained_conf)

                    advantages_sp_br, targets_sp_br = _compute_advantages_and_targets(
                        env_state_sp, policy, train_state.params, None,
                        last_obs_sp, last_dones_sp, traj_batch_sp_agent1, "agent_1", value_idx=num_prev_trained_conf)

                    # 5c) Compute advantages from MP interactions
                    advantages_mp_conf, targets_mp_conf = _compute_advantages_and_targets(
                        env_state_mp2, policy, train_state.params, None,
                        last_obs_mp2, last_dones_mp2, traj_batch_smp0, "agent_0", value_idx=num_prev_trained_conf)

                    advantages_mp_br, targets_mp_br = _compute_advantages_and_targets(
                        env_state_mp2, policy, train_state.params, None,
                        last_obs_mp2, last_dones_mp2, traj_batch_smp1, "agent_1", value_idx=num_prev_trained_conf)

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
                            minbatch_xp, minbatch_sp1, minbatch_sp2, minbatch_mp1,  minbatch_mp2, xp_id, sp_id = batch_infos
                            _, traj_batch_xp, advantages_xp, returns_xp = minbatch_xp
                            _, traj_batch_sp1, advantages_sp1, returns_sp1 = minbatch_sp1
                            _, traj_batch_sp2, advantages_sp2, returns_sp2 = minbatch_sp2
                            _, traj_batch_mp1, advantages_mp1, returns_mp1 = minbatch_mp1
                            _, traj_batch_mp2, advantages_mp2, returns_mp2 = minbatch_mp2

                            def _loss_fn_conf(params, traj_batch_xp, gae_xp, target_v_xp,
                                            traj_batch_sp, gae_sp, target_v_sp,
                                            traj_batch_sp2, gae_sp2, target_v_sp2,
                                            traj_batch_mp, gae_mp, target_v_mp,
                                            traj_batch_mp2, gae_mp2, target_v_mp2):
                                # get policy and value of confederate versus ego and best response agents respectively
                                xp_one_hot_id = jnp.eye(config["POP_SIZE"])[xp_id]
                                xp_one_hot_id = jnp.expand_dims(
                                    jnp.expand_dims(
                                        xp_one_hot_id, 0
                                    ), 0
                                )

                                sp_one_hot_id = jnp.eye(config["POP_SIZE"])[sp_id]
                                sp_one_hot_id = jnp.expand_dims(
                                    jnp.expand_dims(
                                        sp_one_hot_id, 0
                                    ), 0
                                )

                                # Agent_0 (confederate) action using policy interface
                                aux_obs_xp = jnp.repeat(xp_one_hot_id, traj_batch_xp.obs.shape[1], axis=1)
                                aux_obs_xp = jnp.repeat(aux_obs_xp, traj_batch_xp.obs.shape[0], axis=0)

                                _, value_xp, pi_xp, _ = policy.get_action_value_policy(
                                    params=params,
                                    obs=traj_batch_xp.obs,
                                    done=traj_batch_xp.done,
                                    avail_actions=traj_batch_xp.avail_actions,
                                    hstate=None,
                                    rng=jax.random.PRNGKey(0), # only used for action sampling, which is not used here
                                    aux_obs=aux_obs_xp
                                )

                                aux_obs_sp = jnp.repeat(xp_one_hot_id, traj_batch_sp.obs.shape[1], axis=1)
                                aux_obs_sp = jnp.repeat(aux_obs_sp, traj_batch_sp.obs.shape[0], axis=0)
                                _, value_sp, pi_sp, _ = policy.get_action_value_policy(
                                    params=params,
                                    obs=traj_batch_sp.obs,
                                    done=traj_batch_sp.done,
                                    avail_actions=traj_batch_sp.avail_actions,
                                    hstate=None,
                                    rng=jax.random.PRNGKey(0), # only used for action sampling, which is not used here
                                    aux_obs=aux_obs_sp
                                )

                                _, value_sp2, pi_sp2, _ = policy.get_action_value_policy(
                                    params=params,
                                    obs=traj_batch_sp2.obs,
                                    done=traj_batch_sp2.done,
                                    avail_actions=traj_batch_sp2.avail_actions,
                                    hstate=None,
                                    rng=jax.random.PRNGKey(0), # only used for action sampling, which is not used here
                                    aux_obs=aux_obs_sp
                                )

                                _, value_mp, pi_mp, _ = policy.get_action_value_policy(
                                    params=params,
                                    obs=traj_batch_mp.obs,
                                    done=traj_batch_mp.done,
                                    avail_actions=traj_batch_mp.avail_actions,
                                    hstate=None,
                                    rng=jax.random.PRNGKey(0), # only used for action sampling, which is not used here
                                    aux_obs=aux_obs_sp
                                )

                                _, value_mp2, pi_mp2, _ = policy.get_action_value_policy(
                                    params=params,
                                    obs=traj_batch_mp2.obs,
                                    done=traj_batch_mp2.done,
                                    avail_actions=traj_batch_mp2.avail_actions,
                                    hstate=None,
                                    rng=jax.random.PRNGKey(0), # only used for action sampling, which is not used here
                                    aux_obs=aux_obs_sp
                                )

                                log_prob_xp = pi_xp.log_prob(traj_batch_xp.action)
                                log_prob_sp = pi_sp.log_prob(traj_batch_sp.action)
                                log_prob_sp2 = pi_sp2.log_prob(traj_batch_sp2.action)
                                log_prob_mp = pi_mp.log_prob(traj_batch_mp.action)
                                log_prob_mp2 = pi_mp2.log_prob(traj_batch_mp2.action)


                                value_loss_xp = _compute_ppo_value_loss(value_xp, traj_batch_xp, target_v_xp)
                                value_loss_sp = _compute_ppo_value_loss(value_sp, traj_batch_sp, target_v_sp)
                                value_loss_sp2 = _compute_ppo_value_loss(value_sp2, traj_batch_sp2, target_v_sp2)
                                value_loss_mp = _compute_ppo_value_loss(value_mp, traj_batch_mp, target_v_mp)
                                value_loss_mp2 = _compute_ppo_value_loss(value_mp2, traj_batch_mp2, target_v_mp2)

                                pg_loss_xp = _compute_ppo_pg_loss(log_prob_xp, traj_batch_xp, gae_xp)
                                pg_loss_sp = _compute_ppo_pg_loss(log_prob_sp, traj_batch_sp, gae_sp)
                                pg_loss_sp2 = _compute_ppo_pg_loss(log_prob_sp2, traj_batch_sp2, gae_sp2)
                                pg_loss_mp = _compute_ppo_pg_loss(log_prob_mp, traj_batch_mp, gae_mp)
                                pg_loss_mp2 = _compute_ppo_pg_loss(log_prob_mp2, traj_batch_mp2, gae_mp2)


                                # Entropy for interaction with ego agent
                                entropy_xp = jnp.mean(pi_xp.entropy())
                                entropy_sp = jnp.mean(pi_sp.entropy())
                                entropy_sp2 = jnp.mean(pi_sp2.entropy())
                                entropy_mp = jnp.mean(pi_mp.entropy())
                                entropy_mp2 = jnp.mean(pi_mp2.entropy())

                                xp_pg_weight = -config["COMEDI_ALPHA"] # negate to minimize the ego agent's PG objective
                                sp_pg_weight = 1.0
                                mp2_pg_weight = config["COMEDI_BETA"]

                                xp_loss = xp_pg_weight * pg_loss_xp + config["VF_COEF"] * value_loss_xp - config["ENT_COEF"] * entropy_xp
                                sp_loss = sp_pg_weight * pg_loss_sp + config["VF_COEF"] * value_loss_sp - config["ENT_COEF"] * entropy_sp
                                sp2_loss = sp_pg_weight * pg_loss_sp2 + config["VF_COEF"] * value_loss_sp2 - config["ENT_COEF"] * entropy_sp2
                                mp_loss = mp2_pg_weight * pg_loss_mp + config["VF_COEF"] * value_loss_mp - config["ENT_COEF"] * entropy_mp
                                mp2_loss = mp2_pg_weight * pg_loss_mp2 + config["VF_COEF"] * value_loss_mp2 - config["ENT_COEF"] * entropy_mp2

                                total_loss = sp_loss + sp2_loss + xp_loss + mp2_loss + mp_loss
                                return total_loss, (value_loss_xp, value_loss_sp + value_loss_sp2, value_loss_mp + value_loss_mp2,
                                                    pg_loss_xp, pg_loss_sp + pg_loss_sp2, pg_loss_mp + pg_loss_mp2,
                                                    entropy_xp, entropy_sp + entropy_sp2, entropy_mp + entropy_mp2)

                            grad_fn = jax.value_and_grad(_loss_fn_conf, has_aux=True)
                            (loss_val, aux_vals), grads = grad_fn(
                                train_state_conf.params,
                                traj_batch_xp, advantages_xp, returns_xp,
                                traj_batch_sp1, advantages_sp1, returns_sp1,
                                traj_batch_sp2, advantages_sp2, returns_sp2,
                                traj_batch_mp1, advantages_mp1, returns_mp1,
                                traj_batch_mp2, advantages_mp2, returns_mp2)
                            train_state_conf = train_state_conf.apply_gradients(grads=grads)
                            return train_state_conf, (loss_val, aux_vals)

                        (
                            train_state_conf, traj_batch_xp,
                            traj_batch_sp_conf, traj_batch_sp_br,
                            traj_batch_mp_conf, traj_batch_mp_br,
                            advantages_xp_conf, advantages_sp_conf,
                            advantages_sp_br, advantages_mp_conf,
                            advantages_mp_br, targets_xp_conf,
                            targets_sp_conf, targets_sp_br,
                            targets_mp_conf, targets_mp_br,
                            rng, xp_id, sp_id
                        ) = update_state

                        rng, perm_rng_xp, perm_rng_sp_conf, perm_rng_sp_br, perm_rng_mp2_conf, perm_rng_mp2_br = jax.random.split(rng, 6)

                        # Create minibatches for each agent and interaction type
                        minibatches_xp = _create_minibatches(
                            traj_batch_xp, advantages_xp_conf, targets_xp_conf, None,
                            config["NUM_ENVS"], config["NUM_MINIBATCHES"], perm_rng_xp
                        )
                        minibatches_sp_conf = _create_minibatches(
                            traj_batch_sp_conf, advantages_sp_conf, targets_sp_conf, None,
                            config["NUM_ENVS"], config["NUM_MINIBATCHES"], perm_rng_sp_conf
                        )
                        minibatches_sp_br = _create_minibatches(
                            traj_batch_sp_br, advantages_sp_br, targets_sp_br, None,
                            config["NUM_ENVS"], config["NUM_MINIBATCHES"], perm_rng_sp_br
                        )
                        minibatches_mp_conf = _create_minibatches(
                            traj_batch_mp_conf, advantages_mp_conf, targets_mp_conf, None,
                            config["NUM_ENVS"], config["NUM_MINIBATCHES"], perm_rng_mp2_conf
                        )
                        minibatches_mp_br = _create_minibatches(
                            traj_batch_mp_br, advantages_mp_br, targets_mp_br, None,
                            config["NUM_ENVS"], config["NUM_MINIBATCHES"], perm_rng_mp2_br
                        )

                        # Update confederate
                        repeated_xp_id = jnp.repeat(xp_id, minibatches_xp[1].obs.shape[0], axis=0)
                        repeated_sp_id = jnp.repeat(sp_id, minibatches_sp_br[1].obs.shape[0], axis=0)
                        train_state_conf, total_loss_conf = jax.lax.scan(
                            _update_minbatch_conf, train_state_conf, (
                                minibatches_xp, minibatches_sp_conf, minibatches_sp_br,
                                minibatches_mp_conf, minibatches_mp_br, repeated_xp_id, repeated_sp_id
                            )
                        )

                        update_state = (train_state_conf,
                            traj_batch_xp, traj_batch_sp_conf, traj_batch_sp_br, traj_batch_mp_conf, traj_batch_mp_br,
                            advantages_xp_conf, advantages_sp_conf, advantages_sp_br, advantages_mp_conf, advantages_mp_br,
                            targets_xp_conf, targets_sp_conf, targets_sp_br, targets_mp_conf, targets_mp_br,
                            rng, xp_id, sp_id
                        )
                        return update_state, total_loss_conf

                    # 3) PPO update
                    rng, sub_rng = jax.random.split(rng, 2)
                    update_state = (
                        train_state,
                        traj_batch_xp, traj_batch_sp_agent0,
                        traj_batch_sp_agent1,
                        traj_batch_smp0, traj_batch_smp1,
                        advantages_xp_conf,
                        advantages_sp_conf, advantages_sp_br,
                        advantages_mp_conf, advantages_mp_br,
                        targets_xp_conf, targets_sp_conf,
                        targets_sp_br, targets_mp_conf,
                        targets_mp_br, sub_rng,
                        max_means_id, num_prev_trained_conf
                    )
                    update_state, conf_losses = jax.lax.scan(
                        _update_epoch, update_state, None, config["UPDATE_EPOCHS"])
                    train_state = update_state[0]

                    (
                        conf_value_loss_xp, conf_value_loss_sp, conf_value_loss_mp,
                        conf_pg_loss_xp, conf_pg_loss_sp, conf_pg_loss_mp,
                        conf_entropy_xp, conf_entropy_sp, conf_entropy_mp
                    ) = conf_losses[1]

                    new_update_runner_state = (
                        train_state, pop_buffer,
                        env_state_sp, last_obs_sp,
                        env_state_xp, last_obs_xp,
                        env_state_mp, last_obs_mp,
                        env_state_mp2, last_obs_mp2,
                        last_dones_xp, last_dones_sp,
                        last_dones_mp, last_dones_mp2,
                        rng, update_steps+1, num_prev_trained_conf
                    )

                    # Metrics
                    metric = traj_batch_xp.info
                    metric["update_steps"] = update_steps
                    metric["value_loss_conf_xp"] = conf_value_loss_xp
                    metric["value_loss_conf_sp"] = conf_value_loss_sp
                    metric["value_loss_conf_mp"] = conf_value_loss_mp

                    metric["pg_loss_conf_xp"] = conf_pg_loss_xp
                    metric["pg_loss_conf_sp"] = conf_pg_loss_sp
                    metric["pg_loss_conf_mp"] = conf_pg_loss_mp

                    metric["entropy_conf_xp"] = conf_entropy_xp
                    metric["entropy_conf_sp"] = conf_entropy_sp
                    metric["entropy_conf_mp"] = conf_entropy_mp

                    metric["average_rewards_ego"] = jnp.mean(traj_batch_xp.reward)
                    metric["average_rewards_br_sp"] = jnp.mean(traj_batch_sp_agent1.reward)
                    metric["average_rewards_br_mp2"] = jnp.mean(traj_batch_smp1.reward)

                    return (new_update_runner_state, checkpoint_array, ckpt_idx+1), metric

                # XP eval against all policies in the buffer
                xp_eval_returns = jax.vmap(per_id_run_episode_fixed_rng, in_axes=(None, 0))(
                        train_state.params,jnp.arange(config["POP_SIZE"]))

                # SP performance against itself
                sp_eval_returns = run_episodes(
                    eval_rng, env,
                    agent_0_param=train_state.params, agent_0_policy=policy,
                    agent_1_param=train_state.params, agent_1_policy=policy,
                    max_episode_steps=config["ROLLOUT_LENGTH"],
                    num_eps=config["NUM_EVAL_EPISODES"]
                )


                update_steps = 0
                init_done_xp = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}
                init_done_sp = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}
                init_done_mp = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}
                init_done_mp2 = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}

                update_runner_state = (
                    train_state, pop_buffer,
                    env_state_sp, obsv_sp,
                    env_state_xp, obsv_xp,
                    env_state_mp, obsv_mp,
                    env_state_mp2, obsv_mp2,
                    init_done_xp, init_done_sp,
                    init_done_mp, init_done_mp2,
                    rng, update_steps,
                    num_existing_agents
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
                    rng, update_steps = new_update_runner_state[-3], new_update_runner_state[-2]

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
                new_update_runner_state, new_checkpoint_array, _, _ ,_ = new_update_with_ckpt_runner_state
                final_train_state = new_update_runner_state[0]

                updated_pop_buffer = partner_population.add_agent(pop_buffer, final_train_state.params)
                conf_checkpoints = new_checkpoint_array
                return updated_pop_buffer, (conf_checkpoints, metric, xp_eval_returns, sp_eval_returns)

            rngs = jax.random.split(rng, config["PARTNER_POP_SIZE"])
            rng, add_conf_iter_rngs = rngs[0], rngs[1:]

            iter_ids = jnp.arange(1, config["PARTNER_POP_SIZE"])
            final_population_buffer, (conf_checkpoints, metric, xp_eval_returns, sp_eval_returns) = jax.lax.scan(
                add_conf_policy, population_buffer, (iter_ids, add_conf_iter_rngs)
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

    train_fn = make_comedi_agents(config)
    out = train_fn(train_rng)
    return out

def get_comedi_population(config, out, env):
    '''
    Get the partner params and partner population for ego training.
    '''
    comedi_pop_size = config["algorithm"]["PARTNER_POP_SIZE"]

    # partner_params has shape (num_seeds, comedi_pop_size, ...)
    partner_params = out['final_params_conf']

    partner_policy = ActorWithConditionalCriticPolicy(
        action_dim=env.action_space(env.agents[1]).n,
        obs_dim=env.observation_space(env.agents[1]).shape[0],
        pop_size=comedi_pop_size, # used to create onehot agent id
        activation=config["algorithm"].get("ACTIVATION", "tanh")
    )

    # Create partner population
    partner_population = AgentPopulation(
        pop_size=comedi_pop_size,
        policy_cls=partner_policy
    )

    return partner_params, partner_population

def run_comedi(config, wandb_logger):
    algorithm_config = dict(config["algorithm"])

    env = make_env(algorithm_config["ENV_NAME"], algorithm_config["ENV_KWARGS"])
    env = LogWrapper(env)

    log.info("Starting CoMeDi training...")
    start = time.time()

    # Generate multiple random seeds from the base seed
    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"])
    rngs = jax.random.split(rng, algorithm_config["NUM_SEEDS"])

    # Create a vmapped version of train_comedi_partners
    with jax.disable_jit(False):
        vmapped_train_fn = jax.jit(
            jax.vmap(
                partial(train_comedi_partners, env=env, config=algorithm_config)
            )
        )
        out = vmapped_train_fn(rngs)

    end = time.time()
    log.info(f"CoMeDi training complete in {end - start} seconds")

    metric_names = get_metric_names(algorithm_config["ENV_NAME"])

    log_metrics(config, out, wandb_logger, metric_names)
    partner_params, partner_population = get_comedi_population(config, out, env)
    return partner_params, partner_population

def compute_sp_mask_and_ids(pop_size):
    cross_product = np.meshgrid(
        np.arange(pop_size),
        np.arange(pop_size)
    )
    agent_id_cartesian_product = np.stack([g.ravel() for g in cross_product], axis=-1)
    conf_ids = agent_id_cartesian_product[:, 0]
    ego_ids = agent_id_cartesian_product[:, 1]
    sp_mask = (conf_ids == ego_ids)
    return sp_mask, agent_id_cartesian_product

def log_metrics(config, outs, logger, metric_names: tuple):
    metrics = outs["metrics"]
    num_seeds, pop_size, num_updates, _, _ = metrics["pg_loss_conf_sp"].shape
    # TODO: add the eval_ep_last_info metrics

    ### Log evaluation metrics
    # we plot XP return curves separately from SP return curves
    # shape (num_seeds, num_updates, pop_size,  num_eval_episodes, num_agents_per_game)
    all_returns_sp = np.asarray(outs["last_ep_infos_sp"]["returned_episode_returns"])
    # shape (num_seeds, num_updates, pop_size, pop_size, num_eval_episodes, num_agents_per_game)
    all_returns_xp = np.asarray(outs["last_ep_infos_xp"]["returned_episode_returns"])
    xs = list(range(num_updates))

    # Average over seeds, then over agent pairs, episodes and num_agents_per_game
    sp_return_curve = all_returns_sp.mean(axis=(0, 3, 4))
    xp_return_curve = all_returns_xp.mean(axis=(0, 4, 5))

    for num_add_policies in range(pop_size):
        for step in range(num_updates):
            logger.log_item("Eval/AvgSPReturnCurve", sp_return_curve[num_add_policies, step], train_step=step)
            mean_xp_returns = xp_return_curve[num_add_policies][:, :(num_add_policies+1)].mean(axis=-1)
            logger.log_item("Eval/AvgXPReturnCurve", mean_xp_returns[step], train_step=step)
    logger.commit()

    ### Log population loss as multi-line plots, where each line is a different population member
    # both xp and xp metrics has shape (num_seeds, pop_size, num_updates, update_epochs, num_minibatches)
    # Average over seeds
    processed_losses = {
        "ConfPGLossSP": np.asarray(metrics["pg_loss_conf_sp"]).mean(axis=(0, 3, 4)), # desired shape (pop_size, num_updates)
        "ConfPGLossXP": np.asarray(metrics["pg_loss_conf_xp"]).mean(axis=(0, 3, 4)),
        "ConfPGLossMP": np.asarray(metrics["pg_loss_conf_mp"]).mean(axis=(0, 3, 4)),
        "ConfValLossSP": np.asarray(metrics["value_loss_sp"]).mean(axis=(0, 3, 4)),
        "ConfValLossXP": np.asarray(metrics["value_loss_xp"]).mean(axis=(0, 3, 4)),
        "ConfValLossMP": np.asarray(metrics["value_loss_mp"]).mean(axis=(0, 3, 4)),
        "EntropySP": np.asarray(metrics["entropy_conf_sp"]).mean(axis=(0, 3, 4)),
        "EntropyXP": np.asarray(metrics["entropy_conf_xp"]).mean(axis=(0, 3, 4)),
        "EntropyMP": np.asarray(metrics["entropy_conf_mp"]).mean(axis=(0, 3, 4)),
    }

    xs = list(range(num_updates))
    keys = [f"pair {i}" for i in range(pop_size)]

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

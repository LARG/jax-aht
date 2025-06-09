'''ROTATE w/o population buffer'''
import copy
from functools import partial
import logging
from typing import NamedTuple
import shutil
import time

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
import hydra

from agents.population_interface import AgentPopulation
from agents.initialize_agents import initialize_s5_agent, initialize_mlp_agent
from agents.agent_interface import ActorWithDoubleCriticPolicy, MLPActorCriticPolicy, S5ActorCriticPolicy
from common.plot_utils import get_metric_names, get_stats
from common.run_episodes import run_episodes
from marl.ppo_utils import Transition, unbatchify, _create_minibatches
from common.save_load_utils import save_train_run
from envs import make_env
from envs.log_wrapper import LogWrapper, LogEnvState
from ego_agent_training.ppo_ego import train_ppo_ego_agent


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

class ConfTransition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    other_value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray

def train_regret_maximizing_partners(config, env, 
                                     ego_params, ego_policy, 
                                     conf_params, conf_policy, 
                                     br_params, br_policy, partner_rng):
    '''
    Train regret-maximizing confederate/best-response pairs using the given ego agent policy and IPPO.
    Return model checkpoints and metrics. 
    '''
    def make_regret_maximizing_partner_train(config):
        num_agents = env.num_agents
        assert num_agents == 2, "This code assumes the environment has exactly 2 agents."

        # Define different minibatch sizes for interactions with ego agent and one with BR agent
        config["NUM_ACTORS"] = num_agents * config["NUM_ENVS"]

        # Right now assume control of just 1 agent
        config["NUM_CONTROLLED_ACTORS"] = config["NUM_ENVS"]
        config["NUM_UNCONTROLLED_AGENTS"] = config["NUM_ENVS"]

        # Divide by 4 because there are 4 types of rollouts 
        config["NUM_UPDATES"] = config["TIMESTEPS_PER_ITER_PARTNER"] // (config["ROLLOUT_LENGTH"] * 4 * config["NUM_ENVS"] * config["PARTNER_POP_SIZE"])
        config["MINIBATCH_SIZE"] = config["ROLLOUT_LENGTH"] * config["NUM_CONTROLLED_ACTORS"]

        assert config["MINIBATCH_SIZE"] % config["NUM_MINIBATCHES"] == 0, "MINIBATCH_SIZE must be divisible by NUM_MINIBATCHES"
        assert config["MINIBATCH_SIZE"] >= config["NUM_MINIBATCHES"], "MINIBATCH_SIZE must be >= NUM_MINIBATCHES"

        def linear_schedule(count):
            frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
            return config["LR"] * frac

        def train(rng, init_params_conf, init_params_br):
            confederate_policy = conf_policy
            
            # Define optimizers for both confederate and BR policy
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule if config["ANNEAL_LR"] else config["LR"], 
                eps=1e-5),
            )
            tx_br = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule if config["ANNEAL_LR"] else config["LR"], eps=1e-5),
            )
            train_state_conf = TrainState.create(
                apply_fn=confederate_policy.network.apply,
                params=init_params_conf,
                tx=tx,
            )

            train_state_br = TrainState.create(
                apply_fn=br_policy.network.apply,
                params=init_params_br,
                tx=tx_br,
            )

            def _reset_to_states(reset_traj_batch, env_state, last_obs, last_dones, last_conf_h, last_partner_h, 
                                 init_partner_hstate, rng, partner_is_br: bool):
                '''Resets the env_state and the hstates to the values in reset_traj_batch for done environments.'''

                def gather_sampled(data_pytree, flat_indices, first_nonbatch_dim: int):
                    '''Will treat all dimensions up to the first_nonbatch_dim as batch dimensions. '''
                    batch_size = config["ROLLOUT_LENGTH"] * config["NUM_ENVS"]
                    flat_data = jax.tree.map(lambda x: x.reshape(batch_size, *x.shape[first_nonbatch_dim:]), data_pytree)
                    sampled_data = jax.tree.map(lambda x: x[flat_indices], flat_data) # Shape (N, ...)
                    return sampled_data
                
                rng, sample_rng = jax.random.split(rng)
                needs_resample = last_dones["__all__"] # shape (N,) bool

                total_reset_states = config["ROLLOUT_LENGTH"] * config["NUM_ENVS"]
                sampled_indices = jax.random.randint(sample_rng, shape=(config["NUM_ENVS"],), minval=0, 
                                                        maxval=total_reset_states)
                
                # Gather sampled leaves from each data pytree
                sampled_env_state = gather_sampled(reset_traj_batch.env_state, sampled_indices, first_nonbatch_dim=2)
                sampled_conf_obs = gather_sampled(reset_traj_batch.conf_obs, sampled_indices, first_nonbatch_dim=2)
                sampled_partner_obs = gather_sampled(reset_traj_batch.partner_obs, sampled_indices, first_nonbatch_dim=2)
                sampled_conf_done = gather_sampled(reset_traj_batch.conf_done, sampled_indices, first_nonbatch_dim=2)
                sampled_partner_done = gather_sampled(reset_traj_batch.partner_done, sampled_indices, first_nonbatch_dim=2)
                
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
                obs_1 = jnp.where(needs_resample[:, jnp.newaxis], sampled_partner_obs, last_obs["agent_1"])

                dones_0 = jnp.where(needs_resample, sampled_conf_done, last_dones["agent_0"])
                dones_1 = jnp.where(needs_resample, sampled_partner_done, last_dones["agent_1"])

                if last_conf_h is not None: # has a leading (1, ) dimension
                    sampled_conf_hstate = gather_sampled(reset_traj_batch.conf_hstate, sampled_indices, first_nonbatch_dim=3)
                    sampled_conf_hstate = jax.tree.map(lambda x: x[jnp.newaxis, ...], sampled_conf_hstate)
                    last_conf_h = jax.tree.map(lambda sampled, original: jnp.where(needs_resample[jnp.newaxis, :, jnp.newaxis], 
                                                                                   sampled, original), sampled_conf_hstate, last_conf_h)
                
                if last_partner_h is not None: # has a leading (1, ) dimension
                    if config["REINIT_BR_TO_EGO"] and partner_is_br: # we know br is compatible with the ego hstate
                        sample_partner_hstate = gather_sampled(reset_traj_batch.partner_hstate, sampled_indices, first_nonbatch_dim=3)                     
                        sample_partner_hstate = jax.tree.map(lambda x: x[jnp.newaxis, ...], sample_partner_hstate)
                    else: # otherwise, just reset to the provided partner initial hstate
                        sample_partner_hstate = init_partner_hstate # Use the initial state passed in

                    last_partner_h = jax.tree.map(lambda sampled, original: jnp.where(needs_resample[jnp.newaxis, :, jnp.newaxis], 
                                                                                    sampled, original), sample_partner_hstate, last_partner_h)
                return env_state, obs_0, obs_1, dones_0, dones_1, last_conf_h, last_partner_h
            
            def _env_step_conf_ego(runner_state, unused):
                """
                agent_0 = confederate, agent_1 = ego
                Returns updated runner_state, a ConfTransition for the confederate, and a ResetTransition.
                """
                train_state_conf, env_state, last_obs, last_dones, last_conf_h, last_ego_h, reset_traj_batch, rng = runner_state
                rng, act_rng, partner_rng, step_rng = jax.random.split(rng, 4)

                # Reset conf-ego data collection from conf-br states
                if reset_traj_batch is not None:
                    env_state, obs_0, obs_1, dones_0, dones_1, last_conf_h, last_ego_h = _reset_to_states(
                        reset_traj_batch, env_state, last_obs, last_dones, 
                        last_conf_h, last_ego_h, init_ego_hstate, rng, partner_is_br=False)
                else: # Original logic if not resetting
                    obs_0, obs_1 = last_obs["agent_0"], last_obs["agent_1"]
                    dones_0, dones_1 = last_dones["agent_0"], last_dones["agent_1"]

                # Get available actions for agent 0 from environment state
                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
                avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

                # Agent_0 (confederate) action using policy interface
                act_0, (val_ego, val_br), pi_0, new_conf_h = confederate_policy.get_action_value_policy(
                    params=train_state_conf.params,
                    obs=obs_0.reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                    done=dones_0.reshape(1, config["NUM_CONTROLLED_ACTORS"]),
                    avail_actions=jax.lax.stop_gradient(avail_actions_0),
                    hstate=last_conf_h,
                    rng=act_rng
                )
                logp_0 = pi_0.log_prob(act_0)

                act_0 = act_0.squeeze()
                logp_0 = logp_0.squeeze()
                val_ego = val_ego.squeeze()
                val_br = val_br.squeeze()

                # Agent_1 (ego) action using policy interface
                act_1, _, _, new_ego_h = ego_policy.get_action_value_policy(
                    params=ego_params,
                    obs=obs_1.reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                    done=dones_1.reshape(1, config["NUM_CONTROLLED_ACTORS"]),
                    avail_actions=jax.lax.stop_gradient(avail_actions_1),
                    hstate=last_ego_h,
                    rng=partner_rng
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
                transition = ConfTransition(
                    done=done["agent_0"],
                    action=act_0,
                    value=val_ego,
                    other_value=val_br,
                    reward=reward["agent_1"],
                    log_prob=logp_0,
                    obs=obs_0,
                    info=info_0,
                    avail_actions=avail_actions_0
                )
                reset_transition = ResetTransition(
                    # all of these are from before env step
                    env_state=env_state,
                    conf_obs=obs_0,
                    partner_obs=obs_1,
                    conf_done=last_dones["agent_0"],
                    partner_done=last_dones["agent_1"],
                    conf_hstate=last_conf_h,
                    partner_hstate=last_ego_h
                )

                new_runner_state = (train_state_conf, env_state_next, obs_next, done, new_conf_h, new_ego_h, reset_traj_batch, rng)
                return new_runner_state, (transition, reset_transition)

            def _env_step_conf_br(runner_state, unused):
                """
                agent_0 = confederate, agent_1 = best response
                Returns updated runner_state, ConfTransition for the confederate, 
                Transition for the best response, and a ResetTransition.
                """
                train_state_conf, train_state_br, env_state, last_obs, last_dones, \
                    last_conf_h, last_br_h, reset_traj_batch, rng = runner_state
                rng, conf_rng, br_rng, step_rng = jax.random.split(rng, 4)
                
                # Reset conf-br data collection from conf-ego states
                if reset_traj_batch is not None:
                    env_state, obs_0, obs_1, dones_0, dones_1, last_conf_h, last_br_h = _reset_to_states(
                        reset_traj_batch, env_state, last_obs, last_dones, 
                        last_conf_h, last_br_h, init_br_hstate, rng, partner_is_br=True)
                else: # Original logic if not resetting
                    obs_0, obs_1 = last_obs["agent_0"], last_obs["agent_1"]
                    dones_0, dones_1 = last_dones["agent_0"], last_dones["agent_1"]

                # Get available actions for agent 0 from environment state
                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
                avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

                # Agent_0 (confederate) action
                act_0, (val_ego, val_br), pi_0, new_conf_h = confederate_policy.get_action_value_policy(
                    params=train_state_conf.params,
                    obs=obs_0.reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                    done=dones_0.reshape(1, config["NUM_CONTROLLED_ACTORS"]),
                    avail_actions=jax.lax.stop_gradient(avail_actions_0),
                    hstate=last_conf_h,
                    rng=conf_rng
                )
                logp_0 = pi_0.log_prob(act_0)

                act_0 = act_0.squeeze()
                logp_0 = logp_0.squeeze()
                val_ego = val_ego.squeeze()
                val_br = val_br.squeeze()

                # Agent 1 (best response) action
                act_1, val_1, pi_1, new_br_h = br_policy.get_action_value_policy(
                    params=train_state_br.params,
                    obs=obs_1.reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                    done=dones_1.reshape(1, config["NUM_CONTROLLED_ACTORS"]),
                    avail_actions=jax.lax.stop_gradient(avail_actions_1),
                    hstate=last_br_h,
                    rng=br_rng
                )
                logp_1 = pi_1.log_prob(act_1)

                act_1 = act_1.squeeze()
                logp_1 = logp_1.squeeze()
                val_1 = val_1.squeeze()

                # Combine actions into the env format
                combined_actions = jnp.concatenate([act_0, act_1], axis=0)
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
                transition_0 = ConfTransition(
                    done=done["agent_0"],
                    action=act_0,
                    value=val_br,
                    other_value=val_ego,
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
                reset_transition = ResetTransition(
                    # all of these are from before env step
                    env_state=env_state,
                    conf_obs=obs_0,
                    partner_obs=obs_1,
                    conf_done=last_dones["agent_0"],
                    partner_done=last_dones["agent_1"],
                    conf_hstate=last_conf_h,
                    partner_hstate=last_br_h
                )
                
                # Pass reset_traj_batch and init_br_hstate through unchanged in the state tuple
                new_runner_state = (train_state_conf, train_state_br, env_state_next, obs_next, done, new_conf_h, new_br_h, reset_traj_batch, rng)
                return new_runner_state, (transition_0, transition_1, reset_transition)
            
            # --------------------------
            # 3d) GAE & update step
            # --------------------------
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
            
                def _compute_ppo_pg_loss(objective, log_prob, traj_batch):
                    '''Policy gradient loss function for PPO'''
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    obj_norm = (objective - objective.mean()) / (objective.std() + 1e-8)
                    pg_loss_1 = ratio * obj_norm
                    pg_loss_2 = jnp.clip(
                        ratio, 
                        1.0 - config["CLIP_EPS"], 
                        1.0 + config["CLIP_EPS"]) * obj_norm
                    pg_loss = -jnp.mean(jnp.minimum(pg_loss_1, pg_loss_2))
                    return pg_loss

                def _update_minbatch_conf(train_state_conf, batch_infos):
                    '''
                    Teammate update function. Note that this implementation gathers both XSP (XP data from SP start states)
                    and SXP (SP data from XP start states) data to enable experimenting with different objectives. 
                    ROTATE uses the "sreg-xp_ret-sp_ret-sxp" objective, which only requires SP, XP and SXP data.
                    '''
                    minbatch_xp, minbatch_sp, minbatch_xsp, minbatch_sxp = batch_infos

                    def _loss_fn_conf(params, minbatch_xp, minbatch_sp, minbatch_xsp, minbatch_sxp):
                        # doesn't really matter which init_conf_hstate we use here since it's all the same
                        init_conf_hstate, traj_batch_xp, gae_xp, target_v_xp = minbatch_xp
                        _, traj_batch_sp, gae_sp, target_v_sp = minbatch_sp
                        _, traj_batch_xsp, gae_xsp, target_v_xsp = minbatch_xsp
                        _, traj_batch_sxp, gae_sxp, target_v_sxp = minbatch_sxp

                        # get policy and value for all 4 interaction types
                        _, (value_xp_on_xp_data, value_sp_on_xp_data), pi_xp, _ = confederate_policy.get_action_value_policy(
                            params=params, 
                            obs=traj_batch_xp.obs, 
                            done=traj_batch_xp.done,
                            avail_actions=traj_batch_xp.avail_actions,
                            hstate=init_conf_hstate,
                            rng=jax.random.PRNGKey(0) # only used for action sampling, which is not used here 
                        )
                        _, (value_xp_on_sp_data, value_sp_on_sp_data), pi_sp, _ = confederate_policy.get_action_value_policy(
                            params=params, 
                            obs=traj_batch_sp.obs, 
                            done=traj_batch_sp.done,
                            avail_actions=traj_batch_sp.avail_actions,
                            hstate=init_conf_hstate,
                            rng=jax.random.PRNGKey(0) # only used for action sampling, which is not used here 
                        )

                        _, (value_xp_on_xsp_data, value_sp_on_xsp_data), pi_xsp, _ = confederate_policy.get_action_value_policy(
                            params=params, 
                            obs=traj_batch_xsp.obs, 
                            done=traj_batch_xsp.done,
                            avail_actions=traj_batch_xsp.avail_actions,
                            hstate=init_conf_hstate,
                            rng=jax.random.PRNGKey(0) # only used for action sampling, which is not used here 
                        )

                        _, (value_xp_on_sxp_data, value_sp_on_sxp_data), pi_sxp, _ = confederate_policy.get_action_value_policy(
                            params=params, 
                            obs=traj_batch_sxp.obs, 
                            done=traj_batch_sxp.done,
                            avail_actions=traj_batch_sxp.avail_actions,
                            hstate=init_conf_hstate,
                            rng=jax.random.PRNGKey(0) # only used for action sampling, which is not used here 
                        )

                        log_prob_xp = pi_xp.log_prob(traj_batch_xp.action)
                        log_prob_sp = pi_sp.log_prob(traj_batch_sp.action)
                        log_prob_xsp = pi_xsp.log_prob(traj_batch_xsp.action)
                        log_prob_sxp = pi_sxp.log_prob(traj_batch_sxp.action)
                        
                        value_loss_xp = _compute_ppo_value_loss(value_xp_on_xp_data, traj_batch_xp, target_v_xp)
                        value_loss_sp = _compute_ppo_value_loss(value_sp_on_sp_data, traj_batch_sp, target_v_sp)
                        value_loss_xsp = _compute_ppo_value_loss(value_xp_on_xsp_data, traj_batch_xsp, target_v_xsp)
                        value_loss_sxp = _compute_ppo_value_loss(value_sp_on_sxp_data, traj_batch_sxp, target_v_sxp)

                        # Compute policy objectives

                        # optimize per-state regret for all interaction types
                        if config["CONF_OBJ_TYPE"] == "per_state_regret":
                            xp_return_to_go_xp_data = value_xp_on_xp_data + gae_xp
                            sp_return_to_go_sp_data = value_sp_on_sp_data + gae_sp
                            xsp_return_to_go_xsp_data = value_xp_on_xsp_data + gae_xsp
                            sxp_return_to_go_sxp_data = value_sp_on_sxp_data + gae_sxp
                            # regret as the total objective
                            total_xp_objective = config["REGRET_SP_WEIGHT"] * value_sp_on_xp_data - xp_return_to_go_xp_data
                            total_sp_objective = config["SP_WEIGHT"] * config["REGRET_SP_WEIGHT"] * sp_return_to_go_sp_data - value_xp_on_sp_data
                            total_xsp_objective = config["REGRET_SP_WEIGHT"] * value_sp_on_xsp_data - xsp_return_to_go_xsp_data
                            total_sxp_objective = config["SP_WEIGHT"] * config["REGRET_SP_WEIGHT"] * sxp_return_to_go_sxp_data - value_xp_on_sxp_data

                        # optimize per-state regret for all interaction types   
                        elif config["CONF_OBJ_TYPE"] == "per_state_regret_target":
                            # use target returns and values to compute the regret objective
                            total_xp_objective = config["REGRET_SP_WEIGHT"] * traj_batch_xp.other_value - target_v_xp
                            total_sp_objective = config["SP_WEIGHT"] * config["REGRET_SP_WEIGHT"] * target_v_sp - traj_batch_sp.other_value
                            total_xsp_objective = config["REGRET_SP_WEIGHT"] * traj_batch_xsp.other_value - target_v_xsp
                            total_sxp_objective = config["SP_WEIGHT"] * config["REGRET_SP_WEIGHT"] * target_v_sxp - traj_batch_sxp.other_value
                        
                        # optimize per-state regret on ego rollouts only, return for both types of br interactions
                        elif config["CONF_OBJ_TYPE"] == "sreg-xp_ret-sp_ret-sxp":
                            xp_return_to_go_xp_data = value_xp_on_xp_data + gae_xp

                            total_xp_objective = config["REGRET_SP_WEIGHT"] * value_sp_on_xp_data - xp_return_to_go_xp_data
                            total_sp_objective = config["SP_WEIGHT"] * gae_sp
                            total_xsp_objective = jnp.array(0.0) # no PG loss term on ego rollouts from conf-br states
                            total_sxp_objective = config["SP_WEIGHT"] * gae_sxp

                        # optimize per-state regret on ego and sp rollouts, return on sxp, and nothing on xsp
                        elif config["CONF_OBJ_TYPE"] == "sreg-xp_sreg-sp_ret-sxp":
                            xp_return_to_go_xp_data = value_xp_on_xp_data + gae_xp
                            sp_return_to_go_sp_data = value_sp_on_sp_data + gae_sp

                            total_xp_objective = config["REGRET_SP_WEIGHT"] * value_sp_on_xp_data - xp_return_to_go_xp_data
                            total_sp_objective = config["SP_WEIGHT"] * config["REGRET_SP_WEIGHT"] * sp_return_to_go_sp_data - value_xp_on_sp_data
                            total_xsp_objective = jnp.array(0.0) # no PG loss term on ego rollouts from conf-br states
                            total_sxp_objective = config["SP_WEIGHT"] * gae_sxp

                        # optimize trajectory-level regret for all interaction types
                        elif config["CONF_OBJ_TYPE"] == "gae_per_state_regret":
                            total_xp_objective = -gae_xp
                            total_sp_objective = config["SP_WEIGHT"] * gae_sp
                            total_xsp_objective = jnp.array(0.0)
                            total_sxp_objective = config["SP_WEIGHT"] * gae_sxp

                        elif config["CONF_OBJ_TYPE"] == "traj_regret":
                            total_xp_objective = -gae_xp
                            total_sp_objective = config["SP_WEIGHT"] * gae_sp
                            total_xsp_objective = jnp.array(0.0)
                            total_sxp_objective = jnp.array(0.0)

                        pg_loss_xp = _compute_ppo_pg_loss(total_xp_objective, log_prob_xp, traj_batch_xp)
                        pg_loss_sp = _compute_ppo_pg_loss(total_sp_objective, log_prob_sp, traj_batch_sp)
                        pg_loss_xsp = _compute_ppo_pg_loss(total_xsp_objective, log_prob_xsp, traj_batch_xsp)
                        pg_loss_sxp = _compute_ppo_pg_loss(total_sxp_objective, log_prob_sxp, traj_batch_sxp)

                        entropy_xp = jnp.mean(pi_xp.entropy())
                        entropy_sp = jnp.mean(pi_sp.entropy())
                        entropy_xsp = jnp.mean(pi_xsp.entropy())
                        entropy_sxp = jnp.mean(pi_sxp.entropy())

                        xp_loss = pg_loss_xp + config["VF_COEF"] * value_loss_xp - config["ENT_COEF"] * entropy_xp
                        sp_loss = pg_loss_sp + config["VF_COEF"] * value_loss_sp - config["ENT_COEF"] * entropy_sp
                        xsp_loss = pg_loss_xsp + config["VF_COEF"] * value_loss_xsp - config["ENT_COEF"] * entropy_xsp
                        sxp_loss = pg_loss_sxp + config["VF_COEF"] * value_loss_sxp - config["ENT_COEF"] * entropy_sxp

                        total_loss = sp_loss + xp_loss + xsp_loss + sxp_loss

                        return total_loss, ((value_loss_xp, pg_loss_xp, entropy_xp),
                                             (value_loss_sp, pg_loss_sp, entropy_sp),
                                             (value_loss_xsp, pg_loss_xsp, entropy_xsp),
                                             (value_loss_sxp, pg_loss_sxp, entropy_sxp))


                    grad_fn = jax.value_and_grad(_loss_fn_conf, has_aux=True)
                    (loss_val, aux_vals), grads = grad_fn(
                        train_state_conf.params, 
                        minbatch_xp, minbatch_sp, minbatch_xsp, minbatch_sxp)
                    train_state_conf = train_state_conf.apply_gradients(grads=grads)
                    return train_state_conf, (loss_val, aux_vals)
                
                def _update_minbatch_br(train_state_br, batch_infos):
                    minbatch_sp, minbatch_sxp = batch_infos
                    init_br_hstate, traj_batch_sp, advantages_sp, returns_sp = minbatch_sp
                    init_br_hstate, traj_batch_sxp, advantages_sxp, returns_sxp = minbatch_sxp

                    # merge the two sources of data since the BR is always return-maximizing
                    # axis 0 is the time dimension, axis 1 is the minibatch dimension
                    traj_batch_sp_merged = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=1), traj_batch_sp, traj_batch_sxp)
                    init_br_hstate_merged = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=1), init_br_hstate, init_br_hstate)                    
                    advantages_sp_merged = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=1), advantages_sp, advantages_sxp)
                    returns_sp_merged = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=1), returns_sp, returns_sxp)

                    def _loss_fn_br(params, traj_batch, gae, target_v):
                        _, value, pi, _ = br_policy.get_action_value_policy(
                            params=params, 
                            obs=traj_batch.obs, 
                            done=traj_batch.done,
                            avail_actions=traj_batch.avail_actions,
                            hstate=init_br_hstate_merged,
                            rng=jax.random.PRNGKey(0) # only used for action sampling, which is not used here 
                        )

                        log_prob = pi.log_prob(traj_batch.action)

                        value_loss = _compute_ppo_value_loss(value, traj_batch, target_v)
                        pg_loss = _compute_ppo_pg_loss(gae, log_prob, traj_batch)
                        entropy = jnp.mean(pi.entropy())

                        sp_loss = pg_loss + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy

                        total_loss = sp_loss
                        return total_loss, (value_loss, pg_loss, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn_br, has_aux=True)
                    (loss_val, aux_vals), grads = grad_fn(
                        train_state_br.params, 
                        traj_batch_sp_merged, advantages_sp_merged, returns_sp_merged)
                    train_state_br = train_state_br.apply_gradients(grads=grads)
                    return train_state_br, (loss_val, aux_vals)

                (
                    train_state_conf, train_state_br, 
                    conf_update_data, br_update_data,
                    rng
                ) = update_state

                (
                    traj_batch_xp, traj_batch_sp_conf, traj_batch_xsp, traj_batch_sxp_conf,
                    advantages_xp_conf, advantages_sp_conf, advantages_xsp_conf, advantages_sxp_conf,
                    targets_xp_conf, targets_sp_conf, targets_xsp_conf, targets_sxp_conf,
                ) = conf_update_data

                (
                    traj_batch_sp_br, traj_batch_sxp_br,
                    advantages_sp_br, advantages_sxp_br,
                    targets_sp_br, targets_sxp_br,
                ) = br_update_data

                rng, perm_rng_xp, perm_rng_sp_conf, perm_rng_sp_br, \
                    perm_rng_xsp, perm_rng_sxp_conf, perm_rng_sxp_br = jax.random.split(rng, 7)

                # Create minibatches for each agent and interaction type
                # 1) Conf-ego interaction (XP)
                minibatches_xp = _create_minibatches(
                    traj_batch_xp, advantages_xp_conf, targets_xp_conf, init_conf_hstate, 
                    config["NUM_CONTROLLED_ACTORS"], config["NUM_MINIBATCHES"], perm_rng_xp
                )
                # 2) Conf-br interaction (SP)
                minibatches_sp_conf = _create_minibatches(
                    traj_batch_sp_conf, advantages_sp_conf, targets_sp_conf, init_conf_hstate, 
                    config["NUM_CONTROLLED_ACTORS"], config["NUM_MINIBATCHES"], perm_rng_sp_conf
                )
                minibatches_sp_br = _create_minibatches(
                    traj_batch_sp_br, advantages_sp_br, targets_sp_br, init_br_hstate, 
                    config["NUM_CONTROLLED_ACTORS"], config["NUM_MINIBATCHES"], perm_rng_sp_br
                )
                # 3) Conf-ego interaction from conf-br states (XSP)
                minibatches_xsp = _create_minibatches(
                    traj_batch_xsp, advantages_xsp_conf, targets_xsp_conf, init_conf_hstate, 
                    config["NUM_CONTROLLED_ACTORS"], config["NUM_MINIBATCHES"], perm_rng_xsp
                )
                # 4) Conf-br interaction from conf-ego states (SXP)
                minibatches_sxp_conf = _create_minibatches(
                    traj_batch_sxp_conf, advantages_sxp_conf, targets_sxp_conf, init_conf_hstate, 
                    config["NUM_CONTROLLED_ACTORS"], config["NUM_MINIBATCHES"], perm_rng_sxp_conf
                )
                minibatches_sxp_br = _create_minibatches(
                    traj_batch_sxp_br, advantages_sxp_br, targets_sxp_br, init_br_hstate, 
                    config["NUM_CONTROLLED_ACTORS"], config["NUM_MINIBATCHES"], perm_rng_sxp_br
                )

                # Update confederate
                train_state_conf, total_loss_conf = jax.lax.scan(
                    _update_minbatch_conf, train_state_conf, (minibatches_xp, minibatches_sp_conf, 
                                                              minibatches_xsp, minibatches_sxp_conf)
                )

                # Update best response
                train_state_br, total_loss_br = jax.lax.scan(
                    _update_minbatch_br, train_state_br, (minibatches_sp_br, minibatches_sxp_br)
                )

                update_state = (train_state_conf, train_state_br, 
                    conf_update_data, br_update_data, rng)
                return update_state, (total_loss_conf, total_loss_br)

            def _update_step(update_runner_state, unused):
                """
                1. Collect confederate-ego rollout (XP).
                2. Collect confederate-br rollout (SP).
                3. Collect confederate-ego rollout from confederate-br states (XSP).
                4. Collect confederate-br rollout from confederate-ego states (SXP). 
                5. Compute advantages for XP, XSP, SP, and SXP interactions.
                6. PPO updates for best response and confederate policies.
                """
                (
                    train_state_conf, train_state_br, 
                    env_state_xp, env_state_sp, env_state_xsp, env_state_sxp, 
                    last_obs_xp, last_obs_sp, last_obs_xsp, last_obs_sxp, 
                    last_dones_xp, last_dones_sp, last_dones_xsp, last_dones_sxp, 
                    conf_hstate_xp, ego_hstate_xp, 
                    conf_hstate_sp, br_hstate_sp, 
                    conf_hstate_xsp, ego_hstate_xsp, 
                    conf_hstate_sxp, br_hstate_sxp, 
                    rng_update, update_steps
                ) = update_runner_state

                rng_update, rng_xp, rng_sp, rng_xsp, rng_sxp = jax.random.split(rng_update, 5)

                # 1) rollout for conf-ego interaction (XP)
                runner_state_xp = (train_state_conf, env_state_xp, last_obs_xp, last_dones_xp,
                                    conf_hstate_xp, ego_hstate_xp, None, rng_xp)
                runner_state_xp, (traj_batch_xp, reset_traj_batch_xp) = jax.lax.scan(
                    _env_step_conf_ego, runner_state_xp, None, config["ROLLOUT_LENGTH"])
                (train_state_conf, env_state_xp, last_obs_xp, last_dones_xp, 
                 conf_hstate_xp, ego_hstate_xp, _, rng_xp) = runner_state_xp
            
                # 2) rollout for conf-br interaction (SP)
                runner_state_sp = (train_state_conf, train_state_br, env_state_sp, last_obs_sp, 
                                   last_dones_sp, conf_hstate_sp, br_hstate_sp, None, rng_sp)
                runner_state_sp, (traj_batch_sp_conf, traj_batch_sp_br, reset_traj_batch_sp) = jax.lax.scan(
                    _env_step_conf_br, runner_state_sp, None, config["ROLLOUT_LENGTH"])
                (train_state_conf, train_state_br, env_state_sp, last_obs_sp, last_dones_sp,
                conf_hstate_sp, br_hstate_sp, _, rng_sp) = runner_state_sp
                
                # 3) rollout for conf-ego interaction from conf-br states (XSP)
                runner_state_xsp = (train_state_conf, env_state_xsp, last_obs_xsp, last_dones_xsp,
                                    conf_hstate_xsp, ego_hstate_xsp, 
                                    reset_traj_batch_sp, rng_xsp)
                runner_state_xsp, (traj_batch_xsp, _) = jax.lax.scan(
                    _env_step_conf_ego, runner_state_xsp, None, config["ROLLOUT_LENGTH"])
                (train_state_conf, env_state_xsp, last_obs_xsp, last_dones_xsp, 
                 conf_hstate_xsp, ego_hstate_xsp, _, rng_xsp) = runner_state_xsp

                # 4) rollout for conf-br interaction from conf-ego states (SXP)
                runner_state_sxp = (train_state_conf, train_state_br, env_state_sxp, last_obs_sxp, 
                                    last_dones_sxp, conf_hstate_sxp, br_hstate_sxp, 
                                    reset_traj_batch_xp, rng_sxp)
                runner_state_sxp, (traj_batch_sxp_conf, traj_batch_sxp_br, _) = jax.lax.scan(
                    _env_step_conf_br, runner_state_sxp, None, config["ROLLOUT_LENGTH"])
                (train_state_conf, train_state_br, env_state_sxp, last_obs_sxp, last_dones_sxp,
                 conf_hstate_sxp, br_hstate_sxp, _, rng_sxp) = runner_state_sxp

                def _compute_advantages_and_targets(batch_size, env_state, policy, policy_params, policy_hstate, 
                                                   last_obs, last_dones, traj_batch, agent_name, value_idx=None):
                    '''Value_idx argument is to support the ActorWithDoubleCritic (confederate) policy, which 
                    has two value heads. Value head 0 models the ego agent while value head 1 models the best response.'''
                    avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)[agent_name].astype(jnp.float32)
                    _, vals, _, _ = policy.get_action_value_policy(
                        params=policy_params,
                        obs=last_obs[agent_name].reshape(1, batch_size, -1),
                        done=last_dones[agent_name].reshape(1, batch_size),
                        avail_actions=jax.lax.stop_gradient(avail_actions),
                        hstate=policy_hstate,
                        rng=jax.random.PRNGKey(0)  # dummy key as we don't sample actions
                    )
                    if value_idx is None:
                        last_val = vals.squeeze()
                    else:
                        last_val = vals[value_idx].squeeze()
                    advantages, targets = _calculate_gae(traj_batch, last_val)
                    return advantages, targets
                
                # 5a) Compute conf advantages for XP (conf-ego) interaction
                advantages_xp_conf, targets_xp_conf = _compute_advantages_and_targets(
                    config["NUM_CONTROLLED_ACTORS"],
                    env_state_xp, conf_policy, train_state_conf.params, conf_hstate_xp, 
                    last_obs_xp, last_dones_xp, traj_batch_xp, "agent_0", value_idx=0)

                # 5b) Compute conf and br advantages for SP (conf-br) interaction
                advantages_sp_conf, targets_sp_conf = _compute_advantages_and_targets(
                    config["NUM_CONTROLLED_ACTORS"],
                    env_state_sp, conf_policy, train_state_conf.params, conf_hstate_sp, 
                    last_obs_sp, last_dones_sp, traj_batch_sp_conf, "agent_0", value_idx=1)
                
                advantages_sp_br, targets_sp_br = _compute_advantages_and_targets(
                    config["NUM_CONTROLLED_ACTORS"],
                    env_state_sp, br_policy, train_state_br.params, br_hstate_sp, 
                    last_obs_sp, last_dones_sp, traj_batch_sp_br, "agent_1", value_idx=None)

                # 5c) Compute conf advantages for XSP (conf-ego from conf-br states) interaction
                advantages_xsp_conf, targets_xsp_conf = _compute_advantages_and_targets(
                    config["NUM_CONTROLLED_ACTORS"],
                    env_state_xsp, conf_policy, train_state_conf.params, conf_hstate_xsp, 
                    last_obs_xsp, last_dones_xsp, traj_batch_xsp, "agent_0", value_idx=0)
                
                # 5d) Compute conf and br advantages for SXP (conf-br from conf-ego states) interaction
                advantages_sxp_conf, targets_sxp_conf = _compute_advantages_and_targets(
                    config["NUM_CONTROLLED_ACTORS"],
                    env_state_sxp, conf_policy, train_state_conf.params, conf_hstate_sxp, 
                    last_obs_sxp, last_dones_sxp, traj_batch_sxp_conf, "agent_0", value_idx=1)
                
                advantages_sxp_br, targets_sxp_br = _compute_advantages_and_targets(
                    config["NUM_CONTROLLED_ACTORS"],
                    env_state_sxp, br_policy, train_state_br.params, br_hstate_sxp, 
                    last_obs_sxp, last_dones_sxp, traj_batch_sxp_br, "agent_1", value_idx=None)
                
                # 6) PPO update
                conf_update_data = (
                    traj_batch_xp, traj_batch_sp_conf, traj_batch_xsp, traj_batch_sxp_conf,
                    advantages_xp_conf, advantages_sp_conf, advantages_xsp_conf, advantages_sxp_conf,
                    targets_xp_conf, targets_sp_conf, targets_xsp_conf, targets_sxp_conf,
                )
                br_update_data = (
                    traj_batch_sp_br, traj_batch_sxp_br,
                    advantages_sp_br, advantages_sxp_br,
                    targets_sp_br, targets_sxp_br,
                )
                rng_update, sub_rng = jax.random.split(rng_update, 2)

                update_state = (
                    train_state_conf, train_state_br, 
                    conf_update_data, br_update_data,
                    sub_rng
                )
                update_state, all_losses = jax.lax.scan(
                    _update_epoch, update_state, None, config["UPDATE_EPOCHS"])
                train_state_conf = update_state[0]
                train_state_br = update_state[1]

                conf_losses, br_losses = all_losses

                (conf_loss_xp, conf_loss_sp, conf_loss_xsp, conf_loss_sxp) = conf_losses[1]
                conf_value_loss_xp = (conf_loss_xp[0] + conf_loss_xsp[0])/2.0
                conf_pg_loss_xp = (conf_loss_xp[1] + conf_loss_xsp[1])/2.0
                conf_entropy_xp = (conf_loss_xp[2] + conf_loss_xsp[2])/2.0

                conf_value_loss_sp = (conf_loss_sp[0] + conf_loss_sxp[0])/2.0
                conf_pg_loss_sp = (conf_loss_sp[1] + conf_loss_sxp[1])/2.0
                conf_entropy_sp = (conf_loss_sp[2] + conf_loss_sxp[2])/2.0

                conf_avg_reward_ego = jnp.mean(jnp.array([traj_batch_xp.reward, traj_batch_xsp.reward]))
                conf_avg_reward_br = jnp.mean(jnp.array([traj_batch_sp_br.reward, traj_batch_sxp_br.reward]))

                (br_value_loss, br_pg_loss, br_entropy) = br_losses[1]
                
                # Metrics
                metric = traj_batch_xp.info
                metric["update_steps"] = update_steps
                metric["value_loss_conf_against_ego"] = conf_value_loss_xp
                metric["value_loss_conf_against_br"] = conf_value_loss_sp

                metric["pg_loss_conf_against_ego"] = conf_pg_loss_xp
                metric["pg_loss_conf_against_br"] = conf_pg_loss_sp

                metric["entropy_conf_against_ego"] = conf_entropy_xp
                metric["entropy_conf_against_br"] = conf_entropy_sp
                
                metric["average_rewards_ego"] = conf_avg_reward_ego
                metric["average_rewards_br"] = conf_avg_reward_br

                metric["value_loss_br"] = br_value_loss
                metric["pg_loss_br"] = br_pg_loss
                metric["entropy_loss_br"] = br_entropy

                new_update_runner_state = (
                    train_state_conf, train_state_br, 
                    env_state_xp, env_state_sp, env_state_xsp, env_state_sxp, 
                    last_obs_xp, last_obs_sp, last_obs_xsp, last_obs_sxp, 
                    last_dones_xp, last_dones_sp, last_dones_xsp, last_dones_sxp, 
                    conf_hstate_xp, ego_hstate_xp, 
                    conf_hstate_sp, br_hstate_sp, 
                    conf_hstate_xsp, ego_hstate_xsp, 
                    conf_hstate_sxp, br_hstate_sxp, 
                    rng_update, update_steps + 1
                )
                return (new_update_runner_state, metric)

            # --------------------------
            # PPO Update and Checkpoint saving
            # --------------------------
            ckpt_and_eval_interval = config["NUM_UPDATES"] // max(1, config["NUM_CHECKPOINTS"] - 1) # -1 because we store a ckpt at the last update
            num_ckpts = config["NUM_CHECKPOINTS"]

            # Build a PyTree that holds parameters for all conf agent checkpoints
            def init_ckpt_array(params_pytree):
                return jax.tree.map(
                    lambda x: jnp.zeros((num_ckpts,) + x.shape, x.dtype), 
                    params_pytree)
        
            def _update_step_with_ckpt(state_with_ckpt, unused):
                (update_runner_state, checkpoint_array_conf, checkpoint_array_br, ckpt_idx, 
                    eval_info_br, eval_info_ego) = state_with_ckpt

                # Single PPO update
                (new_update_runner_state, metric) = _update_step(
                    update_runner_state,
                    None
                )

                rng, update_steps = new_update_runner_state[-2], new_update_runner_state[-1]

                # Decide if we store a checkpoint
                # update steps is 1-indexed because it was incremented at the end of the update step
                to_store = jnp.logical_or(jnp.equal(jnp.mod(update_steps-1, ckpt_and_eval_interval), 0),
                                          jnp.equal(update_steps, config["NUM_UPDATES"]))
                      
                def store_and_eval_ckpt(args):
                    ckpt_arr_and_ep_infos, rng, cidx = args
                    ckpt_arr_conf, ckpt_arr_br, prev_ep_infos_br, prev_ep_infos_ego = ckpt_arr_and_ep_infos
                    new_ckpt_arr_conf = jax.tree.map(
                        lambda c_arr, p: c_arr.at[cidx].set(p),
                        ckpt_arr_conf, train_state_conf.params
                    )
                    new_ckpt_arr_br = jax.tree.map(
                        lambda c_arr, p: c_arr.at[cidx].set(p),
                        ckpt_arr_br, train_state_br.params
                    )

                    # run eval episodes
                    rng, eval_rng, = jax.random.split(rng)
                    # conf vs ego
                    last_ep_info_with_ego = run_episodes(eval_rng, env, 
                        agent_0_param=train_state_conf.params, agent_0_policy=confederate_policy,
                        agent_1_param=ego_params, agent_1_policy=ego_policy, 
                        max_episode_steps=config["ROLLOUT_LENGTH"], num_eps=config["NUM_EVAL_EPISODES"]
                    )
                    # conf vs br
                    last_ep_info_with_br = run_episodes(eval_rng, env, 
                        agent_0_param=train_state_br.params, agent_0_policy=br_policy,
                        agent_1_param=train_state_conf.params, agent_1_policy=confederate_policy, 
                        max_episode_steps=config["ROLLOUT_LENGTH"], num_eps=config["NUM_EVAL_EPISODES"]
                    )
                    
                    return ((new_ckpt_arr_conf, new_ckpt_arr_br, last_ep_info_with_br, last_ep_info_with_ego), rng, cidx + 1)

                def skip_ckpt(args):
                    return args
                rng, store_and_eval_rng = jax.random.split(rng, 2)
                (checkpoint_array_and_infos, store_and_eval_rng, ckpt_idx) = jax.lax.cond(
                    to_store, 
                    store_and_eval_ckpt, 
                    skip_ckpt, 
                    ((checkpoint_array_conf, checkpoint_array_br, eval_info_br, eval_info_ego), store_and_eval_rng, ckpt_idx)
                )
                checkpoint_array_conf, checkpoint_array_br, ep_info_br, ep_info_ego = checkpoint_array_and_infos
                
                metric["eval_ep_last_info_br"] = ep_info_br
                metric["eval_ep_last_info_ego"] = ep_info_ego

                return (new_update_runner_state,
                        checkpoint_array_conf, checkpoint_array_br, ckpt_idx, 
                        ep_info_br, ep_info_ego), metric

            # --------------------------
            # Init all variables for train loop
            # --------------------------
            rng, reset_rng_xp, reset_rng_sp, reset_rng_xsp, reset_rng_sxp = jax.random.split(rng, 5)
            reset_rngs_xp = jax.random.split(reset_rng_xp, config["NUM_ENVS"])
            reset_rngs_sp = jax.random.split(reset_rng_sp, config["NUM_ENVS"])
            reset_rngs_xsp = jax.random.split(reset_rng_xsp, config["NUM_ENVS"])
            reset_rngs_sxp = jax.random.split(reset_rng_sxp, config["NUM_ENVS"])

            obsv_xp, env_state_xp = jax.vmap(env.reset, in_axes=(0,))(reset_rngs_xp)
            obsv_sp, env_state_sp = jax.vmap(env.reset, in_axes=(0,))(reset_rngs_sp)
            obsv_xsp, env_state_xsp = jax.vmap(env.reset, in_axes=(0,))(reset_rngs_xsp)
            obsv_sxp, env_state_sxp = jax.vmap(env.reset, in_axes=(0,))(reset_rngs_sxp)

            # Initialize hidden states
            init_ego_hstate = ego_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])
            init_conf_hstate = confederate_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])
            init_br_hstate = br_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])

            # init checkpoint array
            checkpoint_array_conf = init_ckpt_array(train_state_conf.params)
            checkpoint_array_br = init_ckpt_array(train_state_br.params)
            ckpt_idx = 0

            # initial ep_infos for scan over _update_step_with_ckpt
            rng, rng_eval_ego, rng_eval_br = jax.random.split(rng, 3)
            ep_infos_ego = run_episodes(rng_eval_ego, env, 
                agent_0_param=train_state_conf.params, agent_0_policy=confederate_policy,
                agent_1_param=ego_params, agent_1_policy=ego_policy, 
                max_episode_steps=config["ROLLOUT_LENGTH"], num_eps=config["NUM_EVAL_EPISODES"]
            )
            ep_infos_br = run_episodes(rng_eval_br, env, 
                agent_0_param=train_state_br.params, agent_0_policy=br_policy,
                agent_1_param=ego_params, agent_1_policy=ego_policy, 
                max_episode_steps=config["ROLLOUT_LENGTH"], num_eps=config["NUM_EVAL_EPISODES"])

            # Initialize done flags
            init_dones_xp = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}
            init_dones_sp = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}
            init_dones_xsp = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}
            init_dones_sxp = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}
            
            # Initialize update runner state
            rng, rng_update = jax.random.split(rng, 2)
            update_steps = 0

            update_runner_state = (
                train_state_conf, train_state_br, 
                env_state_xp, env_state_sp, env_state_xsp, env_state_sxp,
                obsv_xp, obsv_sp, obsv_xsp, obsv_sxp, 
                init_dones_xp, init_dones_sp, init_dones_xsp, init_dones_sxp,
                init_conf_hstate, init_ego_hstate, # hstates for conf-ego XP interaction
                init_conf_hstate, init_br_hstate, # hstates for conf-br SP interaction
                init_conf_hstate, init_ego_hstate, # hstates for conf-ego xsp interaction
                init_conf_hstate, init_br_hstate, # hstates for conf-br sxp interaction
                rng_update, update_steps
            )

            state_with_ckpt = (
                update_runner_state, checkpoint_array_conf, checkpoint_array_br, 
                ckpt_idx, ep_infos_br, ep_infos_ego
            )
            # run training
            state_with_ckpt, metrics = jax.lax.scan(
                _update_step_with_ckpt,
                state_with_ckpt,
                xs=None,
                length=config["NUM_UPDATES"]
            )
            (
                final_runner_state, checkpoint_array_conf, checkpoint_array_br, 
                final_ckpt_idx, last_ep_infos_br, last_ep_infos_ego
            ) = state_with_ckpt

            out = {
                "final_params_conf": final_runner_state[0].params,
                "final_params_br": final_runner_state[1].params,
                "checkpoints_conf": checkpoint_array_conf,
                "checkpoints_br": checkpoint_array_br,
                "metrics": metrics,  # shape (NUM_UPDATES, ...)
            }
            return out

        return train
    # ------------------------------
    # Actually run the adversarial teammate training
    # ------------------------------
    rngs = jax.random.split(partner_rng, config["PARTNER_POP_SIZE"])
    train_fn = jax.jit(jax.vmap(make_regret_maximizing_partner_train(config)))
    out = train_fn(rngs, conf_params, br_params)
    return out

def open_ended_training_step(carry, ego_policy, conf_policy, br_policy, partner_population, 
                             oe_config, ego_config,env):
    '''
    Train the ego agent against the regret-maximizing partners. 
    Note: Currently training fcp agent against **all** adversarial partner checkpoints
    '''
    prev_ego_params, prev_conf_params, prev_br_params, rng = carry
    rng, partner_rng, ego_rng, conf_init_rng, br_init_rng = jax.random.split(rng, 5)
    
    if oe_config["REINIT_CONF"]:
        init_rngs = jax.random.split(conf_init_rng, oe_config["PARTNER_POP_SIZE"])
        conf_params = jax.vmap(conf_policy.init_params)(init_rngs)
    else:
        conf_params = prev_conf_params

    if oe_config["REINIT_BR_TO_BR"]:
        init_rngs = jax.random.split(br_init_rng, oe_config["PARTNER_POP_SIZE"])
        br_params = jax.vmap(br_policy.init_params)(init_rngs)
    elif oe_config["REINIT_BR_TO_EGO"]:
        br_params = jax.tree.map(lambda x: x[jnp.newaxis, ...].repeat(oe_config["PARTNER_POP_SIZE"], axis=0), prev_ego_params)
    else:
        br_params = prev_br_params

    # Train partner agents with ego_policy
    train_out = train_regret_maximizing_partners(oe_config, env, 
                                                 ego_params=prev_ego_params, ego_policy=ego_policy, 
                                                 conf_params=conf_params, conf_policy=conf_policy, 
                                                 br_params=br_params, br_policy=br_policy,
                                                 partner_rng=partner_rng
                                                 )
        
    if oe_config["EGO_TEAMMATE"] == "final":
        train_partner_params = train_out["final_params_conf"]

    elif oe_config["EGO_TEAMMATE"] == "all":
        n_ckpts = oe_config["PARTNER_POP_SIZE"] * oe_config["NUM_CHECKPOINTS"]
        train_partner_params = jax.tree.map(
            lambda x: x.reshape((n_ckpts,) + x.shape[2:]), 
            train_out["checkpoints_conf"]
        )
        
    # Train ego agent using train_ppo_ego_agents
    ego_out = train_ppo_ego_agent(
        config=ego_config,
        env=env,
        train_rng=ego_rng,
        ego_policy=ego_policy,
        init_ego_params=prev_ego_params,
        n_ego_train_seeds=1,
        partner_population=partner_population,
        partner_params=train_partner_params
    )
    
    updated_ego_parameters = ego_out["final_params"]
    updated_conf_parameters = train_out["final_params_conf"]
    updated_br_parameters = train_out["final_params_br"]

    # remove initial dimension of 1, to ensure that input and output ego parameters have the same dimension
    updated_ego_parameters = jax.tree.map(lambda x: x.squeeze(axis=0), updated_ego_parameters)

    carry = (updated_ego_parameters, updated_conf_parameters, updated_br_parameters, rng)
    return carry, (train_out, ego_out)


def train_rotate_without_pop(rng, env, algorithm_config, ego_config):
    rng, init_ego_rng, init_conf_rng, init_br_rng, train_rng = jax.random.split(rng, 5)
    
    # initialize ego policy and config
    ego_policy, init_ego_params = initialize_s5_agent(algorithm_config, env, init_ego_rng)
    # ego_policy, init_ego_params = initialize_mlp_agent(algorithm_config, env, init_ego_rng)
    
    # initialize PARTNER_POP_SIZE conf and br params
    conf_policy = ActorWithDoubleCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=env.observation_space(env.agents[0]).shape[0],
    )
    init_conf_rngs = jax.random.split(init_conf_rng, algorithm_config["PARTNER_POP_SIZE"])
    init_conf_params = jax.vmap(conf_policy.init_params)(init_conf_rngs)

    assert not (algorithm_config["REINIT_BR_TO_EGO"] and algorithm_config["REINIT_BR_TO_BR"]), "Cannot reinitialize br to both ego and br"
    if algorithm_config["REINIT_BR_TO_EGO"]:
        # initialize br policy to have same architecture as ego policy
        # a bit hacky
        br_policy = S5ActorCriticPolicy(
            action_dim=env.action_space(env.agents[0]).n,
            obs_dim=env.observation_space(env.agents[0]).shape[0],
            d_model=algorithm_config.get("S5_D_MODEL", 128),
            ssm_size=algorithm_config.get("S5_SSM_SIZE", 128),
            n_layers=algorithm_config.get("S5_N_LAYERS", 2),
            blocks=algorithm_config.get("S5_BLOCKS", 1),
            fc_hidden_dim=algorithm_config.get("S5_ACTOR_CRITIC_HIDDEN_DIM", 1024),
            fc_n_layers=algorithm_config.get("FC_N_LAYERS", 3),
            s5_activation=algorithm_config.get("S5_ACTIVATION", "full_glu"),
            s5_do_norm=algorithm_config.get("S5_DO_NORM", True),
            s5_prenorm=algorithm_config.get("S5_PRENORM", True),
            s5_do_gtrxl_norm=algorithm_config.get("S5_DO_GTRXL_NORM", True),
        )
    else:
        br_policy = MLPActorCriticPolicy(
            action_dim=env.action_space(env.agents[0]).n,
            obs_dim=env.observation_space(env.agents[0]).shape[0],
        )

    init_br_rngs = jax.random.split(init_br_rng, algorithm_config["PARTNER_POP_SIZE"])
    init_br_params = jax.vmap(br_policy.init_params)(init_br_rngs)

    # Create partner population
    if algorithm_config["EGO_TEAMMATE"] == "all":
        pop_size = algorithm_config["PARTNER_POP_SIZE"] * algorithm_config["NUM_CHECKPOINTS"]
    elif algorithm_config["EGO_TEAMMATE"] == "final":
        pop_size = algorithm_config["PARTNER_POP_SIZE"]
    else:
        raise ValueError(f"Invalid value for EGO_TEAMMATE: {algorithm_config['EGO_TEAMMATE']}")
    
    partner_population = AgentPopulation(
        pop_size=pop_size,
        policy_cls=conf_policy
    )

    @jax.jit
    def open_ended_step_fn(carry, unused):
        return open_ended_training_step(carry, ego_policy, conf_policy, br_policy, 
                                        partner_population, algorithm_config, ego_config, env)
    
    init_carry = (init_ego_params, init_conf_params, init_br_params, train_rng)
    final_carry, outs = jax.lax.scan(
        open_ended_step_fn, 
        init_carry, 
        xs=None,
        length=algorithm_config["NUM_OPEN_ENDED_ITERS"]
    )
    return outs

def run_rotate_without_pop(config, wandb_logger):
    algorithm_config = dict(config["algorithm"])

    # Create only one environment instance
    env = make_env(algorithm_config["ENV_NAME"], algorithm_config["ENV_KWARGS"])
    env = LogWrapper(env)
    
    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"])
    rng, init_ego_rng = jax.random.split(rng)
    rngs = jax.random.split(rng, algorithm_config["NUM_SEEDS"])

    # initialize ego config
    ego_config = copy.deepcopy(algorithm_config)
    ego_config["TOTAL_TIMESTEPS"] = algorithm_config["TIMESTEPS_PER_ITER_EGO"]
    EGO_ARGS = algorithm_config.get("EGO_ARGS", {})
    ego_config.update(EGO_ARGS)

    log.info("Starting ROTATE w/o population buffer training...")
    start_time = time.time()

    DEBUG = False
    with jax.disable_jit(DEBUG):
        train_fn = jax.jit(jax.vmap(partial(train_rotate_without_pop, 
                env=env, algorithm_config=algorithm_config, ego_config=ego_config
                )
            )
        )
        outs = train_fn(rngs)
    
    end_time = time.time()
    log.info(f"ROTATE w/o population buffer training completed in {end_time - start_time} seconds.")

    # Log metrics
    metric_names = get_metric_names(algorithm_config["ENV_NAME"])
    log_metrics(config, wandb_logger, outs, metric_names)

    # Prepare return values for heldout evaluation
    _ , ego_outs = outs
    ego_params = jax.tree.map(lambda x: x[:, :, 0], ego_outs["final_params"]) # shape (num_seeds, num_open_ended_iters, 1, num_ckpts, leaf_dim)
    ego_policy, init_ego_params = initialize_s5_agent(algorithm_config, env, init_ego_rng)

    return ego_policy, ego_params, init_ego_params


def log_metrics(config, logger, outs, metric_names: tuple):
    """Process training metrics and log them using the provided logger.
    
    Args:
        config: dict, the configuration
        outs: tuple, contains (teammate_outs, ego_outs) for each iteration
        logger: Logger, instance to log metrics
        metric_names: tuple, names of metrics to extract from training logs
    """
    teammate_outs, ego_outs = outs
    teammate_metrics = teammate_outs["metrics"]
    ego_metrics = ego_outs["metrics"]

    num_seeds, num_open_ended_iters, _, num_ego_updates = ego_metrics["returned_episode_returns"].shape[:4]
    num_partner_updates = teammate_metrics["returned_episode_returns"].shape[3]

    ### Process/extract PAIRED-specific losses    
    # Conf vs ego, conf vs br, br losses
    # shape (num_seeds, num_open_ended_iters, num_partner_seeds, num_updates, num_eval_episodes, num_agents_per_env)
    teammate_mean_dims = (0, 2, 4, 5)
    avg_teammate_sp_returns = np.asarray(teammate_metrics["eval_ep_last_info_br"]["returned_episode_returns"]).mean(axis=teammate_mean_dims)
    avg_teammate_xp_returns = np.asarray(teammate_metrics["eval_ep_last_info_ego"]["returned_episode_returns"]).mean(axis=teammate_mean_dims)

    #  shape (num_open_ended_iters, num_partner_seeds, num_partner_updates, update_epochs, num_minibatches)
    avg_value_losses_teammate_against_ego = np.asarray(teammate_metrics["value_loss_conf_against_ego"]).mean(axis=teammate_mean_dims)
    avg_value_losses_teammate_against_br = np.asarray(teammate_metrics["value_loss_conf_against_br"]).mean(axis=teammate_mean_dims) 
    avg_value_losses_br = np.asarray(teammate_metrics["value_loss_br"]).mean(axis=teammate_mean_dims)
    
    avg_actor_losses_teammate_against_ego = np.asarray(teammate_metrics["pg_loss_conf_against_ego"]).mean(axis=teammate_mean_dims) 
    avg_actor_losses_teammate_against_br = np.asarray(teammate_metrics["pg_loss_conf_against_br"]).mean(axis=teammate_mean_dims)
    avg_actor_losses_br = np.asarray(teammate_metrics["pg_loss_br"]).mean(axis=teammate_mean_dims)
    
    avg_entropy_losses_teammate_against_ego = np.asarray(teammate_metrics["entropy_conf_against_ego"]).mean(axis=teammate_mean_dims)
    avg_entropy_losses_teammate_against_br = np.asarray(teammate_metrics["entropy_conf_against_br"]).mean(axis=teammate_mean_dims)
    avg_entropy_losses_br = np.asarray(teammate_metrics["entropy_loss_br"]).mean(axis=teammate_mean_dims)
    
    # shape (num_seeds, num_open_ended_iters, num_partner_seeds, num_partner_updates)
    avg_rewards_teammate_against_br = np.asarray(teammate_metrics["average_rewards_br"]).mean(axis=(0, 2))
    avg_rewards_teammate_against_ego = np.asarray(teammate_metrics["average_rewards_ego"]).mean(axis=(0, 2))
    
    # Process ego-specific metrics
    # shape (num_seeds, num_open_ended_iters, num_ego_seeds, num_ego_updates, num_partners, num_eval_episodes, num_agents_per_env)
    avg_ego_returns = np.asarray(ego_metrics["eval_ep_last_info"]["returned_episode_returns"]).mean(axis=(0, 2, 4, 5, 6))
    # shape (num_seeds, num_open_ended_iters, num_ego_seeds, num_ego_updates, update_epochs, num_minibatches)
    avg_ego_value_losses = np.asarray(ego_metrics["value_loss"]).mean(axis=(0, 2, 4, 5))
    avg_ego_actor_losses = np.asarray(ego_metrics["actor_loss"]).mean(axis=(0, 2, 4, 5))
    avg_ego_entropy_losses = np.asarray(ego_metrics["entropy_loss"]).mean(axis=(0, 2, 4, 5))
    avg_ego_grad_norms = np.asarray(ego_metrics["avg_grad_norm"]).mean(axis=(0, 2, 4, 5))
    
    # extract teammate-vs-ego stats 
    teammate_stats = get_stats(teammate_metrics, metric_names) # shape (num_seeds, num_open_ended_iters, num_partner_seeds, num_partner_updates, 2)
    teammate_stat_means = jax.tree.map(lambda x: np.mean(x, axis=(0, 2))[..., 0], teammate_stats) # shape (num_open_ended_iters, num_partner_updates)
    
    # extract ego stats
    ego_stats = get_stats(ego_metrics, metric_names) # shape (num_seeds, num_open_ended_iters, num_ego_seeds, num_ego_updates, 2)
    ego_stat_means = jax.tree.map(lambda x: np.mean(x, axis=(0, 2))[..., 0], ego_stats) # shape (num_open_ended_iters, num_ego_updates)

    for iter_idx in range(num_open_ended_iters):        
        # Log all partner metrics
        for step in range(num_partner_updates):
            global_step = iter_idx * num_partner_updates + step
            
            # Log standard partner stats from get_stats
            for stat_name, stat_data in teammate_stat_means.items():
                logger.log_item(f"Train/Conf-Against-Ego_{stat_name}", stat_data[iter_idx, step], train_step=global_step)
            
            # Log paired-specific metrics
            # Eval metrics
            logger.log_item("Eval/ConfReturn-Against-Ego", avg_teammate_xp_returns[iter_idx][step], train_step=global_step)
            logger.log_item("Eval/ConfReturn-Against-BR", avg_teammate_sp_returns[iter_idx][step], train_step=global_step)
            logger.log_item("Eval/EgoRegret", avg_teammate_sp_returns[iter_idx][step] - avg_teammate_xp_returns[iter_idx][step], train_step=global_step)
            
            # Confederate losses
            logger.log_item("Losses/ConfValLoss-Against-Ego", avg_value_losses_teammate_against_ego[iter_idx][step], train_step=global_step)
            logger.log_item("Losses/ConfActorLoss-Against-Ego", avg_actor_losses_teammate_against_ego[iter_idx][step], train_step=global_step)
            logger.log_item("Losses/ConfEntropy-Against-Ego", avg_entropy_losses_teammate_against_ego[iter_idx][step], train_step=global_step)
            logger.log_item("Losses/ConfValLoss-Against-BR", avg_value_losses_teammate_against_br[iter_idx][step], train_step=global_step)
            logger.log_item("Losses/ConfActorLoss-Against-BR", avg_actor_losses_teammate_against_br[iter_idx][step], train_step=global_step)
            logger.log_item("Losses/ConfEntropy-Against-BR", avg_entropy_losses_teammate_against_br[iter_idx][step], train_step=global_step)
            
            # Best response losses
            logger.log_item("Losses/BRValLoss", avg_value_losses_br[iter_idx][step], train_step=global_step)
            logger.log_item("Losses/BRActorLoss", avg_actor_losses_br[iter_idx][step], train_step=global_step)
            logger.log_item("Losses/BREntropyLoss", avg_entropy_losses_br[iter_idx][step], train_step=global_step)
        
            # Rewards
            logger.log_item("Losses/AvgConfEgoRewards", avg_rewards_teammate_against_ego[iter_idx][step], train_step=global_step)
            logger.log_item("Losses/AvgConfBRRewards", avg_rewards_teammate_against_br[iter_idx][step], train_step=global_step)

        ### Ego metrics processing
        for step in range(num_ego_updates):
            global_step = iter_idx * num_ego_updates + step
 
            # Standard ego stats from get_stats
            for stat_name, stat_data in ego_stat_means.items():
                logger.log_item(f"Train/Ego_{stat_name}", stat_data[iter_idx, step], train_step=global_step)

            # Ego eval metrics
            logger.log_item("Eval/EgoReturn-Against-Conf", avg_ego_returns[iter_idx][step], train_step=global_step)

            # Ego agent losses
            logger.log_item("Losses/EgoValueLoss", avg_ego_value_losses[iter_idx][step], train_step=global_step)
            logger.log_item("Losses/EgoActorLoss", avg_ego_actor_losses[iter_idx][step], train_step=global_step)
            logger.log_item("Losses/EgoEntropyLoss", avg_ego_entropy_losses[iter_idx][step], train_step=global_step)
            logger.log_item("Losses/EgoGradNorm", avg_ego_grad_norms[iter_idx][step], train_step=global_step)
            
    logger.commit()

    # Saving artifacts
    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    out_savepath = save_train_run(outs, savedir, savename="saved_train_run")
    if config["logger"]["log_train_out"]:
        logger.log_artifact(name="saved_train_run", path=out_savepath, type_name="train_run")
    
    # Cleanup locally logged out file
    if not config["local_logger"]["save_train_out"]:
        shutil.rmtree(out_savepath)
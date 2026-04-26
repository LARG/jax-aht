'''Implemented to be as faithful to the original PAIRED as possible.'''
import shutil
import time
import logging

import hydra
import jax
import jax.nn
import jax.numpy as jnp
import numpy as np
import optax
from typing import NamedTuple
from flax.training.train_state import TrainState

from agents.mlp_actor_critic_agent import ActorWithDoubleCriticPolicy, MLPActorCriticPolicy, ActorWithConditionalCriticPolicy
from agents.initialize_agents import initialize_s5_agent
from common.plot_utils import get_stats, get_metric_names
from common.save_load_utils import save_train_run
from common.run_episodes import run_episodes
from marl.ppo_utils import Transition, unbatchify, _create_minibatches
from envs import make_env
from envs.log_wrapper import LogWrapper

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    agent_onehot_id: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray
    episode_id: jnp.ndarray
    time_id: jnp.ndarray

class TransitionEgo(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray

def gather_params(partner_params_pytree, idx_vec):
    """
    partner_params_pytree: pytree with all partner params. Each leaf has shape (n_seeds, m_ckpts, ...).
    idx_vec: a vector of indices with shape (num_envs,) each in [0, n_seeds*m_ckpts).

    Return a new pytree where each leaf has shape (num_envs, ...). Each leaf has a sampled
    partner's parameters for each environment.
    """
    # We'll define a function that gathers from each leaf
    # where leaf has shape (n_seeds, m_ckpts, ...), we want [idx_vec[i]] for each i.
    # We'll vmap a slicing function.
    def gather_leaf(leaf):
        def slice_one(idx):
            return leaf[idx]  # shape (...)
        return jax.vmap(slice_one)(idx_vec)

    return jax.tree.map(gather_leaf, partner_params_pytree)

def train_trajedi_partners(config, env, partner_rng):
    '''
    Train regret-maximizing confederate/best-response pairs, and an ego agent.
    Return model checkpoints and metrics.
    '''
    def make_train(config):
        num_agents = env.num_agents
        assert num_agents == 2, "This code assumes the environment has exactly 2 agents."

        # Right now assume control of just 1 agent
        config["NUM_CONF_ACTORS"] = config["NUM_ENVS_CONFS"]
        config["NUM_EGO_ACTORS"] = config["NUM_ENVS_BR"]

        config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // (config["ROLLOUT_LENGTH"] * (2*config["NUM_ENVS_CONFS"]+config["NUM_ENVS_BR"]))
        assert config["NUM_CONF_ACTORS"] % config["NUM_MINIBATCHES"] == 0, "NUM_CONF_ACTORS must be divisible by NUM_MINIBATCHES"
        assert config["NUM_EGO_ACTORS"] % config["NUM_MINIBATCHES"] == 0, "NUM_EGO_ACTORS must be divisible by NUM_MINIBATCHES"
        assert config["NUM_CONF_ACTORS"] >= config["NUM_MINIBATCHES"], "NUM_CONTROLLED_ACTORS must be >= NUM_MINIBATCHES"
        assert config["NUM_EGO_ACTORS"] >= config["NUM_MINIBATCHES"], "NUM_EGO_ACTORS must be >= NUM_MINIBATCHES"

        def linear_schedule(count):
            frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
            return config["LR"] * frac

        def train(rng):
            # Initialize all three policies: ego, confederate, and best response
            rng, init_ego_rng, init_conf_rng = jax.random.split(rng, 3)
            all_conf_init_rngs = jax.random.split(init_conf_rng, config["PARTNER_POP_SIZE"])

            # Initialize ego agent policy
            ego_policy, init_ego_params = initialize_s5_agent(config, env, init_ego_rng)
            init_ego_hstate_xp = ego_policy.init_hstate(config["NUM_CONF_ACTORS"])
            init_ego_hstate_sp = ego_policy.init_hstate(config["NUM_EGO_ACTORS"])

            # Define optimizers for ego policy
            tx_ego = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule if config["ANNEAL_LR"] else config["LR"], eps=1e-5),
            )

            train_state_ego = TrainState.create(
                apply_fn=ego_policy.network.apply,
                params=init_ego_params,
                tx=tx_ego,
            )

            # Initialize confederate policy using ActorWithDoubleCriticPolicy
            confederate_policy = ActorWithDoubleCriticPolicy(
                action_dim=env.action_space(env.agents[0]).n,
                obs_dim=env.observation_space(env.agents[0]).shape[0],
            )
            init_conf_hstate = confederate_policy.init_hstate(config["NUM_CONF_ACTORS"])

            # Initialize parameters using the policy interfaces
            init_params_conf = confederate_policy.init_params(init_conf_rng)

            def init_train_states(rng_agents):
                def init_single_pop_member_optimizers(rng_agent):
                    init_params_conf = confederate_policy.init_params(rng_agent)
                    return init_params_conf

                init_all_networks_and_optimizers = jax.vmap(init_single_pop_member_optimizers)
                all_conf_params = init_all_networks_and_optimizers(rng_agents)

                # Define optimizers for both confederate and BR policy
                tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(learning_rate=linear_schedule if config["ANNEAL_LR"] else config["LR"],
                    eps=1e-5),
                )

                train_state_conf = TrainState.create(
                    apply_fn=confederate_policy.network.apply,
                    params=all_conf_params,
                    tx=tx,
                )

                return train_state_conf
            
            train_state_conf = init_train_states(all_conf_init_rngs)

            def forward_pass_conf(params, obs, done, avail_actions, hstate, rng):
                act, val, pi, new_hstate = confederate_policy.get_action_value_policy(
                    params=params,
                    obs=obs[jnp.newaxis, ...],
                    done=done[jnp.newaxis, ...],
                    avail_actions=avail_actions,
                    hstate=hstate,
                    rng=rng
                )
                return act, val, pi, new_hstate

            # Init DONE

            # --------------------------
            # 3b) Init envs and hidden states
            # --------------------------
            # rng, reset_rng_conf, reset_rng_br = jax.random.split(rng, 3)
            # reset_rngs_conf = jax.random.split(reset_rng_conf, config["NUM_ENVS"])
            # reset_rngs_br = jax.random.split(reset_rng_br, config["NUM_ENVS"])

            # --------------------------
            # 3c) Define env step
            # --------------------------
            def _env_step_confs_sp(runner_state, unused):
                """
                agent_0 = confederate, agent_1 = br
                Returns updated runner_state, and Transitions for agent_0 and agent_1
                """
                (
                    all_train_state_conf, last_conf_ids,
                    env_state, 
                    last_obs, last_done, last_conf_h_p1, last_conf_h_p2,
                    last_eps_counter, 
                    last_timestep_counter,
                    rng
                ) = runner_state
                rng, act0_rng_sp, act1_rng_sp, step_rng_sp, conf_sampling_rng = jax.random.split(rng, 5)

                # For done envs, resample both conf and brs
                needs_resample = last_done["__all__"]
                resampled_conf_ids = jax.random.randint(conf_sampling_rng, (config["NUM_CONF_ACTORS"],), 0, config["PARTNER_POP_SIZE"])
                identity_matrix = jnp.eye(config["PARTNER_POP_SIZE"])
                
                # Determine final indices based on whether resampling was needed for each env
                updated_conf_ids = jnp.where(
                    needs_resample,
                    resampled_conf_ids,     # Use newly sampled index if True
                    last_conf_ids           # Else, keep index from previous step
                )

                # Reset the hidden states for resampled conf and br if they are not None
                # WARNING: BRDiv was not tested with recurrent actors, so the code for if the hstate is not None may not work
                if (last_conf_h_p1 is not None) and (last_conf_h_p2 is not None):
                    updated_conf_h_p1 = jnp.where(
                        needs_resample,
                        init_conf_hstate,
                        last_conf_h_p1
                    )
                    updated_conf_h_p2 = jnp.where(
                        needs_resample,
                        init_conf_hstate,
                        last_conf_h_p2
                    )
                else:
                    updated_conf_h_p1 = last_conf_h_p1
                    updated_conf_h_p2 = last_conf_h_p2

                # Get the corresponding conf and br params
                updated_conf_params = gather_params(all_train_state_conf.params, updated_conf_ids)
                updated_conf_onehot_ids = identity_matrix[updated_conf_ids]

                # Get available actions for agent 0 from environment state
                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(avail_actions)
                avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
                avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

                # Agent_0 action
                act0_rng_sp = jax.random.split(act0_rng_sp, config["NUM_CONF_ACTORS"])
                act_0, (val_0, val_0_unused), pi_0, new_conf_h_p1 = jax.vmap(forward_pass_conf)(
                    updated_conf_params, last_obs["agent_0"], 
                    last_done["agent_0"], avail_actions_0, 
                    updated_conf_h_p1, act0_rng_sp
                )
                logp_0 = pi_0.log_prob(act_0)
                act_0, val_0, logp_0 = act_0.squeeze(), val_0.squeeze(), logp_0.squeeze()

                # Agent_1 action
                act1_rng_sp = jax.random.split(act1_rng_sp, config["NUM_CONF_ACTORS"])
                act_1, (val_1, val_1_unused), pi_1, new_conf_h_p2 = jax.vmap(forward_pass_conf)(
                    updated_conf_params, last_obs["agent_1"], 
                    last_done["agent_1"], avail_actions_1, 
                    updated_conf_h_p2, act1_rng_sp
                )
                logp_1 = pi_1.log_prob(act_1)
                act_1, val_1, logp_1 = act_1.squeeze(), val_1.squeeze(), logp_1.squeeze()

                # Combine actions into the env format
                combined_actions = jnp.concatenate([act_0, act_1], axis=0)
                env_act = unbatchify(combined_actions, env.agents, config["NUM_CONF_ACTORS"], num_agents)
                env_act = {k: v.flatten() for k, v in env_act.items()}

                # Step env
                step_rngs = jax.random.split(step_rng_sp, config["NUM_CONF_ACTORS"])
                obs_next, env_state_next, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0))(
                    step_rngs, env_state, env_act
                )

                # note that num_actors = num_envs * num_agents
                info_0 = jax.tree.map(lambda x: x[:, 0], info)
                info_1 = jax.tree.map(lambda x: x[:, 1], info)
                agent_0_rews = reward["agent_0"]
                agent_1_rews = reward["agent_1"]

                # Store agent_0 data in transition
                transition_0 = Transition(
                    done=done["agent_0"],
                    action=act_0,
                    value=val_0,
                    agent_onehot_id=updated_conf_onehot_ids,
                    reward=agent_0_rews,
                    log_prob=logp_0,
                    obs=last_obs["agent_0"],
                    info=info_0,
                    avail_actions=avail_actions_0,
                    episode_id=last_eps_counter,
                    time_id=last_timestep_counter
                )

                transition_1 = Transition(
                    done=done["agent_1"],
                    action=act_1,
                    value=val_1,
                    agent_onehot_id=updated_conf_onehot_ids,
                    reward=agent_1_rews,
                    log_prob=logp_1,
                    obs=last_obs["agent_1"],
                    info=info_1,
                    avail_actions=avail_actions_1,
                    episode_id=last_eps_counter,
                    time_id=last_timestep_counter
                )

                last_eps_counter = last_eps_counter + done["agent_0"].astype(int)
                last_timestep_counter = (1-done["agent_0"].astype(int)) * (last_timestep_counter+1)

                new_runner_state = (all_train_state_conf, updated_conf_ids, 
                                    env_state_next, obs_next, done, new_conf_h_p1, 
                                    new_conf_h_p2, last_eps_counter, last_timestep_counter, 
                                    rng)
                return new_runner_state, (transition_0, transition_1)

            def _env_step_confs_xp(runner_state, unused):
                """
                agent_0 = confederate, agent_1 = br
                Returns updated runner_state, and Transitions for agent_0 and agent_1
                """
                (
                    all_train_state_conf, train_state_ego, last_conf_ids,
                    env_state, last_obs, last_done, last_conf_h, last_ego_h,
                    rng
                ) = runner_state
                rng, act0_rng_xp, act1_rng_xp, step_rng_xp, conf_sampling_rng = jax.random.split(rng, 5)

                # For done envs, resample both conf and brs
                needs_resample = last_done["__all__"]
                resampled_conf_ids = jax.random.randint(conf_sampling_rng, (config["NUM_CONF_ACTORS"],), 0, config["PARTNER_POP_SIZE"])
                identity_matrix = jnp.eye(config["PARTNER_POP_SIZE"])
                
                # Determine final indices based on whether resampling was needed for each env
                updated_conf_ids = jnp.where(
                    needs_resample,
                    resampled_conf_ids,     # Use newly sampled index if True
                    last_conf_ids           # Else, keep index from previous step
                )

                # Reset the hidden states for resampled conf and br if they are not None
                # WARNING: BRDiv was not tested with recurrent actors, so the code for if the hstate is not None may not work
                if (last_conf_h is not None):
                    updated_conf_h = jnp.where(
                        needs_resample,
                        init_conf_hstate,
                        last_conf_h
                    )
                    updated_ego_h = jnp.where(
                        needs_resample,
                        init_ego_hstate_xp,
                        last_ego_h
                    )
                else:
                    updated_conf_h = last_conf_h
                    updated_ego_h = last_ego_h

                
                # Get the corresponding conf and br params
                updated_conf_params = gather_params(all_train_state_conf.params, updated_conf_ids)
                updated_conf_onehot_ids = identity_matrix[updated_conf_ids]

                # Get available actions for agent 0 from environment state
                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(avail_actions)
                avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
                avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

                # Agent_0 action
                act0_rng_xp = jax.random.split(act0_rng_xp, config["NUM_CONF_ACTORS"])
                act_0, (val_0_unused, val_0), pi_0, new_conf_h = jax.vmap(forward_pass_conf)(
                    updated_conf_params, last_obs["agent_0"],  
                    last_done["agent_0"], avail_actions_0,
                    updated_conf_h, act0_rng_xp
                )
                logp_0 = pi_0.log_prob(act_0)
                act_0, val_0, logp_0 = act_0.squeeze(), val_0.squeeze(), logp_0.squeeze()

                # Agent_1 action

                act1_rng_xp, _ = jax.random.split(act1_rng_xp, 2)
                act_1, val_1, pi_1, new_ego_h = ego_policy.get_action_value_policy(
                    params=train_state_ego.params,
                    obs=last_obs["agent_1"].reshape(1, config["NUM_CONF_ACTORS"], -1),
                    done=last_done["agent_1"].reshape(1, config["NUM_CONF_ACTORS"]),
                    avail_actions=jax.lax.stop_gradient(avail_actions_1),
                    hstate=updated_ego_h,
                    rng=act1_rng_xp
                )

                logp_1 = pi_1.log_prob(act_1)
                act_1, val_1, logp_1 = act_1.squeeze(), val_1.squeeze(), logp_1.squeeze()

                # Combine actions into the env format
                combined_actions = jnp.concatenate([act_0, act_1], axis=0)
                env_act = unbatchify(combined_actions, env.agents, config["NUM_CONF_ACTORS"], num_agents)
                env_act = {k: v.flatten() for k, v in env_act.items()}

                # Step env
                step_rngs = jax.random.split(step_rng_xp, config["NUM_CONF_ACTORS"])
                obs_next, env_state_next, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0))(
                    step_rngs, env_state, env_act
                )

                # note that num_actors = num_envs * num_agents
                info_0 = jax.tree.map(lambda x: x[:, 0], info)
                info_1 = jax.tree.map(lambda x: x[:, 1], info)
                agent_0_rews = reward["agent_0"]
                agent_1_rews = reward["agent_1"]

                # Store agent_0 data in transition
                transition_0 = Transition(
                    done=done["agent_0"],
                    action=act_0,
                    value=val_0,
                    agent_onehot_id=updated_conf_onehot_ids,
                    reward=agent_0_rews,
                    log_prob=logp_0,
                    obs=last_obs["agent_0"],
                    info=info_0,
                    avail_actions=avail_actions_0,
                    episode_id=jnp.zeros_like(done["agent_0"].astype(int)),
                    time_id=jnp.zeros_like(done["agent_0"].astype(int))

                )

                transition_1 = Transition(
                    done=done["agent_1"],
                    action=act_1,
                    value=val_1,
                    agent_onehot_id=updated_conf_onehot_ids,
                    reward=agent_1_rews,
                    log_prob=logp_1,
                    obs=last_obs["agent_1"],
                    info=info_1,
                    avail_actions=avail_actions_1,
                    episode_id=jnp.zeros_like(done["agent_1"].astype(int)),
                    time_id=jnp.zeros_like(done["agent_1"].astype(int))
                )

                new_runner_state = (all_train_state_conf, train_state_ego,
                                    updated_conf_ids, env_state_next, 
                                    obs_next, done, new_conf_h, 
                                    new_ego_h, rng)
                return new_runner_state, (transition_0, transition_1)

            def _env_step_br_sp(runner_state, unused):
                """
                agent_0 = confederate, agent_1 = br
                Returns updated runner_state, and Transitions for agent_0 and agent_1
                """
                (
                    train_state_ego, env_state_br_sp, last_obs_br_sp, 
                    last_done_br_sp, last_br_h_p1, last_br_h_p2, rng
                ) = runner_state
                rng, act0_rng_br_sp, act1_rng_br_sp, step_rng_sp = jax.random.split(rng, 4)

                # For done envs, resample both conf and brs
                needs_resample = last_done_br_sp["__all__"]
                
                # Reset the hidden states for resampled conf and br if they are not None
                # WARNING: BRDiv was not tested with recurrent actors, so the code for if the hstate is not None may not work
                updated_br_h_p1 = last_br_h_p1
                updated_br_h_p2 = last_br_h_p2

                # Get available actions for agent 0 from environment state
                avail_actions = jax.vmap(env.get_avail_actions)(env_state_br_sp.env_state)
                avail_actions = jax.lax.stop_gradient(avail_actions)
                avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
                avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

                # Agent_0 action
                act0_rng_br_sp, _ = jax.random.split(act0_rng_br_sp, 2)
                act_0, val_0, pi_0, new_br_h_p1 = ego_policy.get_action_value_policy(
                    params=train_state_ego.params,
                    obs=last_obs_br_sp["agent_0"].reshape(1, config["NUM_EGO_ACTORS"], -1),
                    done=last_done_br_sp["agent_0"].reshape(1, config["NUM_EGO_ACTORS"]),
                    avail_actions=jax.lax.stop_gradient(avail_actions_0),
                    hstate=updated_br_h_p1,
                    rng=act0_rng_br_sp
                )
                logp_0 = pi_0.log_prob(act_0)
                act_0, val_0, logp_0 = act_0.squeeze(), val_0.squeeze(), logp_0.squeeze()

                # Agent_1 action
                act1_rng_br_sp, _ = jax.random.split(act1_rng_br_sp, 2)
                act_1, val_1, pi_1, new_br_h_p2 = ego_policy.get_action_value_policy(
                    params=train_state_ego.params,
                    obs=last_obs_br_sp["agent_1"].reshape(1, config["NUM_EGO_ACTORS"], -1),
                    done=last_done_br_sp["agent_1"].reshape(1, config["NUM_EGO_ACTORS"]),
                    avail_actions=jax.lax.stop_gradient(avail_actions_1),
                    hstate=updated_br_h_p2,
                    rng=act1_rng_br_sp
                )
                logp_1 = pi_1.log_prob(act_1)
                act_1, val_1, logp_1 = act_1.squeeze(), val_1.squeeze(), logp_1.squeeze()

                # Combine actions into the env format
                combined_actions = jnp.concatenate([act_0, act_1], axis=0)
                env_act = unbatchify(combined_actions, env.agents, config["NUM_EGO_ACTORS"], num_agents)
                env_act = {k: v.flatten() for k, v in env_act.items()}

                # Step env
                step_rngs = jax.random.split(step_rng_sp, config["NUM_EGO_ACTORS"])
                obs_next, env_state_next, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0))(
                    step_rngs, env_state_br_sp, env_act
                )

                # note that num_actors = num_envs * num_agents
                info_0 = jax.tree.map(lambda x: x[:, 0], info)
                info_1 = jax.tree.map(lambda x: x[:, 1], info)
                agent_0_rews = reward["agent_0"]
                agent_1_rews = reward["agent_1"]

                # Store agent_0 data in transition
                transition_0 = TransitionEgo(
                    done=done["agent_0"],
                    action=act_0,
                    value=val_0,
                    reward=agent_0_rews,
                    log_prob=logp_0,
                    obs=last_obs_br_sp["agent_0"],
                    info=info_0,
                    avail_actions=avail_actions_0
                )

                transition_1 = TransitionEgo(
                    done=done["agent_1"],
                    action=act_1,
                    value=val_1,
                    reward=agent_1_rews,
                    log_prob=logp_1,
                    obs=last_obs_br_sp["agent_1"],
                    info=info_1,
                    avail_actions=avail_actions_1
                )

                (
                    train_state_ego, env_state_br_sp, last_obs_br_sp, 
                    last_done_br_sp, last_br_h_p1, last_br_h_p2, rng
                ) = runner_state
                new_runner_state = (train_state_ego, env_state_next, 
                                    obs_next, done, 
                                    new_br_h_p1, new_br_h_p2, rng)
                return new_runner_state, (transition_0, transition_1)
            
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
                def _update_minbatch_conf(train_state_conf, batch_infos):
                    minbatch_conf_xp, minbatch_conf_sp1, minbatch_conf_sp2 = batch_infos
                    init_hstate_conf_xp, traj_batch_conf_xp, gae_conf_xp, target_v_conf_xp = minbatch_conf_xp   
                    init_hstate_conf_sp1, traj_batch_conf_sp1, gae_conf_sp1, target_v_conf_sp1 = minbatch_conf_sp1
                    init_hstate_conf_sp2, traj_batch_conf_sp2, gae_conf_sp2, target_v_conf_sp2 = minbatch_conf_sp2

                    def _loss_fn_policy(
                        params, init_hstate_conf_sp1, traj_batch_conf_sp1, 
                        gae_conf_sp1, target_v_conf_sp1, 
                        init_hstate_conf_sp2, traj_batch_conf_sp2, 
                        gae_conf_sp2, target_v_conf_sp2,
                        init_hstate_conf_xp, 
                        traj_batch_conf_xp, gae_conf_xp, target_v_conf_xp,
                        agent_id
                    ):
                        # get policy and value of confederate versus ego and best response agents respectively

                        is_relevant_sp1 = jnp.equal(
                            jnp.argmax(traj_batch_conf_sp1.agent_onehot_id, axis=-1),
                            agent_id
                        )
                        loss_weights_sp1 = jnp.where(is_relevant_sp1, 1, 0).astype(jnp.float32)

                        is_relevant_sp2 = jnp.equal(
                            jnp.argmax(traj_batch_conf_sp2.agent_onehot_id, axis=-1),
                            agent_id
                        )
                        loss_weights_sp2 = jnp.where(is_relevant_sp2, 1, 0).astype(jnp.float32)

                        is_relevant_xp = jnp.equal(
                            jnp.argmax(traj_batch_conf_xp.agent_onehot_id, axis=-1),
                            agent_id
                        )
                        loss_weights_xp = jnp.where(is_relevant_xp, 1, 0).astype(jnp.float32)

                        def compute_mean_weighted_losses(loss_tensor, weights):
                            return jax.lax.cond(
                                weights.sum() == 0,
                                lambda x: jnp.zeros_like(x).astype(jnp.float32),
                                lambda x: x,
                                (
                                    weights*loss_tensor
                                ).sum()/(weights.sum() + 1e-8)
                            )

                        def _compute_indiv_pol_sp_log_probs_and_vals_and_entropy(agent_id):                    
                            param = gather_params(params, agent_id)
                            _, (vals_sp1, _), pi_conf_sp1, _ = confederate_policy.get_action_value_policy(
                                params=jax.tree.map(lambda x: jnp.squeeze(x, axis=0), param),
                                obs=traj_batch_conf_sp1.obs,
                                done=traj_batch_conf_sp1.done,
                                avail_actions=traj_batch_conf_sp1.avail_actions,
                                hstate=init_hstate_conf_sp1,
                                rng=jax.random.PRNGKey(0) # only used for action sampling, which is not used here
                            )
                            _, (vals_sp2, _), pi_conf_sp2, _ = confederate_policy.get_action_value_policy(
                                params=jax.tree.map(lambda x: jnp.squeeze(x, axis=0), param),
                                obs=traj_batch_conf_sp2.obs,
                                done=traj_batch_conf_sp2.done,
                                avail_actions=traj_batch_conf_sp2.avail_actions,
                                hstate=init_hstate_conf_sp2,
                                rng=jax.random.PRNGKey(0) # only used for action sampling, which is not used here
                            )

                            log_prob_conf_sp1 = pi_conf_sp1.log_prob(traj_batch_conf_sp1.action)
                            log_prob_conf_sp2 = pi_conf_sp2.log_prob(traj_batch_conf_sp2.action)

                            sp1_entropy = pi_conf_sp1.entropy()
                            sp2_entropy = pi_conf_sp2.entropy()

                            return log_prob_conf_sp1, log_prob_conf_sp2, vals_sp1, vals_sp2, sp1_entropy, sp2_entropy
                        
                        def _compute_indiv_pol_xp_log_probs_and_vals_and_entropy(agent_id):                    
                            param = gather_params(params, agent_id)
                            _, (_, value_conf_xp), pi_conf_xp, _ = confederate_policy.get_action_value_policy(
                                params=jax.tree.map(lambda x: jnp.squeeze(x, axis=0), param),
                                obs=traj_batch_conf_xp.obs,
                                done=traj_batch_conf_xp.done,
                                avail_actions=traj_batch_conf_xp.avail_actions,
                                hstate=init_hstate_conf_xp,
                                rng=jax.random.PRNGKey(0) # only used for action sampling, which is not used here
                            )
                            log_prob_conf_xp = pi_conf_xp.log_prob(traj_batch_conf_xp.action)
                            all_xp_entropy = pi_conf_xp.entropy()

                            return log_prob_conf_xp, value_conf_xp, all_xp_entropy
                        
                        possible_agent_ids = jnp.expand_dims(jnp.arange(config["PARTNER_POP_SIZE"]), 1)
                        all_sp_log_probs_sp1, all_sp_log_probs_sp2, all_vals_sp1, all_vals_sp2, all_sp1_entropy, all_sp2_entropy = jax.vmap(_compute_indiv_pol_sp_log_probs_and_vals_and_entropy)(possible_agent_ids)
                        all_xp_log_probs, all_vals_xp, all_xp_entropy = jax.vmap(_compute_indiv_pol_xp_log_probs_and_vals_and_entropy)(possible_agent_ids)

                        # pop x time x batch
                        selected_indices_sp1 = jnp.expand_dims(
                            jnp.argmax(traj_batch_conf_sp1.agent_onehot_id, axis=-1), axis=0
                        )
                        selected_indices_sp2 = jnp.expand_dims(
                            jnp.argmax(traj_batch_conf_sp2.agent_onehot_id, axis=-1), axis=0
                        )
                        selected_indices_xp = jnp.expand_dims(
                            jnp.argmax(traj_batch_conf_xp.agent_onehot_id, axis=-1), axis=0
                        )

                        selected_log_probs_sp1 = jnp.squeeze(
                            jnp.take_along_axis(all_sp_log_probs_sp1, selected_indices_sp1, axis=0), axis=0
                        )
                        selected_log_probs_sp2 = jnp.squeeze(
                            jnp.take_along_axis(all_sp_log_probs_sp2, selected_indices_sp2, axis=0), axis=0
                        )
                        selected_log_probs_xp = jnp.squeeze(
                            jnp.take_along_axis(all_xp_log_probs, selected_indices_xp, axis=0), axis=0
                        )

                        log_prob_conf_sp1 = selected_log_probs_sp1
                        log_prob_conf_sp2 = selected_log_probs_sp2
                        log_prob_conf_xp = selected_log_probs_xp

                        value_conf_sp1 = jnp.squeeze(
                            jnp.take_along_axis(all_vals_sp1, selected_indices_sp1, axis=0), axis=0
                        )
                        value_conf_sp2 = jnp.squeeze(
                            jnp.take_along_axis(all_vals_sp2, selected_indices_sp2, axis=0), axis=0
                        )
                        value_conf_xp = jnp.squeeze(
                            jnp.take_along_axis(all_vals_xp, selected_indices_xp, axis=0), axis=0
                        )

                        entropy_conf_sp1 = jnp.squeeze(
                            jnp.take_along_axis(all_sp1_entropy, selected_indices_sp1, axis=0), axis=0
                        )
                        entropy_conf_sp2 = jnp.squeeze(
                            jnp.take_along_axis(all_sp2_entropy, selected_indices_sp2, axis=0), axis=0
                        )
                        entropy_conf_xp = jnp.squeeze(
                            jnp.take_along_axis(all_xp_entropy, selected_indices_xp, axis=0), axis=0
                        )

                        copied_episode_counters_sp1 = jnp.tile(traj_batch_conf_sp1.episode_id[None, ...], (jnp.shape(selected_log_probs_sp1)[0], 1, 1)) 
                        copied_episode_counters_sp2 = jnp.tile(traj_batch_conf_sp2.episode_id[None, ...], (jnp.shape(selected_log_probs_sp2)[0], 1, 1)) 
                        copied_timestep_counters_sp1 = jnp.tile(traj_batch_conf_sp1.time_id[None, ...], (jnp.shape(selected_log_probs_sp1)[0], 1, 1)) 
                        copied_timestep_counters_sp2 = jnp.tile(traj_batch_conf_sp2.time_id[None, ...], (jnp.shape(selected_log_probs_sp2)[0], 1, 1)) 
                        copied_all_sp_log_probs_sp1 = jnp.tile(all_sp_log_probs_sp1[None, ...], (jnp.shape(selected_log_probs_sp1)[0], 1, 1, 1))
                        copied_all_sp_log_probs_sp2 = jnp.tile(all_sp_log_probs_sp2[None, ...], (jnp.shape(selected_log_probs_sp2)[0], 1, 1, 1))
                        copied_selected_sp_log_probs_sp1 = jnp.tile(selected_log_probs_sp1[None, ...], (jnp.shape(selected_log_probs_sp1)[0], 1, 1))
                        copied_selected_sp_log_probs_sp2 = jnp.tile(selected_log_probs_sp2[None, ...], (jnp.shape(selected_log_probs_sp2)[0], 1, 1))
                        
                        def per_step_aggregate(
                                all_eps_id_sp1, all_eps_id_sp2, 
                                eps_id_sp1, eps_id_sp2,
                                all_time_id_sp1, all_time_id_sp2, 
                                time_id_sp1, time_id_sp2,
                                all_sp_log_probs_sp1, all_sp_log_probs_sp2,
                                selected_log_probs_sp1, selected_log_probs_sp2,
                                original_log_probs_sp1, original_log_probs_sp2
                        ):
                            is_relevant_weight_sp1 = (
                                all_eps_id_sp1 == jnp.tile(eps_id_sp1[None, ...], (jnp.shape(all_eps_id_sp1)[0], 1))
                            ).astype(int)

                            is_relevant_weight_sp2 = (
                                all_eps_id_sp2 == jnp.tile(eps_id_sp2[None, ...], (jnp.shape(all_eps_id_sp2)[0], 1))
                            ).astype(int)
                            
                            time_diff_sp1 = jnp.tile(time_id_sp1[None, ...], (jnp.shape(all_time_id_sp1)[0], 1)) - all_time_id_sp1
                            time_diff_sp2 = jnp.tile(time_id_sp2[None, ...], (jnp.shape(all_time_id_sp2)[0], 1)) - all_time_id_sp2

                            abs_diff_time_id_sp1 = jnp.abs(time_diff_sp1)
                            abs_diff_time_id_sp2 = jnp.abs(time_diff_sp2)
                            log_mult1_weight = is_relevant_weight_sp1*(config["GAMMA"]**abs_diff_time_id_sp1)
                            log_mult2_weight = is_relevant_weight_sp2*(config["GAMMA"]**abs_diff_time_id_sp2)
                            
                            # Sum selected log prob on the timestep axis (i.e., 0)
                            summed_selected_log_probs_sp1 = jnp.sum(
                                selected_log_probs_sp1*is_relevant_weight_sp1,
                                axis=0
                            )

                            # Sum selected log prob on the timestep axis (i.e., 0)
                            summed_selected_log_probs_sp2 = jnp.sum(
                                selected_log_probs_sp2*is_relevant_weight_sp2,
                                axis=0
                            )

                            summed_original_log_probs_sp1 = jnp.sum(
                                original_log_probs_sp1*is_relevant_weight_sp1,
                                axis=0
                            )

                            # Sum selected log prob on the timestep axis (i.e., 0)
                            summed_original_log_probs_sp2 = jnp.sum(
                                original_log_probs_sp2*is_relevant_weight_sp2,
                                axis=0
                            )

                            log_delta_i_t_sp1 = jnp.sum(
                                selected_log_probs_sp1*log_mult1_weight,
                                axis=0
                            )

                            log_delta_i_t_sp2 = jnp.sum(
                                selected_log_probs_sp2*log_mult2_weight,
                                axis=0
                            )

                            # Compute mean policy trajectory log probs
                            copied_weight_sp1 = jnp.tile(is_relevant_weight_sp1[None, ...], (config["PARTNER_POP_SIZE"], 1, 1))
                            copied_weight_sp2 = jnp.tile(is_relevant_weight_sp2[None, ...], (config["PARTNER_POP_SIZE"], 1, 1))
                            copied_mult1_weight = jnp.tile(log_mult1_weight[None, ...], (config["PARTNER_POP_SIZE"], 1, 1))
                            copied_mult2_weight = jnp.tile(log_mult2_weight[None, ...], (config["PARTNER_POP_SIZE"], 1, 1))

                            def logmeanexp(inp_array, axis=0):
                                return jnp.log(jnp.mean(jnp.exp(inp_array), axis=axis))
                            
                            traj_log_prob_sp1 = jnp.sum(all_sp_log_probs_sp1*copied_weight_sp1, axis=1)
                            traj_log_prob_sp2 = jnp.sum(all_sp_log_probs_sp2*copied_weight_sp2, axis=1)
                            avg_pol_traj_log_prob_sp1 = logmeanexp(traj_log_prob_sp1, axis=0)
                            avg_pol_traj_log_prob_sp2 = logmeanexp(traj_log_prob_sp2, axis=0)

                            log_delta_hat_t_sp1 = logmeanexp(
                                jnp.sum(
                                    all_sp_log_probs_sp1*copied_mult1_weight,
                                    axis=1
                                ), axis=0
                            )

                            log_delta_hat_t_sp2 = logmeanexp(
                                jnp.sum(
                                    all_sp_log_probs_sp2*copied_mult2_weight,
                                    axis=1
                                ), axis=0
                            )

                            return summed_selected_log_probs_sp1, summed_selected_log_probs_sp2, summed_original_log_probs_sp1, summed_original_log_probs_sp2, avg_pol_traj_log_prob_sp1, avg_pol_traj_log_prob_sp2, log_delta_i_t_sp1, log_delta_i_t_sp2, log_delta_hat_t_sp1, log_delta_hat_t_sp2

                        log_traj_pi_sp1, log_traj_pi_sp2, log_traj_ori_pi_sp1, log_traj_ori_pi_sp2, log_traj_pi_hat_sp1, log_traj_pi_hat_sp2, log_delta_i_t_sp1, log_delta_i_t_sp2, log_delta_hat_t_sp1, log_delta_hat_t_sp2 = jax.vmap(per_step_aggregate)(
                            copied_episode_counters_sp1, copied_episode_counters_sp2,
                            traj_batch_conf_sp1.episode_id, traj_batch_conf_sp2.episode_id,
                            copied_timestep_counters_sp1, copied_timestep_counters_sp2,
                            traj_batch_conf_sp1.time_id, traj_batch_conf_sp2.time_id,
                            copied_all_sp_log_probs_sp1, copied_all_sp_log_probs_sp2,
                            copied_selected_sp_log_probs_sp1, copied_selected_sp_log_probs_sp2,
                            traj_batch_conf_sp1.log_prob, traj_batch_conf_sp2.log_prob
                        )

                        delta_hat_traj_sp1 = jnp.exp(log_delta_hat_t_sp1)
                        delta_hat_traj_sp2 = jnp.exp(log_delta_hat_t_sp2)
                        delta_i_traj_sp1 = jnp.exp(log_delta_i_t_sp1)
                        delta_i_traj_sp2 = jnp.exp(log_delta_i_t_sp2)
                        
                        pol_ratios_sp1 = jnp.exp(log_traj_pi_hat_sp1-log_traj_ori_pi_sp1)
                        pol_ratios_sp2 = jnp.exp(log_traj_pi_hat_sp2-log_traj_ori_pi_sp2)

                        pol_ratios2_sp1 = jnp.exp(log_traj_pi_sp1-log_traj_ori_pi_sp1)
                        pol_ratios2_sp2 = jnp.exp(log_traj_pi_sp2-log_traj_ori_pi_sp2)

                        pi_multiplier_sp1 = jax.lax.stop_gradient(pol_ratios2_sp1*(delta_hat_traj_sp1 - ((1.0/config["PARTNER_POP_SIZE"]) * log_delta_i_t_sp1)))
                        pi_multiplier_sp2 = jax.lax.stop_gradient(pol_ratios2_sp2*(delta_hat_traj_sp2 - ((1.0/config["PARTNER_POP_SIZE"]) * log_delta_i_t_sp2)))

                        delta_multiplier_sp1 = jax.lax.stop_gradient(pol_ratios_sp1 * delta_i_traj_sp1)
                        delta_multiplier_sp2 = jax.lax.stop_gradient(pol_ratios_sp2 * delta_i_traj_sp2)

                        trajedi_loss_sp1 = pi_multiplier_sp1 * log_traj_pi_sp1 + delta_multiplier_sp1 * log_delta_i_t_sp1
                        trajedi_loss_sp2 = pi_multiplier_sp2 * log_traj_pi_sp2 + delta_multiplier_sp2 * log_delta_i_t_sp2

                        trajedi_loss_sp1 = compute_mean_weighted_losses(trajedi_loss_sp1, loss_weights_sp1)
                        trajedi_loss_sp2 = compute_mean_weighted_losses(trajedi_loss_sp2, loss_weights_sp2)


                        # Value loss for interaction with ego agent
                        value_pred_conf_xp_clipped = traj_batch_conf_xp.value + (
                            value_conf_xp - traj_batch_conf_xp.value
                            ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses_conf_xp = jnp.square(value_conf_xp - target_v_conf_xp)
                        value_losses_clipped_conf_xp = jnp.square(value_pred_conf_xp_clipped - target_v_conf_xp)
                        value_loss_conf_xp = compute_mean_weighted_losses(
                            jnp.maximum(value_losses_conf_xp, value_losses_clipped_conf_xp),
                            loss_weights_xp
                        )

                        # Value loss for interaction with best response agent
                        value_pred_conf_sp1_clipped = traj_batch_conf_sp1.value + (
                            value_conf_sp1 - traj_batch_conf_sp1.value
                            ).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses_conf_sp1 = jnp.square(value_conf_sp1 - target_v_conf_sp1)
                        value_losses_clipped_conf_sp1 = jnp.square(value_pred_conf_sp1_clipped - target_v_conf_sp1)
                        value_loss_conf_sp1 = compute_mean_weighted_losses(
                            jnp.maximum(value_losses_conf_sp1, value_losses_clipped_conf_sp1),
                            loss_weights_sp1
                        )

                        value_pred_conf_sp2_clipped = traj_batch_conf_sp2.value + (
                            value_conf_sp2 - traj_batch_conf_sp2.value
                            ).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses_conf_sp2 = jnp.square(value_conf_sp2 - target_v_conf_sp2)
                        value_losses_clipped_conf_sp2 = jnp.square(value_pred_conf_sp2_clipped - target_v_conf_sp2)
                        value_loss_conf_sp2 = compute_mean_weighted_losses(
                            jnp.maximum(value_losses_conf_sp2, value_losses_clipped_conf_sp2),
                            loss_weights_sp2
                        )

                        # Policy gradient loss for interaction with ego agent
                        ratio_conf_xp = jnp.exp(log_prob_conf_xp - traj_batch_conf_xp.log_prob)
                        gae_norm_conf_xp = (gae_conf_xp - gae_conf_xp.mean()) / (gae_conf_xp.std() + 1e-8)
                        pg_loss_1_conf_xp = ratio_conf_xp * gae_norm_conf_xp
                        pg_loss_2_conf_xp = jnp.clip(
                            ratio_conf_xp,
                            1.0 - config["CLIP_EPS"],
                            1.0 + config["CLIP_EPS"]) * gae_norm_conf_xp
                        pg_loss_conf_xp = compute_mean_weighted_losses(
                            -jnp.minimum(pg_loss_1_conf_xp, pg_loss_2_conf_xp),
                            loss_weights_xp
                        )

                        # Policy gradient loss for interaction with best response agent
                        ratio_conf_sp1 = jnp.exp(log_prob_conf_sp1 - traj_batch_conf_sp1.log_prob)
                        gae_norm_conf_sp1 = (gae_conf_sp1 - gae_conf_sp1.mean()) / (gae_conf_sp1.std() + 1e-8)
                        pg_loss_1_conf_sp1 = ratio_conf_sp1 * gae_norm_conf_sp1
                        pg_loss_2_conf_sp1 = jnp.clip(
                            ratio_conf_sp1,
                            1.0 - config["CLIP_EPS"],
                            1.0 + config["CLIP_EPS"]) * gae_norm_conf_sp1
                        pg_loss_conf_sp1 = compute_mean_weighted_losses(
                            -jnp.minimum(pg_loss_1_conf_sp1, pg_loss_2_conf_sp1),
                            loss_weights_sp1
                        )

                        ratio_conf_sp2 = jnp.exp(log_prob_conf_sp2 - traj_batch_conf_sp2.log_prob)
                        gae_norm_conf_sp2 = (gae_conf_sp2 - gae_conf_sp2.mean()) / (gae_conf_sp2.std() + 1e-8)
                        pg_loss_1_conf_sp2 = ratio_conf_sp2 * gae_norm_conf_sp2
                        pg_loss_2_conf_sp2 = jnp.clip(
                            ratio_conf_sp2,
                            1.0 - config["CLIP_EPS"],
                            1.0 + config["CLIP_EPS"]) * gae_norm_conf_sp2
                        pg_loss_conf_sp2 = compute_mean_weighted_losses(
                            -jnp.minimum(pg_loss_1_conf_sp2, pg_loss_2_conf_sp2),
                            loss_weights_sp2
                        )

                        # Entropy for interaction with ego agent
                        entropy_conf_xp = compute_mean_weighted_losses(
                            entropy_conf_xp,
                            loss_weights_xp
                        )
                        # Entropy for interaction with best response agent
                        entropy_conf_sp1 = compute_mean_weighted_losses(
                            entropy_conf_sp1,
                            loss_weights_sp1
                        )
                        entropy_conf_sp2 = compute_mean_weighted_losses(
                            entropy_conf_sp1,
                            loss_weights_sp2
                        )
                        
                        # We negate the pg_loss_conf_ego to minimize the ego agent's objective
                        conf_xp_loss = pg_loss_conf_xp + config["VF_COEF"] * value_loss_conf_xp - config["ENT_COEF"] * entropy_conf_xp
                        conf_sp_loss = pg_loss_conf_sp1 + pg_loss_conf_sp2 + config["VF_COEF"] * (value_loss_conf_sp1 + value_loss_conf_sp2) - config["ENT_COEF"] * (entropy_conf_sp1 + entropy_conf_sp2) + config["TRAJEDI_COEF"] * (trajedi_loss_sp1 + trajedi_loss_sp2)
                        
                        total_loss = conf_xp_loss + conf_sp_loss

                        return total_loss, (value_loss_conf_xp, value_loss_conf_sp1+value_loss_conf_sp2, pg_loss_conf_xp, pg_loss_conf_sp1+pg_loss_conf_sp2, entropy_conf_xp, entropy_conf_sp1+entropy_conf_sp2, trajedi_loss_sp1+trajedi_loss_sp2)

                    grad_fn = jax.value_and_grad(_loss_fn_policy, has_aux=True)

                    def gather_conf_params_and_return_grads(agent_id):
                        (loss_val_conf, aux_vals_conf), grads_conf = grad_fn(
                            train_state_conf.params, 
                            init_hstate_conf_sp1, traj_batch_conf_sp1, gae_conf_sp1, target_v_conf_sp1, 
                            init_hstate_conf_sp2, traj_batch_conf_sp2, gae_conf_sp2, target_v_conf_sp2,
                            init_hstate_conf_xp, traj_batch_conf_xp, gae_conf_xp, target_v_conf_xp,
                            agent_id
                        )
                        return (loss_val_conf, aux_vals_conf), grads_conf

                    possible_agent_ids = jnp.expand_dims(jnp.arange(config["PARTNER_POP_SIZE"]), 1)
                    (loss_val_conf, aux_vals_conf), grads_conf = jax.vmap(gather_conf_params_and_return_grads)(possible_agent_ids)
                    #grads_conf_new = jax.tree.map(lambda x: jnp.squeeze(x, 1), grads_conf)
                    
                    train_state_conf = train_state_conf.apply_gradients(grads=jax.tree.map(lambda x: jnp.sum(x, axis=0), grads_conf))
                    return train_state_conf, (loss_val_conf, aux_vals_conf)

                def _update_minbatch_ego(train_state_ego, batch_info):
                    minibatches_ego_xp, minibatches_ego_sp1, minibatches_ego_sp2 = batch_info
                    init_hstate_ego_xp, traj_batch_ego_xp, advantages_xp, returns_xp = minibatches_ego_xp
                    init_hstate_ego_sp1, traj_batch_ego_sp1, advantages_sp1, returns_sp1 = minibatches_ego_sp1
                    init_hstate_ego_sp2, traj_batch_ego_sp2, advantages_sp2, returns_sp2 = minibatches_ego_sp2

                    def _loss_fn_ego(
                        params, init_hstate_ego_xp, traj_batch_ego_xp, advantages_xp, returns_xp,
                        init_hstate_ego_sp1, traj_batch_ego_sp1, advantages_sp1, returns_sp1,
                        init_hstate_ego_sp2, traj_batch_ego_sp2, advantages_sp2, returns_sp2
                    ):
                        def compute_single_minibatch_loss(init_hstate_ego, traj_batch_ego, gae, target_v):
                            _, value, pi, _ = ego_policy.get_action_value_policy(
                                params=params, # (64,)
                                obs=traj_batch_ego.obs, # (512, 15)
                                done=traj_batch_ego.done, # (512,)
                                avail_actions=traj_batch_ego.avail_actions, # (512, 6)
                                hstate=init_hstate_ego, # (1, 16, 8)
                                rng=jax.random.PRNGKey(0) # only used for action sampling, which is not used here
                            )
                            log_prob = pi.log_prob(traj_batch_ego.action)

                            # Value loss
                            value_pred_clipped = traj_batch_ego.value + (
                                value - traj_batch_ego.value
                                ).clip(
                                -config["CLIP_EPS"], config["CLIP_EPS"])
                            value_losses = jnp.square(value - target_v)
                            value_losses_clipped = jnp.square(value_pred_clipped - target_v)
                            value_loss = jnp.maximum(value_losses, value_losses_clipped).mean()

                            # Policy gradient loss
                            ratio = jnp.exp(log_prob - traj_batch_ego.log_prob)
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
                        
                        xp_total_loss, (xp_value_loss, xp_pg_loss, xp_entropy) = compute_single_minibatch_loss(
                            init_hstate_ego_xp, traj_batch_ego_xp, advantages_xp, returns_xp
                        )
                        sp1_total_loss, (sp1_value_loss, sp1_pg_loss, sp1_entropy) = compute_single_minibatch_loss(
                            init_hstate_ego_sp1, traj_batch_ego_sp1, advantages_sp1, returns_sp1
                        )
                        sp2_total_loss, (sp2_value_loss, sp2_pg_loss, sp2_entropy) = compute_single_minibatch_loss(
                            init_hstate_ego_sp2, traj_batch_ego_sp2, advantages_sp2, returns_sp2
                        )

                        return xp_total_loss+sp1_total_loss+sp2_total_loss, (
                            xp_value_loss, sp1_value_loss+sp2_value_loss,
                            xp_pg_loss, sp1_pg_loss+sp2_pg_loss,
                            xp_entropy, sp1_entropy+sp2_entropy
                        )

                    grad_fn = jax.value_and_grad(_loss_fn_ego, has_aux=True)
                    (loss_val, aux_vals), grads = grad_fn(
                        train_state_ego.params, 
                        init_hstate_ego_xp, traj_batch_ego_xp, advantages_xp, returns_xp,
                        init_hstate_ego_sp1, traj_batch_ego_sp1, advantages_sp1, returns_sp1,
                        init_hstate_ego_sp2, traj_batch_ego_sp2, advantages_sp2, returns_sp2,
                    )
                    train_state_ego = train_state_ego.apply_gradients(grads=grads)
                    return train_state_ego, (loss_val, aux_vals)

                # TODO Update this part of code to adjust to the types of interactions
                # used during training
                (
                    train_state_conf, train_state_ego,
                    traj_batch_conf_sp1, traj_batch_conf_sp2, traj_batch_conf_xp,
                    traj_batch_br_sp1, traj_batch_br_sp2, traj_batch_br_xp,
                    advantages_conf_sp1, advantages_conf_sp2, advantages_conf_xp,
                    advantages_br_sp1, advantages_br_sp2, advantages_br_xp,
                    targets_conf_sp1, targets_conf_sp2, targets_conf_xp,
                    targets_br_sp1, targets_br_sp2, targets_br_xp,
                    rng_conf, rng_br
                ) = update_state

                init_hstate_conf = confederate_policy.init_hstate(config["NUM_CONF_ACTORS"])
                init_hstate_br_xp = ego_policy.init_hstate(config["NUM_CONF_ACTORS"])
                init_hstate_br_sp = ego_policy.init_hstate(config["NUM_EGO_ACTORS"])

                rng_conf, perm_rng_conf_sp1, perm_rng_conf_sp2, perm_rng_conf_xp = jax.random.split(rng_conf, 4)
                rng_br, perm_rng_br_sp1, perm_rng_br_sp2, perm_rng_br_xp = jax.random.split(rng_br, 4)
                # Create minibatches for each agent and interaction type
                minibatches_conf_sp1 = _create_minibatches(
                    traj_batch_conf_sp1, advantages_conf_sp1, targets_conf_sp1, init_hstate_conf,
                    config["NUM_CONF_ACTORS"], config["NUM_MINIBATCHES"], perm_rng_conf_sp1
                )

                minibatches_conf_sp2 = _create_minibatches(
                    traj_batch_conf_sp2, advantages_conf_sp2, targets_conf_sp2, init_hstate_conf,
                    config["NUM_CONF_ACTORS"], config["NUM_MINIBATCHES"], perm_rng_conf_sp2
                )

                minibatches_conf_xp = _create_minibatches(
                    traj_batch_conf_xp, advantages_conf_xp, targets_conf_xp, init_hstate_conf,
                    config["NUM_CONF_ACTORS"], config["NUM_MINIBATCHES"], perm_rng_conf_xp
                )

                minibatches_br_sp1 = _create_minibatches(
                    traj_batch_br_sp1, advantages_br_sp1, targets_br_sp1, init_hstate_br_sp,
                    config["NUM_EGO_ACTORS"], config["NUM_MINIBATCHES"], perm_rng_br_sp1
                )

                minibatches_br_sp2 = _create_minibatches(
                    traj_batch_br_sp2, advantages_br_sp2, targets_br_sp2, init_hstate_br_sp,
                    config["NUM_EGO_ACTORS"], config["NUM_MINIBATCHES"], perm_rng_br_sp2
                )

                minibatches_br_xp = _create_minibatches(
                    traj_batch_br_xp, advantages_br_xp, targets_br_xp, init_hstate_br_xp,
                    config["NUM_CONF_ACTORS"], config["NUM_MINIBATCHES"], perm_rng_br_xp
                )

                # Update confederates
                train_state_conf, all_losses_conf = jax.lax.scan(
                    _update_minbatch_conf, train_state_conf, (minibatches_conf_xp, minibatches_conf_sp1, minibatches_conf_sp2)
                )

                # Update ego agent
                train_state_ego, all_losses_ego = jax.lax.scan(
                    _update_minbatch_ego, train_state_ego, (minibatches_br_xp, minibatches_br_sp1, minibatches_br_sp2)
                )

                update_state = (
                    train_state_conf, train_state_ego,
                    traj_batch_conf_sp1, traj_batch_conf_sp2, traj_batch_conf_xp,
                    traj_batch_br_sp1, traj_batch_br_sp2, traj_batch_br_xp,
                    advantages_conf_sp1, advantages_conf_sp2, advantages_conf_xp,
                    advantages_br_sp1, advantages_br_sp2, advantages_br_xp,
                    targets_conf_sp1, targets_conf_sp2, targets_conf_xp,
                    targets_br_sp1, targets_br_sp2, targets_br_xp,
                    rng_conf, rng_br
                )
                return update_state, (all_losses_conf, all_losses_ego)

            def _update_step(update_runner_state, unused):
                """
                1. Collect rollout for interactions against ego agent.
                2. Collect rollout for interactions against br agent.
                3. Compute advantages for ego-conf and conf-br interactions.
                4. PPO updates for best response and confederate policies.
                """
                (
                    all_train_state_conf, train_state_ego,
                    last_env_state_confs_sp, last_env_state_confs_xp, last_env_state_br_sp,
                    last_obs_confs_sp, last_obs_confs_xp, last_obs_br_sp, 
                    last_done_confs_sp, last_done_confs_xp, last_done_br_sp,
                    last_conf_h_sp_p1, last_conf_h_sp_p2, 
                    last_conf_h_xp, last_br_h_xp,
                    last_br_h_sp_p1, last_br_h_sp_p2,
                    last_eps_counter, last_timestep_counter,
                    rng_conf, rng_br, update_steps
                ) = update_runner_state

                rng_conf, conf_sampling_sp_rng, conf_sampling_xp_rng = jax.random.split(rng_conf, 3)
                conf_ids_sp = jax.random.randint(conf_sampling_sp_rng, (config["NUM_ENVS_CONFS"],), 0, config["PARTNER_POP_SIZE"])
                conf_ids_xp = jax.random.randint(conf_sampling_xp_rng, (config["NUM_ENVS_CONFS"],), 0, config["PARTNER_POP_SIZE"])

                # 1) rollout for self-play interactions between confederates
                runner_state_conf_sp = (
                    all_train_state_conf, conf_ids_sp,
                    last_env_state_confs_sp, last_obs_confs_sp, 
                    last_done_confs_sp, last_conf_h_sp_p1, 
                    last_conf_h_sp_p2, last_eps_counter, 
                    last_timestep_counter, rng_conf
                )

                runner_state_conf_sp, (traj_batch_conf_p0, traj_batch_conf_p1) = jax.lax.scan(
                    _env_step_confs_sp, runner_state_conf_sp, None, config["ROLLOUT_LENGTH"])
                (
                    all_train_state_conf, last_conf_ids, last_env_state_confs_sp, 
                    last_obs_confs_sp, last_done_confs_sp, last_conf_h_sp_p1, 
                    last_conf_h_sp_p2, last_eps_counter, last_timestep_counter, rng_conf
                ) = runner_state_conf_sp

                # 2) rollout for interactions of confederate against br agent
                runner_state_conf_xp = (
                    all_train_state_conf, train_state_ego, conf_ids_xp,
                    last_env_state_confs_xp, last_obs_confs_xp, 
                    last_done_confs_xp, last_conf_h_xp, last_br_h_xp,
                    rng_conf
                )
                runner_state_conf_xp, (traj_batch_conf_xp, traj_batch_br_xp) = jax.lax.scan(
                    _env_step_confs_xp, runner_state_conf_xp, None, config["ROLLOUT_LENGTH"])
                (
                    all_train_state_conf, train_state_ego, last_conf_ids_xp,
                    last_env_state_confs_xp, last_obs_confs_xp, last_done_confs_xp,
                    last_conf_h_xp, last_br_h_xp, rng_conf
                ) = runner_state_conf_xp

                # 3) rollout self-play interactions of br agent
                runner_state_br_sp = (
                    train_state_ego, last_env_state_br_sp, last_obs_br_sp, 
                    last_done_br_sp, last_br_h_sp_p1, last_br_h_sp_p2,
                    rng_br
                )
                runner_state_br_sp, (traj_batch_br_sp_p1, traj_batch_br_sp_p2) = jax.lax.scan(
                    _env_step_br_sp, runner_state_br_sp, None, config["ROLLOUT_LENGTH"])
                (
                    train_state_ego, last_env_state_br_sp, last_obs_br_sp, 
                    last_done_br_sp, last_br_h_sp_p1, last_br_h_sp_p2,
                    rng_br
                ) = runner_state_br_sp

                def _compute_advantages_and_targets_conf(batch_size, env_state, policy_params, policy_hstate,
                                                   last_obs, last_dones, last_conf_ids, traj_batch, 
                                                   agent_name, value_idx=None):
                    '''Value_idx argument is to support the ActorWithDoubleCritic (confederate) policy, which
                    has two value heads. Value head 0 models the ego agent while value head 1 models the best response.'''
                    avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)[agent_name].astype(jnp.float32)
                    rng_key = jax.random.PRNGKey(0)  # dummy key as we don't sample actions
                    rng_keys = jax.random.split(rng_key, last_obs[agent_name].shape[0])
                    specific_policy_params = gather_params(policy_params, last_conf_ids)
                    _, vals, _, _ = jax.vmap(forward_pass_conf)(
                        specific_policy_params, last_obs[agent_name], 
                        last_dones[agent_name], jax.lax.stop_gradient(avail_actions), 
                        policy_hstate, rng_keys  # dummy key as we don't sample actions
                    )
                
                    if value_idx is None:
                        last_val = vals.squeeze()
                    else:
                        last_val = vals[value_idx].squeeze()
                    advantages, targets = _calculate_gae(traj_batch, last_val)
                    return advantages, targets

                def _compute_advantages_and_targets_br(batch_size, env_state, policy_params, policy_hstate,
                                                   last_obs, last_dones, traj_batch, agent_name):
                    '''Value_idx argument is to support the ActorWithDoubleCritic (confederate) policy, which
                    has two value heads. Value head 0 models the ego agent while value head 1 models the best response.'''
                    avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)[agent_name].astype(jnp.float32)
                    _, vals, _, _ = ego_policy.get_action_value_policy(
                        params=policy_params,
                        obs=last_obs[agent_name].reshape(1, batch_size, -1),
                        done=last_dones[agent_name].reshape(1, batch_size),
                        avail_actions=jax.lax.stop_gradient(avail_actions),
                        hstate=policy_hstate,
                        rng=jax.random.PRNGKey(0)  # dummy key as we don't sample actions
                    )
                    last_val = vals.squeeze()
                    advantages, targets = _calculate_gae(traj_batch, last_val)
                    return advantages, targets


                # 4a) compute advantage for confederate agent from interaction with itself
                advantages_sp_conf_p1, targets_sp_conf_p1 = _compute_advantages_and_targets_conf(
                    config["NUM_CONF_ACTORS"],
                    last_env_state_confs_sp, all_train_state_conf.params, 
                    last_conf_h_sp_p1, last_obs_confs_sp, last_done_confs_sp, last_conf_ids,
                    traj_batch_conf_p0, "agent_0", value_idx=0
                )

                advantages_sp_conf_p2, targets_sp_conf_p2 = _compute_advantages_and_targets_conf(
                    config["NUM_CONF_ACTORS"],
                    last_env_state_confs_sp, all_train_state_conf.params, 
                    last_conf_h_sp_p2, last_obs_confs_sp, last_done_confs_sp, last_conf_ids, 
                    traj_batch_conf_p1, "agent_1", value_idx=0
                )

                # 4b) compute advantage for confederate agent from interaction with BR Policy
                advantages_xp_conf, targets_xp_conf = _compute_advantages_and_targets_conf(
                    config["NUM_CONF_ACTORS"],
                    last_env_state_confs_xp, all_train_state_conf.params, 
                    last_conf_h_xp, last_obs_confs_xp, last_done_confs_xp, last_conf_ids_xp,
                    traj_batch_conf_xp, "agent_0", value_idx=1
                )
            
                # 5a) compute advantage for ego agent from interaction with confederates
                advantages_xp_br, targets_xp_br = _compute_advantages_and_targets_br(
                    config["NUM_CONF_ACTORS"],
                    last_env_state_confs_xp, train_state_ego.params, 
                    last_br_h_xp, last_obs_confs_xp, last_done_confs_xp, 
                    traj_batch_br_xp, "agent_1"
                )

                # 5b) compute advantage for ego agent from interaction with itself
                advantages_sp_br_p1, targets_sp_br_p1 = _compute_advantages_and_targets_br(
                    config["NUM_EGO_ACTORS"],
                    last_env_state_br_sp, train_state_ego.params, 
                    last_br_h_sp_p1, last_obs_br_sp, last_done_br_sp, 
                    traj_batch_br_sp_p1, "agent_0"
                )

                advantages_sp_br_p2, targets_sp_br_p2 = _compute_advantages_and_targets_br(
                    config["NUM_EGO_ACTORS"],
                    last_env_state_br_sp, train_state_ego.params, 
                    last_br_h_sp_p2, last_obs_br_sp, last_done_br_sp, 
                    traj_batch_br_sp_p2, "agent_1"
                )
                
                # 3) PPO update
                update_state = (
                    all_train_state_conf, train_state_ego,
                    traj_batch_conf_p0, traj_batch_conf_p1, traj_batch_conf_xp,
                    traj_batch_br_sp_p1, traj_batch_br_sp_p2, traj_batch_br_xp,
                    advantages_sp_conf_p1, advantages_sp_conf_p2, advantages_xp_conf,
                    advantages_sp_br_p1, advantages_sp_br_p2, advantages_xp_br,
                    targets_sp_conf_p1, targets_sp_conf_p2, targets_xp_conf,
                    targets_sp_br_p1, targets_sp_br_p2, targets_xp_br,
                    rng_conf, rng_br
                )
                update_state, (all_losses_conf, all_losses_ego) = jax.lax.scan(
                    _update_epoch, update_state, None, config["UPDATE_EPOCHS"])
                all_train_state_conf = update_state[0]
                train_state_ego = update_state[1]

                # Metrics
                metric = traj_batch_conf_p0.info
                metric["update_steps"] = update_steps

                # Conf agent losses: value_loss_ego, value_loss_br, pg_loss_ego, pg_loss_br, entropy_ego, entropy_br
                metric["value_loss_conf_against_br"] = all_losses_conf[1][0]
                metric["value_loss_conf_self_play"] = all_losses_conf[1][1]
                metric["pg_loss_conf_against_br"] = all_losses_conf[1][2]
                metric["pg_loss_conf_self_play"] = all_losses_conf[1][3]
                metric["entropy_conf_against_br"] = all_losses_conf[1][4]
                metric["entropy_conf_self_play"] = all_losses_conf[1][5]
                metric["trajedi_loss"] = all_losses_conf[1][6]

                # Ego agent losses
                metric["value_loss_ego_against_conf"] = all_losses_ego[1][0]
                metric["value_loss_ego_self_play"] = all_losses_ego[1][1]
                metric["pg_loss_ego_against_conf"] = all_losses_ego[1][2]
                metric["pg_loss_ego_self_play"] = all_losses_ego[1][3]
                metric["entropy_loss_ego_against_conf"] = all_losses_ego[1][4]
                metric["entropy_loss_ego_self_play"] = all_losses_ego[1][5]

                metric["average_rewards_conf_sp"] = (jnp.mean(traj_batch_conf_p0.reward) + jnp.mean(traj_batch_conf_p1.reward))/2.0
                metric["average_rewards_conf_xp"] = jnp.mean(traj_batch_conf_xp.reward) 
                metric["average_rewards_ego_sp"] = (jnp.mean(traj_batch_br_sp_p1.reward) + jnp.mean(traj_batch_br_sp_p2.reward))/2.0
                metric["average_rewards_br"] = jnp.mean(traj_batch_br_xp.reward)

                new_runner_state = (
                    all_train_state_conf, train_state_ego,
                    last_env_state_confs_sp, last_env_state_confs_xp, last_env_state_br_sp,
                    last_obs_confs_sp, last_obs_confs_xp, last_obs_br_sp, 
                    last_done_confs_sp, last_done_confs_xp, last_done_br_sp,
                    last_conf_h_sp_p1, last_conf_h_sp_p2, 
                    last_conf_h_xp, last_br_h_xp,
                    last_br_h_sp_p1, last_br_h_sp_p2,
                    last_eps_counter, last_timestep_counter,
                    rng_conf, rng_br, update_steps
                )
                return (new_runner_state, metric)

            # --------------------------
            # PPO Update and Checkpoint saving
            # --------------------------
            ckpt_and_eval_interval = config["NUM_UPDATES"] // max(1, config["NUM_CHECKPOINTS"] - 1)  # -1 because we store a ckpt at the last update
            num_ckpts = config["NUM_CHECKPOINTS"]

            # Build a PyTree that holds parameters for all conf agent checkpoints
            def init_ckpt_array(params_pytree):
                return jax.tree.map(
                    lambda x: jnp.zeros((num_ckpts,) + x.shape, x.dtype),
                    params_pytree)

            def _update_step_with_ckpt(state_with_ckpt, unused):
                (
                    (
                        all_train_state_conf, train_state_ego,
                        env_state_confs_sp, env_state_confs_xp, env_state_br_sp,
                        obs_confs_sp, obs_confs_xp, obs_br_sp, 
                        done_confs_sp, done_confs_xp, done_br_sp,
                        conf_h_sp_p1, conf_h_sp_p2, 
                        conf_h_xp, br_h_xp,
                        br_h_sp_p1, br_h_sp_p2,
                        eps_counter, timestep_counter,
                        rng_conf, rng_br, update_steps
                    ), checkpoint_array_conf, checkpoint_array_ego, ckpt_idx,
                    eval_info_ego
                ) = state_with_ckpt


                # Single PPO update
                (new_runner_state, metric) = _update_step(
                    (
                        all_train_state_conf, train_state_ego,
                        env_state_confs_sp, env_state_confs_xp, env_state_br_sp,
                        obs_confs_sp, obs_confs_xp, obs_br_sp, 
                        done_confs_sp, done_confs_xp, done_br_sp,
                        conf_h_sp_p1, conf_h_sp_p2, 
                        conf_h_xp, br_h_xp,
                        br_h_sp_p1, br_h_sp_p2,
                        eps_counter, timestep_counter,
                        rng_conf, rng_br, update_steps
                    ),
                    None
                )

                (
                    all_train_state_conf, train_state_ego,
                    env_state_confs_sp, env_state_confs_xp, env_state_br_sp,
                    obs_confs_sp, obs_confs_xp, obs_br_sp, 
                    done_confs_sp, done_confs_xp, done_br_sp,
                    conf_h_sp_p1, conf_h_sp_p2, 
                    conf_h_xp, br_h_xp,
                    br_h_sp_p1, br_h_sp_p2,
                    eps_counter, timestep_counter,
                    rng_conf, rng_br, update_steps
                ) = new_runner_state

                # Decide if we store a checkpoint
                # update steps is 1-indexed because it was incremented at the end of the update step
                to_store = jnp.logical_or(jnp.equal(jnp.mod(update_steps-1, ckpt_and_eval_interval), 0),
                                        jnp.equal(update_steps, config["NUM_UPDATES"]))

                def store_and_eval_ckpt(args):
                    ckpt_arr_and_ep_infos, rng, cidx = args
                    ckpt_arr_conf, ckpt_arr_ego, prev_ep_infos_ego = ckpt_arr_and_ep_infos
                    new_ckpt_arr_conf = jax.tree.map(
                        lambda c_arr, p: c_arr.at[cidx].set(p),
                        ckpt_arr_conf, all_train_state_conf.params
                    )
                    
                    new_ckpt_arr_ego = jax.tree.map(
                        lambda c_arr, p: c_arr.at[cidx].set(p),
                        ckpt_arr_ego, train_state_ego.params
                    )

                    # run eval episodes
                    rng, eval_rng, = jax.random.split(rng)
                    # conf vs ego

                    def run_all_episodes_xp(rng, train_state_conf, train_state_ego):
                        conf_ids = jnp.arange(config["PARTNER_POP_SIZE"])
                        gathered_conf_model_params = gather_params(train_state_conf.params, conf_ids)

                        rng, eval_rng = jax.random.split(rng)
                        def run_episodes_fixed_rng(conf_param):
                            return run_episodes(
                                eval_rng, env,
                                conf_param, confederate_policy,
                                train_state_ego.params, ego_policy,
                                config["ROLLOUT_LENGTH"], config["NUM_EVAL_EPISODES"],
                            )
                        ep_infos = jax.vmap(run_episodes_fixed_rng)(
                            gathered_conf_model_params # leaves where shape is (pop_size*pop_size, ...)
                        )
                        return ep_infos
            
                    last_ep_info_with_ego = run_all_episodes_xp(
                        eval_rng,
                        all_train_state_conf, 
                        train_state_ego
                    )

                    return ((new_ckpt_arr_conf, new_ckpt_arr_ego, last_ep_info_with_ego), rng, cidx + 1)

                def skip_ckpt(args):
                    return args

                (checkpoint_array_and_infos, rng_br, ckpt_idx) = jax.lax.cond(
                    to_store,
                    store_and_eval_ckpt,
                    skip_ckpt,
                    ((checkpoint_array_conf, checkpoint_array_ego, eval_info_ego), rng_br, ckpt_idx)
                )
                checkpoint_array_conf, checkpoint_array_ego, ep_info_ego = checkpoint_array_and_infos
                metric["eval_ep_last_info_ego"] = ep_info_ego

                return ((all_train_state_conf, train_state_ego, 
                         env_state_confs_sp, env_state_confs_xp, env_state_br_sp,
                         obs_confs_sp, obs_confs_xp, obs_br_sp, 
                         done_confs_sp, done_confs_xp, done_br_sp,
                         conf_h_sp_p1, conf_h_sp_p2, 
                         conf_h_xp, br_h_xp,
                         br_h_sp_p1, br_h_sp_p2,
                         eps_counter, timestep_counter,
                         rng_conf, rng_br, update_steps),
                         checkpoint_array_conf, checkpoint_array_ego, ckpt_idx,
                         ep_info_ego), metric

            # init checkpoint array
            checkpoint_array_conf = init_ckpt_array(train_state_conf.params)
            checkpoint_array_ego = init_ckpt_array(train_state_ego.params)
            ckpt_idx = 0

            # initial state for scan over _update_step_with_ckpt
            update_steps = 0
            
            def run_all_episodes(rng, train_state_conf, train_state_ego):
                conf_ids = jnp.arange(config["PARTNER_POP_SIZE"])
                gathered_conf_model_params = gather_params(train_state_conf.params, conf_ids)

                rng, eval_rng = jax.random.split(rng)
                def run_episodes_fixed_rng(conf_param):
                    return run_episodes(
                        eval_rng, env,
                        conf_param, confederate_policy,
                        train_state_ego.params, ego_policy,
                        config["ROLLOUT_LENGTH"], config["NUM_EVAL_EPISODES"],
                    )
                ep_infos = jax.vmap(run_episodes_fixed_rng)(
                    gathered_conf_model_params # leaves where shape is (pop_size*pop_size, ...)
                )
                return ep_infos
            
            rng, rng_eval = jax.random.split(rng, 2)
            ep_infos = run_all_episodes(
                rng_eval, 
                train_state_conf, 
                train_state_ego
            )

            rng, reset_rng_conf_sp, reset_rng_xp, reset_rng_br_sp= jax.random.split(rng, 4)
            reset_rngs_xp = jax.random.split(reset_rng_xp, config["NUM_CONF_ACTORS"])
            reset_rngs_conf_sp = jax.random.split(reset_rng_conf_sp, config["NUM_CONF_ACTORS"])
            reset_rngs_br_sp = jax.random.split(reset_rng_br_sp, config["NUM_EGO_ACTORS"])

            obs_xp, env_state_xp = jax.vmap(env.reset, in_axes=(0,))(reset_rngs_xp)
            obs_confs_sp, env_state_confs_sp = jax.vmap(env.reset, in_axes=(0,))(reset_rngs_conf_sp)
            obs_br_sp, env_state_br_sp = jax.vmap(env.reset, in_axes=(0,))(reset_rngs_br_sp)
            
            # Initialize hidden states
            init_conf_hstate_sp1 = confederate_policy.init_hstate(config["NUM_CONF_ACTORS"])
            init_conf_hstate_sp2 = confederate_policy.init_hstate(config["NUM_CONF_ACTORS"])
            init_conf_hstate_xp = confederate_policy.init_hstate(config["NUM_CONF_ACTORS"])
            init_ego_hstate_sp1 = ego_policy.init_hstate(config["NUM_EGO_ACTORS"])
            init_ego_hstate_sp2 = ego_policy.init_hstate(config["NUM_EGO_ACTORS"])
            init_ego_hstate_xp = ego_policy.init_hstate(config["NUM_CONF_ACTORS"])

            # Initialize done flags
            init_dones_conf_sp = {k: jnp.zeros((config["NUM_CONF_ACTORS"]), dtype=bool) for k in env.agents + ["__all__"]}
            init_dones_xp = {k: jnp.zeros((config["NUM_CONF_ACTORS"]), dtype=bool) for k in env.agents + ["__all__"]}
            init_dones_br_sp = {k: jnp.zeros((config["NUM_EGO_ACTORS"]), dtype=bool) for k in env.agents + ["__all__"]}

            init_eps_counter = jnp.zeros((config["NUM_CONF_ACTORS"]), dtype=int)
            init_timestep_counter = jnp.zeros((config["NUM_CONF_ACTORS"]), dtype=int)

            rng, rng_conf, rng_br = jax.random.split(rng, 3)

            update_runner_state = (
                train_state_conf, train_state_ego,
                env_state_confs_sp, env_state_xp, env_state_br_sp,
                obs_confs_sp, obs_xp, obs_br_sp, 
                init_dones_conf_sp, init_dones_xp, init_dones_br_sp,
                init_conf_hstate_sp1, init_conf_hstate_sp2, 
                init_conf_hstate_xp, init_ego_hstate_xp,
                init_ego_hstate_sp1, init_ego_hstate_sp2,
                init_eps_counter, init_timestep_counter,
                rng_conf, rng_br, update_steps
            )

            state_with_ckpt = (
                update_runner_state, checkpoint_array_conf, checkpoint_array_ego,
                ckpt_idx, ep_infos
            )
            # run training
            state_with_ckpt, metrics = jax.lax.scan(
                _update_step_with_ckpt,
                state_with_ckpt,
                xs=None,
                length=config["NUM_UPDATES"]
            )
            (
                final_runner_state, checkpoint_array_conf, checkpoint_array_ego,
                final_ckpt_idx, last_ep_infos_ego
            ) = state_with_ckpt

            out = {
                "final_params_conf": final_runner_state[0].params,
                "final_params_br": final_runner_state[1].params,
                "checkpoints_conf": checkpoint_array_conf,
                "checkpoints_ego": checkpoint_array_ego,
                "metrics": metrics,  # shape (NUM_UPDATES, ...)
            }
            return out

        return train
    # ------------------------------
    # Actually run the adversarial teammate training
    # ------------------------------
    rngs = jax.random.split(partner_rng, config["NUM_SEEDS"])
    train_fn = jax.jit(jax.vmap(make_train(config)))
    out = train_fn(rngs)
    return out


def log_metrics(config, logger, outs, metric_names: tuple):
    """Process training metrics and log them using the provided logger.

    Args:
        config: dict, the configuration
        outs: the output of train_paired
        logger: Logger, instance to log metrics
        metric_names: tuple, names of metrics to extract from training logs
    """
    metrics = outs["metrics"]

    # Extract metrics for all agents
    # shape (num_seeds, num_updates, num_eval_episodes, num_agents_per_env)
    avg_returns_confs_vs_br = np.asarray(metrics["eval_ep_last_info_ego"]["returned_episode_returns"]).mean(axis=(0, 2, 3))
    
    # Value losses
    # shape (num_seeds, num_updates, update_epochs, num_minibatches)
    avg_value_losses_conf_self_play = np.asarray(metrics["value_loss_conf_self_play"]).mean(axis=(0, 2, 3, 4))
    avg_value_losses_conf_vs_br = np.asarray(metrics["value_loss_conf_against_br"]).mean(axis=(0, 2, 3, 4))
    avg_value_losses_br_sp = np.asarray(metrics["value_loss_ego_self_play"]).mean(axis=(0, 2, 3))
    avg_value_losses_br_xp = np.asarray(metrics["value_loss_ego_against_conf"]).mean(axis=(0, 2, 3))

    # Actor losses
    # shape (num_seeds, num_updates, update_epochs, num_minibatches)
    avg_actor_losses_conf_self_play = np.asarray(metrics["pg_loss_conf_self_play"]).mean(axis=(0, 2, 3, 4))
    avg_actor_losses_conf_vs_br = np.asarray(metrics["pg_loss_conf_against_br"]).mean(axis=(0, 2, 3, 4))
    avg_actor_losses_ego_sp = np.asarray(metrics["pg_loss_ego_self_play"]).mean(axis=(0, 2, 3))
    avg_actor_losses_ego_xp = np.asarray(metrics["pg_loss_ego_against_conf"]).mean(axis=(0, 2, 3))

    # Entropy losses
    #  shape (num_seeds, num_updates, update_epochs, num_minibatches)
    avg_entropy_losses_conf_self_play = np.asarray(metrics["entropy_conf_self_play"]).mean(axis=(0, 2, 3, 4))
    avg_entropy_losses_conf_vs_br = np.asarray(metrics["entropy_conf_against_br"]).mean(axis=(0, 2, 3, 4))
    avg_entropy_losses_ego_sp = np.asarray(metrics["entropy_loss_ego_self_play"]).mean(axis=(0, 2, 3))
    avg_entropy_losses_ego_xp = np.asarray(metrics["entropy_loss_ego_against_conf"]).mean(axis=(0, 2, 3))

    # Entropy losses
    #  shape (num_seeds, num_updates, update_epochs, num_minibatches)
    avg_sp_trajedi_loss = np.asarray(metrics["trajedi_loss"]).mean(axis=(0, 2, 3, 4))

    # Rewards
    # shape (num_seeds, num_updates)
    avg_rewards_conf_sp = np.asarray(metrics["average_rewards_conf_sp"]).mean(axis=0)
    avg_rewards_conf_xp = np.asarray(metrics["average_rewards_conf_xp"]).mean(axis=0)
    avg_rewards_br_sp = np.asarray(metrics["average_rewards_ego_sp"]).mean(axis=0)
    avg_rewards_br_xp = np.asarray(metrics["average_rewards_br"]).mean(axis=0)

    # Get standard stats
    stats = get_stats(metrics, metric_names)
    stats = {k: np.mean(np.array(v), axis=0) for k, v in stats.items()}

    num_updates = metrics["update_steps"].shape[1]

    # Log all metrics
    for step in range(num_updates):
        # Log standard stats from get_stats, which all belong to the ego agent
        for stat_name, stat_data in stats.items():
            if step < stat_data.shape[0]:  # Ensure step is within bounds
                stat_mean = stat_data[step, 0]
                logger.log_item(f"Train/Ego_{stat_name}", stat_mean, train_step=step)

        # Log returns for different agent interactions
        logger.log_item("Eval/ConfReturn-Against-BR", avg_returns_confs_vs_br[step], train_step=step)
        

        # Confederate losses
        logger.log_item("Losses/ConfValLoss-Self-Play", avg_value_losses_conf_self_play[step], train_step=step)
        logger.log_item("Losses/ConfActorLoss-Self-Play", avg_actor_losses_conf_self_play[step], train_step=step)
        logger.log_item("Losses/ConfEntropy-Self-Play", avg_entropy_losses_conf_self_play[step], train_step=step)
        logger.log_item("Losses/ConfTrajeDiLoss-Self-Play", avg_sp_trajedi_loss[step], train_step=step)

        logger.log_item("Losses/ConfValLoss-Against-BR", avg_value_losses_conf_vs_br[step], train_step=step)
        logger.log_item("Losses/ConfActorLoss-Against-BR", avg_actor_losses_conf_vs_br[step], train_step=step)
        logger.log_item("Losses/ConfEntropy-Against-BR", avg_entropy_losses_conf_vs_br[step], train_step=step)

        # Best response losses
        logger.log_item("Losses/BRValLoss-Self-Play", avg_value_losses_br_sp[step], train_step=step)
        logger.log_item("Losses/BRActorLoss-Self-Play", avg_actor_losses_ego_sp[step], train_step=step)
        logger.log_item("Losses/BREntropyLoss-Self-Play", avg_entropy_losses_ego_sp[step], train_step=step)

        # Ego agent losses
        logger.log_item("Losses/EgoValLoss-Against-Confs", avg_value_losses_br_xp[step], train_step=step)
        logger.log_item("Losses/EgoActorLoss-Against-Confs", avg_actor_losses_ego_xp[step], train_step=step)
        logger.log_item("Losses/EgoEntropyLoss-Against-Confs", avg_entropy_losses_ego_xp[step], train_step=step)

        # Rewards
        logger.log_item("Losses/AvgConfRewards-Against-BR", avg_rewards_conf_xp[step], train_step=step)
        logger.log_item("Losses/AvgConfRewards-Self-Play", avg_rewards_conf_sp[step], train_step=step)
        logger.log_item("Losses/AvgBRRewards-Against-Conf", avg_rewards_br_xp[step], train_step=step)
        logger.log_item("Losses/AvgEgoRewards-Self-Play", avg_rewards_br_sp[step], train_step=step)

    logger.commit()

    # Saving artifacts
    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    out_savepath = save_train_run(outs, savedir, savename="saved_train_run")
    if config["logger"]["log_train_out"]:
        logger.log_artifact(name="saved_train_run", path=out_savepath, type_name="train_run")

    # Cleanup locally logged out file
    if not config["local_logger"]["save_train_out"]:
        shutil.rmtree(out_savepath)

def run_trajedi(config, wandb_logger):
    algorithm_config = dict(config["algorithm"])

    # Create only one environment instance
    env = make_env(algorithm_config["ENV_NAME"], algorithm_config["ENV_KWARGS"])
    env = LogWrapper(env)

    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"])
    rng, train_rng, eval_rng = jax.random.split(rng, 3)

    # Train using TrajeDi algorithm
    log.info("Starting TrajeDi training...")
    start_time = time.time()
    DEBUG = False
    with jax.disable_jit(DEBUG):
        outs = train_trajedi_partners(algorithm_config, env, train_rng)
    end_time = time.time()
    log.info(f"PAIRED training completed in {end_time - start_time} seconds.")

    # Prepare return values for heldout evaluation
    env = make_env(algorithm_config["ENV_NAME"], algorithm_config["ENV_KWARGS"])
    ego_policy, init_ego_params = initialize_s5_agent(algorithm_config, env, eval_rng)

    # Log metrics
    metric_names = get_metric_names(algorithm_config["ENV_NAME"])
    log_metrics(config, wandb_logger, outs, metric_names)

    return ego_policy, outs["final_params_br"], init_ego_params

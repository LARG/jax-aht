import shutil
import time
import logging
from typing import NamedTuple
from functools import partial

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
import wandb

from envs import make_env
from envs.log_wrapper import LogWrapper
from agents.agent_interface import ActorWithConditionalCriticPolicy
from agents.population_interface import AgentPopulation
from agents.mlp_actor_critic import ActorWithConditionalCritic
from common.plot_utils import get_metric_names
from common.run_episodes import run_episodes
from marl.ppo_utils import unbatchify
from common.save_load_utils import save_train_run

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class XPTransition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    self_id: jnp.ndarray
    oppo_id: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray

def train_brdiv_partners(train_rng, env, config):
    num_agents = env.num_agents
    assert num_agents == 2, "This code assumes the environment has exactly 2 agents."

    # Define different minibatch sizes for interactions with ego agent and one with BR agent
    config["NUM_ENVS"] = config["NUM_ENVS_XP"] + config["NUM_ENVS_SP"]
    config["NUM_GAME_AGENTS"] = num_agents
    config["NUM_ACTORS"] = num_agents * config["NUM_ENVS"]

    # Right now assume control of both agent and its BR
    config["NUM_CONTROLLED_ACTORS"] = config["NUM_ACTORS"]

    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // (num_agents * config["ROLLOUT_LENGTH"])// config["NUM_ENVS"]
    config["MINIBATCH_SIZE_EGO"] = ((config["NUM_GAME_AGENTS"]-1) * config["NUM_ACTORS"] * config["ROLLOUT_LENGTH"]) // config["NUM_MINIBATCHES"]
    config["MINIBATCH_SIZE_BR"] = (config["NUM_ACTORS"] * config["ROLLOUT_LENGTH"]) // config["NUM_MINIBATCHES"]

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

    def make_brdiv_agents(config):
        def linear_schedule(count):
            frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
            return config["LR"] * frac
        
        def train(rng):
            # initialize confederate
            conf_agent_net = ActorWithConditionalCritic(env.action_space(env.agents[0]).n)
            # initialize best response
            br_agent_net = ActorWithConditionalCritic(env.action_space(env.agents[0]).n)
            
            rng, init_conf_rng, init_br_rng = jax.random.split(rng, 3)
            all_conf_init_rngs = jax.random.split(init_conf_rng, config["PARTNER_POP_SIZE"])
            all_br_init_rngs = jax.random.split(init_br_rng, config["PARTNER_POP_SIZE"])

            def init_train_states(rng_agents, rng_brs):
                def init_single_pair_optimizers(rng_agent, rng_br):

                    # Initialize parameters of the generated confederate and BR policy
                    init_x = ( # init obs, ids, avail_actions
                        jnp.zeros(env.observation_space(env.agents[0]).shape),
                        jnp.zeros(config["PARTNER_POP_SIZE"]),
                        jnp.ones(env.action_space(env.agents[0]).n),
                    )
                    init_params = conf_agent_net.init_with_output(rng_agent, init_x)[1]

                    init_x_br = ( # init obs, avail_actions
                        jnp.zeros(env.observation_space(env.agents[1]).shape),
                        jnp.zeros(config["PARTNER_POP_SIZE"]),
                        jnp.ones(env.action_space(env.agents[1]).n),
                    )
                    init_params_br = br_agent_net.init_with_output(rng_br, init_x_br)[1]

                    return init_params, init_params_br

                init_all_networks_and_optimizers = jax.vmap(init_single_pair_optimizers)
                all_conf_params, all_br_params = init_all_networks_and_optimizers(rng_agents, rng_brs)
            
                # Define optimizers for both confederate and BR policy
                tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(learning_rate=linear_schedule if config["ANNEAL_LR"] else config["LR"], 
                    eps=1e-5),
                )
                tx_br = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(learning_rate=linear_schedule if config["ANNEAL_LR"] else config["LR"], 
                    eps=1e-5),
                )

                train_state_conf = TrainState.create(
                    apply_fn=conf_agent_net.apply,
                    params=all_conf_params,
                    tx=tx,
                )

                train_state_br = TrainState.create(
                    apply_fn=br_agent_net.apply,
                    params=all_br_params,
                    tx=tx_br,
                )

                return train_state_conf, train_state_br

            all_conf_optims, all_br_optims = init_train_states(
                all_conf_init_rngs, all_br_init_rngs
            )
            # --------------------------
            # 3b) Init envs
            # --------------------------
            
            rng, reset_rng= jax.random.split(rng, 2)
            reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])

            obsv_ego, env_state_ego = jax.vmap(env.reset, in_axes=(0,))(reset_rngs)

            # --------------------------
            # 3c) Define env step
            # --------------------------
            # Implement Rollout Against Ego Agent
            def _env_step(runner_state, unused):
                """
                agent_0 = confederate, agent_1 = ego
                Returns updated runner_state, and a Transition for agent_0.
                """
                conf_params, br_params, conf_agent_id, br_agent_id, env_state, last_obs, rng = runner_state
                rng, actor_rng, partner_rng, step_rng = jax.random.split(rng, 4)

                obs_0 = last_obs["agent_0"]
                obs_1 = last_obs["agent_1"]

                # Get available actions for agent 0 from environment state
                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(avail_actions)
                avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
                avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

                # Agent_0 action
                forward_pass_conf = lambda param, ob, id, avail_act: conf_agent_net.apply(param, (ob, id, avail_act))
                # TODO: what is this being vmapped across? 
                pi_0, val_0 = jax.vmap(forward_pass_conf)(conf_params, obs_0, br_agent_id, avail_actions_0)
                act_0 = pi_0.sample(seed=actor_rng)
                logp_0 = pi_0.log_prob(act_0)

                # Agent_1 action
                forward_pass_br = lambda param, ob, id, avail_act: br_agent_net.apply(param, (ob, id, avail_act))
                pi_1, val_1 = jax.vmap(forward_pass_br)(br_params, obs_1, conf_agent_id, avail_actions_1)
                act_1 = pi_1.sample(seed=partner_rng)
                logp_1 = pi_1.log_prob(act_1)

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
                info_1 = jax.tree.map(lambda x: x[:, 1], info)

                indiv_conf_rew_compute = lambda conf_id, br_id, agent1_rew: jax.lax.cond(jnp.equal(
                    jnp.argmax(conf_id, axis=-1), jnp.argmax(br_id, axis=-1)
                ), lambda x: x, lambda x: -x, agent1_rew)

                indiv_ego_rew_compute = lambda conf_id, br_id, agent0_rew: jax.lax.cond(jnp.equal(
                    jnp.argmax(conf_id, axis=-1), jnp.argmax(br_id, axis=-1)
                ), lambda x: x, lambda x: -x, agent0_rew)

                agent_0_rews = jax.vmap(indiv_conf_rew_compute)(conf_agent_id, br_agent_id, reward["agent_1"])
                agent_1_rews = jax.vmap(indiv_ego_rew_compute)(conf_agent_id, br_agent_id, reward["agent_0"])
                
                # Store agent_0 data in transition
                transition_0 = XPTransition(
                    done=done["agent_0"],
                    action=act_0,
                    value=val_0,
                    self_id=conf_agent_id,
                    oppo_id=br_agent_id,
                    reward=agent_0_rews,
                    log_prob=logp_0,
                    obs=obs_0,
                    info=info_0,
                    avail_actions=avail_actions_0
                )

                transition_1 = XPTransition(
                    done=done["agent_1"],
                    action=act_1,
                    value=val_1,
                    self_id=br_agent_id,
                    oppo_id=conf_agent_id,
                    reward=agent_1_rews,
                    log_prob=logp_1,
                    obs=obs_1,
                    info=info_1,
                    avail_actions=avail_actions_1
                )
                new_runner_state = (conf_params, br_params, conf_agent_id, br_agent_id, env_state_next, obs_next, rng)
                return new_runner_state, (transition_0, transition_1)
            
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
            
            def run_single_episode(ep_rng, br_param, conf_param, br_id, conf_id):
                '''agent_0 is the confederate, agent 1 is the best response'''
                # Reset the env.
                ep_rng, reset_rng = jax.random.split(ep_rng)
                obs, env_state = env.reset(reset_rng)
                # Get available actions for agent 0 from environment state
                avail_actions = env.get_avail_actions(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(avail_actions)
                avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
                avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)
                against_br_return = jnp.zeros(1, dtype=float)
                
                # Do one step to get a dummy info structure.
                ep_rng, act_rng, part_rng, step_rng = jax.random.split(ep_rng, 4)
                pi0, _ = conf_agent_net.apply(conf_param, (obs["agent_0"], br_id, avail_actions_0))
                act0 = pi0.sample(seed=act_rng)

                pi1, _ = br_agent_net.apply(br_param, (obs["agent_1"], conf_id, avail_actions_1))
                act1 = pi1.sample(seed=part_rng)
                    
                both_actions = [act0, act1]
                env_act = {k: both_actions[i] for i, k in enumerate(env.agents)}
                _, _, reward, done, dummy_info = env.step(step_rng, env_state, env_act)
                against_br_return = against_br_return + reward["agent_0"]

                # We'll use a scan to iterate steps until the episode is done.
                ep_ts = 1
                ep_rng, remaining_steps_rng = jax.random.split(ep_rng)
                init_carry = (ep_ts, env_state, obs, remaining_steps_rng, done, dummy_info, against_br_return)
                def scan_step(carry, _):
                    def take_step(carry_step):
                        ep_ts, env_state, obs, ep_rng, done, info_next, against_br_return = carry_step
                        ep_rng, act_rng, part_rng, step_rng = jax.random.split(ep_rng, 4)
                        
                        # Get available actions for agent 0 from environment state
                        avail_actions = env.get_avail_actions(env_state.env_state)
                        avail_actions = jax.lax.stop_gradient(avail_actions)
                        avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
                        avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

                        pi0, _ = conf_agent_net.apply(conf_param, (obs["agent_0"], br_id, avail_actions_0))
                        act0 = pi0.sample(seed=act_rng) # sample because mode does worse on LBF

                        pi1, _ = br_agent_net.apply(br_param, (obs["agent_1"], conf_id, avail_actions_1))
                        act1 = pi1.sample(seed=part_rng)

                        both_actions = [act0, act1]
                        env_act = {k: both_actions[i] for i, k in enumerate(env.agents)}

                        obs_next, env_state_next, reward, done_next, info_next = env.step(step_rng, env_state, env_act)
                        against_br_return = against_br_return + reward["agent_0"]

                        return (ep_ts + 1, env_state_next, obs_next, ep_rng, done_next, info_next, against_br_return)
                            
                    ep_ts, env_state, obs, ep_rng, done, info_next, against_br_return = carry
                    new_carry = jax.lax.cond(
                        done["__all__"],
                        # if done, execute true function(operand). else, execute false function(operand).
                        lambda curr_carry: curr_carry, # True fn
                        take_step, # False fn
                        operand=carry
                    )
                    return new_carry, None

                final_carry, _ = jax.lax.scan(
                    scan_step, init_carry, None, length=max_episode_steps)
                # Return the final info (which includes the episode return via LogWrapper).
                return (final_carry[-2], final_carry[-1])
                    
            def run_episodes(ep_rng, br_param, conf_param, br_id, conf_id, num_eps):
                '''TODO: convert to vmap'''
                def body_fn(carry, _):
                    ep_rng = carry
                    ep_rng, ep_rng_step = jax.random.split(ep_rng)
                    all_outs = run_single_episode(ep_rng_step, br_param, conf_param, br_id, conf_id)
                    return ep_rng, all_outs
                ep_rng, all_outs = jax.lax.scan(body_fn, ep_rng, None, length=num_eps)
                return all_outs  # each leaf has shape (num_eps, ...)
            
            def run_all_episodes(rng, train_state_conf, train_state_br, max_eval_episodes):
                cross_product = jnp.meshgrid(
                    jnp.arange(config["PARTNER_POP_SIZE"]),
                    jnp.arange(config["PARTNER_POP_SIZE"])
                )
                agent_id_cartesian_product = jnp.stack([g.ravel() for g in cross_product], axis=-1)
                possible_one_hot_ids = jnp.eye(config["PARTNER_POP_SIZE"])

                conf_ids = agent_id_cartesian_product[:, 0]
                conf_ids_one_hot = possible_one_hot_ids[conf_ids]
                ego_ids = agent_id_cartesian_product[:, 1]
                ego_ids_one_hot = possible_one_hot_ids[ego_ids]

                gathered_conf_model_params = gather_params(train_state_conf.params, conf_ids)
                gathered_br_model_params = gather_params(train_state_br.params, ego_ids)

                # run eval episodes
                rng, eval_rng = jax.random.split(rng)

                run_episodes_fixed_rng = lambda a, b, c, d: run_episodes(eval_rng, a, b, c, d, max_eval_episodes)
                ep_infos = jax.vmap(run_episodes_fixed_rng)(
                    gathered_br_model_params, gathered_conf_model_params, 
                    ego_ids_one_hot, conf_ids_one_hot
                )

                return ep_infos

            def _update_epoch(update_state, unused):
                def _update_minbatch(all_train_states, all_data):
                    train_state_conf, train_state_br = all_train_states
                    conf_batch_data, br_batch_data = all_data

                    traj_batch_conf, advantages_conf, returns_conf = conf_batch_data
                    traj_batch_br, advantages_br, returns_br = br_batch_data

                    def _loss_fn(param, agent_net, traj_batch, gae, target_v, agent_id):
                        # get policy and value of confederate versus ego and best response agents respectively
                        param = jax.tree.map(lambda x: jnp.squeeze(x, 0), param)
                        pi, value = agent_net.apply(param, (traj_batch.obs, traj_batch.oppo_id, traj_batch.avail_actions))
                        log_prob = pi.log_prob(traj_batch.action)

                        is_relevant = jnp.equal(
                            jnp.argmax(traj_batch.self_id, axis=-1), 
                            agent_id
                        )
                        loss_weights = jnp.where(is_relevant, 1, 0).astype(jnp.float32)
                        
                        # Value loss
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                            ).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - target_v)
                        value_losses_clipped = jnp.square(value_pred_clipped - target_v)
                        value_loss = jax.lax.cond(
                            loss_weights.sum() == 0, 
                            lambda x: jnp.zeros_like(x).astype(jnp.float32), 
                            lambda x: x,
                            (loss_weights * jnp.maximum(value_losses, value_losses_clipped)).sum() / loss_weights.sum()
                        )
                        
                        choose_actor_weight = lambda self_id, other_id, rew: jax.lax.cond(
                            jnp.equal(jnp.argmax(self_id, axis=-1), jnp.argmax(other_id, axis=-1)), 
                            lambda x: (1 + 2*config["XP_LOSS_WEIGHTS"]) * jnp.ones_like(x), 
                            lambda x: config["XP_LOSS_WEIGHTS"] * jnp.ones_like(x), 
                            rew
                        )

                        self_agent_id, other_agent_id = traj_batch.self_id, traj_batch.oppo_id
                        actor_weights = jax.vmap(choose_actor_weight)(self_agent_id, other_agent_id, traj_batch.reward)         

                        # Policy gradient loss
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae_norm = (gae - gae.mean()) / (gae.std() + 1e-8)
                        pg_loss_1 = ratio * gae_norm * actor_weights
                        pg_loss_2 = jnp.clip(
                            ratio, 
                            1.0 - config["CLIP_EPS"], 
                            1.0 + config["CLIP_EPS"]) * gae_norm * actor_weights
                        pg_loss = jax.lax.cond(
                            loss_weights.sum() == 0, 
                            lambda x: jnp.zeros_like(x).astype(jnp.float32), 
                            lambda x: x, 
                            -(
                                loss_weights*jnp.minimum(pg_loss_1, pg_loss_2)
                            ).sum()/loss_weights.sum()
                        )

                        # Entropy
                        entropy = jax.lax.cond(
                            loss_weights.sum() == 0, 
                            lambda x: jnp.zeros_like(x).astype(jnp.float32), 
                            lambda x: x,
                            (loss_weights * pi.entropy()).sum()/loss_weights.sum()
                        )
                        
                        total_loss = pg_loss + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
                        return total_loss, (value_loss, pg_loss, entropy)

                    possible_agent_ids = jnp.expand_dims(jnp.arange(config["PARTNER_POP_SIZE"]), 1)
                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    def gather_conf_params_and_return_grads(train_state_params, agent_id):
                        param_vector = gather_params(train_state_params, agent_id)
                        (loss_val_conf, aux_vals_conf), grads_conf = grad_fn(
                            param_vector, conf_agent_net, traj_batch_conf, 
                            advantages_conf, returns_conf, agent_id
                        )
                        return (loss_val_conf, aux_vals_conf), grads_conf
                    
                    def gather_br_params_and_return_grads(train_state_params, agent_id):
                        param_vector = gather_params(train_state_params, agent_id)
                        (loss_val_br, aux_vals_br), grads_br = grad_fn(
                            param_vector, br_agent_net, traj_batch_br, 
                            advantages_br, returns_br, agent_id
                        )
                        return (loss_val_br, aux_vals_br), grads_br

                    compute_conf_grads = lambda x: gather_conf_params_and_return_grads(train_state_conf.params, x)
                    compute_br_grads = lambda x: gather_br_params_and_return_grads(train_state_br.params, x)

                    (loss_val_conf, aux_vals_conf), grads_conf = jax.vmap(compute_conf_grads)(possible_agent_ids)
                    (loss_val_br, aux_vals_br), grads_br = jax.vmap(compute_br_grads)(possible_agent_ids)
                    
                    grads_conf_new = jax.tree.map(lambda x: jnp.squeeze(x, 1), grads_conf)
                    grads_br_new = jax.tree.map(lambda x: jnp.squeeze(x, 1), grads_br)
                    train_state_conf = train_state_conf.apply_gradients(grads=grads_conf_new)
                    train_state_br = train_state_br.apply_gradients(grads=grads_br_new)
                    return (train_state_conf, train_state_br), ((loss_val_conf, aux_vals_conf), (loss_val_br, aux_vals_br))
                
                (
                    train_state_conf, train_state_br, 
                    traj_batch_conf, traj_batch_br, 
                    advantages_conf, advantages_br, 
                    targets_conf, targets_br, 
                    rng_ego, rng_br
                ) = update_state

                rng_ego, perm_rng_conf = jax.random.split(rng_ego)
                rng_br, perm_rng_br = jax.random.split(rng_br)

                batch_size_conf = config["MINIBATCH_SIZE_EGO"] * config["NUM_MINIBATCHES"] // config["NUM_GAME_AGENTS"]
                batch_size_br = config["MINIBATCH_SIZE_BR"] * config["NUM_MINIBATCHES"] // config["NUM_GAME_AGENTS"]
                
                assert (
                    batch_size_conf == (config["NUM_GAME_AGENTS"]-1) * config["ROLLOUT_LENGTH"] * config["NUM_ACTORS"] // config["NUM_GAME_AGENTS"]
                ), "batch size must be equal to number of steps * number of actors"
                assert (
                    batch_size_br == config["ROLLOUT_LENGTH"] * config["NUM_ACTORS"] // config["NUM_GAME_AGENTS"]
                ), "batch size must be equal to number of steps * number of actors"

                permutation_conf = jax.random.permutation(perm_rng_conf, batch_size_conf)
                permutation_br = jax.random.permutation(perm_rng_br, batch_size_br)

                batch_conf = (traj_batch_conf, advantages_conf, targets_conf)
                batch_br = (traj_batch_br, advantages_br, targets_br)
                
                batch_conf_reshaped = jax.tree.map(
                    lambda x: x.reshape((batch_size_conf,) + x.shape[2:]), batch_conf
                )
                
                batch_br_reshaped = jax.tree.map(
                    lambda x: x.reshape((batch_size_br,) + x.shape[2:]), batch_br
                )

                shuffled_batch_conf = jax.tree.map(
                    lambda x: jnp.take(x, permutation_conf, axis=0), batch_conf_reshaped
                )
                shuffled_batch_br = jax.tree.map(
                    lambda x: jnp.take(x, permutation_br, axis=0), batch_br_reshaped
                )

                minibatches_conf = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch_conf,
                )

                minibatches_br = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch_br,
                )
                # Update both policies
                updated_train_states, total_loss = jax.lax.scan(
                    _update_minbatch, (train_state_conf, train_state_br), (minibatches_conf, minibatches_br)
                )
                train_state_conf = updated_train_states[0]
                train_state_br = updated_train_states[1]
                
                update_state = (train_state_conf, train_state_br, 
                    traj_batch_conf, traj_batch_br, 
                    advantages_conf, advantages_br, 
                    targets_conf, targets_conf,
                    rng_ego, rng_br
                )
                return update_state, total_loss

            def _update_step(update_runner_state, unused):
                (
                    all_train_state_conf, all_train_state_br, rng, update_steps
                ) = update_runner_state

                rng, reset_rng= jax.random.split(rng, 2)
                reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
                last_obs, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rngs)

                rng, conf_sampling_sp_rng, conf_sampling_rng, br_sampling_rng = jax.random.split(rng, 4)

                # Sampling IDs for SP data collection
                ids_sp = jax.random.randint(conf_sampling_sp_rng, (config["NUM_ENVS_SP"],), 0, config["PARTNER_POP_SIZE"])

                # Sampling IDs for XP data collection
                conf_ids_xp = jax.random.randint(conf_sampling_rng, (config["NUM_ENVS_XP"],), 0, config["PARTNER_POP_SIZE"])

                br_sampling_rngs = jax.random.split(br_sampling_rng, config["NUM_ENVS_XP"]+1)
                br_sampling_rng = br_sampling_rngs[0]

                # Sample BR IDs that are different from conf id
                br_ids_xp = jax.random.randint(br_sampling_rng, (config["NUM_ENVS_XP"],), 0, config["PARTNER_POP_SIZE"])
                sample_new_id = lambda a: (jax.random.split(a[0], 2)[1], jax.random.randint(a[0], a[1].shape, minval=0, maxval=config["PARTNER_POP_SIZE"]))
                loop_logic = lambda z, x, y: jax.lax.while_loop(lambda a: jnp.equal(z,a[1]), sample_new_id, (x, y))
                _, br_ids_xp = jax.vmap(loop_logic)(conf_ids_xp, br_sampling_rngs[1:], br_ids_xp)
                
                conf_ids = jnp.concatenate([ids_sp, conf_ids_xp], axis=-1)
                br_ids = jnp.concatenate([ids_sp, br_ids_xp], axis=-1)

                identity_matrix = jnp.eye(config["PARTNER_POP_SIZE"])
                conf_one_hots = identity_matrix[conf_ids]
                br_one_hots = identity_matrix[br_ids]
                
                gather_conf_params = gather_params(all_train_state_conf.params, conf_ids)
                gather_br_params = gather_params(all_train_state_br.params, br_ids)

                runner_state = (
                    gather_conf_params, gather_br_params, conf_one_hots, br_one_hots,
                    env_state, last_obs, rng
                )
                runner_state, traj_batch = jax.lax.scan(
                    _env_step, runner_state, None, config["ROLLOUT_LENGTH"])
                (gather_conf_params, gather_br_params, conf_one_hots, br_one_hots, env_state, last_obs, rng) = runner_state

                # Get agent 0 and agent 1 trajectories from interaction between conf policy and its BR policy.
                traj_batch_conf, traj_batch_br = traj_batch

                # Compute advantage for confederate agent from interaction with br policy
                last_obs_conf = last_obs["agent_0"]
                forward_pass_conf = lambda param, ob, id, avail_act: conf_agent_net.apply(param, (ob, id, avail_act))
                avail_actions_0 = jax.vmap(env.get_avail_actions)(env_state.env_state)["agent_0"].astype(jnp.float32)
                _, last_val_conf = jax.vmap(forward_pass_conf)(gather_conf_params, last_obs_conf, br_one_hots, avail_actions_0)

                advantages_conf, targets_conf = _calculate_gae(traj_batch_conf, last_val_conf)

                # 3c) compute advantage for br policy from interaction with confederate agent
                last_obs_br = last_obs["agent_1"]

                forward_pass_br = lambda param, ob, id, avail_act: br_agent_net.apply(param, (ob, id, avail_act))
                avail_actions_1 = jax.vmap(env.get_avail_actions)(env_state.env_state)["agent_1"].astype(jnp.float32)
                _, last_val_br = jax.vmap(forward_pass_br)(gather_br_params, last_obs_br, conf_one_hots, avail_actions_1)
                advantages_br, targets_br = _calculate_gae(traj_batch_br, last_val_br)

                # 3) PPO update
                rng, conf_batch_sampling_rng, br_batch_sampling_rng = jax.random.split(rng, 3)
                update_state = (
                    all_train_state_conf, all_train_state_br, traj_batch_conf, 
                    traj_batch_br, advantages_conf, advantages_br, 
                    targets_conf, targets_br, 
                    conf_batch_sampling_rng, br_batch_sampling_rng
                )

                update_state, all_losses = jax.lax.scan(
                    _update_epoch, update_state, None, config["UPDATE_EPOCHS"])
                all_train_state_conf = update_state[0]
                all_train_state_br = update_state[1]

                # Metrics
                metric = traj_batch_conf.info
                metric["update_steps"] = update_steps
                metric["value_loss_conf_agent"] = all_losses[0][1][0]
                metric["value_loss_br_agent"] = all_losses[1][1][0]
                metric["pg_loss_conf_agent"] = all_losses[0][1][1]
                metric["pg_loss_br_agent"] = all_losses[1][1][1]
                metric["entropy_conf"] = all_losses[0][1][2]
                metric["entropy_br"] = all_losses[1][1][2]

                new_runner_state = (
                    all_train_state_conf, all_train_state_br, rng, update_steps + 1
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
            
            max_episode_steps = config["ROLLOUT_LENGTH"]
            
            def _update_step_with_ckpt(state_with_ckpt, unused):
                ((
                    train_state_conf, train_state_br, rng, update_steps
                ), checkpoint_array_conf, checkpoint_array_br, ckpt_idx, 
                    eval_info) = state_with_ckpt

                # Single PPO update
                (new_runner_state, metric) = _update_step(
                    (train_state_conf, train_state_br, rng, update_steps),
                    None
                )

                (
                    train_state_conf, train_state_br, rng, update_steps
                ) = new_runner_state

                # Decide if we store a checkpoint
                # update steps is 1-indexed because it was incremented at the end of the update step
                to_store = jnp.logical_or(jnp.equal(jnp.mod(update_steps-1, ckpt_and_eval_interval), 0),
                                        jnp.equal(update_steps, config["NUM_UPDATES"]))
                max_eval_episodes = config["NUM_EVAL_EPISODES"]
                
                def store_and_eval_ckpt(args):
                    ckpt_arr_and_ep_infos, rng, cidx = args
                    ckpt_arr_conf, ckpt_arr_br, _ = ckpt_arr_and_ep_infos
                    new_ckpt_arr_conf = jax.tree.map(
                        lambda c_arr, p: c_arr.at[cidx].set(p),
                        ckpt_arr_conf, train_state_conf.params
                    )
                    new_ckpt_arr_br = jax.tree.map(
                        lambda c_arr, p: c_arr.at[cidx].set(p),
                        ckpt_arr_br, train_state_br.params
                    )

                    rng, eval_rng = jax.random.split(rng)
                    ep_infos = run_all_episodes(eval_rng, train_state_conf, train_state_br, max_eval_episodes)
                    
                    return ((new_ckpt_arr_conf, new_ckpt_arr_br, ep_infos), rng, cidx + 1)

                def skip_ckpt(args):
                    return args

                (checkpoint_array_and_infos, rng, ckpt_idx) = jax.lax.cond(
                    to_store, 
                    store_and_eval_ckpt, 
                    skip_ckpt, 
                    ((checkpoint_array_conf, checkpoint_array_br, eval_info), rng, ckpt_idx)
                )
                checkpoint_array_conf, checkpoint_array_br, ckpt_infos = checkpoint_array_and_infos
                
                metric["real_eval_ep_last_info"] = ckpt_infos[0] 
                metric["eval_ep_last_info"] = ckpt_infos[1] # return of confederate

                return ((train_state_conf, train_state_br, rng, update_steps),
                        checkpoint_array_conf, checkpoint_array_br, ckpt_idx, 
                        ckpt_infos), metric

            # init checkpoint array
            checkpoint_array_conf = init_ckpt_array(all_conf_optims.params)
            checkpoint_array_br = init_ckpt_array(all_br_optims.params)
            ckpt_idx = 0

            # initial state for scan over _update_step_with_ckpt
            update_steps = 0

            rng, rng_eval = jax.random.split(rng, 2)
            max_eval_episodes = config["NUM_EVAL_EPISODES"]
            
            ep_infos = run_all_episodes(rng_eval, all_conf_optims, all_br_optims, max_eval_episodes)
            
            update_runner_state = (
                all_conf_optims, all_br_optims, rng, update_steps
            )

            state_with_ckpt = (
                update_runner_state, checkpoint_array_conf, 
                checkpoint_array_br, ckpt_idx, ep_infos
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
                final_ckpt_idx, all_ep_infos
            ) = state_with_ckpt

            out = {
                "final_params_conf": final_runner_state[0].params,
                "final_params_br": final_runner_state[1].params,
                "checkpoints_conf": checkpoint_array_conf,
                "checkpoints_br": checkpoint_array_br,
                "metrics": metrics, # metrics is from the perspective of the confederate agent (averaged over population)
                "all_pair_returns": all_ep_infos
            }
            return out

        return train
    # ------------------------------
    # Actually run the adversarial teammate training
    # ------------------------------
    train_fn = make_brdiv_agents(config)
    out = train_fn(train_rng)
    return out

def get_brdiv_population(config, out, env):
    '''
    Get the partner params and partner population for ego training.
    '''
    brdiv_pop_size = config["algorithm"]["PARTNER_POP_SIZE"]

    # partner_params has shape (num_seeds, brdiv_pop_size, ...)
    partner_params = out['final_params_conf']
    
    partner_policy = ActorWithConditionalCriticPolicy(
        action_dim=env.action_space(env.agents[1]).n,
        obs_dim=env.observation_space(env.agents[1]).shape[0],
        pop_size=brdiv_pop_size, # used to create onehot agent id
        activation=config["algorithm"].get("ACTIVATION", "tanh")
    )

    # Create partner population
    partner_population = AgentPopulation( 
        pop_size=brdiv_pop_size,
        policy_cls=partner_policy
    )

    return partner_params, partner_population

def run_brdiv(config, wandb_logger):
    algorithm_config = dict(config["algorithm"])

    env = make_env(algorithm_config["ENV_NAME"], algorithm_config["ENV_KWARGS"])
    env = LogWrapper(env)

    log.info("Starting BRDiv training...")
    start = time.time()
    
    # Generate multiple random seeds from the base seed
    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"])
    rngs = jax.random.split(rng, algorithm_config["NUM_SEEDS"])
    
    # Create a vmapped version of train_brdiv_partners
    with jax.disable_jit(False):
        vmapped_train_fn = jax.jit(
            jax.vmap(
                partial(train_brdiv_partners, env=env, config=algorithm_config)
            )
        )
        out = vmapped_train_fn(rngs)
    
    end = time.time()
    log.info(f"BRDiv training complete in {end - start} seconds")

    metric_names = get_metric_names(algorithm_config["ENV_NAME"])
    log_metrics(config, out, wandb_logger, metric_names)

    partner_params, partner_population = get_brdiv_population(config, out, env)

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
    # metrics now has shape (num_seeds, num_updates, _, _, pop_size)
    num_seeds, num_updates, _, _, pop_size = metrics["pg_loss_conf_agent"].shape # number of trained pairs

    ### Log evaluation metrics
    # we plot XP return curves separately from SP return curves 
    # shape (num_seeds, num_updates, (pop_size)^2, num_eval_episodes, 1)
    all_returns = np.asarray(metrics["eval_ep_last_info"])
    xs = list(range(num_updates))
    
    sp_mask, agent_id_cartesian_product = compute_sp_mask_and_ids(pop_size)
    sp_returns = all_returns[:, :, sp_mask]
    xp_returns = all_returns[:, :, ~sp_mask]
    
    # Average over seeds, then over agent pairs, episodes and num_agents_per_game
    sp_return_curve = sp_returns.mean(axis=(0, 2, 3, 4))
    xp_return_curve = xp_returns.mean(axis=(0, 2, 3, 4))

    for step in range(num_updates):
        logger.log_item("Eval/AvgSPReturnCurve", sp_return_curve[step], train_step=step)
        logger.log_item("Eval/AvgXPReturnCurve", xp_return_curve[step], train_step=step)
    logger.commit()

    # log final XP matrix to wandb - average over seeds
    last_returns_array = all_returns[:, -1].mean(axis=(0, 2, 3))
    last_returns_array = np.reshape(last_returns_array, (pop_size, pop_size))
    logger.log_xp_matrix("Eval/LastXPMatrix", last_returns_array)

    ### Log population loss as multi-line plots, where each line is a different population member
    # shape (num_seeds, num_updates, update_epochs, num_minibatches, pop_size)
    # Average over seeds
    processed_losses = {
        "ConfPGLoss": np.asarray(metrics["pg_loss_conf_agent"]).mean(axis=(0, 2, 3)).transpose(),
        "BRPGLoss": np.asarray(metrics["pg_loss_br_agent"]).mean(axis=(0, 2, 3)).transpose(),
        "ConfValLoss": np.asarray(metrics["value_loss_conf_agent"]).mean(axis=(0, 2, 3)).transpose(),
        "BRValLoss": np.asarray(metrics["value_loss_br_agent"]).mean(axis=(0, 2, 3)).transpose(),
        "ConfEntropy": np.asarray(metrics["entropy_conf"]).mean(axis=(0, 2, 3)).transpose(),
        "BREntropy": np.asarray(metrics["entropy_br"]).mean(axis=(0, 2, 3)).transpose(),
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
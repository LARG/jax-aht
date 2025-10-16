'''Implemented to be as faithful to the original PAIRED as possible.'''
import shutil
import time
import logging

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from agents.mlp_actor_critic_agent import ActorWithDoubleCriticPolicy, MLPActorCriticPolicy
from agents.initialize_agents import initialize_s5_agent
from common.plot_utils import get_stats, get_metric_names
from common.save_load_utils import save_train_run
from common.run_episodes import run_episodes
from marl.ppo_utils import Transition, unbatchify, _create_minibatches
from envs import make_env
from envs.log_wrapper import LogWrapper

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def train_paired(config, env, partner_rng):
    '''
    Train regret-maximizing confederate/best-response pairs, and an ego agent.
    Return model checkpoints and metrics.
    '''
    def make_train(config):
        num_agents = env.num_agents
        assert num_agents == 2, "This code assumes the environment has exactly 2 agents."

        # Right now assume control of just 1 agent
        config["NUM_CONTROLLED_ACTORS"] = config["NUM_ENVS"]
        config["NUM_UNCONTROLLED_AGENTS"] = config["NUM_ENVS"]

        config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // (config["ROLLOUT_LENGTH"] *3 * config["NUM_ENVS"])
        assert config["NUM_CONTROLLED_ACTORS"] % config["NUM_MINIBATCHES"] == 0, "NUM_CONTROLLED_ACTORS must be divisible by NUM_MINIBATCHES"
        assert config["NUM_CONTROLLED_ACTORS"] >= config["NUM_MINIBATCHES"], "NUM_CONTROLLED_ACTORS must be >= NUM_MINIBATCHES"

        def linear_schedule(count):
            frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
            return config["LR"] * frac

        def train(rng):
            # Initialize all three policies: ego, confederate, and best response
            rng, init_ego_rng, init_conf_rng, init_br_rng = jax.random.split(rng, 4)

            # Initialize ego agent policy
            ego_policy, init_ego_params = initialize_s5_agent(config, env, init_ego_rng)

            # Initialize confederate policy using ActorWithDoubleCriticPolicy
            confederate_policy = ActorWithDoubleCriticPolicy(
                action_dim=env.action_space(env.agents[0]).n,
                obs_dim=env.observation_space(env.agents[0]).shape[0]
            )

            # Initialize best response policy using MLPActorCriticPolicy
            br_policy = MLPActorCriticPolicy(
                action_dim=env.action_space(env.agents[1]).n,
                obs_dim=env.observation_space(env.agents[1]).shape[0]
            )

            # Initialize parameters using the policy interfaces
            init_params_conf = confederate_policy.init_params(init_conf_rng)
            init_params_br = br_policy.init_params(init_br_rng)

            # Define optimizers for all three policies
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule if config["ANNEAL_LR"] else config["LR"],
                eps=1e-5),
            )
            tx_br = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule if config["ANNEAL_LR"] else config["LR"], eps=1e-5),
            )
            tx_ego = optax.chain(
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

            train_state_ego = TrainState.create(
                apply_fn=ego_policy.network.apply,
                params=init_ego_params,
                tx=tx_ego,
            )

            # --------------------------
            # 3b) Init envs and hidden states
            # --------------------------
            rng, reset_rng_ego, reset_rng_br = jax.random.split(rng, 3)
            reset_rngs_ego = jax.random.split(reset_rng_ego, config["NUM_ENVS"])
            reset_rngs_br = jax.random.split(reset_rng_br, config["NUM_ENVS"])

            obsv_ego, env_state_ego = jax.vmap(env.reset, in_axes=(0,))(reset_rngs_ego)
            obsv_br, env_state_br = jax.vmap(env.reset, in_axes=(0,))(reset_rngs_br)
            # --------------------------
            # 3c) Define env step
            # --------------------------
            def _env_step_ego(runner_state, unused):
                """
                agent_0 = confederate, agent_1 = ego
                Returns updated runner_state, and two Transitions: one for agent_0 (confederate) and one for agent_1 (ego).
                """
                train_state_conf, train_state_ego, env_state, last_obs, last_dones, last_conf_h, last_ego_h, rng = runner_state
                rng, act_rng, ego_rng, step_rng = jax.random.split(rng, 4)

                obs_0 = last_obs["agent_0"]
                obs_1 = last_obs["agent_1"]

                # Get available actions for both agents from environment state
                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
                avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

                # Agent_0 (confederate) action using policy interface
                act_0, (val_0, _), pi_0, new_conf_h = confederate_policy.get_action_value_policy(
                    params=train_state_conf.params,
                    obs=obs_0.reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                    done=last_dones["agent_0"].reshape(1, config["NUM_CONTROLLED_ACTORS"]),  # confederate has same done status
                    avail_actions=jax.lax.stop_gradient(avail_actions_0),
                    hstate=last_conf_h,
                    rng=act_rng
                )
                logp_0 = pi_0.log_prob(act_0)

                act_0 = act_0.squeeze()
                logp_0 = logp_0.squeeze()
                val_0 = val_0.squeeze()

                # Agent_1 (ego) action using policy interface
                act_1, val_1, pi_1, new_ego_h = ego_policy.get_action_value_policy(
                    params=train_state_ego.params,
                    obs=obs_1.reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                    done=last_dones["agent_1"].reshape(1, config["NUM_CONTROLLED_ACTORS"]),
                    avail_actions=jax.lax.stop_gradient(avail_actions_1),
                    hstate=last_ego_h,
                    rng=ego_rng
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
                # note that num_actors = num_envs * num_agents
                info_0 = jax.tree.map(lambda x: x[:, 0], info)
                info_1 = jax.tree.map(lambda x: x[:, 1], info)

                # Store agent_0 (confederate) data in transition
                transition_0 = Transition(
                    done=done["agent_0"],
                    action=act_0,
                    value=val_0,
                    reward=reward["agent_1"],
                    log_prob=logp_0,
                    obs=obs_0,
                    info=info_0,
                    avail_actions=avail_actions_0
                )

                # Store agent_1 (ego) data in transition
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

                new_runner_state = (train_state_conf, train_state_ego, env_state_next, obs_next, done, new_conf_h, new_ego_h, rng)
                return new_runner_state, (transition_0, transition_1)

            def _env_step_br(runner_state, unused):
                """
                agent_0 = confederate, agent_1 = best response
                Returns updated runner_state, and two Transitions: one for agent_0 (confederate) and one for agent_1 (best response).
                """
                train_state_conf, train_state_br, env_state, last_obs, last_dones, last_conf_h, last_br_h, rng = runner_state
                rng, conf_rng, br_rng, step_rng = jax.random.split(rng, 4)

                obs_0 = last_obs["agent_0"]
                obs_1 = last_obs["agent_1"]

                # Get available actions for agent 0 from environment state
                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
                avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

                # Agent_0 (confederate) action
                act_0, (_, val_0), pi_0, new_conf_h = confederate_policy.get_action_value_policy(
                    params=train_state_conf.params,
                    obs=obs_0.reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                    done=last_dones["agent_0"].reshape(1, config["NUM_CONTROLLED_ACTORS"]),  # Get done flag with right shape
                    avail_actions=jax.lax.stop_gradient(avail_actions_0),
                    hstate=last_conf_h,
                    rng=conf_rng
                )
                logp_0 = pi_0.log_prob(act_0)

                act_0 = act_0.squeeze()
                logp_0 = logp_0.squeeze()
                val_0 = val_0.squeeze()

                # Agent 1 (best response) action
                act_1, val_1, pi_1, new_br_h = br_policy.get_action_value_policy(
                    params=train_state_br.params,
                    obs=obs_1.reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                    done=last_dones["agent_1"].reshape(1, config["NUM_CONTROLLED_ACTORS"]),  # Get done flag with right shape
                    avail_actions=jax.lax.stop_gradient(avail_actions_1),
                    hstate=last_br_h,
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
                new_runner_state = (train_state_conf, train_state_br, env_state_next, obs_next, done, new_conf_h, new_br_h, rng)
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
                    minbatch_conf_ego, minbatch_conf_br = batch_infos
                    init_hstate_conf_ego, traj_batch_conf_ego, advantages_conf_ego, returns_conf_ego = minbatch_conf_ego
                    init_hstate_conf_br, traj_batch_conf_br, advantages_conf_br, returns_conf_br = minbatch_conf_br

                    def _loss_fn_conf(params,
                        init_hstate_conf_ego, traj_batch_conf_ego, gae_conf_ego, target_v_conf_ego,
                        init_hstate_conf_br, traj_batch_conf_br, gae_conf_br, target_v_conf_br):
                        # get policy and value of confederate versus ego and best response agents respectively

                        _, (value_conf_ego, _), pi_conf_ego, _ = confederate_policy.get_action_value_policy(
                            params=params,
                            obs=traj_batch_conf_ego.obs,
                            done=traj_batch_conf_ego.done,
                            avail_actions=traj_batch_conf_ego.avail_actions,
                            hstate=init_hstate_conf_ego,
                            rng=jax.random.PRNGKey(0) # only used for action sampling, which is not used here
                        )
                        _, (_, value_conf_br), pi_conf_br, _ = confederate_policy.get_action_value_policy(
                            params=params,
                            obs=traj_batch_conf_br.obs,
                            done=traj_batch_conf_br.done,
                            avail_actions=traj_batch_conf_br.avail_actions,
                            hstate=init_hstate_conf_br,
                            rng=jax.random.PRNGKey(0) # only used for action sampling, which is not used here
                        )

                        log_prob_conf_ego = pi_conf_ego.log_prob(traj_batch_conf_ego.action)
                        log_prob_conf_br = pi_conf_br.log_prob(traj_batch_conf_br.action)

                        # Value loss for interaction with ego agent
                        value_pred_conf_ego_clipped = traj_batch_conf_ego.value + (
                            value_conf_ego - traj_batch_conf_ego.value
                            ).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses_conf_ego = jnp.square(value_conf_ego - target_v_conf_ego)
                        value_losses_clipped_conf_ego = jnp.square(value_pred_conf_ego_clipped - target_v_conf_ego)
                        value_loss_conf_ego = jnp.maximum(value_losses_conf_ego, value_losses_clipped_conf_ego).mean()

                        # Value loss for interaction with best response agent
                        value_pred_conf_br_clipped = traj_batch_conf_br.value + (
                            value_conf_br - traj_batch_conf_br.value
                            ).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses_conf_br = jnp.square(value_conf_br - target_v_conf_br)
                        value_losses_clipped_conf_br = jnp.square(value_pred_conf_br_clipped - target_v_conf_br)
                        value_loss_conf_br = jnp.maximum(value_losses_conf_br, value_losses_clipped_conf_br).mean()

                        # Policy gradient loss for interaction with ego agent
                        ratio_conf_ego = jnp.exp(log_prob_conf_ego - traj_batch_conf_ego.log_prob)
                        gae_norm_conf_ego = (gae_conf_ego - gae_conf_ego.mean()) / (gae_conf_ego.std() + 1e-8)
                        pg_loss_1_conf_ego = ratio_conf_ego * gae_norm_conf_ego
                        pg_loss_2_conf_ego = jnp.clip(
                            ratio_conf_ego,
                            1.0 - config["CLIP_EPS"],
                            1.0 + config["CLIP_EPS"]) * gae_norm_conf_ego
                        pg_loss_conf_ego = -jnp.mean(jnp.minimum(pg_loss_1_conf_ego, pg_loss_2_conf_ego))

                        # Policy gradient loss for interaction with best response agent
                        ratio_conf_br = jnp.exp(log_prob_conf_br - traj_batch_conf_br.log_prob)
                        gae_norm_conf_br = (gae_conf_br - gae_conf_br.mean()) / (gae_conf_br.std() + 1e-8)
                        pg_loss_1_conf_br = ratio_conf_br * gae_norm_conf_br
                        pg_loss_2_conf_br = jnp.clip(
                            ratio_conf_br,
                            1.0 - config["CLIP_EPS"],
                            1.0 + config["CLIP_EPS"]) * gae_norm_conf_br
                        pg_loss_conf_br = -jnp.mean(jnp.minimum(pg_loss_1_conf_br, pg_loss_2_conf_br))

                        # Entropy for interaction with ego agent
                        entropy_conf_ego = jnp.mean(pi_conf_ego.entropy())

                        # Entropy for interaction with best response agent
                        entropy_conf_br = jnp.mean(pi_conf_br.entropy())

                        # We negate the pg_loss_conf_ego to minimize the ego agent's objective
                        conf_ego_loss = - pg_loss_conf_ego + config["VF_COEF"] * value_loss_conf_ego - config["ENT_COEF"] * entropy_conf_ego
                        conf_br_loss = pg_loss_conf_br + config["VF_COEF"] * value_loss_conf_br - config["ENT_COEF"] * entropy_conf_br
                        total_loss = conf_ego_loss + conf_br_loss
                        return total_loss, (value_loss_conf_ego, value_loss_conf_br, pg_loss_conf_ego, pg_loss_conf_br, entropy_conf_ego, entropy_conf_br)

                    grad_fn = jax.value_and_grad(_loss_fn_conf, has_aux=True)
                    (loss_val, aux_vals), grads = grad_fn(
                        train_state_conf.params,
                        init_hstate_conf_ego, traj_batch_conf_ego, advantages_conf_ego, returns_conf_ego,
                        init_hstate_conf_br, traj_batch_conf_br, advantages_conf_br, returns_conf_br)
                    train_state_conf = train_state_conf.apply_gradients(grads=grads)
                    return train_state_conf, (loss_val, aux_vals)

                def _update_minbatch_br(train_state_br, batch_info):
                    init_hstate_br, traj_batch_br, advantages, returns = batch_info
                    def _loss_fn_br(params, init_hstate_br, traj_batch_br, gae, target_v):
                        _, value, pi, _ = br_policy.get_action_value_policy(
                            params=params,
                            obs=traj_batch_br.obs,
                            done=traj_batch_br.done,
                            avail_actions=traj_batch_br.avail_actions,
                            hstate=init_hstate_br,
                            rng=jax.random.PRNGKey(0) # only used for action sampling, which is not used here
                        )
                        log_prob = pi.log_prob(traj_batch_br.action)

                        # Value loss
                        value_pred_clipped = traj_batch_br.value + (
                            value - traj_batch_br.value
                            ).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - target_v)
                        value_losses_clipped = jnp.square(value_pred_clipped - target_v)
                        value_loss = jnp.maximum(value_losses, value_losses_clipped).mean()

                        # Policy gradient loss
                        ratio = jnp.exp(log_prob - traj_batch_br.log_prob)
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

                    grad_fn = jax.value_and_grad(_loss_fn_br, has_aux=True)
                    (loss_val, aux_vals), grads = grad_fn(
                        train_state_br.params, init_hstate_br, traj_batch_br, advantages, returns)
                    train_state_br = train_state_br.apply_gradients(grads=grads)
                    return train_state_br, (loss_val, aux_vals)

                def _update_minbatch_ego(train_state_ego, batch_info):
                    init_hstate_ego, traj_batch_ego, advantages, returns = batch_info
                    def _loss_fn_ego(params, init_hstate_ego, traj_batch_ego, gae, target_v):
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

                    grad_fn = jax.value_and_grad(_loss_fn_ego, has_aux=True)
                    (loss_val, aux_vals), grads = grad_fn(
                        train_state_ego.params, init_hstate_ego, traj_batch_ego, advantages, returns)
                    train_state_ego = train_state_ego.apply_gradients(grads=grads)
                    return train_state_ego, (loss_val, aux_vals)

                (
                    train_state_conf, train_state_br, train_state_ego,
                    traj_batch_conf_ego, traj_batch_ego, traj_batch_conf_br, traj_batch_br,
                    advantages_conf, advantages_ego, advantages_conf_br, advantages_br,
                    targets_conf, targets_ego, targets_conf_br, targets_br,
                    rng_ego, rng_br
                ) = update_state

                init_hstate_ego = ego_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])
                init_hstate_br = br_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])
                init_hstate_conf = confederate_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])

                rng_ego, perm_rng_conf_ego, perm_rng_ego, perm_rng_conf_br, perm_rng_br = jax.random.split(rng_ego, 5)
                # Create minibatches for each agent and interaction type
                minibatches_conf_ego = _create_minibatches(
                    traj_batch_conf_ego, advantages_conf, targets_conf, init_hstate_conf,
                    config["NUM_CONTROLLED_ACTORS"], config["NUM_MINIBATCHES"], perm_rng_conf_ego
                )

                minibatches_ego = _create_minibatches(
                    traj_batch_ego, advantages_ego, targets_ego, init_hstate_ego,
                    config["NUM_CONTROLLED_ACTORS"], config["NUM_MINIBATCHES"], perm_rng_ego
                )

                minibatches_conf_br = _create_minibatches(
                    traj_batch_conf_br, advantages_conf_br, targets_conf_br, init_hstate_conf,
                    config["NUM_CONTROLLED_ACTORS"], config["NUM_MINIBATCHES"], perm_rng_conf_br
                )

                minibatches_br = _create_minibatches(
                    traj_batch_br, advantages_br, targets_br, init_hstate_br,
                    config["NUM_CONTROLLED_ACTORS"], config["NUM_MINIBATCHES"], perm_rng_br
                )

                # Update confederate based on interaction with ego and br
                train_state_conf, all_losses_conf = jax.lax.scan(
                    _update_minbatch_conf, train_state_conf, (minibatches_conf_ego, minibatches_conf_br)
                )

                # Update best response
                train_state_br, all_losses_br = jax.lax.scan(
                    _update_minbatch_br, train_state_br, minibatches_br
                )

                # Update ego agent
                train_state_ego, all_losses_ego = jax.lax.scan(
                    _update_minbatch_ego, train_state_ego, minibatches_ego
                )

                update_state = (
                    train_state_conf, train_state_br, train_state_ego,
                    traj_batch_conf_ego, traj_batch_ego, traj_batch_conf_br, traj_batch_br,
                    advantages_conf, advantages_ego, advantages_conf_br, advantages_br,
                    targets_conf, targets_ego, targets_conf_br, targets_br,
                    rng_ego, rng_br
                )
                return update_state, (all_losses_conf, all_losses_br, all_losses_ego)

            def _update_step(update_runner_state, unused):
                """
                1. Collect rollout for interactions against ego agent.
                2. Collect rollout for interactions against br agent.
                3. Compute advantages for ego-conf and conf-br interactions.
                4. PPO updates for best response and confederate policies.
                """
                (
                    train_state_conf, train_state_br, train_state_ego,
                    env_state_ego, env_state_br,
                    last_obs_ego, last_obs_br,
                    last_dones_ego, last_dones_br,
                    conf_hstate_ego, ego_hstate,
                    conf_hstate_br, br_hstate,
                    rng_ego, rng_br, update_steps
                ) = update_runner_state

                # 1) rollout for interactions of confederate against ego agent
                runner_state_ego = (train_state_conf, train_state_ego, env_state_ego, last_obs_ego, last_dones_ego,
                                    conf_hstate_ego, ego_hstate, rng_ego)
                runner_state_ego, (traj_batch_conf_ego, traj_batch_ego) = jax.lax.scan(
                    _env_step_ego, runner_state_ego, None, config["ROLLOUT_LENGTH"])
                (train_state_conf, train_state_ego, env_state_ego, last_obs_ego, last_dones_ego,
                 conf_hstate_ego, ego_hstate, rng_ego) = runner_state_ego

                # 2) rollout for interactions of confederate against br agent
                runner_state_br = (train_state_conf, train_state_br, env_state_br, last_obs_br,
                                   last_dones_br, conf_hstate_br, br_hstate, rng_br)
                runner_state_br, (traj_batch_conf_br, traj_batch_br) = jax.lax.scan(
                    _env_step_br, runner_state_br, None, config["ROLLOUT_LENGTH"])
                (train_state_conf, train_state_br, env_state_br, last_obs_br, last_dones_br,
                conf_hstate_br, br_hstate, rng_br) = runner_state_br

                # 3a) compute advantage for confederate agent from interaction with ego agent

                # Get available actions for agent 0 from environment state
                avail_actions_0_ego = jax.vmap(env.get_avail_actions)(env_state_ego.env_state)["agent_0"].astype(jnp.float32)

                # Get last value for confederate
                _, (last_val_0_conf_ego, _), _, _ = confederate_policy.get_action_value_policy(
                    params=train_state_conf.params,
                    obs=last_obs_ego["agent_0"].reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                    done=last_dones_ego["agent_0"].reshape(1, config["NUM_CONTROLLED_ACTORS"]),
                    avail_actions=jax.lax.stop_gradient(avail_actions_0_ego),
                    hstate=conf_hstate_ego,
                    rng=jax.random.PRNGKey(0)  # dummy key as we don't sample actions
                )
                last_val_0_conf_ego = last_val_0_conf_ego.squeeze()
                advantages_conf, targets_conf = _calculate_gae(traj_batch_conf_ego, last_val_0_conf_ego)

                # 3b) compute advantage for ego agent from interaction with confederate

                # Get available actions for agent 1 from environment state
                avail_actions_1_ego = jax.vmap(env.get_avail_actions)(env_state_ego.env_state)["agent_1"].astype(jnp.float32)

                # Get last value for ego agent
                _, last_val_1_ego, _, _ = ego_policy.get_action_value_policy(
                    params=train_state_ego.params,
                    obs=last_obs_ego["agent_1"].reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                    done=last_dones_ego["agent_1"].reshape(1, config["NUM_CONTROLLED_ACTORS"]),
                    avail_actions=jax.lax.stop_gradient(avail_actions_1_ego),
                    hstate=ego_hstate,
                    rng=jax.random.PRNGKey(0)  # dummy key as we don't sample actions
                )
                last_val_1_ego = last_val_1_ego.squeeze()
                advantages_ego, targets_ego = _calculate_gae(traj_batch_ego, last_val_1_ego)

                # 3c) compute advantage for confederate agent from interaction with br policy

                # Get available actions for agent 0 from environment state
                avail_actions_0_br = jax.vmap(env.get_avail_actions)(env_state_br.env_state)["agent_0"].astype(jnp.float32)

                # Get last value using agent interface
                _, (_, last_val_0_br), _, _ = confederate_policy.get_action_value_policy(
                    params=train_state_conf.params,
                    obs=last_obs_br["agent_0"].reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                    done=last_dones_br["agent_0"].reshape(1, config["NUM_CONTROLLED_ACTORS"]),
                    avail_actions=jax.lax.stop_gradient(avail_actions_0_br),
                    hstate=conf_hstate_br,
                    rng=jax.random.PRNGKey(0)  # dummy key as we don't sample actions
                )
                last_val_0_br = last_val_0_br.squeeze()
                advantages_conf_br, targets_conf_br = _calculate_gae(traj_batch_conf_br, last_val_0_br)

                # 3d) compute advantage for br policy from interaction with confederate agent
                avail_actions_1_br = jax.vmap(env.get_avail_actions)(env_state_br.env_state)["agent_1"].astype(jnp.float32)
                # Get last value using agent interface
                _, last_val_1_br, _, _ = br_policy.get_action_value_policy(
                    params=train_state_br.params,
                    obs=last_obs_br["agent_1"].reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                    done=last_dones_br["agent_1"].reshape(1, config["NUM_CONTROLLED_ACTORS"]),
                    avail_actions=jax.lax.stop_gradient(avail_actions_1_br),
                    hstate=br_hstate,
                    rng=jax.random.PRNGKey(0)  # dummy key as we don't sample actions
                )
                last_val_1_br = last_val_1_br.squeeze()
                advantages_br, targets_br = _calculate_gae(traj_batch_br, last_val_1_br)

                # 3) PPO update
                update_state = (
                    train_state_conf, train_state_br, train_state_ego,
                    traj_batch_conf_ego, traj_batch_ego, traj_batch_conf_br, traj_batch_br,
                    advantages_conf, advantages_ego, advantages_conf_br, advantages_br,
                    targets_conf, targets_ego, targets_conf_br, targets_br,
                    rng_ego, rng_br
                )
                update_state, (all_losses_conf, all_losses_br, all_losses_ego) = jax.lax.scan(
                    _update_epoch, update_state, None, config["UPDATE_EPOCHS"])
                train_state_conf = update_state[0]
                train_state_br = update_state[1]
                train_state_ego = update_state[2]

                # Metrics
                metric = traj_batch_ego.info
                metric["update_steps"] = update_steps

                # Conf agent losses: value_loss_ego, value_loss_br, pg_loss_ego, pg_loss_br, entropy_ego, entropy_br
                metric["value_loss_conf_against_ego"] = all_losses_conf[1][0]
                metric["value_loss_conf_against_br"] = all_losses_conf[1][1]
                metric["pg_loss_conf_against_ego"] = all_losses_conf[1][2]
                metric["pg_loss_conf_against_br"] = all_losses_conf[1][3]
                metric["entropy_conf_against_ego"] = all_losses_conf[1][4]
                metric["entropy_conf_against_br"] = all_losses_conf[1][5]

                # Ego agent losses
                metric["value_loss_ego"] = all_losses_ego[1][0]
                metric["pg_loss_ego"] = all_losses_ego[1][1]
                metric["entropy_loss_ego"] = all_losses_ego[1][2]

                # br agent losses
                metric["value_loss_br"] = all_losses_br[1][0]
                metric["pg_loss_br"] = all_losses_br[1][1]
                metric["entropy_loss_br"] = all_losses_br[1][2]

                metric["average_rewards_conf_against_ego"] = jnp.mean(traj_batch_conf_ego.reward)
                metric["average_rewards_conf_against_br"] = jnp.mean(traj_batch_conf_br.reward) # redundant with br reward
                metric["average_rewards_ego"] = jnp.mean(traj_batch_ego.reward)
                metric["average_rewards_br"] = jnp.mean(traj_batch_br.reward)


                new_runner_state = (
                    train_state_conf, train_state_br, train_state_ego, env_state_ego, env_state_br,
                    last_obs_ego, last_obs_br, last_dones_ego, last_dones_br,
                    conf_hstate_ego, ego_hstate, conf_hstate_br, br_hstate,
                    rng_ego, rng_br, update_steps + 1
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
                ((
                    train_state_conf, train_state_br, train_state_ego, env_state_ego, env_state_br,
                    last_obs_ego, last_obs_br , last_dones_ego, last_dones_br,
                    conf_hstate_ego, ego_hstate, conf_hstate_br, br_hstate,
                    rng_ego, rng_br, update_steps
                ), checkpoint_array_conf, checkpoint_array_br, checkpoint_array_ego, ckpt_idx,
                    eval_info_br, eval_info_ego) = state_with_ckpt

                # Single PPO update
                (new_runner_state, metric) = _update_step(
                    (train_state_conf, train_state_br, train_state_ego, env_state_ego, env_state_br,
                    last_obs_ego, last_obs_br, last_dones_ego, last_dones_br,
                    conf_hstate_ego, ego_hstate, conf_hstate_br, br_hstate,
                    rng_ego, rng_br, update_steps),
                    None
                )

                (
                    train_state_conf, train_state_br, train_state_ego, env_state_ego, env_state_br,
                    last_obs_ego, last_obs_br, last_dones_ego, last_dones_br,
                    conf_hstate_ego, ego_hstate, conf_hstate_br, br_hstate,
                    rng_ego, rng_br, update_steps
                ) = new_runner_state

                # Decide if we store a checkpoint
                # update steps is 1-indexed because it was incremented at the end of the update step
                to_store = jnp.logical_or(jnp.equal(jnp.mod(update_steps-1, ckpt_and_eval_interval), 0),
                                        jnp.equal(update_steps, config["NUM_UPDATES"]))

                def store_and_eval_ckpt(args):
                    ckpt_arr_and_ep_infos, rng, cidx = args
                    ckpt_arr_conf, ckpt_arr_br, ckpt_arr_ego, prev_ep_infos_br, prev_ep_infos_ego = ckpt_arr_and_ep_infos
                    new_ckpt_arr_conf = jax.tree.map(
                        lambda c_arr, p: c_arr.at[cidx].set(p),
                        ckpt_arr_conf, train_state_conf.params
                    )
                    new_ckpt_arr_br = jax.tree.map(
                        lambda c_arr, p: c_arr.at[cidx].set(p),
                        ckpt_arr_br, train_state_br.params
                    )
                    new_ckpt_arr_ego = jax.tree.map(
                        lambda c_arr, p: c_arr.at[cidx].set(p),
                        ckpt_arr_ego, train_state_ego.params
                    )

                    # run eval episodes
                    rng, eval_rng, = jax.random.split(rng)
                    # conf vs ego
                    last_ep_info_with_ego = run_episodes(eval_rng, env,
                        agent_0_param=train_state_conf.params, agent_0_policy=confederate_policy,
                        agent_1_param=train_state_ego.params, agent_1_policy=ego_policy,
                        max_episode_steps=config["ROLLOUT_LENGTH"], num_eps=config["NUM_EVAL_EPISODES"]
                    )
                    # conf vs br
                    last_ep_info_with_br = run_episodes(eval_rng, env,
                        agent_0_param=train_state_br.params, agent_0_policy=br_policy,
                        agent_1_param=train_state_conf.params, agent_1_policy=confederate_policy,
                        max_episode_steps=config["ROLLOUT_LENGTH"], num_eps=config["NUM_EVAL_EPISODES"]
                    )

                    return ((new_ckpt_arr_conf, new_ckpt_arr_br, new_ckpt_arr_ego, last_ep_info_with_br, last_ep_info_with_ego), rng, cidx + 1)

                def skip_ckpt(args):
                    return args

                (checkpoint_array_and_infos, rng_ego, ckpt_idx) = jax.lax.cond(
                    to_store,
                    store_and_eval_ckpt,
                    skip_ckpt,
                    ((checkpoint_array_conf, checkpoint_array_br, checkpoint_array_ego, eval_info_br, eval_info_ego), rng_ego, ckpt_idx)
                )
                checkpoint_array_conf, checkpoint_array_br, checkpoint_array_ego, ep_info_br, ep_info_ego = checkpoint_array_and_infos

                metric["eval_ep_last_info_br"] = ep_info_br
                metric["eval_ep_last_info_ego"] = ep_info_ego

                return ((train_state_conf, train_state_br, train_state_ego, env_state_ego, env_state_br,
                         last_obs_ego, last_obs_br, last_dones_ego, last_dones_br,
                         conf_hstate_ego, ego_hstate, conf_hstate_br, br_hstate,
                         rng_ego, rng_br, update_steps),
                         checkpoint_array_conf, checkpoint_array_br, checkpoint_array_ego, ckpt_idx,
                         ep_info_br, ep_info_ego), metric

            # init checkpoint array
            checkpoint_array_conf = init_ckpt_array(train_state_conf.params)
            checkpoint_array_br = init_ckpt_array(train_state_br.params)
            checkpoint_array_ego = init_ckpt_array(train_state_ego.params)
            ckpt_idx = 0

            # initial state for scan over _update_step_with_ckpt
            update_steps = 0
            rng, rng_eval_ego, rng_eval_br = jax.random.split(rng, 3)
            ep_infos_ego = run_episodes(rng_eval_ego, env,
                agent_0_param=train_state_conf.params, agent_0_policy=confederate_policy,
                agent_1_param=train_state_ego.params, agent_1_policy=ego_policy,
                max_episode_steps=config["ROLLOUT_LENGTH"], num_eps=config["NUM_EVAL_EPISODES"]
            )
            ep_infos_br = run_episodes(rng_eval_br, env,
                agent_0_param=train_state_br.params, agent_0_policy=br_policy,
                agent_1_param=train_state_ego.params, agent_1_policy=ego_policy,
                max_episode_steps=config["ROLLOUT_LENGTH"], num_eps=config["NUM_EVAL_EPISODES"])

            # Initialize hidden states
            init_ego_hstate = ego_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])
            init_conf_hstate_ego = confederate_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])
            init_conf_hstate_br = confederate_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])
            init_br_hstate = br_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])


            # Initialize done flags
            init_dones_ego = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}
            init_dones_br = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}

            rng, rng_ego, rng_br = jax.random.split(rng, 3)
            update_runner_state = (
                train_state_conf, train_state_br, train_state_ego, env_state_ego, env_state_br,
                obsv_ego, obsv_br, init_dones_ego, init_dones_br,
                init_conf_hstate_ego, init_ego_hstate, init_conf_hstate_br, init_br_hstate,
                rng_ego, rng_br, update_steps
            )
            state_with_ckpt = (
                update_runner_state, checkpoint_array_conf, checkpoint_array_br, checkpoint_array_ego,
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
                final_runner_state, checkpoint_array_conf, checkpoint_array_br, checkpoint_array_ego,
                final_ckpt_idx, last_ep_infos_br, last_ep_infos_ego
            ) = state_with_ckpt

            out = {
                "final_params_conf": final_runner_state[0].params,
                "final_params_br": final_runner_state[1].params,
                "final_params_ego": final_runner_state[2].params,
                "checkpoints_conf": checkpoint_array_conf,
                "checkpoints_br": checkpoint_array_br,
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
    avg_conf_returns_vs_ego = np.asarray(metrics["eval_ep_last_info_ego"]["returned_episode_returns"]).mean(axis=(0, 2, 3))
    avg_conf_returns_vs_br = np.asarray(metrics["eval_ep_last_info_br"]["returned_episode_returns"]).mean(axis=(0, 2, 3))

    # Value losses
    # shape (num_seeds, num_updates, update_epochs, num_minibatches)
    avg_value_losses_conf_vs_ego = np.asarray(metrics["value_loss_conf_against_ego"]).mean(axis=(0, 2, 3))
    avg_value_losses_conf_vs_br = np.asarray(metrics["value_loss_conf_against_br"]).mean(axis=(0, 2, 3))
    avg_value_losses_br = np.asarray(metrics["value_loss_br"]).mean(axis=(0, 2, 3))
    avg_value_losses_ego = np.asarray(metrics["value_loss_ego"]).mean(axis=(0, 2, 3))

    # Actor losses
    # shape (num_seeds, num_updates, update_epochs, num_minibatches)
    avg_actor_losses_conf_vs_ego = np.asarray(metrics["pg_loss_conf_against_ego"]).mean(axis=(0, 2, 3))
    avg_actor_losses_conf_vs_br = np.asarray(metrics["pg_loss_conf_against_br"]).mean(axis=(0, 2, 3))
    avg_actor_losses_br = np.asarray(metrics["pg_loss_br"]).mean(axis=(0, 2, 3))
    avg_actor_losses_ego = np.asarray(metrics["pg_loss_ego"]).mean(axis=(0, 2, 3))

    # Entropy losses
    #  shape (num_seeds, num_updates, update_epochs, num_minibatches)
    avg_entropy_losses_conf_vs_ego = np.asarray(metrics["entropy_conf_against_ego"]).mean(axis=(0, 2, 3))
    avg_entropy_losses_conf_vs_br = np.asarray(metrics["entropy_conf_against_br"]).mean(axis=(0, 2, 3))
    avg_entropy_losses_br = np.asarray(metrics["entropy_loss_br"]).mean(axis=(0, 2, 3))
    avg_entropy_losses_ego = np.asarray(metrics["entropy_loss_ego"]).mean(axis=(0, 2, 3))

    # Rewards
    # shape (num_seeds, num_updates)
    avg_rewards_conf_vs_br = np.asarray(metrics["average_rewards_conf_against_br"]).mean(axis=0)
    avg_rewards_conf_vs_ego = np.asarray(metrics["average_rewards_conf_against_ego"]).mean(axis=0)
    avg_rewards_br = np.asarray(metrics["average_rewards_br"]).mean(axis=0)
    avg_rewards_ego = np.asarray(metrics["average_rewards_ego"]).mean(axis=0)

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
        logger.log_item("Eval/ConfReturn-Against-Ego", avg_conf_returns_vs_ego[step], train_step=step)
        logger.log_item("Eval/ConfReturn-Against-BR", avg_conf_returns_vs_br[step], train_step=step)
        logger.log_item("Eval/EgoRegret", avg_conf_returns_vs_br[step] - avg_conf_returns_vs_ego[step], train_step=step)

        # Confederate losses
        logger.log_item("Losses/ConfValLoss-Against-Ego", avg_value_losses_conf_vs_ego[step], train_step=step)
        logger.log_item("Losses/ConfActorLoss-Against-Ego", avg_actor_losses_conf_vs_ego[step], train_step=step)
        logger.log_item("Losses/ConfEntropy-Against-Ego", avg_entropy_losses_conf_vs_ego[step], train_step=step)

        logger.log_item("Losses/ConfValLoss-Against-BR", avg_value_losses_conf_vs_br[step], train_step=step)
        logger.log_item("Losses/ConfActorLoss-Against-BR", avg_actor_losses_conf_vs_br[step], train_step=step)
        logger.log_item("Losses/ConfEntropy-Against-BR", avg_entropy_losses_conf_vs_br[step], train_step=step)

        # Best response losses
        logger.log_item("Losses/BRValLoss", avg_value_losses_br[step], train_step=step)
        logger.log_item("Losses/BRActorLoss", avg_actor_losses_br[step], train_step=step)
        logger.log_item("Losses/BREntropyLoss", avg_entropy_losses_br[step], train_step=step)

        # Ego agent losses
        logger.log_item("Losses/EgoValLoss", avg_value_losses_ego[step], train_step=step)
        logger.log_item("Losses/EgoActorLoss", avg_actor_losses_ego[step], train_step=step)
        logger.log_item("Losses/EgoEntropyLoss", avg_entropy_losses_ego[step], train_step=step)

        # Rewards
        logger.log_item("Losses/AvgConfEgoRewards", avg_rewards_conf_vs_ego[step], train_step=step)
        logger.log_item("Losses/AvgConfBRRewards", avg_rewards_conf_vs_br[step], train_step=step)
        logger.log_item("Losses/AvgBRRewards", avg_rewards_br[step], train_step=step)
        logger.log_item("Losses/AvgEgoRewards", avg_rewards_ego[step], train_step=step)

    logger.commit()

    # Saving artifacts
    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    out_savepath = save_train_run(outs, savedir, savename="saved_train_run")
    if config["logger"]["log_train_out"]:
        logger.log_artifact(name="saved_train_run", path=out_savepath, type_name="train_run")

    # Cleanup locally logged out file
    if not config["local_logger"]["save_train_out"]:
        shutil.rmtree(out_savepath)

def run_paired(config, wandb_logger):
    algorithm_config = dict(config["algorithm"])

    # Create only one environment instance
    env = make_env(algorithm_config["ENV_NAME"], algorithm_config["ENV_KWARGS"])
    env = LogWrapper(env)

    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"])
    rng, train_rng, eval_rng = jax.random.split(rng, 3)

    # Train using PAIRED algorithm (unified training of ego, confederate, and best response)
    log.info("Starting PAIRED training...")
    start_time = time.time()
    DEBUG = False
    with jax.disable_jit(DEBUG):
        outs = train_paired(algorithm_config, env, train_rng)
    end_time = time.time()
    log.info(f"PAIRED training completed in {end_time - start_time} seconds.")

    # Prepare return values for heldout evaluation
    env = make_env(algorithm_config["ENV_NAME"], algorithm_config["ENV_KWARGS"])
    ego_policy, init_ego_params = initialize_s5_agent(algorithm_config, env, eval_rng)

    # Log metrics
    metric_names = get_metric_names(algorithm_config["ENV_NAME"])
    log_metrics(config, wandb_logger, outs, metric_names)

    return ego_policy, outs["final_params_ego"], init_ego_params

'''
Given an ego agent policy, train a regret-maximizing confederate and best-response pair against the ego policy.
'''
import os
import logging
import time

import hydra
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax
import numpy as np

from agents.agent_interface import ActorWithDoubleCriticPolicy, MLPActorCriticPolicy
from common.plot_utils import get_metric_names, get_stats
from common.save_load_utils import save_train_run
from common.wandb_visualizations import Logger
from common.run_episodes import run_episodes
from common.ppo_utils import Transition, unbatchify
from envs import make_env
from envs.log_wrapper import LogWrapper
from evaluation.vis_episodes import save_video
from evaluation.agent_loader_from_config import initialize_rl_agent_from_config

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
# suppress info logging from matplotlib.animation
logging.getLogger('matplotlib.animation').setLevel(logging.CRITICAL)


def train_regret_maximizing_partners(config, ego_params, ego_policy, env, partner_rng):
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

        config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // (config["ROLLOUT_LENGTH"] + config["ROLLOUT_LENGTH"])// config["NUM_ENVS"]
        config["MINIBATCH_SIZE_EGO"] = (config["NUM_ACTORS"] * config["ROLLOUT_LENGTH"]) // config["NUM_MINIBATCHES"]
        config["MINIBATCH_SIZE_BR"] = (config["NUM_ACTORS"] * config["ROLLOUT_LENGTH"]) // config["NUM_MINIBATCHES"]

        def linear_schedule(count):
            frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
            return config["LR"] * frac

        def train(rng):
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
            
            rng, init_conf_rng, init_br_rng = jax.random.split(rng, 3)
            
            # Initialize parameters using the policy interfaces
            init_params_conf = confederate_policy.init_params(init_conf_rng)
            init_params_br = br_policy.init_params(init_br_rng)

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
                Returns updated runner_state, and a Transition for agent_0.
                """
                train_state_conf, env_state, last_obs, last_dones, last_conf_h, last_ego_h, rng = runner_state
                rng, act_rng, partner_rng, step_rng = jax.random.split(rng, 4)

                obs_0 = last_obs["agent_0"]
                obs_1 = last_obs["agent_1"]

                # Get available actions for agent 0 from environment state
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
                act_1, _, _, new_ego_h = ego_policy.get_action_value_policy(
                    params=ego_params,
                    obs=obs_1.reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                    done=last_dones["agent_1"].reshape(1, config["NUM_CONTROLLED_ACTORS"]),
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
                transition = Transition(
                    done=done["agent_0"],
                    action=act_0,
                    value=val_0,
                    reward=-reward["agent_1"],
                    log_prob=logp_0,
                    obs=obs_0,
                    info=info_0,
                    avail_actions=avail_actions_0
                )
                new_runner_state = (train_state_conf, env_state_next, obs_next, done, new_conf_h, new_ego_h, rng)
                return new_runner_state, transition
            
            def _env_step_br(runner_state, unused):
                """
                agent_0 = confederate, agent_1 = best response
                Returns updated runner_state, and a Transition for agent_0 and agent_1.
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
                    minbatch_ego, minbatch_br = batch_infos
                    traj_batch_ego, advantages_ego, returns_ego = minbatch_ego
                    traj_batch_br, advantages_br, returns_br = minbatch_br

                    init_conf_hstate = confederate_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])
                    def _loss_fn_conf(params, traj_batch_ego, gae_ego, target_v_ego, traj_batch_br, gae_br, target_v_br):
                        # get policy and value of confederate versus ego and best response agents respectively

                        _, (value_ego, _), pi_ego, _ = confederate_policy.get_action_value_policy(
                            params=params, 
                            obs=traj_batch_ego.obs, 
                            done=traj_batch_ego.done,
                            avail_actions=traj_batch_ego.avail_actions,
                            hstate=init_conf_hstate,
                            rng=jax.random.PRNGKey(0) # only used for action sampling, which is not used here 
                        )
                        _, (_, value_br), pi_br, _ = confederate_policy.get_action_value_policy(
                            params=params, 
                            obs=traj_batch_br.obs, 
                            done=traj_batch_br.done,
                            avail_actions=traj_batch_br.avail_actions,
                            hstate=init_conf_hstate,
                            rng=jax.random.PRNGKey(0) # only used for action sampling, which is not used here 
                        )

                        log_prob_ego = pi_ego.log_prob(traj_batch_ego.action)
                        log_prob_br = pi_br.log_prob(traj_batch_br.action)

                        # Value loss for interaction with ego agent
                        value_pred_ego_clipped = traj_batch_ego.value + (
                            value_ego - traj_batch_ego.value
                            ).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses_ego = jnp.square(value_ego - target_v_ego)
                        value_losses_clipped_ego = jnp.square(value_pred_ego_clipped - target_v_ego)
                        value_loss_ego = (
                            0.5 * jnp.maximum(value_losses_ego, value_losses_clipped_ego).mean()
                        )

                        # Value loss for interaction with best response agent
                        value_pred_br_clipped = traj_batch_br.value + (
                            value_br - traj_batch_br.value
                            ).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses_br = jnp.square(value_br - target_v_br)
                        value_losses_clipped_br = jnp.square(value_pred_br_clipped - target_v_br)
                        value_loss_br = (
                            0.5 * jnp.maximum(value_losses_br, value_losses_clipped_br).mean()
                        )

                        # Policy gradient loss for interaction with ego agent
                        ratio_ego = jnp.exp(log_prob_ego - traj_batch_ego.log_prob)
                        gae_norm_ego = (gae_ego - gae_ego.mean()) / (gae_ego.std() + 1e-8)
                        pg_loss_1_ego = ratio_ego * gae_norm_ego
                        pg_loss_2_ego = jnp.clip(
                            ratio_ego, 
                            1.0 - config["CLIP_EPS"], 
                            1.0 + config["CLIP_EPS"]) * gae_norm_ego
                        pg_loss_ego = -jnp.mean(jnp.minimum(pg_loss_1_ego, pg_loss_2_ego))

                        # Policy gradient loss for interaction with best response agent
                        ratio_br = jnp.exp(log_prob_br - traj_batch_br.log_prob)
                        gae_norm_br = (gae_br - gae_br.mean()) / (gae_br.std() + 1e-8)
                        pg_loss_1_br = ratio_br * gae_norm_br
                        pg_loss_2_br = jnp.clip(
                            ratio_br, 
                            1.0 - config["CLIP_EPS"], 
                            1.0 + config["CLIP_EPS"]) * gae_norm_br
                        pg_loss_br = -jnp.mean(jnp.minimum(pg_loss_1_br, pg_loss_2_br))

                        # Entropy for interaction with ego agent
                        entropy_ego = jnp.mean(pi_ego.entropy())
                        
                        # Entropy for interaction with best response agent
                        entropy_br = jnp.mean(pi_br.entropy())
                        ego_loss = pg_loss_ego + config["VF_COEF"] * value_loss_ego - config["ENT_COEF"] * entropy_ego
                        br_loss = pg_loss_br + config["VF_COEF"] * value_loss_br - config["ENT_COEF"] * entropy_br
                        total_loss = (1 - config["CONF_BR_WEIGHT"]) * ego_loss + config["CONF_BR_WEIGHT"] * br_loss
                        return total_loss, (value_loss_ego, value_loss_br, pg_loss_ego, pg_loss_br, entropy_ego, entropy_br)

                    grad_fn = jax.value_and_grad(_loss_fn_conf, has_aux=True)
                    (loss_val, aux_vals), grads = grad_fn(
                        train_state_conf.params, 
                        traj_batch_ego, advantages_ego, returns_ego, 
                        traj_batch_br, advantages_br, returns_br)
                    train_state_conf = train_state_conf.apply_gradients(grads=grads)
                    return train_state_conf, (loss_val, aux_vals)
                
                def _update_minbatch_br(train_state_br, batch_info):
                    traj_batch_br, advantages, returns = batch_info
                    init_br_hstate = br_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])
                    def _loss_fn_br(params, traj_batch_br, gae, target_v):
                        _, value, pi, _ = br_policy.get_action_value_policy(
                            params=params, 
                            obs=traj_batch_br.obs, 
                            done=traj_batch_br.done,
                            avail_actions=traj_batch_br.avail_actions,
                            hstate=init_br_hstate,
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
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

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
                        train_state_br.params, traj_batch_br, advantages, returns)
                    train_state_br = train_state_br.apply_gradients(grads=grads)
                    return train_state_br, (loss_val, aux_vals)


                (
                    train_state_conf, train_state_br, 
                    traj_batch_ego, traj_batch_br_0, traj_batch_br_1, 
                    advantages_conf_ego, advantages_conf_br, advantages_br, 
                    targets_conf_ego, targets_conf_br, targets_br, 
                    rng_ego, rng_br
                ) = update_state

                rng_ego, perm_rng_ego, perm_rng_br0, perm_rng_br1 = jax.random.split(rng_ego, 4)

                # Divide batch size by TWO because we are only training on data of agent_0
                batch_size_ego = config["MINIBATCH_SIZE_EGO"] * config["NUM_MINIBATCHES"] // 2 
                batch_size_br = config["MINIBATCH_SIZE_BR"] * config["NUM_MINIBATCHES"] // 2 
                assert (
                    batch_size_ego == config["ROLLOUT_LENGTH"] * config["NUM_ACTORS"] // 2
                ), "batch size must be equal to number of steps * number of actors"

                permutation_ego = jax.random.permutation(perm_rng_ego, batch_size_ego)
                permutation_br0 = jax.random.permutation(perm_rng_br0, batch_size_br)
                permutation_br1 = jax.random.permutation(perm_rng_br1, batch_size_br)

                batch_ego = (traj_batch_ego, advantages_conf_ego, targets_conf_ego)
                batch_br0 = (traj_batch_br_0, advantages_conf_br, targets_conf_br)
                batch_br1 = (traj_batch_br_1, advantages_br, targets_br)
                
                batch_ego_reshaped = jax.tree.map(
                    lambda x: x.reshape((batch_size_ego,) + x.shape[2:]), batch_ego
                )
                batch_br0_reshaped = jax.tree.map(
                    lambda x: x.reshape((batch_size_br,) + x.shape[2:]), batch_br0
                )
                batch_br1_reshaped = jax.tree.map(
                    lambda x: x.reshape((batch_size_br,) + x.shape[2:]), batch_br1
                )

                shuffled_batch_ego = jax.tree.map(
                    lambda x: jnp.take(x, permutation_ego, axis=0), batch_ego_reshaped
                )
                shuffled_batch_br0 = jax.tree.map(
                    lambda x: jnp.take(x, permutation_br0, axis=0), batch_br0_reshaped
                )
                shuffled_batch_br1 = jax.tree.map(
                    lambda x: jnp.take(x, permutation_br1, axis=0), batch_br1_reshaped
                )

                minibatches_ego = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch_ego,
                )

                minibatches_br0 = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch_br0,
                )

                minibatches_br1 = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch_br1,
                )

                # Update confederate
                train_state_conf, total_loss = jax.lax.scan(
                    _update_minbatch_conf, train_state_conf, (minibatches_ego, minibatches_br0)
                )

                # Update best response
                train_state_br, total_loss_br = jax.lax.scan(
                    _update_minbatch_br, train_state_br, minibatches_br1
                )

                update_state = (train_state_conf, train_state_br, 
                    traj_batch_ego, traj_batch_br_0, traj_batch_br_1, 
                    advantages_conf_ego, advantages_conf_br, advantages_br,
                    targets_conf_ego, targets_conf_br, targets_br, 
                    rng_ego, rng_br)
                return update_state, (total_loss, total_loss_br)

            def _update_step(update_runner_state, unused):
                """
                1. Collect rollout for interactions against ego agent.
                2. Collect rollout for interactions against br agent.
                3. Compute advantages for ego-conf and conf-br interactions.
                4. PPO updates for best response and confederate policies.
                """
                (
                    train_state_conf, train_state_br, 
                    env_state_ego, env_state_br,
                    last_obs_ego, last_obs_br, 
                    last_dones_ego, last_dones_br, 
                    conf_hstate_ego, ego_hstate, 
                    conf_hstate_br, br_hstate, 
                    rng_ego, rng_br, update_steps
                ) = update_runner_state

                # 1) rollout for interactions against ego agent
                runner_state_ego = (train_state_conf, env_state_ego, last_obs_ego, last_dones_ego, 
                                    conf_hstate_ego, ego_hstate, rng_ego)
                runner_state_ego, traj_batch_ego = jax.lax.scan(
                    _env_step_ego, runner_state_ego, None, config["ROLLOUT_LENGTH"])
                (train_state_conf, env_state_ego, last_obs_ego, last_dones_ego, 
                 conf_hstate_ego, ego_hstate, rng_ego) = runner_state_ego

                # 2) rollout for interactions against br agent
                runner_state_br = (train_state_conf, train_state_br, env_state_br, last_obs_br, 
                                   last_dones_br, conf_hstate_br, br_hstate, rng_br)
                runner_state_br, traj_batch_br = jax.lax.scan(
                    _env_step_br, runner_state_br, None, config["ROLLOUT_LENGTH"])
                (train_state_conf, train_state_br, env_state_br, last_obs_br, last_dones_br, 
                conf_hstate_br, br_hstate, rng_br) = runner_state_br

                traj_batch_br_0, traj_batch_br_1 = traj_batch_br

                # 3a) compute advantage for confederate agent from interaction with ego agent

                # Get available actions for agent 0 from environment state
                avail_actions_0_ego = jax.vmap(env.get_avail_actions)(env_state_ego.env_state)["agent_0"].astype(jnp.float32)
                
                # Get last value
                _, (last_val_0_ego, _), _, _ = confederate_policy.get_action_value_policy(
                    params=train_state_conf.params,
                    obs=last_obs_ego["agent_0"].reshape(1, config["NUM_CONTROLLED_ACTORS"], -1), 
                    done=last_dones_ego["agent_0"].reshape(1, config["NUM_CONTROLLED_ACTORS"]),
                    avail_actions=jax.lax.stop_gradient(avail_actions_0_ego),
                    hstate=conf_hstate_ego,
                    rng=jax.random.PRNGKey(0)  # dummy key as we don't sample actions
                )
                last_val_0_ego = last_val_0_ego.squeeze()
                advantages_conf_ego, targets_conf_ego = _calculate_gae(traj_batch_ego, last_val_0_ego)

                # 3b) compute advantage for confederate agent from interaction with br policy

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
                advantages_conf_br, targets_conf_br = _calculate_gae(traj_batch_br_0, last_val_0_br)

                # 3c) compute advantage for br policy from interaction with confederate agent

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
                advantages_br, targets_br = _calculate_gae(traj_batch_br_1, last_val_1_br)

                # 3) PPO update
                update_state = (
                    train_state_conf, train_state_br, 
                    traj_batch_ego, traj_batch_br_0, traj_batch_br_1, 
                    advantages_conf_ego, advantages_conf_br, advantages_br,
                    targets_conf_ego, targets_conf_br, targets_br, 
                    rng_ego, rng_br
                )
                update_state, all_losses = jax.lax.scan(
                    _update_epoch, update_state, None, config["UPDATE_EPOCHS"])
                train_state_conf = update_state[0]
                train_state_br = update_state[1]

                # Metrics
                metric = traj_batch_ego.info
                metric["update_steps"] = update_steps
                metric["value_loss_conf_against_ego"] = all_losses[0][1][0]
                metric["value_loss_conf_against_br"] = all_losses[0][1][1]
                metric["pg_loss_conf_against_ego"] = all_losses[0][1][2]
                metric["pg_loss_conf_against_br"] = all_losses[0][1][3]
                metric["entropy_conf_against_ego"] = all_losses[0][1][4]
                metric["entropy_conf_against_br"] = all_losses[0][1][5]
                metric["average_rewards_br"] = jnp.mean(traj_batch_br_0.reward)
                metric["average_rewards_ego"] = jnp.mean(traj_batch_ego.reward)

                #value_loss_br, pg_loss_br, entropy_loss_br = total_loss_br[1]
                metric["value_loss_br"] = all_losses[1][1][0]
                metric["pg_loss_br"] = all_losses[1][1][1]
                metric["entropy_loss_br"] = all_losses[1][1][2]

                new_runner_state = (
                    train_state_conf, train_state_br, env_state_ego, env_state_br, 
                    last_obs_ego, last_obs_br, last_dones_ego, last_dones_br, 
                    conf_hstate_ego, ego_hstate, conf_hstate_br, br_hstate,  
                    rng_ego, rng_br, update_steps + 1
                )
                return (new_runner_state, metric)

            # --------------------------
            # PPO Update and Checkpoint saving
            # --------------------------
            checkpoint_interval = max(1, config["NUM_UPDATES"] // config["NUM_CHECKPOINTS"])
            num_ckpts = config["NUM_CHECKPOINTS"]

            # Build a PyTree that holds parameters for all conf agent checkpoints
            def init_ckpt_array(params_pytree):
                return jax.tree.map(
                    lambda x: jnp.zeros((num_ckpts,) + x.shape, x.dtype), 
                    params_pytree)
        
            def _update_step_with_ckpt(state_with_ckpt, unused):
                ((
                    train_state_conf, train_state_br, env_state_ego, env_state_br, 
                    last_obs_ego, last_obs_br , last_dones_ego, last_dones_br, 
                    conf_hstate_ego, ego_hstate, conf_hstate_br, br_hstate, 
                    rng_ego, rng_br, update_steps
                ), checkpoint_array_conf, checkpoint_array_br, ckpt_idx, 
                    eval_info_br, eval_info_ego) = state_with_ckpt

                # Single PPO update
                (new_runner_state, metric) = _update_step(
                    (train_state_conf, train_state_br, env_state_ego, env_state_br, 
                    last_obs_ego, last_obs_br, last_dones_ego, last_dones_br, 
                    conf_hstate_ego, ego_hstate, conf_hstate_br, br_hstate, 
                    rng_ego, rng_br, update_steps),
                    None
                )

                (
                    train_state_conf, train_state_br, env_state_ego, env_state_br,
                    last_obs_ego, last_obs_br, last_dones_ego, last_dones_br, 
                    conf_hstate_ego, ego_hstate, conf_hstate_br, br_hstate, 
                    rng_ego, rng_br, update_steps
                ) = new_runner_state

                # Decide if we store a checkpoint
                to_store = jnp.equal(jnp.mod(update_steps, checkpoint_interval), 0)
                
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
                    last_ep_info_with_ego = run_episodes(rng, env, 
                        agent_0_param=train_state_conf.params, agent_0_policy=confederate_policy,
                        agent_1_param=ego_params, agent_1_policy=ego_policy, 
                        max_episode_steps=config["ROLLOUT_LENGTH"], num_eps=config["NUM_EVAL_EPISODES"]
                    )
                    # conf vs br
                    last_ep_info_with_br = run_episodes(rng, env, 
                        agent_0_param=train_state_br.params, agent_0_policy=br_policy,
                        agent_1_param=ego_params, agent_1_policy=ego_policy, 
                        max_episode_steps=config["ROLLOUT_LENGTH"], num_eps=config["NUM_EVAL_EPISODES"]
                    )
                    
                    return ((new_ckpt_arr_conf, new_ckpt_arr_br, last_ep_info_with_br, last_ep_info_with_ego), rng, cidx + 1)

                def skip_ckpt(args):
                    return args

                (checkpoint_array_and_infos, rng_ego, ckpt_idx) = jax.lax.cond(
                    to_store, 
                    store_and_eval_ckpt, 
                    skip_ckpt, 
                    ((checkpoint_array_conf, checkpoint_array_br, eval_info_br, eval_info_ego), rng_ego, ckpt_idx)
                )
                checkpoint_array_conf, checkpoint_array_br, ep_info_br, ep_info_ego = checkpoint_array_and_infos
                
                metric["eval_ep_last_info_br"] = ep_info_br
                metric["eval_ep_last_info_ego"] = ep_info_ego

                return ((train_state_conf, train_state_br, env_state_ego, env_state_br, 
                         last_obs_ego, last_obs_br, last_dones_ego, last_dones_br, 
                         conf_hstate_ego, ego_hstate, conf_hstate_br, br_hstate, 
                         rng_ego, rng_br, update_steps),
                         checkpoint_array_conf, checkpoint_array_br, ckpt_idx, 
                         ep_info_br, ep_info_ego), metric

            # init checkpoint array
            checkpoint_array_conf = init_ckpt_array(train_state_conf.params)
            checkpoint_array_br = init_ckpt_array(train_state_br.params)
            ckpt_idx = 0

            # initial state for scan over _update_step_with_ckpt
            update_steps = 0
            rng, rng_eval_ego, rng_eval_br = jax.random.split(rng, 3)
            # conf vs ego
            ep_infos_ego = run_episodes(rng_eval_ego, env, 
                agent_0_param=train_state_conf.params, agent_0_policy=confederate_policy,
                agent_1_param=ego_params, agent_1_policy=ego_policy, 
                max_episode_steps=config["ROLLOUT_LENGTH"], num_eps=config["NUM_EVAL_EPISODES"]
            )
            # br vs conf
            ep_infos_br = run_episodes(rng_eval_br, env, 
                agent_0_param=train_state_br.params, agent_0_policy=br_policy,
                agent_1_param=train_state_conf.params, agent_1_policy=confederate_policy, 
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
                train_state_conf, train_state_br, env_state_ego, env_state_br, 
                obsv_ego, obsv_br, init_dones_ego, init_dones_br, 
                init_conf_hstate_ego, init_ego_hstate, init_conf_hstate_br, init_br_hstate, 
                rng_ego, rng_br, update_steps
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
    out = train_fn(rngs)
    return out


def run_regret_evaluation(config):
    algorithm_config = dict(config["algorithm"])
    wandb_logger = Logger(config)

    # Create only one environment instance
    env = make_env(algorithm_config["ENV_NAME"], algorithm_config["ENV_KWARGS"])
    env = LogWrapper(env)
    
    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"])
    rng, init_rng, train_rng = jax.random.split(rng, 3)
    ego_agent_config = dict(config["ego_agent"])
    ego_policy, ego_params, _, idx_labels = initialize_rl_agent_from_config(ego_agent_config, "ego", env, init_rng)
    n_ego_agents = len(np.array(idx_labels).flatten())
    assert n_ego_agents == 1, "Regret evaluation only supports a single ego agent currently."
    # TODO: update regret evaluation to vmap over multiple ego agents
    ego_params = jax.tree.map(lambda x: x[0], ego_params)

    # run evaluation
    log.info("Starting regret-maximizing evaluation...")
    start_time = time.time()
    train_out = train_regret_maximizing_partners(algorithm_config, ego_params, ego_policy, env, train_rng)
    end_time = time.time()
    log.info(f"Regret-maximizing evaluation completed in {end_time - start_time} seconds.")
    
    metric_names = get_metric_names(algorithm_config["ENV_NAME"])
    log_metrics(config, env, wandb_logger, train_out, metric_names)
    
    if config["logger"]["log_video"] or config["local_logger"]["save_video"]:
        log_video(config, wandb_logger, train_out, ego_params, ego_policy)
    
    wandb_logger.close()


def log_video(config, logger, train_out, ego_params, ego_policy):
    '''Log video of confederate vs ego and confederate vs br to wandb.'''
    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    env = make_env(config["algorithm"]["ENV_NAME"], config["algorithm"]["ENV_KWARGS"])
    
    # Get the 0th population member
    br_params = jax.tree.map(lambda x: x[0], train_out["final_params_br"])
    conf_params = jax.tree.map(lambda x: x[0], train_out["final_params_conf"])

    # Initialize best response and confederate policies
    env_action_size = env.action_space(env.agents[0]).n
    env_obs_size = env.observation_space(env.agents[0]).shape[0]
    br_policy = MLPActorCriticPolicy(action_dim=env_action_size, obs_dim=env_obs_size)
    conf_policy = ActorWithDoubleCriticPolicy(action_dim=env_action_size, obs_dim=env_obs_size)

    savepath_ego_vs_conf = save_video(
        env, env_name=config["algorithm"]["ENV_NAME"], 
        agent_0_param=ego_params, agent_0_policy=ego_policy, 
        agent_1_param=conf_params, agent_1_policy=conf_policy, 
        max_episode_steps=config["algorithm"]["ROLLOUT_LENGTH"], num_eps=5, 
        savevideo=True, save_dir=savedir, save_name="ego-vs-conf"
    )

    savepath_conf_vs_br = save_video(
        env, env_name=config["algorithm"]["ENV_NAME"], 
        agent_0_param=conf_params, agent_0_policy=conf_policy, 
        agent_1_param=br_params, agent_1_policy=br_policy, 
        max_episode_steps=config["algorithm"]["ROLLOUT_LENGTH"], num_eps=5, 
        savevideo=True, save_dir=savedir, save_name="conf-vs-br"
    )

    if config["logger"]["log_video"]: # log to wandb
        logger.log_video("Eval/Ego-vs-Conf-Video", savepath_ego_vs_conf)
        logger.log_video("Eval/Conf-vs-BR-Video", savepath_conf_vs_br)

    if not config["local_logger"]["save_video"]: # remove video from savepath
        os.remove(savepath_ego_vs_conf)
        os.remove(savepath_conf_vs_br)


def log_metrics(config, env, logger, train_out, metric_names: tuple):
    '''Log metrics to wandb.'''
    metrics = train_out["metrics"]
    num_updates = metrics["returned_episode_returns"].shape[1]

    # shape (num_partner_seeds, num_updates, num_eval_episodes, num_agents_per_env)
    avg_teammate_sp_returns = np.asarray(metrics["eval_ep_last_info_br"]["returned_episode_returns"])[..., 0].mean(axis=(0, 2))
    avg_teammate_xp_returns = np.asarray(metrics["eval_ep_last_info_ego"]["returned_episode_returns"])[..., 0].mean(axis=(0, 2))

    # Conf vs ego, conf vs br, br losses
    #  shape (num_partner_seeds, num_updates, update_epochs, num_minibatches)
    avg_value_losses_teammate_against_ego = np.asarray(metrics["value_loss_conf_against_ego"]).mean(axis=(0, 2, 3))
    avg_value_losses_teammate_against_br = np.asarray(metrics["value_loss_conf_against_br"]).mean(axis=(0, 2, 3)) 
    avg_value_losses_br = np.asarray(metrics["value_loss_br"]).mean(axis=(0, 2, 3))
    
    avg_actor_losses_teammate_against_ego = np.asarray(metrics["pg_loss_conf_against_ego"]).mean(axis=(0, 2, 3)) 
    avg_actor_losses_teammate_against_br = np.asarray(metrics["pg_loss_conf_against_br"]).mean(axis=(0, 2, 3))
    avg_actor_losses_br = np.asarray(metrics["pg_loss_br"]).mean(axis=(0, 2, 3))
    
    avg_entropy_losses_teammate_against_ego = np.asarray(metrics["entropy_conf_against_ego"]).mean(axis=(0, 2, 3))
    avg_entropy_losses_teammate_against_br = np.asarray(metrics["entropy_conf_against_br"]).mean(axis=(0, 2, 3))
    avg_entropy_losses_br = np.asarray(metrics["entropy_loss_br"]).mean(axis=(0, 2, 3))
    
    # shape (num_partner_seeds, num_updates)
    avg_rewards_teammate_against_br = np.asarray(metrics["average_rewards_br"]).mean(axis=0)
    avg_rewards_teammate_against_ego = np.asarray(metrics["average_rewards_ego"]).mean(axis=0)
    
    #########################################################
    all_stats = get_stats(metrics, metric_names)
    all_stats = {k: np.mean(np.array(v), axis=0) for k, v in all_stats.items()}
    
    # Log all partner metrics
    for step in range(num_updates):        
        # Log standard partner stats from get_stats
        for stat_name, stat_data in all_stats.items():
            if step < stat_data.shape[0]:  # Ensure step is within bounds
                stat_mean = stat_data[step, 0]
                logger.log_item(f"Train/Conf-Against-Ego_{stat_name}", stat_mean, train_step=step)

        # Eval metrics
        logger.log_item("Eval/ConfReturn-Against-Ego", avg_teammate_xp_returns[step], train_step=step)
        logger.log_item("Eval/ConfReturn-Against-BR", avg_teammate_sp_returns[step], train_step=step)
        
        # Confederate losses
        logger.log_item("Losses/ConfValLoss-Against-Ego", avg_value_losses_teammate_against_ego[step], train_step=step)
        logger.log_item("Losses/ConfActorLoss-Against-Ego", avg_actor_losses_teammate_against_ego[step], train_step=step)
        logger.log_item("Losses/ConfEntropy-Against-Ego", avg_entropy_losses_teammate_against_ego[step], train_step=step)
        
        logger.log_item("Losses/ConfValLoss-Against-BR", avg_value_losses_teammate_against_br[step], train_step=step)
        logger.log_item("Losses/ConfActorLoss-Against-BR", avg_actor_losses_teammate_against_br[step], train_step=step)
        logger.log_item("Losses/ConfEntropy-Against-BR", avg_entropy_losses_teammate_against_br[step], train_step=step)
        
        # Best response losses
        logger.log_item("Losses/BRValLoss", avg_value_losses_br[step], train_step=step)
        logger.log_item("Losses/BRActorLoss", avg_actor_losses_br[step], train_step=step)
        logger.log_item("Losses/BREntropyLoss", avg_entropy_losses_br[step], train_step=step)
    
        # Rewards
        logger.log_item("Losses/AvgConfEgoRewards", avg_rewards_teammate_against_ego[step], train_step=step)
        logger.log_item("Losses/AvgConfBRRewards", avg_rewards_teammate_against_br[step], train_step=step)

    logger.commit()

    # Saving artifacts
    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    out_savepath = save_train_run(train_out, savedir, savename="saved_train_run")
    if config["logger"]["log_train_out"]:
        logger.log_artifact(name="saved_train_run", path=out_savepath, type_name="train_run")
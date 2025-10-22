import shutil
import time
import logging
import copy
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from functools import partial

from agents.mlp_actor_critic_agent import MLPActorCriticPolicy
from agents.population_interface import AgentPopulation
from agents.initialize_agents import initialize_s5_agent
from common.plot_utils import get_stats, get_metric_names
from common.save_load_utils import save_train_run
from common.run_episodes import run_episodes
from marl.ppo_utils import Transition, unbatchify
from envs import make_env
from envs.log_wrapper import LogWrapper
from ego_agent_training.ppo_ego import train_ppo_ego_agent

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def train_minimax_partners(config, ego_params, ego_policy, env, partner_rng):
    '''
    Train confederate that minimizes the given ego agent using IPPO.
    Return model checkpoints and metrics.
    '''
    def make_minimax_partner_train(config):
        num_agents = env.num_agents
        assert num_agents == 2, "This code assumes the environment has exactly 2 agents."

        config["NUM_ACTORS"] = num_agents * config["NUM_ENVS"]

        # Right now assume control of just 1 agent
        config["NUM_CONTROLLED_ACTORS"] = config["NUM_ENVS"]
        config["NUM_UNCONTROLLED_AGENTS"] = config["NUM_ENVS"]

        config["NUM_UPDATES"] = config["TIMESTEPS_PER_ITER_PARTNER"] // (config["ROLLOUT_LENGTH"] * config["NUM_ENVS"] * config["PARTNER_POP_SIZE"])
        config["MINIBATCH_SIZE"] = (config["NUM_ACTORS"] * config["ROLLOUT_LENGTH"]) // config["NUM_MINIBATCHES"]

        def linear_schedule(count):
            frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
            return config["LR"] * frac

        def train(rng):
            # Initialize confederate policy using ActorWithDoubleCriticPolicy
            confederate_policy = MLPActorCriticPolicy(
                action_dim=env.action_space(env.agents[0]).n,
                obs_dim=env.observation_space(env.agents[0]).shape[0]
            )

            rng, init_conf_rng = jax.random.split(rng, 2)

            # Initialize parameters using the policy interfaces
            init_params_conf = confederate_policy.init_params(init_conf_rng)

            # Define optimizers for both confederate and BR policy
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule if config["ANNEAL_LR"] else config["LR"],
                eps=1e-5),
            )
            train_state_conf = TrainState.create(
                apply_fn=confederate_policy.network.apply,
                params=init_params_conf,
                tx=tx,
            )

            # --------------------------
            # 3b) Init envs and hidden states
            # --------------------------
            rng, reset_rng_ego = jax.random.split(rng, 2)
            reset_rngs_ego = jax.random.split(reset_rng_ego, config["NUM_ENVS"])

            obsv_ego, env_state_ego = jax.vmap(env.reset, in_axes=(0,))(reset_rngs_ego)
            # --------------------------
            # 3c) Define env step
            # --------------------------
            def _env_step(runner_state, unused):
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
                act_0, val_0, pi_0, new_conf_h = confederate_policy.get_action_value_policy(
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
                    reward=reward["agent_1"],
                    log_prob=logp_0,
                    obs=obs_0,
                    info=info_0,
                    avail_actions=avail_actions_0
                )
                new_runner_state = (train_state_conf, env_state_next, obs_next, done, new_conf_h, new_ego_h, rng)
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

            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state_conf, minibatch_ego):
                    traj_batch_ego, advantages_ego, returns_ego = minibatch_ego

                    init_conf_hstate = confederate_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])

                    def _loss_fn(params, traj_batch_ego, gae_ego, target_v_ego):
                        _, value_ego, pi_ego, _ = confederate_policy.get_action_value_policy(
                            params=params,
                            obs=traj_batch_ego.obs,
                            done=traj_batch_ego.done,
                            avail_actions=traj_batch_ego.avail_actions,
                            hstate=init_conf_hstate,
                            rng=jax.random.PRNGKey(0) # only used for action sampling, which is not used here
                        )
                        log_prob_ego = pi_ego.log_prob(traj_batch_ego.action)

                        # Value loss for interaction with ego agent
                        value_pred_ego_clipped = traj_batch_ego.value + (
                            value_ego - traj_batch_ego.value
                            ).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses_ego = jnp.square(value_ego - target_v_ego)
                        value_losses_clipped_ego = jnp.square(value_pred_ego_clipped - target_v_ego)
                        value_loss_ego = jnp.maximum(value_losses_ego, value_losses_clipped_ego).mean()

                        # Policy gradient loss for interaction with ego agent
                        ratio_ego = jnp.exp(log_prob_ego - traj_batch_ego.log_prob)
                        gae_norm_ego = (gae_ego - gae_ego.mean()) / (gae_ego.std() + 1e-8)
                        pg_loss_1_ego = ratio_ego * gae_norm_ego
                        pg_loss_2_ego = jnp.clip(
                            ratio_ego,
                            1.0 - config["CLIP_EPS"],
                            1.0 + config["CLIP_EPS"]) * gae_norm_ego
                        pg_loss_ego = -jnp.mean(jnp.minimum(pg_loss_1_ego, pg_loss_2_ego))

                        # Entropy for interaction with ego agent
                        entropy_ego = jnp.mean(pi_ego.entropy())

                        # We negate the pg loss to minimize the ego agent's objective
                        total_loss = -pg_loss_ego + config["VF_COEF"] * value_loss_ego - config["ENT_COEF"] * entropy_ego
                        return total_loss, (value_loss_ego, pg_loss_ego, entropy_ego)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (loss_val, aux_vals), grads = grad_fn(
                        train_state_conf.params,
                        traj_batch_ego, advantages_ego, returns_ego)
                    train_state_conf = train_state_conf.apply_gradients(grads=grads)
                    return train_state_conf, (loss_val, aux_vals)


                (
                    train_state_conf,
                    traj_batch_ego,
                    advantages_conf_ego,
                    targets_conf_ego,
                    rng_ego
                ) = update_state

                rng_ego, perm_rng_ego = jax.random.split(rng_ego, 2)

                # Divide batch size by TWO because we are only training on data of agent_0
                batch_size_ego = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"] // 2
                assert (
                    batch_size_ego == config["ROLLOUT_LENGTH"] * config["NUM_ACTORS"] // 2
                ), "batch size must be equal to number of steps * number of actors"

                permutation_ego = jax.random.permutation(perm_rng_ego, batch_size_ego)
                batch_ego = (traj_batch_ego, advantages_conf_ego, targets_conf_ego)

                batch_ego_reshaped = jax.tree.map(
                    lambda x: x.reshape((batch_size_ego,) + x.shape[2:]), batch_ego
                )

                shuffled_batch_ego = jax.tree.map(
                    lambda x: jnp.take(x, permutation_ego, axis=0), batch_ego_reshaped
                )

                minibatches_ego = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch_ego,
                )

                # Update confederate
                train_state_conf, (total_loss, aux_vals) = jax.lax.scan(
                    _update_minbatch, train_state_conf, minibatches_ego
                )

                update_state = (train_state_conf,
                    traj_batch_ego,
                    advantages_conf_ego,
                    targets_conf_ego,
                    rng_ego)
                return update_state, (total_loss, aux_vals)

            def _update_step(update_runner_state, unused):
                """
                1. Collect rollout for interactions against ego agent.
                2. Compute advantages for ego-conf interactions.
                3. PPO updates for confederate policy.
                """
                (
                    train_state_conf,
                    env_state_ego,
                    last_obs_ego, last_dones_ego,
                    conf_hstate_ego, ego_hstate,
                    rng_ego, update_steps
                ) = update_runner_state

                # 1) rollout for interactions against ego agent
                runner_state_ego = (train_state_conf, env_state_ego, last_obs_ego, last_dones_ego,
                                    conf_hstate_ego, ego_hstate, rng_ego)
                runner_state_ego, traj_batch_ego = jax.lax.scan(
                    _env_step, runner_state_ego, None, config["ROLLOUT_LENGTH"])
                (train_state_conf, env_state_ego, last_obs_ego, last_dones_ego,
                 conf_hstate_ego, ego_hstate, rng_ego) = runner_state_ego

                # 2) compute advantage for confederate agent from interaction with ego agent

                # Get available actions for agent 0 from environment state
                avail_actions_0_ego = jax.vmap(env.get_avail_actions)(env_state_ego.env_state)["agent_0"].astype(jnp.float32)

                # Get last value
                _, last_val_0_ego, _, _ = confederate_policy.get_action_value_policy(
                    params=train_state_conf.params,
                    obs=last_obs_ego["agent_0"].reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                    done=last_dones_ego["agent_0"].reshape(1, config["NUM_CONTROLLED_ACTORS"]),
                    avail_actions=jax.lax.stop_gradient(avail_actions_0_ego),
                    hstate=conf_hstate_ego,
                    rng=jax.random.PRNGKey(0)  # dummy key as we don't sample actions
                )
                last_val_0_ego = last_val_0_ego.squeeze()
                advantages_conf_ego, targets_conf_ego = _calculate_gae(traj_batch_ego, last_val_0_ego)

                # 3) PPO update
                update_state = (
                    train_state_conf,
                    traj_batch_ego,
                    advantages_conf_ego,
                    targets_conf_ego,
                    rng_ego
                )
                update_state, (total_loss, aux_vals) = jax.lax.scan(
                    _update_epoch, update_state, None, config["UPDATE_EPOCHS"])
                train_state_conf = update_state[0]
                value_loss_ego, pg_loss_ego, entropy_ego = aux_vals

                # Metrics
                metric = traj_batch_ego.info
                metric["update_steps"] = update_steps
                metric["value_loss_conf_against_ego"] = value_loss_ego
                metric["pg_loss_conf_against_ego"] = pg_loss_ego
                metric["entropy_conf_against_ego"] = entropy_ego
                metric["average_rewards_ego"] = jnp.mean(traj_batch_ego.reward)

                new_runner_state = (
                    train_state_conf, env_state_ego,
                    last_obs_ego, last_dones_ego,
                    conf_hstate_ego, ego_hstate,
                    rng_ego, update_steps + 1
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
                    train_state_conf, env_state_ego,
                    last_obs_ego, last_dones_ego,
                    conf_hstate_ego, ego_hstate,
                    rng_ego, update_steps
                ), checkpoint_array_conf, ckpt_idx,
                    eval_info_ego) = state_with_ckpt

                # Single PPO update
                (new_runner_state, metric) = _update_step(
                    (train_state_conf, env_state_ego,
                    last_obs_ego, last_dones_ego,
                    conf_hstate_ego, ego_hstate,
                    rng_ego, update_steps),
                    None
                )

                (
                    train_state_conf, env_state_ego,
                    last_obs_ego, last_dones_ego,
                    conf_hstate_ego, ego_hstate,
                    rng_ego, update_steps
                ) = new_runner_state

                # Decide if we store a checkpoint
                # update steps is 1-indexed because it was incremented at the end of the update step
                to_store = jnp.logical_or(jnp.equal(jnp.mod(update_steps-1, ckpt_and_eval_interval), 0),
                                        jnp.equal(update_steps, config["NUM_UPDATES"]))

                def store_and_eval_ckpt(args):
                    ckpt_arr_and_ep_infos, rng, cidx = args
                    ckpt_arr_conf, prev_ep_infos_ego = ckpt_arr_and_ep_infos
                    new_ckpt_arr_conf = jax.tree.map(
                        lambda c_arr, p: c_arr.at[cidx].set(p),
                        ckpt_arr_conf, train_state_conf.params
                    )
                    # run eval episodes
                    rng, eval_rng, = jax.random.split(rng)
                    # conf vs ego
                    last_ep_info_with_ego = run_episodes(eval_rng, env,
                        agent_0_param=train_state_conf.params, agent_0_policy=confederate_policy,
                        agent_1_param=ego_params, agent_1_policy=ego_policy,
                        max_episode_steps=config["ROLLOUT_LENGTH"], num_eps=config["NUM_EVAL_EPISODES"]
                    )

                    return ((new_ckpt_arr_conf, last_ep_info_with_ego), rng, cidx + 1)

                def skip_ckpt(args):
                    return args

                (checkpoint_array_and_infos, rng_ego, ckpt_idx) = jax.lax.cond(
                    to_store,
                    store_and_eval_ckpt,
                    skip_ckpt,
                    ((checkpoint_array_conf, eval_info_ego), rng_ego, ckpt_idx)
                )
                checkpoint_array_conf, ep_info_ego = checkpoint_array_and_infos

                metric["eval_ep_last_info_ego"] = ep_info_ego

                return ((train_state_conf, env_state_ego,
                         last_obs_ego, last_dones_ego,
                         conf_hstate_ego, ego_hstate,
                         rng_ego, update_steps),
                         checkpoint_array_conf, ckpt_idx,
                         ep_info_ego), metric

            # init checkpoint array
            checkpoint_array_conf = init_ckpt_array(train_state_conf.params)
            ckpt_idx = 0

            # initial state for scan over _update_step_with_ckpt
            update_steps = 0
            rng, rng_eval_ego = jax.random.split(rng, 2)
            ep_infos_ego = run_episodes(rng_eval_ego, env,
                agent_0_param=train_state_conf.params, agent_0_policy=confederate_policy,
                agent_1_param=ego_params, agent_1_policy=ego_policy,
                max_episode_steps=config["ROLLOUT_LENGTH"], num_eps=config["NUM_EVAL_EPISODES"]
            )

            # Initialize hidden states
            init_ego_hstate = ego_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])
            init_conf_hstate_ego = confederate_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])

            # Initialize done flags
            init_dones_ego = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}

            rng, rng_ego = jax.random.split(rng, 2)
            update_runner_state = (
                train_state_conf, env_state_ego,
                obsv_ego, init_dones_ego,
                init_conf_hstate_ego, init_ego_hstate,
                rng_ego, update_steps
            )
            state_with_ckpt = (
                update_runner_state, checkpoint_array_conf,
                ckpt_idx, ep_infos_ego
            )
            # run training
            state_with_ckpt, metrics = jax.lax.scan(
                _update_step_with_ckpt,
                state_with_ckpt,
                xs=None,
                length=config["NUM_UPDATES"]
            )
            (
                final_runner_state, checkpoint_array_conf,
                final_ckpt_idx, last_ep_infos_ego
            ) = state_with_ckpt

            out = {
                "final_params": final_runner_state[0].params,
                "checkpoints": checkpoint_array_conf,
                "metrics": metrics,  # shape (NUM_UPDATES, ...)
            }
            return out

        return train
    # ------------------------------
    # Actually run the teammate training
    # ------------------------------
    rngs = jax.random.split(partner_rng, config["PARTNER_POP_SIZE"])
    train_fn = jax.jit(jax.vmap(make_minimax_partner_train(config)))
    out = train_fn(rngs)
    return out

def open_ended_training_step(carry, ego_policy, partner_population, config, env):
    '''
    Train the ego agent against the adversarial partner.
    '''
    prev_ego_params, rng = carry
    rng, partner_rng, ego_rng = jax.random.split(rng, 3)

    # Train partner agents with ego_policy
    train_out = train_minimax_partners(config, prev_ego_params, ego_policy, env, partner_rng)
    train_partner_params = train_out["final_params"]

    # Train ego agent using train_ppo_ego_agent
    config["TOTAL_TIMESTEPS"] = config["TIMESTEPS_PER_ITER_EGO"]
    ego_out = train_ppo_ego_agent(
        config=config,
        env=env,
        train_rng=ego_rng,
        ego_policy=ego_policy,
        init_ego_params=prev_ego_params,
        n_ego_train_seeds=1,
        partner_population=partner_population,
        partner_params=train_partner_params
    )

    updated_ego_parameters = ego_out["final_params"]
    # remove initial dimension of 1, to ensure that input and output carry have the same dimension
    updated_ego_parameters = jax.tree.map(lambda x: x.squeeze(axis=0), updated_ego_parameters)

    carry = (updated_ego_parameters, rng)
    return carry, (train_out, ego_out)

def train_minimax(rng, env, algorithm_config, partner_population, ego_config):
    rng, init_rng, train_rng = jax.random.split(rng, 3)

    # Initialize ego agent
    ego_policy, init_ego_params = initialize_s5_agent(ego_config, env, init_rng)

    @jax.jit
    def open_ended_step_fn(carry, unused):
        return open_ended_training_step(carry, ego_policy, partner_population, algorithm_config, env)

    init_carry = (init_ego_params, train_rng)
    final_carry, outs = jax.lax.scan(
        open_ended_step_fn,
        init_carry,
        xs=None,
        length=algorithm_config["NUM_OPEN_ENDED_ITERS"]
    )
    return outs

def run_minimax(config, wandb_logger):
    algorithm_config = dict(config["algorithm"])

    # Create only one environment instance
    env = make_env(algorithm_config["ENV_NAME"], algorithm_config["ENV_KWARGS"])
    env = LogWrapper(env)

    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"])
    rng, init_ego_rng = jax.random.split(rng)
    rngs = jax.random.split(rng, algorithm_config["NUM_SEEDS"])

    # Initialize partner policy once - reused for all iterations
    partner_policy = MLPActorCriticPolicy(
        action_dim=env.action_space(env.agents[1]).n,
        obs_dim=env.observation_space(env.agents[1]).shape[0]
    )

    # Create partner population
    partner_population = AgentPopulation(
        pop_size=algorithm_config["PARTNER_POP_SIZE"],
        policy_cls=partner_policy
    )

    # initialize ego config
    ego_config = copy.deepcopy(algorithm_config)
    ego_config["TOTAL_TIMESTEPS"] = algorithm_config["TIMESTEPS_PER_ITER_EGO"]
    EGO_ARGS = algorithm_config.get("EGO_ARGS", {})
    ego_config.update(EGO_ARGS)

    log.info("Starting open-ended minimax training...")
    start_time = time.time()

    DEBUG = False
    with jax.disable_jit(DEBUG):
        train_fn = jax.jit(jax.vmap(partial(train_minimax,
                env=env, algorithm_config=algorithm_config,
                partner_population=partner_population,
                ego_config=ego_config
                )
            )
        )
        outs = train_fn(rngs)

    end_time = time.time()
    log.info(f"Open-ended minimax training completed in {end_time - start_time} seconds.")

    # Prepare return values for heldout evaluation
    _ , ego_outs = outs
    ego_params = jax.tree.map(lambda x: x[:, :, 0], ego_outs["final_params"]) # shape (num_seeds, num_open_ended_iters, 1, num_ckpts, leaf_dim)
    ego_policy, init_ego_params = initialize_s5_agent(algorithm_config, env, init_ego_rng)

    # Log metrics
    metric_names = get_metric_names(algorithm_config["ENV_NAME"])
    log_metrics(config, wandb_logger, outs, metric_names)
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
    teammate_metrics = teammate_outs["metrics"] # conf vs ego
    ego_metrics = ego_outs["metrics"]

    num_seeds = teammate_metrics["returned_episode_returns"].shape[0]
    num_open_ended_iters = ego_metrics["returned_episode_returns"].shape[1]
    num_partner_updates = teammate_metrics["returned_episode_returns"].shape[3]
    num_ego_updates = ego_metrics["returned_episode_returns"].shape[3]

    # Extract partner train stats
    teammate_metrics = jax.tree.map(lambda x: x, teammate_metrics)
    teammate_stats = get_stats(teammate_metrics, metric_names) # shape (num_seeds, num_open_ended_iters, num_partner_seeds, num_partner_updates, 2)
    teammate_stat_means = jax.tree.map(lambda x: np.mean(x, axis=(0, 2))[..., 0], teammate_stats) # shape (num_open_ended_iters, num_partner_updates)

    # Extract ego train stats
    ego_stats = get_stats(ego_metrics, metric_names) # shape (num_seeds, num_open_ended_iters, num_ego_seeds, num_ego_updates, 2)
    ego_stat_means = jax.tree.map(lambda x: np.mean(x, axis=(0, 2))[..., 0], ego_stats) # shape (num_open_ended_iters, num_ego_updates)

    # Process/extract minimax-specific losses
    # shape (num_seeds, num_open_ended_iters, num_partner_seeds, num_updates, num_eval_episodes, num_agents_per_env)
    avg_teammate_xp_returns = np.asarray(teammate_metrics["eval_ep_last_info_ego"]["returned_episode_returns"]).mean(axis=(0, 2, 4, 5))

    # Conf vs ego losses
    #  shape (num_seeds, num_open_ended_iters, num_partner_seeds, num_updates, update_epochs, num_minibatches)
    avg_value_losses_teammate_against_ego = np.asarray(teammate_metrics["value_loss_conf_against_ego"]).mean(axis=(0, 2, 4, 5))
    avg_actor_losses_teammate_against_ego = np.asarray(teammate_metrics["pg_loss_conf_against_ego"]).mean(axis=(0, 2, 4, 5))
    avg_entropy_losses_teammate_against_ego = np.asarray(teammate_metrics["entropy_conf_against_ego"]).mean(axis=(0, 2, 4, 5))

    # shape (num_seeds, num_open_ended_iters, num_partner_seeds, num_updates)
    avg_rewards_teammate_against_ego = np.asarray(teammate_metrics["average_rewards_ego"]).mean(axis=(0, 2))

    # Process ego-specific metrics
    # shape (num_seeds, num_open_ended_iters, num_ego_seeds, num_updates, num_partners, num_eval_episodes, num_agents_per_env)
    avg_ego_returns = np.asarray(ego_metrics["eval_ep_last_info"]["returned_episode_returns"]).mean(axis=(0, 2, 4, 5, 6))

    # shape (num_seeds, num_open_ended_iters, num_ego_seeds, num_updates, update_epochs, num_minibatches)
    avg_ego_value_losses = np.asarray(ego_metrics["value_loss"]).mean(axis=(0, 2, 4, 5))
    avg_ego_actor_losses = np.asarray(ego_metrics["actor_loss"]).mean(axis=(0, 2, 4, 5))
    avg_ego_entropy_losses = np.asarray(ego_metrics["entropy_loss"]).mean(axis=(0, 2, 4, 5))
    avg_ego_grad_norms = np.asarray(ego_metrics["avg_grad_norm"]).mean(axis=(0, 2, 4, 5))
    for iter_idx in range(num_open_ended_iters):
        # Log all partner metrics
        for step in range(num_partner_updates):
            global_step = iter_idx * num_partner_updates + step

            # Log standard partner stats from get_stats
            for stat_name, stat_data in teammate_stat_means.items():
                logger.log_item(f"Train/Conf-Against-Ego_{stat_name}", stat_data[iter_idx, step], train_step=global_step)

            # Minimax partner eval metrics
            logger.log_item("Eval/ConfReturn-Against-Ego", avg_teammate_xp_returns[iter_idx][step], train_step=global_step)

            # Confederate losses
            logger.log_item("Losses/ConfValLoss-Against-Ego", avg_value_losses_teammate_against_ego[iter_idx][step], train_step=global_step)
            logger.log_item("Losses/ConfActorLoss-Against-Ego", avg_actor_losses_teammate_against_ego[iter_idx][step], train_step=global_step)
            logger.log_item("Losses/ConfEntropy-Against-Ego", avg_entropy_losses_teammate_against_ego[iter_idx][step], train_step=global_step)

            # Rewards
            logger.log_item("Losses/AvgConfEgoRewards", avg_rewards_teammate_against_ego[iter_idx][step], train_step=global_step)

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

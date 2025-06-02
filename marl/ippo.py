'''
Based on the IPPO implementation from JaxMarl. Trains a parameter-shared, MLP IPPO agent on a
fully cooperative multi-agent environment. Note that this code is only compatible with MLP policies.
'''
import shutil

import hydra
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from agents.initialize_agents import initialize_s5_agent, initialize_mlp_agent, \
    initialize_rnn_agent, initialize_pseudo_actor_with_double_critic, initialize_pseudo_actor_with_conditional_critic
from common.plot_utils import get_stats, get_metric_names
from common.save_load_utils import save_train_run
from envs import make_env
from envs.log_wrapper import LogWrapper
from marl.ppo_utils import Transition, batchify, unbatchify, _create_minibatches


def initialize_agent(actor_type, algorithm_config, env, init_rng):
    if actor_type == "s5":
        policy, init_params = initialize_s5_agent(algorithm_config, env, init_rng)
    elif actor_type == "mlp":
        policy, init_params = initialize_mlp_agent(algorithm_config, env, init_rng)
    elif actor_type == "rnn":
        policy, init_params = initialize_rnn_agent(algorithm_config, env, init_rng)
    elif actor_type == "pseudo_actor_with_double_critic":
        policy, init_params = initialize_pseudo_actor_with_double_critic(algorithm_config, env, init_rng)
    elif actor_type == "pseudo_actor_with_conditional_critic":
        policy, init_params = initialize_pseudo_actor_with_conditional_critic(algorithm_config, env, init_rng)
    return policy, init_params

def make_train(config, env):
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["ROLLOUT_LENGTH"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["ROLLOUT_LENGTH"] // config["NUM_MINIBATCHES"]
    )

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        rng, init_rng = jax.random.split(rng)
        policy, init_params = initialize_agent(config["ACTOR_TYPE"], config, env, init_rng)

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

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, last_hstate, rng = runner_state

                rng, act_rng = jax.random.split(rng)

                last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                last_done_batch = batchify(last_done, env.agents, config["NUM_ACTORS"])

                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(batchify(avail_actions, 
                    env.agents, config["NUM_ACTORS"]).astype(jnp.float32))

                action, value, pi, new_hstate = policy.get_action_value_policy(
                    params=train_state.params,
                    obs=last_obs_batch.reshape(1, config["NUM_ACTORS"], -1),
                    done=last_done_batch.reshape(1, config["NUM_ACTORS"]),
                    avail_actions=avail_actions.reshape(1, config["NUM_ACTORS"], -1),
                    hstate=last_hstate,
                    rng=act_rng
                )
                log_prob = pi.log_prob(action)

                action = action.squeeze()
                log_prob = log_prob.squeeze()
                value = value.squeeze()

                env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
                env_act = {k:v.flatten() for k,v in env_act.items()}

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                new_obs, new_env_state, reward, new_done, info = jax.vmap(env.step, in_axes=(0,0,0))(
                    rng_step, env_state, env_act
                )
                
                # note that num_actors = num_envs * num_agents
                info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)

                transition = Transition(
                    batchify(new_done, env.agents, config["NUM_ACTORS"]).squeeze(),
                    action,
                    value,
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob,
                    last_obs_batch,
                    info,
                    avail_actions
                )
                runner_state = (train_state, new_env_state, new_obs, new_done, new_hstate, rng)
                return runner_state, transition
            
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["ROLLOUT_LENGTH"]
            )

            # Get final value estimate for completed trajectory
            train_state, env_state, last_obs, last_done, last_hstate, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            last_obs_batch = last_obs_batch.reshape(1, config["NUM_ACTORS"], -1)
            last_done_batch = batchify(last_done, env.agents, config["NUM_ACTORS"])
            last_done_batch = last_done_batch.reshape(1, config["NUM_ACTORS"])
            last_avail_batch = jax.vmap(env.get_avail_actions)(env_state.env_state)
            last_avail_batch = jax.lax.stop_gradient(batchify(last_avail_batch, 
                env.agents, config["NUM_ACTORS"]).astype(jnp.float32))
            
            _, last_val, _, _ = policy.get_action_value_policy(
                params=train_state.params,
                obs=last_obs_batch,
                done=last_done_batch,
                avail_actions=last_avail_batch,
                hstate=last_hstate,
                rng=jax.random.PRNGKey(0)  # Dummy key since we're just extracting the value
            )
            last_val = last_val.squeeze()

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

            advantages, targets = _calculate_gae(traj_batch, last_val)

            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info
                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, value, pi, _ = policy.get_action_value_policy(
                            params=params,
                            obs=traj_batch.obs,
                            done=traj_batch.done,
                            avail_actions=traj_batch.avail_actions,
                            hstate=init_hstate,
                            rng=jax.random.PRNGKey(0) # only used for action sampling, which is unused here
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, init_hstate, traj_batch, advantages, targets, rng = update_state
                rng, perm_rng = jax.random.split(rng)
                minibatches = _create_minibatches(traj_batch, advantages, targets, init_hstate, 
                                                  config["NUM_ACTORS"], config["NUM_MINIBATCHES"], perm_rng)

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
                return update_state, total_loss
            init_hstate = policy.init_hstate(config["NUM_ACTORS"])
            update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            metric["update_steps"] = update_steps
            
            rng = update_state[-1]
            update_steps += 1
            runner_state = (train_state, env_state, last_obs, last_done, last_hstate, rng)
            return (runner_state, update_steps), metric

        ckpt_and_eval_interval = config["NUM_UPDATES"] // max(1, config["NUM_CHECKPOINTS"] - 1)
        num_ckpts = config["NUM_CHECKPOINTS"]

        # build a pytree that can hold the parameters for all checkpoints.
        def init_ckpt_array(params_pytree):
            return jax.tree.map(
                lambda x: jnp.zeros((num_ckpts,) + x.shape, x.dtype),
                params_pytree
            )

        def _update_step_with_checkpoint(update_with_ckpt_runner_state, unused):
            (update_runner_state, checkpoint_array, ckpt_idx) = update_with_ckpt_runner_state
            # update_runner_state is ((train_state, env_state, obs, done, hstate, rng), update_steps)
            # Run one PPO update step
            update_runner_state, metric = _update_step(update_runner_state, None)
            _, update_steps = update_runner_state
            # update steps is 1-indexed because it was incremented at the end of the update step
            to_store = jnp.logical_or(jnp.equal(jnp.mod(update_steps-1, ckpt_and_eval_interval), 0),
                                      jnp.equal(update_steps, config["NUM_UPDATES"]))

            def store_ckpt_fn(args):
                # Write current runner_state[0].params into checkpoint_array at ckpt_idx
                # and increment ckpt_idx
                _checkpoint_array, _ckpt_idx = args
                new_checkpoint_array = jax.tree.map(
                    lambda c_arr, p: c_arr.at[_ckpt_idx].set(p),
                    _checkpoint_array,
                    update_runner_state[0][0].params
                )
                return new_checkpoint_array, _ckpt_idx + 1 
            # TODO: potential issue is that if this function is always executed regardless of whether to_store is true or false, then _ckpt_idx will be wrong

            def skip_ckpt_fn(args):
                return args  # No changes if we don't store

            checkpoint_array, ckpt_idx = jax.lax.cond(
                to_store, # if to_store, execute true function(operand). else, execute false function(operand).
                store_ckpt_fn, # true fn
                skip_ckpt_fn, # false fn
                (checkpoint_array, ckpt_idx),
            )

            runner_state = (update_runner_state, checkpoint_array, ckpt_idx)
            return runner_state, metric

        # (5) Use lax.scan over NUM_UPDATES
        rng, _rng = jax.random.split(rng)
        update_steps = 0
        init_hstate = policy.init_hstate(config["NUM_ACTORS"])
        init_done = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}
        update_runner_state = ((train_state, env_state, obsv, init_done, init_hstate, _rng), update_steps)
        checkpoint_array = init_ckpt_array(train_state.params)
        ckpt_idx = 0
        update_with_ckpt_runner_state = (update_runner_state, checkpoint_array, ckpt_idx)

        runner_state, metrics = jax.lax.scan(
            _update_step_with_checkpoint,
            update_with_ckpt_runner_state,
            xs=None,  # No per-step input data
            length=config["NUM_UPDATES"],
        )

        update_runner_state, checkpoint_array, final_ckpt_idx = runner_state

        return {
            "final_params": update_runner_state[0][0].params,
            "metrics": metrics,
            "checkpoints": checkpoint_array,
            "final_ckpt_idx": final_ckpt_idx # CLEANUP FLAG
        }
    return train

def run_ippo(config, logger):
    algorithm_config = dict(config.algorithm)
    env = make_env(algorithm_config["ENV_NAME"], algorithm_config["ENV_KWARGS"])
    env = LogWrapper(env)

    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"])
    rngs = jax.random.split(rng, algorithm_config["NUM_SEEDS"])
    
    with jax.disable_jit(False):
        train_jit = jax.jit(jax.vmap(make_train(algorithm_config, env)))
        out = train_jit(rngs)

    log_metrics(config, out, logger)
    return out

def log_metrics(config, out, logger):
    '''Save train run output and log to wandb as artifact.'''    
    train_metrics = out["metrics"]
    metric_names = get_metric_names(config["ENV_NAME"])
    train_stats = get_stats(train_metrics, metric_names)

    # each key in train_stats is a metric name, and the value is an array of shape (num_seeds, num_updates, num_agents_per_game)
    # where the last dimension contains the mean and std of the metric
    train_stats = {k: np.mean(np.array(v), axis=0) for k, v in train_stats.items()}

    # Log metrics for each update step
    num_updates = train_metrics["returned_episode"].shape[1] # shape is (num_seeds, num_updates, rollout_len, num_envs*num_agents_per_game)
    for step in range(num_updates):
        for stat_name, stat_data in train_stats.items():
            # second dimension contains the mean and std of the metric
            stat_mean = stat_data[step, 0]
            logger.log_item(f"Train/{stat_name}", stat_mean, train_step=step, commit=True)

    logger.commit()

    # save artifacts
    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    out_savepath = save_train_run(out, savedir, savename="saved_train_run")
    if config["logger"]["log_train_out"]:
        logger.log_artifact(name="saved_train_run", path=out_savepath, type_name="train_run")
        # Cleanup locally logged out file
    if not config["local_logger"]["save_train_out"]:
        shutil.rmtree(out_savepath)
   
    # # Generate matplotlib plots # TODO: remove this code after creating the demo notebook
    # metric_names = get_metric_names(config.algorithm["ENV_NAME"])
    # all_stats = get_stats(out["metrics"], metric_names)
    # figures, _ = plot_train_metrics(all_stats, 
    #                                 config.algorithm["ROLLOUT_LENGTH"], 
    #                                 config.algorithm["NUM_ENVS"],
    #                                 savedir=savedir if config["local_logger"]["save_figures"] else None,
    #                                 savename="ippo_train_metrics",
    #                                 show_plots=False
    #                                 )
    
    # # Log plots to wandb
    # for stat_name, fig in figures.items():
    #     logger.log({f"train_metrics/{stat_name}": fig})

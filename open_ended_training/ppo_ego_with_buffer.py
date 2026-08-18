'''
Script for training a PPO ego agent against a BufferedPopulation of homogeneous partner agents.
In comparison to ego_agent_training/ppo_ego.py, this script permits a (potentially nonstationary)
sampling distribution over partners and a population of partners that potentially changes in size.
'''
import logging

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from agents.population_buffer import BufferedPopulation, PopulationBuffer
from marl.ppo_utils import Transition, unbatchify, _create_minibatches
from common.run_episodes import run_episodes
from common.n_agent_utils import (
    get_ego_teammate_indices, augment_all_obs, augment_done, split_actions
)
from common.run_n_agent_episodes import run_n_agent_episodes

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Maps the raw per-update ego metric keys (built in the ego update step) to the
# fully-qualified wandb tags used by the end-of-run log_metrics. Consumed by the
# live-logging callback (common.live_log_utils.make_live_log_callback) so live curves
# use the same naming as the log-at-end paradigm. Loss stats live under "Losses/*";
# rollout-return stats (every other metric key) fall through to EGO_LIVE_RETURNS_PREFIX
# under "Train/Ego/". Keep in sync with rotate_without_pop.log_metrics.
EGO_LIVE_LOSS_KEYS = {
    "value_loss": "Losses/EgoValueLoss",
    "actor_loss": "Losses/EgoActorLoss",
    "entropy_loss": "Losses/EgoEntropyLoss",
    "avg_grad_norm": "Losses/EgoGradNorm",
}
EGO_LIVE_RETURNS_PREFIX = "Train/Ego/"

def compute_ego_num_updates(cfg):
    return int(cfg["TOTAL_TIMESTEPS"] // (cfg["ROLLOUT_LENGTH"] * cfg["NUM_ENVS"]))

def train_ppo_ego_agent_with_buffer(config, env, train_rng, 
                           ego_policy, init_ego_params, n_ego_train_seeds,
                           partner_population: BufferedPopulation, 
                           population_buffer: PopulationBuffer,
                           logger=None, progress_callback=None
                           ):
    '''
    Train PPO ego agent using partners from the BufferedPopulation.

    Args:
        config: dict, config for the training
        env: environment
        train_rng: jax.random.PRNGKey, random key for training
        ego_policy: AgentPolicy, policy for the ego agent
        init_ego_params: dict, initial parameters for the ego agent
        n_ego_train_seeds: int, number of ego training seeds
        partner_population: BufferedPopulation, population manager for partner agents
        population_buffer: PopulationBuffer, buffer containing partner agents
    '''
    # ------------------------------
    # Build the PPO training function
    # ------------------------------
    def make_ppo_train(config):
        '''Ego agents occupy ego_indices slots; partner (buffered population) occupies teammate_indices slots.'''
        num_agents = env.num_agents
        ego_indices, teammate_indices = get_ego_teammate_indices(config, env)
        n_ego = len(ego_indices)
        n_teammates = len(teammate_indices)
        
        augment_obs = config.get("EGO_INDICES") is not None

        config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
        config["NUM_CONTROLLED_ACTORS"] = n_ego * config["NUM_ENVS"]        # ego
        config["NUM_UNCONTROLLED_ACTORS"] = n_teammates * config["NUM_ENVS"] # partner
        config["NUM_UPDATES"] = compute_ego_num_updates(config)

        config["NUM_ACTIONS"] = env.action_space(env.agents[0]).n
        assert config["NUM_CONTROLLED_ACTORS"] % config["NUM_MINIBATCHES"] == 0, "NUM_CONTROLLED_ACTORS must be divisible by NUM_MINIBATCHES"
        assert config["NUM_CONTROLLED_ACTORS"] >= config["NUM_MINIBATCHES"], "NUM_CONTROLLED_ACTORS must be >= NUM_MINIBATCHES"

        def linear_schedule(count):
            frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
            return config["LR"] * frac

        def train(rng):
            if config["ANNEAL_LR"]:
                tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(learning_rate=linear_schedule, eps=1e-5),
                )
            else:
                tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(config["LR"], eps=1e-5),
                )

            train_state = TrainState.create(
                apply_fn=ego_policy.network.apply,
                params=init_ego_params,
                tx=tx,
            )

            #  Init ego and partner hstates
            init_ego_hstate = ego_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])
            init_partner_hstate = partner_population.init_hstate(config["NUM_UNCONTROLLED_ACTORS"])

            def _env_step(runner_state, unused):
                """
                One step of the environment with N-agent parameter sharing:
                - ego slots  (ego_indices)      -> shared ego policy
                - partner slots (teammate_indices) -> BufferedPopulation
                """
                train_state, env_state, prev_obs, prev_done, ego_hstate, partner_hstate, population_buffer, partner_indices, rng = runner_state
                rng, actor_rng, partner_rng, partner_sample_rng, step_rng = jax.random.split(rng, 5)

                avail_actions_vmap = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions_vmap = jax.lax.stop_gradient(avail_actions_vmap)

                # Stack obs/dones for ego and partner role groups (augmented with one-hot ID)
                obs_ego, obs_partner = augment_all_obs(
                    prev_obs, ego_indices, teammate_indices, num_agents, config["NUM_ENVS"], augment_obs=augment_obs)
                dones_ego = augment_done(prev_done, ego_indices, config["NUM_ENVS"])
                dones_partner = augment_done(prev_done, teammate_indices, config["NUM_ENVS"])

                avail_actions_ego = jnp.concatenate(
                    [avail_actions_vmap[f'agent_{i}'].astype(jnp.float32) for i in ego_indices], axis=0)
                avail_actions_partner = jnp.concatenate(
                    [avail_actions_vmap[f'agent_{i}'].astype(jnp.float32) for i in teammate_indices], axis=0)

                # Conditionally resample partners on episode end
                needs_resample = prev_done["__all__"]
                sampled_indices_all, updated_buffer = partner_population.sample_agent_indices(
                    population_buffer,
                    config["NUM_UNCONTROLLED_ACTORS"],
                    partner_sample_rng,
                    needs_resample_mask=needs_resample
                )

                # Determine final indices based on whether resampling was needed for each env
                # needs_resample is (NUM_ENVS,); partner_indices is (NUM_UNCONTROLLED_ACTORS,).
                # Tile so all teammate slots for an env reset together.
                needs_resample_tiled = jnp.tile(needs_resample, n_teammates)  # (NUM_UNCONTROLLED_ACTORS,)
                updated_partner_indices = jnp.where(
                    needs_resample_tiled,
                    sampled_indices_all,    # Use newly sampled index if True
                    partner_indices         # Else, keep index from previous step
                )

                # Note that we do not need to reset the hiden states for both the ego and partner agents
                # as the recurrent states are automatically reset when done is True, and the partner indices are only reset when done is True.

                # Ego action (parameter sharing over n_ego slots)
                act_ego, val_ego, pi_ego, new_ego_hstate = ego_policy.get_action_value_policy(
                    params=train_state.params,
                    obs=obs_ego.reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                    done=dones_ego.reshape(1, config["NUM_CONTROLLED_ACTORS"]),
                    avail_actions=avail_actions_ego,
                    hstate=ego_hstate,
                    rng=actor_rng
                )
                logp_ego = pi_ego.log_prob(act_ego)
                act_ego = act_ego.squeeze()
                logp_ego = logp_ego.squeeze()
                val_ego = val_ego.squeeze()

                # Partner action via BufferedPopulation (parameter sharing over n_teammate slots)
                act_partner, new_partner_hstate = partner_population.get_actions(
                    buffer=updated_buffer,
                    agent_indices=updated_partner_indices,
                    obs=obs_partner,
                    done=dones_partner,
                    avail_actions=avail_actions_partner,
                    hstate=partner_hstate,
                    rng=partner_rng,
                    env_state=env_state,
                    aux_obs=None
                )
                act_partner = act_partner.squeeze()

                # Reconstruct action dict and step env
                env_act = split_actions(act_ego, act_partner, ego_indices, teammate_indices, config["NUM_ENVS"])
                step_rngs = jax.random.split(step_rng, config["NUM_ENVS"])
                obs_next, env_state_next, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    step_rngs, env_state, env_act
                )
                # Stack ego reward per slot: (n_ego * NUM_ENVS,) — one entry per ego slot
                ego_reward_stacked = jnp.concatenate(
                    [reward[f'agent_{i}'] for i in ego_indices], axis=0)  # (NUM_CONTROLLED_ACTORS,)
                info_ego = jax.tree.map(
                    lambda x: jnp.concatenate([x[:, i] for i in ego_indices], axis=0), 
                    info
                )

                # Store all NUM_CONTROLLED_ACTORS (n_ego * NUM_ENVS) ego data.
                ego_done_stacked = augment_done(done, ego_indices, config["NUM_ENVS"])  # (NUM_CONTROLLED_ACTORS,)
                transition = Transition(
                    done=ego_done_stacked,
                    action=act_ego,            # (NUM_CONTROLLED_ACTORS,)
                    value=val_ego,             # (NUM_CONTROLLED_ACTORS,)
                    reward=ego_reward_stacked,
                    log_prob=logp_ego,         # (NUM_CONTROLLED_ACTORS,)
                    obs=obs_ego,               # (NUM_CONTROLLED_ACTORS, aug_obs_dim)
                    info=info_ego,
                    avail_actions=avail_actions_ego  # (NUM_CONTROLLED_ACTORS, n_actions)
                )
                new_runner_state = (train_state, env_state_next, obs_next, done,
                                    new_ego_hstate, new_partner_hstate, updated_buffer,
                                    updated_partner_indices, rng)
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
                    unroll=4,
                )
                return advantages, advantages + traj_batch.value

            def _update_minbatch(train_state, batch_info):
                init_ego_hstate, traj_batch, advantages, returns = batch_info
                def _loss_fn(params, init_ego_hstate, traj_batch, gae, target_v):
                    _, value, pi, _ = ego_policy.get_action_value_policy(
                        params=params, 
                        obs=traj_batch.obs,
                        done=traj_batch.done,
                        avail_actions=traj_batch.avail_actions,
                        hstate=init_ego_hstate,
                        rng=jax.random.PRNGKey(0) # only used for action sampling, which is unused here
                    )
                    log_prob = pi.log_prob(traj_batch.action)

                    # Value loss
                    value_pred_clipped = traj_batch.value + (
                        value - traj_batch.value
                        ).clip(
                        -config["CLIP_EPS"], config["CLIP_EPS"])
                    value_losses = jnp.square(value - target_v)
                    value_losses_clipped = jnp.square(value_pred_clipped - target_v)
                    value_loss = jnp.maximum(value_losses, value_losses_clipped).mean()

                    # Policy gradient loss
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
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

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                (loss_val, aux_vals), grads = grad_fn(
                    train_state.params, init_ego_hstate, traj_batch, advantages, returns)
                train_state = train_state.apply_gradients(grads=grads)

                # compute average grad norm
                grad_l2_norms = jax.tree.map(lambda g: jnp.linalg.norm(g.astype(jnp.float32)), grads)
                sum_of_grad_norms = jax.tree.reduce(lambda x, y: x + y, grad_l2_norms)
                n_elements = len(jax.tree.leaves(grad_l2_norms))
                avg_grad_norm = sum_of_grad_norms / n_elements
                
                return train_state, (loss_val, aux_vals, avg_grad_norm)

            def _update_epoch(update_state, unused):
                train_state, init_ego_hstate, traj_batch, advantages, targets, rng = update_state
                rng, perm_rng = jax.random.split(rng)
                minibatches = _create_minibatches(
                    traj_batch, 
                    advantages, 
                    targets, 
                    init_ego_hstate, 
                    config["NUM_CONTROLLED_ACTORS"], 
                    config["NUM_MINIBATCHES"], 
                    perm_rng
                )
                train_state, losses_and_grads = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, init_ego_hstate, traj_batch, advantages, targets, rng)
                return update_state, losses_and_grads

            def _update_step(update_runner_state, unused):
                """
                1. Collect rollouts
                2. Compute advantage
                3. PPO updates
                """
                (train_state, last_buffer, rng, update_steps) = update_runner_state

                # Init envs & partner indices
                rng, reset_rng, p_rng = jax.random.split(rng, 3)
                reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
                init_obs, init_env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rngs)
                init_done = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}
                new_partner_indices, buffer = partner_population.sample_agent_indices(
                    last_buffer, config["NUM_UNCONTROLLED_ACTORS"], p_rng)

                # 1) rollout
                runner_state = (train_state, init_env_state, init_obs, init_done, 
                                init_ego_hstate, init_partner_hstate, 
                                buffer, new_partner_indices, rng)

                runner_state, traj_batch = jax.lax.scan(
                    _env_step, runner_state, None, config["ROLLOUT_LENGTH"])
                (train_state, env_state, obs, done, ego_hstate, partner_hstate, 
                 buffer, partner_indices, rng) = runner_state

                # 2) advantage
                # Stack augmented ego obs for the last value estimate
                obs_ego_last, _ = augment_all_obs(obs, ego_indices, teammate_indices, num_agents, config["NUM_ENVS"], augment_obs=augment_obs)
                dones_ego_last = augment_done(done, ego_indices, config["NUM_ENVS"])
                avail_actions_ego_last = jnp.concatenate(
                    [jax.vmap(env.get_avail_actions)(env_state.env_state)[f'agent_{i}'].astype(jnp.float32)
                     for i in ego_indices], axis=0)

                # Get final value estimate for completed trajectory
                _, last_val, _, _ = ego_policy.get_action_value_policy(
                    params=train_state.params,
                    obs=obs_ego_last.reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                    done=dones_ego_last.reshape(1, config["NUM_CONTROLLED_ACTORS"]),
                    avail_actions=jax.lax.stop_gradient(avail_actions_ego_last),
                    hstate=ego_hstate,
                    rng=jax.random.PRNGKey(0)
                )
                last_val = last_val.squeeze()
                advantages, targets = _calculate_gae(traj_batch, last_val)

                # 3) PPO update
                update_state = (
                    train_state,
                    init_ego_hstate, # shape is (num_controlled_actors, gru_hidden_dim) with all-0s value
                    traj_batch, # obs has shape (rollout_len, num_controlled_actors, -1)
                    advantages,
                    targets,
                    rng
                )
                update_state, losses_and_grads = jax.lax.scan(
                    _update_epoch, update_state, None, config["UPDATE_EPOCHS"])
                train_state = update_state[0]
                _, loss_terms, avg_grad_norm = losses_and_grads

                # Metrics
                def mask_and_mean(x, mask):
                    return jnp.where(mask, x, 0).sum() / jnp.maximum(1, mask.sum())

                mask = traj_batch.info.get("returned_episode", jnp.ones_like(traj_batch.reward))
                metric = jax.tree.map(lambda x: mask_and_mean(x, mask), traj_batch.info)
                metric["update_steps"] = update_steps
                metric["actor_loss"] = loss_terms[1].mean()
                metric["value_loss"] = loss_terms[0].mean()
                metric["entropy_loss"] = loss_terms[2].mean()
                metric["avg_grad_norm"] = avg_grad_norm.mean()

                # Live logging + progress bar are handled by a single stateful callback
                # (see common.live_log_utils.make_live_log_callback) that aggregates over
                # the seed vmap axis and logs under the end-of-run naming scheme.
                if progress_callback is not None:
                    jax.experimental.io_callback(progress_callback, None, metric)

                new_runner_state = (train_state, buffer, rng, update_steps + 1)
                return (new_runner_state, metric)

            # 3e) PPO Update and Checkpoint saving
            ckpt_and_eval_interval = config["NUM_UPDATES"] // max(1, config["NUM_CHECKPOINTS"] - 1)  # -1 because we store a ckpt at the last update
            num_ckpts = config["NUM_CHECKPOINTS"]

            # Build a PyTree that holds parameters for all FCP checkpoints
            def init_ckpt_array(params_pytree):
                return jax.tree.map(
                    lambda x: jnp.zeros((num_ckpts,) + x.shape, x.dtype), 
                    params_pytree)

            max_episode_steps = config["ROLLOUT_LENGTH"]
            
            def _update_step_with_ckpt(state_with_ckpt, unused):
                (update_state, checkpoint_array, ckpt_idx, init_eval_last_info) = state_with_ckpt

                # Single PPO update
                (new_update_state, metric) = _update_step(
                    update_state,
                    None
                )
                (train_state, buffer, rng, update_steps) = new_update_state

                # Decide if we store a checkpoint
                # update steps is 1-indexed because it was incremented at the end of the update step
                to_store = jnp.logical_or(jnp.equal(jnp.mod(update_steps-1, ckpt_and_eval_interval), 0),
                                        jnp.equal(update_steps, config["NUM_UPDATES"]))

                def store_and_eval_ckpt(args):
                    ckpt_arr, cidx, rng, prev_eval_ret_info = args
                    new_ckpt_arr = jax.tree.map(
                        lambda c_arr, p: c_arr.at[cidx].set(p),
                        ckpt_arr, train_state.params
                    )
                    rng, eval_rng, partner_rng = jax.random.split(rng, 3)
                    
                    # Sample partners from buffer for evaluation
                    # we do not return the updated buffer because we don't want evaluation to impact the buffer distribution
                    partner_indices, eval_buffer = partner_population.sample_agent_indices(
                        buffer, config["NUM_EVAL_EPISODES"], partner_rng
                    )
                    
                    # Gather parameters for sampled partners
                    gathered_params = partner_population.gather_agent_params(
                        eval_buffer, partner_indices
                    )
                    
                    # Run evaluation with sampled partners
                    eval_eps_last_infos = jax.vmap(
                        lambda partner_p: run_n_agent_episodes(
                            eval_rng, env,
                            ego_policy=ego_policy, ego_param=train_state.params,
                            teammate_policy=partner_population.policy_cls, teammate_param=partner_p,
                            ego_indices=ego_indices, teammate_indices=teammate_indices,
                            max_episode_steps=max_episode_steps,
                            num_eps=config["NUM_EVAL_EPISODES"]
                        )
                    )(gathered_params)
                    return (new_ckpt_arr, cidx + 1, rng, eval_eps_last_infos)
                
                def skip_ckpt(args):
                    return args

                (checkpoint_array, ckpt_idx, rng, eval_last_infos) = jax.lax.cond(
                    to_store, store_and_eval_ckpt, skip_ckpt, (checkpoint_array, ckpt_idx, rng, init_eval_last_info)
                )
                # Add evaluation info to metrics
                metric["eval_ep_last_info"] = eval_last_infos

                return ((train_state, buffer, rng, update_steps),
                         checkpoint_array, ckpt_idx, eval_last_infos), metric

            # init checkpoint array
            checkpoint_array = init_ckpt_array(train_state.params)
            ckpt_idx = 0

            # Init eval return infos
            rng, rng_eval, partner_rng, rng_train = jax.random.split(rng, 4)
            
            # Sample partners for initial evaluation
            # we do not update the buffer because we don't want evaluation to impact the buffer distribution
            eval_partner_indices, eval_buffer = partner_population.sample_agent_indices(
                population_buffer, config["NUM_EVAL_EPISODES"], partner_rng
            )
            
            # Gather parameters for partners
            gathered_params = partner_population.gather_agent_params(
                eval_buffer, eval_partner_indices
            )
            
            eval_eps_last_infos = jax.vmap(
                lambda partner_p: run_n_agent_episodes(
                    rng_eval, env,
                    ego_policy=ego_policy, ego_param=train_state.params,
                    teammate_policy=partner_population.policy_cls, teammate_param=partner_p,
                    ego_indices=ego_indices, teammate_indices=teammate_indices,
                    max_episode_steps=max_episode_steps,
                    num_eps=config["NUM_EVAL_EPISODES"]
                )
            )(gathered_params)

            # initial runner state for scanning
            update_steps = 0
            update_runner_state = (
                train_state,
                population_buffer,
                rng_train,
                update_steps
            )
            state_with_ckpt = (update_runner_state, checkpoint_array, ckpt_idx, eval_eps_last_infos)
            
            # run training
            state_with_ckpt, metrics = jax.lax.scan(
                _update_step_with_ckpt,
                state_with_ckpt,
                xs=None,
                length=config["NUM_UPDATES"]
            )
            (final_update_state, checkpoint_array, final_ckpt_idx, eval_eps_last_infos) = state_with_ckpt
            final_train_state, final_buffer, _, _ = final_update_state
            out = {
                "final_params": final_train_state.params,
                "metrics": metrics,  # shape (NUM_UPDATES, ...)
                "checkpoints": checkpoint_array,
                "final_buffer": final_buffer,
            }
            return out
        return train

    # ------------------------------
    # Actually run the PPO training
    # ------------------------------
    rngs = jax.random.split(train_rng, n_ego_train_seeds)
    train_fn = jax.jit(jax.vmap(make_ppo_train(config)))
    out = train_fn(rngs)    
    return out 
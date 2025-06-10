'''
Script for training a LIAM ego agent against a population of partner agents. 
Only suliamrts a population of homogeneous RL partner agents.
Warning: modify with caution, as this script is used as the main script for ego training throughout the project.
'''
import shutil
import time
import logging

import jax
import jax.numpy as jnp
import numpy as np
import optax
import hydra
from flax.training.train_state import TrainState

from agents.population_interface import AgentPopulation
from marl.liam_utils import Transition, ScannedLSTM, initialize_encoder_decoder
from common.run_episodes import run_episodes
from common.plot_utils import get_stats, get_metric_names
from marl.ppo_utils import unbatchify, _create_minibatches
from common.save_load_utils import save_train_run
from ego_agent_training.utils import initialize_ego_agent
from envs import make_env
from envs.log_wrapper import LogWrapper
from evaluation.agent_loader_from_config import initialize_rl_agent_from_config

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def train_liam_ego_agent(config, env, train_rng, 
                        ego_policy, init_ego_params, n_ego_train_seeds,
                        encoder, decoder, init_encoder_decoder_params,
                        partner_population: AgentPopulation,
                        partner_params
                        ):
    '''
    Train LIAM ego agent using the given partner checkpoints and initial ego parameters.

    Args:
        config: dict, config for the training
        env: gymnasium environment
        train_rng: jax.random.PRNGKey, random key for training
        ego_policy: AgentPolicy, policy for the ego agent
        init_ego_params: dict, initial parameters for the ego agent
        n_ego_train_seeds: int, number of ego training seeds
        encoder: Encoder, the encoder model
        decoder: Decoder, the decoder model
        init_encoder_decoder_params: dict, initial parameters for the encoder and decoder
        partner_population: AgentPopulation, population of partner agents
        partner_params: pytree of parameters for the population of agents of shape (pop_size, ...).
    '''
    # Get partner parameters from the population
    num_total_partners = partner_population.pop_size

    # ------------------------------
    # Build the LIAM training function
    # ------------------------------
    def make_liam_train(config):
        '''agent 0 is the ego agent while agent 1 is the confederate'''
        num_agents = env.num_agents
        assert num_agents == 2, "This snippet assumes exactly 2 agents."

        config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
        config["NUM_UNCONTROLLED_ACTORS"] = config["NUM_ENVS"] # assumption: we control 1 agent
        config["NUM_CONTROLLED_ACTORS"] = config["NUM_ENVS"] # assumption: we control 1 agent
        config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["ROLLOUT_LENGTH"] // config["NUM_ENVS"]

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

            encoder_decoder_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["ENCODER_DECODER_LR"], eps=1e-8),
            )

            encoder_decoder_train_state = TrainState.create(
                apply_fn=encoder.model.apply,
                params=init_encoder_decoder_params,
                tx=encoder_decoder_tx,
            )

            def _env_step(runner_state, unused):
                """
                One step of the environment:
                1. Get observations, sample actions from all agents
                2. Step environment using sampled actions
                3. Return state, reward, ...
                """
                train_state, encoder_decoder_train_state, env_state, prev_obs, prev_done, ego_hstate, ego_encoder_hstate, partner_hstate, partner_indices, rng = runner_state
                rng, actor_rng, partner_rng, step_rng = jax.random.split(rng, 4)

                 # Get available actions for agent 0 from environment state
                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(avail_actions)
                avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
                avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

                # Agent 0 Encoder
                embedded_obs_0, ego_encoder_hstate = encoder.compute_embedding(
                    params=encoder_decoder_train_state.params['encoder'],
                    hstate=ego_encoder_hstate,
                    obs=prev_obs["agent_0"].reshape(1, config["NUM_CONTROLLED_ACTORS"], -1), 
                    done=prev_done["agent_0"].reshape(1, config["NUM_CONTROLLED_ACTORS"])
                )
                                
                # Agent_0 (ego) action, value, log_prob
                act_0, val_0, pi_0, ego_hstate = ego_policy.get_action_value_policy(
                    params=train_state.params,
                    obs=embedded_obs_0,
                    done=prev_done["agent_0"].reshape(1, config["NUM_CONTROLLED_ACTORS"]),
                    avail_actions=avail_actions_0,
                    hstate=ego_hstate,
                    rng=actor_rng
                )
                logp_0 = pi_0.log_prob(act_0)

                act_0 = act_0.squeeze()
                logp_0 = logp_0.squeeze()
                val_0 = val_0.squeeze()

                # Agent_1 (partner) action using the AgentPopulation interface
                # Conditionally resample partners based on prev_done["__all__"]                
                needs_resample = prev_done["__all__"] # shape (NUM_ENVS,) bool
                sampled_indices_all = partner_population.sample_agent_indices(config["NUM_CONTROLLED_ACTORS"], partner_rng)

                # Determine final indices based on whether resampling was needed for each env
                updated_partner_indices = jnp.where(
                    needs_resample,         # Mask shape (NUM_ENVS,)
                    sampled_indices_all,    # Use newly sampled index if True
                    partner_indices         # Else, keep index from previous step
                )

                act_1, new_partner_hstate = partner_population.get_actions(
                    partner_params,
                    updated_partner_indices,
                    prev_obs["agent_1"].reshape(config["NUM_CONTROLLED_ACTORS"], 1, -1),
                    prev_done["agent_1"].reshape(config["NUM_CONTROLLED_ACTORS"], 1, -1),
                    avail_actions_1,
                    partner_hstate,
                    partner_rng,
                    env_state=env_state,
                    aux_obs=None
                )
                act_1 = act_1.squeeze()

                # Combine actions into the env format
                combined_actions = jnp.concatenate([act_0, act_1], axis=0)  # shape (2*num_envs,)
                env_act = unbatchify(combined_actions, env.agents, config["NUM_ENVS"], num_agents)
                env_act = {k: v.flatten() for k, v in env_act.items()}

                # Step env
                step_rngs = jax.random.split(step_rng, config["NUM_ENVS"])
                obs_next, env_state_next, reward, done_next, info = jax.vmap(env.step, in_axes=(0,0,0))(
                    step_rngs, env_state, env_act
                )
                # note that num_actors = num_envs * num_agents
                info_0 = jax.tree.map(lambda x: x[:, 0], info)

                # Store agent_0 data in transition
                transition = Transition(
                    done=done_next["agent_0"],
                    action=act_0,
                    value=val_0,
                    reward=reward["agent_0"],
                    log_prob=logp_0,
                    obs=prev_obs["agent_0"],
                    info=info_0,
                    avail_actions=avail_actions_0,
                    partner_obs=prev_obs["agent_1"],
                    partner_action=act_1
                )
                new_runner_state = (train_state, encoder_decoder_train_state, env_state_next, obs_next, done_next, 
                                    ego_hstate, new_partner_hstate, updated_partner_indices, rng)
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

            def _update_minbatch(train_state, encoder_decoder_train_state, batch_info):
                init_ego_hstate, init_ego_encoder_hstate, traj_batch, advantages, returns = batch_info
                def _loss_fn(params, encoder_decoder_params, init_ego_hstate, traj_batch, gae, target_v):

                    # Agent 0 Encoder
                    embedded_batch_obs, _ = encoder.compute_embedding(
                        params=encoder_decoder_params['encoder'],
                        hstate=init_ego_encoder_hstate,
                        obs=traj_batch.obs,
                        done=traj_batch.done
                    )

                    _, value, pi, _ = ego_policy.get_action_value_policy(
                        params=params, 
                        obs=jax.lax.stop_gradient(embedded_batch_obs),
                        done=traj_batch.done,
                        avail_actions=traj_batch.avail_actions,
                        hstate=init_ego_hstate,
                        rng=jax.random.PRNGKey(0) # only used for action sampling, which is unused here
                    )
                    log_prob = pi.log_prob(traj_batch.action)

                    # Reconstruction Loss
                    recon_loss1, recon_loss2 = decoder.evaluate(
                        params=encoder_decoder_params['decoder'],
                        embedding=embedded_batch_obs,
                        modelled_agent_obs=traj_batch.partner_obs,
                        modelled_agent_act=traj_batch.partner_action
                    )
                    recon_loss = (recon_loss1.mean() + recon_loss2.mean())

                    # Value loss
                    value_pred_clipped = traj_batch.value + (
                        value - traj_batch.value
                        ).clip(
                        -config["CLIP_EPS"], config["CLIP_EPS"])
                    value_losses = jnp.square(value - target_v)
                    value_losses_clipped = jnp.square(value_pred_clipped - target_v)
                    value_loss = (
                        jnp.maximum(value_losses, value_losses_clipped).mean()
                    )

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
                    return total_loss, (value_loss, pg_loss, entropy, recon_loss)

                grad_fn = jax.value_and_grad(_loss_fn, argnums=(0,1), has_aux=True)
                (loss_val, aux_vals), (grads, encoder_decoder_grads) = grad_fn(
                    train_state.params, encoder_decoder_train_state.params, init_ego_hstate, init_ego_encoder_hstate, traj_batch, advantages, returns)
                train_state = train_state.apply_gradients(grads=grads)
                encoder_decoder_train_state = encoder_decoder_train_state.apply_gradients(grads=encoder_decoder_grads)
                
                # compute average grad norm
                grad_l2_norms = jax.tree.map(lambda g: jnp.linalg.norm(g.astype(jnp.float32)), grads)
                sum_of_grad_norms = jax.tree.reduce(lambda x, y: x + y, grad_l2_norms)
                n_elements = len(jax.tree.leaves(grad_l2_norms))
                avg_grad_norm = sum_of_grad_norms / n_elements

                encoder_decoder_grad_l2_norms = jax.tree.map(lambda g: jnp.linalg.norm(g.astype(jnp.float32)), encoder_decoder_grads)
                encoder_decoder_sum_of_grad_norms = jax.tree.reduce(lambda x, y: x + y, encoder_decoder_grad_l2_norms)
                encoder_decoder_n_elements = len(jax.tree.leaves(encoder_decoder_grad_l2_norms))
                encoder_decoder_avg_grad_norm = encoder_decoder_sum_of_grad_norms / encoder_decoder_n_elements
                
                return train_state, encoder_decoder_train_state, (loss_val, aux_vals, avg_grad_norm, encoder_decoder_avg_grad_norm)

            def _update_epoch(update_state, unused):
                train_state, encoder_decoder_train_state, init_ego_hstate, init_ego_encoder_hstate, traj_batch, advantages, targets, rng = update_state
                rng, perm_rng = jax.random.split(rng)
                minibatches = _create_minibatches(traj_batch, advantages, targets, init_ego_hstate, config["NUM_CONTROLLED_ACTORS"], config["NUM_MINIBATCHES"], perm_rng)
                train_state, encoder_decoder_train_state, losses_and_grads = jax.lax.scan(
                    _update_minbatch, train_state, encoder_decoder_train_state, minibatches
                )
                update_state = (train_state, encoder_decoder_train_state, init_ego_hstate, init_ego_encoder_hstate, traj_batch, advantages, targets, rng)
                return update_state, losses_and_grads

            def _update_step(update_runner_state, unused):
                """
                1. Collect rollouts
                2. Compute advantage
                3. LIAM updates
                """
                (train_state, encoder_decoder_train_state, last_env_state, last_obs, last_done, last_ego_hstate, last_ego_encoder_hstate, last_partner_hstate, last_partner_indices, rng, update_steps) = update_runner_state

                # 1) rollout
                runner_state = (train_state, encoder_decoder_train_state, last_env_state, last_obs, last_done, last_ego_hstate, last_ego_encoder_hstate, last_partner_hstate, last_partner_indices, rng)
                runner_state, traj_batch = jax.lax.scan(
                    _env_step, runner_state, None, config["ROLLOUT_LENGTH"])
                (train_state, encoder_decoder_train_state, env_state, obs, done, ego_hstate, ego_encoder_hstate, partner_hstate, partner_indices, rng) = runner_state

                # 2) advantage
                # Get available actions for agent 0 from environment state
                avail_actions_0 = jax.vmap(env.get_avail_actions)(env_state.env_state)["agent_0"].astype(jnp.float32)

                # Agent 0 Encoder
                embedded_obs, _ = encoder.compute_embedding(
                    params=encoder_decoder_train_state.params['encoder'],
                    hstate=ego_encoder_hstate,
                    obs=obs["agent_0"].reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                    done=done["agent_0"].reshape(1, config["NUM_CONTROLLED_ACTORS"])
                )

                # Get final value estimate for completed trajectory
                _, last_val, _, _ = ego_policy.get_action_value_policy(
                    params=train_state.params, 
                    obs=embedded_obs,
                    done=done["agent_0"].reshape(1, config["NUM_CONTROLLED_ACTORS"]),
                    avail_actions=jax.lax.stop_gradient(avail_actions_0),
                    hstate=ego_hstate,
                    rng=jax.random.PRNGKey(0)  # Dummy key since we're just extracting the value
                )
                last_val = last_val.squeeze()
                advantages, targets = _calculate_gae(traj_batch, last_val)

                # 3) LIAM update
                init_ego_hstate = ego_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])
                init_ego_encoder_hstate = encoder.init_hstate(batch_size=config["NUM_CONTROLLED_ACTORS"])
                update_state = (
                    train_state,
                    encoder_decoder_train_state,
                    init_ego_hstate, # shape is (num_controlled_actors, gru_hidden_dim) with all-0s value
                    init_ego_encoder_hstate, # shape is (num_controlled_actors, encoder_hidden_dim) with all-0s value
                    traj_batch, # obs has shape (rollout_len, num_controlled_actors, -1)
                    advantages,
                    targets,
                    rng
                )
                update_state, losses_and_grads = jax.lax.scan(
                    _update_epoch, update_state, None, config["UPDATE_EPOCHS"])
                train_state = update_state[0]
                encoder_decoder_train_state = update_state[1]
                _, loss_terms, avg_grad_norm, encoder_decoder_avg_grad_norm = losses_and_grads
                
                metric = traj_batch.info
                metric["update_steps"] = update_steps
                metric["actor_loss"] = loss_terms[1]
                metric["value_loss"] = loss_terms[0]
                metric["entropy_loss"] = loss_terms[2]
                metric["reconstruction_loss"] = loss_terms[3]
                metric["avg_grad_norm"] = avg_grad_norm
                metric["encoder_decoder_avg_grad_norm"] = encoder_decoder_avg_grad_norm
                new_runner_state = (train_state, encoder_decoder_train_state, env_state, obs, done, 
                                    ego_hstate, ego_encoder_hstate, partner_hstate, partner_indices, 
                                    rng, update_steps + 1)
                return (new_runner_state, metric)

            # LIAM Update and Checkpoint saving
            ckpt_and_eval_interval = config["NUM_UPDATES"] // max(1, config["NUM_CHECKPOINTS"] - 1)  # -1 because we store a ckpt at the last update
            num_ckpts = config["NUM_CHECKPOINTS"]

            # Build a PyTree that holds parameters for all FCP checkpoints
            def init_ckpt_array(params_pytree):
                return jax.tree.map(
                    lambda x: jnp.zeros((num_ckpts,) + x.shape, x.dtype), 
                    params_pytree)

            max_episode_steps = config["ROLLOUT_LENGTH"]
            
            def _update_step_with_ckpt(state_with_ckpt, unused):
                (update_state, checkpoint_array, encoder_decoder_checkpoint_array, ckpt_idx, init_eval_last_info) = state_with_ckpt

                # Single LIAM update
                new_update_state, metric = _update_step(
                    update_state,
                    None
                )
                (train_state, encoder_decoder_train_state, env_state, obs, done, 
                 ego_hstate, ego_encoder_hstate, partner_hstate, partner_indices, 
                 rng, update_steps) = new_update_state

                # update steps is 1-indexed because it was incremented at the end of the update step
                to_store = jnp.logical_or(jnp.equal(jnp.mod(update_steps-1, ckpt_and_eval_interval), 0),
                                        jnp.equal(update_steps, config["NUM_UPDATES"]))


                def store_and_eval_ckpt(args):
                    ckpt_arr, encode_decode_ckpt_arr, cidx, rng, prev_eval_ret_info = args
                    new_ckpt_arr = jax.tree.map(
                        lambda c_arr, p: c_arr.at[cidx].set(p),
                        ckpt_arr, train_state.params
                    )

                    new_encode_decode_ckpt_arr = jax.tree.map(
                        lambda c_arr, p: c_arr.at[cidx].set(p),
                        encode_decode_ckpt_arr, encoder_decoder_train_state.params
                    )

                    eval_partner_indices = jnp.arange(num_total_partners)
                    gathered_params = partner_population.gather_agent_params(partner_params, eval_partner_indices)
                    
                    rng, eval_rng = jax.random.split(rng)
                    eval_eps_last_infos = jax.vmap(lambda x: run_episodes(
                        eval_rng, env, agent_0_param=train_state.params, agent_0_policy=ego_policy, 
                        agent_1_param=x, agent_1_policy=partner_population.policy_cls, 
                        max_episode_steps=max_episode_steps, 
                        num_eps=config["NUM_EVAL_EPISODES"]))(gathered_params)
                    return (new_ckpt_arr, new_encode_decode_ckpt_arr, cidx + 1, rng, eval_eps_last_infos)
                
                def skip_ckpt(args):
                    return args

                (checkpoint_array, encoder_decoder_checkpoint_array, ckpt_idx, rng, eval_last_infos) = jax.lax.cond(
                    to_store, store_and_eval_ckpt, skip_ckpt, (checkpoint_array, encoder_decoder_checkpoint_array, ckpt_idx, rng, init_eval_last_info)
                )

                metric["eval_ep_last_info"] = eval_last_infos
                return ((train_state, encoder_decoder_train_state, env_state, obs, done, 
                         ego_hstate, ego_encoder_hstate, partner_hstate, partner_indices, 
                         rng, update_steps),
                         checkpoint_array, encoder_decoder_checkpoint_array, ckpt_idx, eval_last_infos), metric

            # Init envs & partner indices
            rng, reset_rng = jax.random.split(rng)
            reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
            init_obs, init_env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rngs)
            init_done = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}

            # Each environment picks a partner index in [0, n_seeds*m_ckpts)
            rng, partner_rng = jax.random.split(rng)
            init_partner_indices = partner_population.sample_agent_indices(config["NUM_CONTROLLED_ACTORS"], partner_rng)
            
            # Init ego and partner hstates
            init_ego_hstate = ego_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])
            init_ego_encoder_hstate = encoder.init_hstate(batch_size=config["NUM_CONTROLLED_ACTORS"])
            init_partner_hstate = partner_population.init_hstate(config["NUM_UNCONTROLLED_ACTORS"])

            checkpoint_array = init_ckpt_array(train_state.params)
            encoder_decoder_checkpoint_array = init_ckpt_array(encoder_decoder_train_state.params)
            ckpt_idx = 0

            rng, rng_eval, rng_train = jax.random.split(rng, 3)
            # Init eval return infos
            eval_partner_indices = jnp.arange(num_total_partners)
            gathered_params = partner_population.gather_agent_params(partner_params, eval_partner_indices)
            eval_eps_last_infos = jax.vmap(lambda x: run_episodes(
                        rng_eval, env, 
                        agent_0_param=train_state.params, agent_0_policy=ego_policy, 
                        agent_1_param=x, agent_1_policy=partner_population.policy_cls, 
                        max_episode_steps=max_episode_steps, 
                        num_eps=config["NUM_EVAL_EPISODES"]))(gathered_params)

            # initial runner state for scanning
            update_steps = 0
            rng_train, partner_rng = jax.random.split(rng_train)

            update_runner_state = (
                train_state,
                encoder_decoder_train_state,
                init_env_state,
                init_obs,
                init_done,
                init_ego_hstate,
                init_ego_encoder_hstate,
                init_partner_hstate,
                init_partner_indices,
                rng_train,
                update_steps
            )
            state_with_ckpt = (update_runner_state, checkpoint_array, encoder_decoder_checkpoint_array, ckpt_idx, eval_eps_last_infos)
            
            state_with_ckpt, metrics = jax.lax.scan(
                _update_step_with_ckpt,
                state_with_ckpt,
                xs=None,
                length=config["NUM_UPDATES"]
            )
            (final_runner_state, checkpoint_array, encoder_decoder_checkpoint_array, final_ckpt_idx, eval_eps_last_infos) = state_with_ckpt
            out = {
                "final_params": final_runner_state[0].params,
                "metrics": metrics,  # shape (NUM_UPDATES, ...)
                "checkpoints": checkpoint_array,
                "encoder_decoder_checkpoints": encoder_decoder_checkpoint_array,
            }
            return out
        return train

    # ------------------------------
    # Actually run the LIAM training
    # ------------------------------
    rngs = jax.random.split(train_rng, n_ego_train_seeds)
    train_fn = jax.jit(jax.vmap(make_liam_train(config)))
    out = train_fn(rngs)    
    return out

def run_ego_training(config, wandb_logger):
    '''Run ego agent training against the population of partner agents.
    
    Args:
        config: dict, config for the training
    '''
    algorithm_config = dict(config["algorithm"])

    # Create only one environment instance
    env = make_env(algorithm_config["ENV_NAME"], algorithm_config["ENV_KWARGS"])
    env = LogWrapper(env)

    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"])
    rng, init_partner_rng, init_ego_rng, init_ego_encoder_decoder_rng, train_rng = jax.random.split(rng, 5)
    

    partner_agent_config = dict(algorithm_config["partner_agent"])
    partner_policy, partner_params, init_partner_params, idx_labels = initialize_rl_agent_from_config(
        partner_agent_config, partner_agent_config["name"], env, init_partner_rng)

    flattened_partner_params = jax.tree.map(lambda x, y: x.reshape((-1,) + y.shape), partner_params, init_partner_params)        
    pop_size = jax.tree.leaves(flattened_partner_params)[0].shape[0]

    # Create partner population
    partner_population = AgentPopulation(
        pop_size=pop_size,
        policy_cls=partner_policy
    )
    
    # Initialize ego agent policy
    ego_policy, init_ego_params = initialize_ego_agent(algorithm_config, env, init_ego_rng)

    # Initialize ego agent encoder and decoder
    encoder, decoder, init_ecoder_decoder_params = initialize_encoder_decoder(algorithm_config, env, init_ego_encoder_decoder_rng)
    
    log.info("Starting ego agent training...")
    start_time = time.time()
    
    # Run the training
    out = train_liam_ego_agent(
        config=algorithm_config,
        env=env,
        train_rng=train_rng,
        ego_policy=ego_policy,
        init_ego_params=init_ego_params,
        encoder=encoder,
        decoder=decoder,
        init_encoder_decoder_params=init_ecoder_decoder_params,
        n_ego_train_seeds=algorithm_config["NUM_EGO_TRAIN_SEEDS"],
        partner_population=partner_population,
        partner_params=flattened_partner_params
    )
    
    log.info(f"Training completed in {time.time() - start_time:.2f} seconds")
    
    # process and log metrics
    metric_names = get_metric_names(config["ENV_NAME"])
    log_metrics(config, out, wandb_logger, metric_names)
    
    return out["final_params"], ego_policy, init_ego_params, encoder, decoder, init_ecoder_decoder_params

def log_metrics(config, train_out, logger, metric_names: tuple):
    """Process training metrics and log them using the provided logger.
    
    Args:
        training_logs: dict, the logs from training
        logger: Logger, instance to log metrics
        metric_names: tuple, names of metrics to extract from training logs
    """
    train_metrics = train_out["metrics"]

    #### Extract train metrics ####
    train_stats = get_stats(train_metrics, metric_names)
    # each key in train_stats is a metric name, and the value is an array of shape (num_seeds, num_updates, 2)
    # where the last dimension contains the mean and std of the metric
    train_stats = {k: np.mean(np.array(v), axis=0) for k, v in train_stats.items()}
    
    all_ego_value_losses = np.asarray(train_metrics["value_loss"]) # shape (n_ego_train_seeds, num_updates, num_partners, num_minibatches)
    all_ego_actor_losses = np.asarray(train_metrics["actor_loss"]) # shape (n_ego_train_seeds, num_updates, num_partners, num_minibatches)
    all_ego_entropy_losses = np.asarray(train_metrics["entropy_loss"]) # shape (n_ego_train_seeds, num_updates, num_partners, num_minibatches)
    all_ego_reconstruction_losses = np.asarray(train_metrics["reconstruction_loss"]) # shape (n_ego_train_seeds, num_updates, num_partners, num_minibatches)
    all_ego_grad_norms = np.asarray(train_metrics["avg_grad_norm"]) # shape (n_ego_train_seeds, num_updates, num_partners, num_minibatches)
    all_ego_encoder_decoder_grad_norms = np.asarray(train_metrics["encoder_decoder_avg_grad_norm"]) # shape (n_ego_train_seeds, num_updates, num_partners, num_minibatches)
    # Process eval return metrics - average across ego seeds, eval episodes,  training partners 
    # and num_agents per game for each checkpoint
    all_ego_returns = np.asarray(train_metrics["eval_ep_last_info"]["returned_episode_returns"]) # shape (n_ego_train_seeds, num_updates, num_partners, num_eval_episodes, nuM_agents_per_game)
    average_ego_rets_per_iter = np.mean(all_ego_returns, axis=(0, 2, 3, 4))

    # Process loss metrics - average across ego seeds, partners and minibatches dims
    # Loss metrics shape should be (n_ego_train_seeds, num_updates, ...)
    average_ego_value_losses = np.mean(all_ego_value_losses, axis=(0, 2, 3))
    average_ego_actor_losses = np.mean(all_ego_actor_losses, axis=(0, 2, 3))
    average_ego_entropy_losses = np.mean(all_ego_entropy_losses, axis=(0, 2, 3))
    average_ego_reconstruction_losses = np.mean(all_ego_reconstruction_losses, axis=(0, 2, 3))
    average_ego_grad_norms = np.mean(all_ego_grad_norms, axis=(0, 2, 3))
    average_ego_encoder_decoder_grad_norms = np.mean(all_ego_encoder_decoder_grad_norms, axis=(0, 2, 3))

    # Log metrics for each update step
    num_updates = len(average_ego_value_losses)
    for step in range(num_updates):
        for stat_name, stat_data in train_stats.items():
            # second dimension contains the mean and std of the metric
            stat_mean = stat_data[step, 0]
            logger.log_item(f"Train/Ego_{stat_name}", stat_mean, train_step=step, commit=True)

        logger.log_item("Eval/EgoReturn", average_ego_rets_per_iter[step], train_step=step, commit=True)
        logger.log_item("Train/EgoValueLoss", average_ego_value_losses[step], train_step=step, commit=True)
        logger.log_item("Train/EgoActorLoss", average_ego_actor_losses[step], train_step=step, commit=True)
        logger.log_item("Train/EgoEntropyLoss", average_ego_entropy_losses[step], train_step=step, commit=True)
        logger.log_item("Train/EgoReconstructionLoss", average_ego_reconstruction_losses[step], train_step=step, commit=True)
        logger.log_item("Train/EgoGradNorm", average_ego_grad_norms[step], train_step=step, commit=True)
        logger.log_item("Train/EgoEncoderDecoderGradNorm", average_ego_encoder_decoder_grad_norms[step], train_step=step, commit=True)
        logger.commit()
    
    # Saving artifacts
    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    out_savepath = save_train_run(train_out, savedir, savename="ego_train_run")
    if config["logger"]["log_train_out"]:
        logger.log_artifact(name="ego_train_run", path=out_savepath, type_name="train_run")
        # Cleanup locally logged out file
    if not config["local_logger"]["save_train_out"]:
        shutil.rmtree(out_savepath)
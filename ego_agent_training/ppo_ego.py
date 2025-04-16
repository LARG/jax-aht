'''
Script for training an PPO ego agent against a fixed partner policy.
This script uses AgentPolicy and AgentPopulation abstractions to 
support training with any type of ego or partner policy.
'''
import os
import time
import logging

import jax
import jax.numpy as jnp
import numpy as np
import optax
import hydra
from flax.training.train_state import TrainState

from envs import make_env
from envs.log_wrapper import LogWrapper
from ppo.ippo import unbatchify, Transition
from common.run_episodes import run_episodes
from common.agent_interface import AgentPopulation, MLPActorCriticPolicy
from common.initialize_agents import initialize_s5_agent, initialize_mlp_agent, initialize_rnn_agent
from common.wandb_visualizations import Logger
from common.plot_utils import get_stats, get_metric_names
from common.save_load_utils import save_train_run

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def train_ppo_ego_agent(config, env, train_rng, 
                        ego_policy, init_ego_params, n_ego_train_seeds,
                        partner_population: AgentPopulation,
                        partner_params
                        ):
    '''
    Train PPO ego agent using the given partner checkpoints and initial ego parameters.

    Args:
        config: dict, config for the training
        env: gymnasium environment
        train_rng: jax.random.PRNGKey, random key for training
        ego_policy: AgentPolicy, policy for the ego agent
        init_ego_params: dict, initial parameters for the ego agent
        n_ego_train_seeds: int, number of ego training seeds
        partner_population: AgentPopulation, population of partner agents
        partner_params: pytree of parameters for the population of agents of shape (pop_size, ...).
    '''
    # Get partner parameters from the population
    num_total_partners = partner_population.pop_size

    # ------------------------------
    # Build the PPO training function
    # ------------------------------
    def make_ppo_train(config):
        '''agent 0 is the ego agent while agent 1 is the confederate'''
        num_agents = env.num_agents
        assert num_agents == 2, "This snippet assumes exactly 2 agents."

        config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
        config["NUM_UNCONTROLLED_ACTORS"] = config["NUM_ENVS"] # assumption: we control 1 agent
        config["NUM_CONTROLLED_ACTORS"] = config["NUM_ENVS"] # assumption: we control 1 agent
        config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["ROLLOUT_LENGTH"] // config["NUM_ENVS"]

        config["NUM_ACTIONS"] = env.action_space(env.agents[0]).n
        
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
            # We need to use network.apply here to maintain the correct function 
            # signature for the TrainState's apply_fn. The higher level policy methods 
            # have different signatures.
            train_state = TrainState.create(
                apply_fn=ego_policy.network.apply,
                params=init_ego_params,
                tx=tx,
            )

            # Init envs & partner indices
            rng, reset_rng = jax.random.split(rng)
            reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
            obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rngs)

            # Each environment picks a partner index in [0, n_seeds*m_ckpts)
            rng, partner_rng = jax.random.split(rng)
            partner_indices = partner_population.sample_agent_indices(config["NUM_CONTROLLED_ACTORS"], partner_rng)
            
            def _env_step(runner_state, unused):
                """
                One step of the environment:
                1. Get observations, sample actions from all agents
                2. Step environment using sampled actions
                3. Return state, reward, ...
                """
                train_state, env_state, prev_obs, prev_done, hstate_0, partner_hstate, partner_indices, rng = runner_state
                rng, actor_rng, partner_rng, step_rng = jax.random.split(rng, 4)

                obs_0 = prev_obs["agent_0"]
                obs_1 = prev_obs["agent_1"]

                 # Get available actions for agent 0 from environment state
                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(avail_actions)
                avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
                avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)
                
                # Reshape inputs for S5 (sequence_length, batch_size, features)
                obs_0_reshaped = obs_0.reshape(1, config["NUM_CONTROLLED_ACTORS"], -1)
                done_0_reshaped = prev_done["agent_0"].reshape(1, config["NUM_CONTROLLED_ACTORS"])
                
                # Agent_0 (ego) action, value, log_prob
                act_0, val_0, pi_0, hstate_0 = ego_policy.get_action_value_policy(
                    params=train_state.params,
                    obs=obs_0_reshaped,
                    done=done_0_reshaped,
                    avail_actions=avail_actions_0,
                    hstate=hstate_0,
                    rng=actor_rng
                )
                logp_0 = pi_0.log_prob(act_0)

                act_0 = act_0.squeeze()
                logp_0 = logp_0.squeeze()
                val_0 = val_0.squeeze()

                # Agent_1 (partner) action using the AgentPopulation interface
                
                # reshape inputs and parameters so that the first dim corresponds to the number of controlled actors
                # (the dim that vmapping will be applied over) and the second dim corresponds to the time dimension of (1,)
                # required by recurrent partner policies
                obs_1_reshaped = obs_1.reshape(config["NUM_CONTROLLED_ACTORS"], 1, -1)
                done_1_reshaped = prev_done["agent_1"].reshape(config["NUM_CONTROLLED_ACTORS"], 1, -1)

                act_1, new_partner_hstate = partner_population.get_actions(
                    partner_params,
                    partner_indices,
                    obs_1_reshaped,
                    done_1_reshaped,
                    avail_actions_1,
                    partner_hstate,
                    partner_rng
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
                    reward=reward["agent_0"],
                    log_prob=logp_0,
                    obs=obs_0,
                    info=info_0,
                    avail_actions=avail_actions_0
                )
                new_runner_state = (train_state, env_state_next, obs_next, done, hstate_0, new_partner_hstate, partner_indices, rng)
                return new_runner_state, transition

            # GAE & update step
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

            def _update_minbatch(train_state, batch_info):
                init_hstate_0, traj_batch, advantages, returns = batch_info
                def _loss_fn(params, init_hstate_0, traj_batch, gae, target_v):
                    _, value, pi, _ = ego_policy.get_action_value_policy(
                        params=params, 
                        obs=traj_batch.obs,
                        done=traj_batch.done,
                        avail_actions=traj_batch.avail_actions,
                        hstate=init_hstate_0,
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
                    value_loss = (
                        0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
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
                    return total_loss, (value_loss, pg_loss, entropy)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                (loss_val, aux_vals), grads = grad_fn(
                    train_state.params, init_hstate_0, traj_batch, advantages, returns)
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, (loss_val, aux_vals)

            def _update_epoch(update_state, unused):
                train_state, init_hstate_0, traj_batch, advantages, targets, rng = update_state
                rng, perm_rng = jax.random.split(rng)
                # batch_size is now config["NUM_ENVS"]
                permutation = jax.random.permutation(perm_rng, config["NUM_CONTROLLED_ACTORS"])

                batch = (
                    init_hstate_0, # shape (1, num_agents, hidden_dim) = (1, 16, 64)
                    traj_batch, # pytree: obs is shape (rollout_len, num_actors, feat_shape) = (128, 16, 15)
                    advantages, # shape (rollout_len, num_agents) = (128, 16)
                    targets # shape (rollout_len, num_agents) = (128, 16)
                )

                # each leaf of shuffled batch has shape (rollout_len, num_agents, feat_shape)
                # except for init_hstate_0 which has shape (1, num_agents, hidden_dim)
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                # each leaf has shape (num_minibatches, rollout_len, num_agents/num_minibatches, feat_shape)
                # except for init_hstate_0 which has shape (num_minibatches, 1, num_agents/num_minibatches, hidden_dim)
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1] 
                            + list(x.shape[2:]),
                    ), 1, 0,),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, init_hstate_0, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            def _update_step(update_runner_state, unused):
                """
                1. Collect rollouts
                2. Compute advantage
                3. PPO updates
                """
                (train_state, env_state, last_obs, partner_indices, rng, update_steps) = update_runner_state
                init_hstate_0 = ego_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])
                init_partner_hstate = partner_population.init_hstate(config["NUM_UNCONTROLLED_ACTORS"])
                init_done = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}

                # 1) rollout
                runner_state = (train_state, env_state, last_obs, init_done, init_hstate_0, init_partner_hstate, partner_indices, rng)
                runner_state, traj_batch = jax.lax.scan(
                    _env_step, runner_state, None, config["ROLLOUT_LENGTH"])
                (train_state, env_state, last_obs, last_done, last_hstate_0, partner_hstate, partner_indices, rng) = runner_state

                # 2) advantage
                last_obs_batch_0 = last_obs["agent_0"]
                # Get available actions for agent 0 from environment state
                avail_actions_0 = jax.vmap(env.get_avail_actions)(env_state.env_state)["agent_0"].astype(jnp.float32)
                
                # Reshape inputs for S5
                obs_0_reshaped = last_obs_batch_0.reshape(1, config["NUM_CONTROLLED_ACTORS"], -1)
                done_0_reshaped = last_done["agent_0"].reshape(1, config["NUM_CONTROLLED_ACTORS"])
                
                # Get final value estimate for completed trajectory
                _, last_val, _, last_hstate_0 = ego_policy.get_action_value_policy(
                    params=train_state.params, 
                    obs=obs_0_reshaped,
                    done=done_0_reshaped,
                    avail_actions=jax.lax.stop_gradient(avail_actions_0),
                    hstate=last_hstate_0,
                    rng=jax.random.PRNGKey(0)  # Dummy key since we're just extracting the value
                )
                last_val = last_val.squeeze()
                advantages, targets = _calculate_gae(traj_batch, last_val)

                # 3) PPO update
                update_state = (
                    train_state,
                    init_hstate_0, # shape is (num_controlled_actors, gru_hidden_dim) with all-0s value
                    traj_batch, # obs has shape (rollout_len, num_controlled_actors, -1)
                    advantages,
                    targets,
                    rng
                )
                update_state, all_losses = jax.lax.scan(
                    _update_epoch, update_state, None, config["UPDATE_EPOCHS"])
                train_state = update_state[0]

                # Resample partner for each env for next rollout
                # Use the AgentPopulation's sample_agent_indices method
                rng, p_rng = jax.random.split(rng)
                new_partner_idx = partner_population.sample_agent_indices(
                    config["NUM_UNCONTROLLED_ACTORS"], 
                    p_rng
                )
                
                # Reset environment due to partner change
                rng, reset_rng = jax.random.split(rng)
                reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
                obs, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rngs)
                
                # Metrics
                metric = traj_batch.info
                metric["update_steps"] = update_steps
                metric["actor_loss"] = all_losses[1][1]
                metric["value_loss"] = all_losses[1][0]
                metric["entropy_loss"] = all_losses[1][2]
                new_runner_state = (train_state, env_state, obs, new_partner_idx, rng, update_steps + 1)
                return (new_runner_state, metric)

            # 3e) PPO Update and Checkpoint saving
            checkpoint_interval = max(1, config["NUM_UPDATES"] // config["NUM_CHECKPOINTS"])
            num_ckpts = config["NUM_CHECKPOINTS"]

            # Build a PyTree that holds parameters for all FCP checkpoints
            def init_ckpt_array(params_pytree):
                return jax.tree.map(
                    lambda x: jnp.zeros((num_ckpts,) + x.shape, x.dtype), 
                    params_pytree)

            max_episode_steps = config["ROLLOUT_LENGTH"]
            
            def _update_step_with_ckpt(state_with_ckpt, unused):
                ((train_state, env_state, last_obs, partner_idx, rng, update_steps),
                 checkpoint_array, ckpt_idx, init_eval_last_info) = state_with_ckpt

                # Single PPO update
                (new_runner_state, metric) = _update_step(
                    (train_state, env_state, last_obs, partner_idx, rng, update_steps),
                    None
                )
                (train_state, env_state, last_obs, partner_idx, rng, update_steps) = new_runner_state

                # Decide if we store a checkpoint
                to_store = jnp.equal(jnp.mod(update_steps, checkpoint_interval), 0)
                

                def store_and_eval_ckpt(args):
                    ckpt_arr, cidx, rng, prev_eval_ret_info = args
                    new_ckpt_arr = jax.tree.map(
                        lambda c_arr, p: c_arr.at[cidx].set(p),
                        ckpt_arr, train_state.params
                    )

                    eval_partner_indices = jnp.arange(num_total_partners)
                    # Use the AgentPopulation's gather_agent_params method
                    gathered_params = partner_population.gather_agent_params(partner_params, eval_partner_indices)
                    
                    rng, eval_rng = jax.random.split(rng)
                    eval_eps_last_infos = jax.vmap(lambda x: run_episodes(
                        eval_rng, env, agent_0_param=train_state.params, agent_0_policy=ego_policy, 
                        agent_1_param=x, agent_1_policy=partner_population.policy_cls, 
                        max_episode_steps=max_episode_steps, 
                        num_eps=config["NUM_EVAL_EPISODES"]))(gathered_params)
                    return (new_ckpt_arr, cidx + 1, rng, eval_eps_last_infos)
                
                def skip_ckpt(args):
                    return args

                (checkpoint_array, ckpt_idx, rng, eval_last_infos) = jax.lax.cond(
                    to_store, store_and_eval_ckpt, skip_ckpt, (checkpoint_array, ckpt_idx, rng, init_eval_last_info)
                )

                metric["eval_ep_last_info"] = eval_last_infos
                return ((train_state, env_state, last_obs, partner_idx, rng, update_steps),
                        checkpoint_array, ckpt_idx, eval_last_infos), metric

            # init checkpoint array
            checkpoint_array = init_ckpt_array(train_state.params)
            ckpt_idx = 0

            # init rngs
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
            # Initialize partner hidden state
            # Sample initial partner indices
            rng_train, partner_rng = jax.random.split(rng_train)
            partner_indices = partner_population.sample_agent_indices(config["NUM_UNCONTROLLED_ACTORS"], partner_rng)

            update_runner_state = (
                train_state,
                env_state,
                obsv,
                partner_indices,
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
            (final_runner_state, checkpoint_array, final_ckpt_idx, eval_eps_last_infos) = state_with_ckpt
            out = {
                "final_params": final_runner_state[0].params,
                "metrics": metrics,  # shape (NUM_UPDATES, ...)
                "checkpoints": checkpoint_array,
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

    #### Extract eval return metrics ####
    # Process eval return metrics - average across ego seeds, eval episodes and training partners for each checkpoint
    all_ego_returns = np.asarray(train_metrics["eval_ep_last_info"]["returned_episode_returns"]) # shape (n_ego_train_seeds, num_updates, num_partners, num_eval_episodes, 1)
    average_ego_rets_per_iter = np.mean(all_ego_returns[..., 0], axis=(0, 2, 3))

    # Process loss metrics - average across ego seeds, partners and minibatches dims
    # Loss metrics shape should be (n_ego_train_seeds, num_updates, ...)
    average_ego_value_losses = np.mean(all_ego_value_losses, axis=(0, 2, 3))
    average_ego_actor_losses = np.mean(all_ego_actor_losses, axis=(0, 2, 3))
    average_ego_entropy_losses = np.mean(all_ego_entropy_losses, axis=(0, 2, 3))
    
    # Log metrics for each update step
    num_updates = len(average_ego_value_losses)
    for step in range(num_updates):
        for stat_name, stat_data in train_stats.items():
            # second dimension contains the mean and std of the metric
            stat_mean = stat_data[step, 0]
            logger.log_item(f"Train/{stat_name}", stat_mean, train_step=step, commit=True)

        logger.log_item("Eval/EgoReturn", average_ego_rets_per_iter[step], checkpoint=step, commit=True)
        logger.log_item("Train/EgoValueLoss", average_ego_value_losses[step], train_step=step, commit=True)
        logger.log_item("Train/EgoActorLoss", average_ego_actor_losses[step], train_step=step, commit=True)
        logger.log_item("Train/EgoEntropyLoss", average_ego_entropy_losses[step], train_step=step, commit=True)
        
        logger.commit()
    
    # Saving artifacts
    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    # TODO: in the future, add video logging feature
    out_savepath = save_train_run(train_out, savedir, savename="saved_train_run")
    if config["logger"]["log_train_out"]:
        logger.log_artifact(name="saved_train_run", path=out_savepath, type_name="train_run")
        # Cleanup locally logged out file
    if not config["local_logger"]["save_train_out"]:
        os.remove(out_savepath)
   
    # Cleanup
    logger.close()

    
def run_ego_training(config, partner_params, pop_size: int):
    '''Run ego agent training against the population of partner agents.
    
    Args:
        config: dict, config for the training
        partner_params: partner parameters pytree with shape (pop_size, ...)
        pop_size: int, number of partner agents in the population
    '''
    algorithm_config = dict(config["algorithm"])
    logger = Logger(config)

    # Create only one environment instance
    env = make_env(algorithm_config["ENV_NAME"], algorithm_config["ENV_KWARGS"])
    env = LogWrapper(env)

    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"])
    rng, init_rng, train_rng = jax.random.split(rng, 3)
    
    partner_policy = MLPActorCriticPolicy(
        action_dim=env.action_space(env.agents[1]).n,
        obs_dim=env.observation_space(env.agents[1]).shape[0],
        activation="tanh"
    )

    # Create partner population
    partner_population = AgentPopulation(
        pop_size=pop_size,
        policy_cls=partner_policy
    )
    
    # Initialize ego agent
    if algorithm_config["ALG"] == "ppo_s5":
        ego_policy, init_params = initialize_s5_agent(algorithm_config, env, init_rng)
    elif algorithm_config["ALG"] == "ppo_mlp":
        ego_policy, init_params = initialize_mlp_agent(algorithm_config, env, init_rng)
    elif algorithm_config["ALG"] == "ppo_rnn":
        # WARNING: currently the RNN policy is not working. 
        # TODO: fix this!
        ego_policy, init_params = initialize_rnn_agent(algorithm_config, env, init_rng)
    
    log.info("Starting ego agent training...")
    start_time = time.time()
    
    # Run the training
    out = train_ppo_ego_agent(
        config=algorithm_config,
        env=env,
        train_rng=train_rng,
        ego_policy=ego_policy,
        init_ego_params=init_params,
        n_ego_train_seeds=algorithm_config["NUM_EGO_TRAIN_SEEDS"],
        partner_population=partner_population,
        partner_params=partner_params
    )
    
    log.info(f"Training completed in {time.time() - start_time:.2f} seconds")
    
    # process and log metrics
    metric_names = get_metric_names(config["ENV_NAME"])
    log_metrics(config, out, logger, metric_names)
    
    return out
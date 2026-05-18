'''
Script for training a PPO ego agent against a *population* of homogeneous, RL-based partner agents. 
Does not support training against heuristic partner agents. 
**Warning**: modify with caution, as this script is used as the main script for ego training throughout the project.

If running the script directly, please specify a partner agent config at 
`ego_agent_training/configs/algorithm/ppo_ego/_base_.yaml`.

Command to run PPO ego training:
python ego_agent_training/run.py algorithm=ppo_ego/lbf/lbf_7x7_nolevels task=lbf/lbf_7x7_nolevels label=test_ppo_ego

Suggested debug command:
python ego_agent_training/run.py algorithm=ppo_ego/lbf/lbf_7x7_nolevels task=lbf/lbf_7x7_nolevels logger.mode=disabled label=debug algorithm.TOTAL_TIMESTEPS=1e5
'''
import shutil
import time
import logging

import jax
import jax.numpy as jnp
import numpy as np
import optax
import hydra
import flax.linen as nn
from flax.training.train_state import TrainState

from agents.population_interface import AgentPopulation
from common.run_episodes import run_episodes
from common.plot_utils import get_stats, get_metric_names
from common.save_load_utils import save_train_run
from ego_agent_training.utils import initialize_ego_agent
from envs import make_env
from envs.log_wrapper import LogWrapper
from common.agent_loader_from_config import initialize_rl_agent_from_config
from marl.ppo_utils import _create_minibatches, Transition, unbatchify

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class GAILDiscriminator(nn.Module):
    action_dim: int
    hidden_dim: int = 64
    activation: str = "tanh"

    @nn.compact
    def __call__(self, obs, action):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        action_one_hot = jax.nn.one_hot(action, self.action_dim)
        x = jnp.concatenate([obs, action_one_hot], axis=-1)
        x = nn.Dense(self.hidden_dim)(x)
        x = activation(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = activation(x)
        return nn.Dense(1)(x).squeeze(-1)


def _pad_last_dim(x, target_dim: int):
    if x.shape[-1] > target_dim:
        raise ValueError(
            f"Human regularization obs_dim={x.shape[-1]} exceeds "
            f"policy obs_dim={target_dim}"
        )
    if x.shape[-1] == target_dim:
        return x
    return jnp.pad(x, [(0, 0), (0, 0), (0, target_dim - x.shape[-1])])


def _load_human_regularization_data(config, policy_obs_dim: int):
    coef = float(config.get("HUMAN_REG_COEF", 0.0))
    gail_coef = float(config.get("GAIL_REWARD_COEF", 0.0))
    mode = config.get("HUMAN_REG_MODE", "dataset_nll")
    if mode == "reference_kl" and gail_coef <= 0.0:
        return None
    if coef <= 0.0 and gail_coef <= 0.0:
        return None
    if config["ENV_NAME"] != "lbf":
        raise ValueError("Human regularization currently supports LBF only")
    if not config.get("HUMAN_REG_LBF_CONFIG"):
        raise ValueError("Set HUMAN_REG_LBF_CONFIG when HUMAN_REG_COEF > 0")

    from human_data_processing.load_lbf_data import load_bc_data_padded

    data = load_bc_data_padded(
        config_name=config["HUMAN_REG_LBF_CONFIG"],
        exclude_uncertain=bool(config.get("HUMAN_REG_EXCLUDE_UNCERTAIN", False)),
    )
    obs = _pad_last_dim(data.obs, policy_obs_dim)
    actions = data.actions
    if config.get("HUMAN_REG_INVALID_ACTION_MODE", "noop") == "noop":
        picked = jnp.take_along_axis(
            data.avail_actions,
            actions[..., None],
            axis=-1,
        ).squeeze(-1)
        actions = jnp.where(data.mask & ~picked, jnp.zeros_like(actions), actions)

    return {
        "obs": obs,
        "actions": actions,
        "avail_actions": data.avail_actions,
        "mask": data.mask,
    }


def _load_human_reference_policy(config):
    if config.get("HUMAN_REG_MODE", "dataset_nll") != "reference_kl":
        return None
    if float(config.get("HUMAN_REG_COEF", 0.0)) <= 0.0:
        return None
    if config["ENV_NAME"] != "lbf":
        raise ValueError("Reference-policy human regularization currently supports LBF only")
    if not config.get("HUMAN_REF_BC_CHECKPOINT"):
        raise ValueError("Set HUMAN_REF_BC_CHECKPOINT when HUMAN_REG_MODE=reference_kl")
    if not config.get("HUMAN_REF_BC_CONFIG"):
        raise ValueError("Set HUMAN_REF_BC_CONFIG when HUMAN_REG_MODE=reference_kl")

    from agents.bc.bc_lstm import BCLSTMAgent
    from agents.bc.evaluate_lbf import load_bc_config

    ref_config = load_bc_config(config["HUMAN_REF_BC_CONFIG"])
    env_kwargs = config["ENV_KWARGS"]
    if ref_config.lbf_feature_mode != "none":
        ref_config = ref_config._replace(
            lbf_grid_size=env_kwargs["grid_size"],
            lbf_num_food=env_kwargs["num_food"],
        )
    ref_agent = BCLSTMAgent(ref_config, weight_path=config["HUMAN_REF_BC_CHECKPOINT"])
    return {
        "config": ref_config,
        "network": ref_agent.network,
        "params": ref_agent.params,
    }


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
    progress_interval = int(config.get("PROGRESS_LOG_INTERVAL", 0) or 0)

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
        local_progress_interval = progress_interval or max(1, int(config["NUM_UPDATES"] // 20))

        config["NUM_ACTIONS"] = env.action_space(env.agents[0]).n
        assert config["NUM_CONTROLLED_ACTORS"] % config["NUM_MINIBATCHES"] == 0, "NUM_CONTROLLED_ACTORS must be divisible by NUM_MINIBATCHES"
        assert config["NUM_CONTROLLED_ACTORS"] >= config["NUM_MINIBATCHES"], "NUM_CONTROLLED_ACTORS must be >= NUM_MINIBATCHES"
        human_reg_mode = config.get("HUMAN_REG_MODE", "dataset_nll")
        human_reg_data = _load_human_regularization_data(
            config,
            ego_policy.obs_dim,
        )
        human_reg_enabled = (
            human_reg_data is not None
            and float(config.get("HUMAN_REG_COEF", 0.0)) > 0.0
            and human_reg_mode == "dataset_nll"
        )
        human_ref_data = _load_human_reference_policy(config)
        human_ref_enabled = human_ref_data is not None
        gail_enabled = (
            human_reg_data is not None
            and float(config.get("GAIL_REWARD_COEF", 0.0)) > 0.0
        )
        gail_discriminator = GAILDiscriminator(
            action_dim=config["NUM_ACTIONS"],
            hidden_dim=config.get("GAIL_DISC_HIDDEN_DIM", 64),
            activation=config.get("GAIL_DISC_ACTIVATION", "tanh"),
        )

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
            rng, disc_rng = jax.random.split(rng)
            disc_state = TrainState.create(
                apply_fn=gail_discriminator.apply,
                params=gail_discriminator.init(
                    disc_rng,
                    jnp.zeros((ego_policy.obs_dim,)),
                    jnp.array(0, dtype=jnp.int32),
                ),
                tx=optax.adam(
                    config.get("GAIL_DISC_LR", 3e-4),
                    eps=1e-5,
                ),
            )
            #  Init ego and partner hstates
            init_ego_hstate = ego_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])
            init_partner_hstate = partner_population.init_hstate(config["NUM_UNCONTROLLED_ACTORS"])
            
            def _env_step(runner_state, unused):
                """
                One step of the environment:
                1. Get observations, sample actions from all agents
                2. Step environment using sampled actions
                3. Return state, reward, ...
                """
                train_state, env_state, prev_obs, prev_done, ego_hstate, partner_hstate, partner_indices, rng = runner_state
                rng, actor_rng, partner_rng, step_rng = jax.random.split(rng, 4)

                 # Get available actions for agent 0 from environment state
                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(avail_actions)
                avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
                avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

                # Conditionally resample partners based on prev_done["__all__"]                
                needs_resample = prev_done["__all__"] # shape (NUM_ENVS,) bool
                sampled_indices_all = partner_population.sample_agent_indices(config["NUM_CONTROLLED_ACTORS"], partner_rng)

                # Determine final indices based on whether resampling was needed for each env
                updated_partner_indices = jnp.where(
                    needs_resample,         # Mask shape (NUM_ENVS,)
                    sampled_indices_all,    # Use newly sampled index if True
                    partner_indices         # Else, keep index from previous step
                )

                # Note that we do not need to reset the hiden states for both the ego and partner agents
                # as the recurrent states are automatically reset when done is True, and the partner indices are only reset when done is True.
                
                # Agent_0 (ego) action, value, log_prob
                act_0, val_0, pi_0, new_ego_hstate = ego_policy.get_action_value_policy(
                    params=train_state.params,
                    obs=prev_obs["agent_0"].reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
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
                    avail_actions=avail_actions_0
                )
                new_runner_state = (train_state, env_state_next, obs_next, done_next, 
                                    new_ego_hstate, new_partner_hstate, updated_partner_indices, rng)
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

            def _sample_human_episodes(rng, batch_episodes):
                idx = jax.random.randint(
                    rng,
                    (batch_episodes,),
                    0,
                    human_reg_data["obs"].shape[0],
                )
                return jax.tree.map(lambda x: x[idx], human_reg_data)

            def _sample_human_minibatches(rng):
                rngs = jax.random.split(rng, config["NUM_MINIBATCHES"])
                return jax.vmap(
                    lambda key: _sample_human_episodes(
                        key,
                        config.get("HUMAN_REG_BATCH_EPISODES", 16),
                    )
                )(rngs)

            def _human_regularization_loss(params, human_batch):
                obs = jnp.swapaxes(human_batch["obs"], 0, 1)
                actions = jnp.swapaxes(human_batch["actions"], 0, 1)
                avail_actions = jnp.swapaxes(human_batch["avail_actions"], 0, 1)
                mask = jnp.swapaxes(human_batch["mask"], 0, 1)
                done = jnp.logical_not(mask)
                human_hstate = ego_policy.init_hstate(obs.shape[1])
                _, _, pi, _ = ego_policy.get_action_value_policy(
                    params=params,
                    obs=obs,
                    done=done,
                    avail_actions=avail_actions,
                    hstate=human_hstate,
                    rng=jax.random.PRNGKey(0),
                )
                action_available = jnp.take_along_axis(
                    avail_actions,
                    actions[..., None],
                    axis=-1,
                ).squeeze(-1)
                valid = mask & action_available
                nll = -pi.log_prob(actions) * valid
                return nll.sum() / jnp.maximum(valid.sum(), 1.0)

            def _human_reference_logits(obs, avail_actions):
                ref_config = human_ref_data["config"]
                ref_obs = obs
                if ref_config.lbf_feature_mode == "path":
                    from agents.bc.lbf_features import augment_lbf_obs
                    ref_obs = augment_lbf_obs(
                        ref_obs,
                        grid_size=ref_config.lbf_grid_size,
                        num_food=ref_config.lbf_num_food,
                    )
                if ref_obs.shape[-1] < ref_config.obs_dim:
                    ref_obs = jnp.pad(
                        ref_obs,
                        [(0, 0), (0, 0), (0, ref_config.obs_dim - ref_obs.shape[-1])],
                    )
                elif ref_obs.shape[-1] > ref_config.obs_dim:
                    raise ValueError(
                        f"Reference BC obs_dim={ref_config.obs_dim} cannot handle "
                        f"obs_dim={ref_obs.shape[-1]}"
                    )

                init_carry = (
                    jnp.zeros((ref_obs.shape[1], ref_config.lstm_dim)),
                    jnp.zeros((ref_obs.shape[1], ref_config.lstm_dim)),
                )

                def step_fn(carry, inputs):
                    obs_t, avail_t = inputs
                    carry, logits = human_ref_data["network"].apply(
                        {"params": human_ref_data["params"]},
                        carry,
                        obs_t,
                    )
                    logits = jnp.where(avail_t > 0, logits, -1e9)
                    return carry, logits

                _, logits = jax.lax.scan(step_fn, init_carry, (ref_obs, avail_actions))
                return jax.lax.stop_gradient(logits)

            def _human_reference_kl_loss(traj_batch, pi):
                ref_logits = _human_reference_logits(
                    traj_batch.obs,
                    traj_batch.avail_actions,
                )
                ref_log_probs = jax.nn.log_softmax(ref_logits, axis=-1)
                rl_log_probs = jax.nn.log_softmax(pi.logits, axis=-1)
                ref_probs = jax.nn.softmax(ref_logits, axis=-1)
                kl = jnp.sum(
                    jnp.where(ref_probs > 0, ref_probs * (ref_log_probs - rl_log_probs), 0.0),
                    axis=-1,
                )
                return kl.mean()

            def _gail_rewards(disc_params, traj_batch):
                logits = gail_discriminator.apply(
                    disc_params,
                    traj_batch.obs,
                    traj_batch.action,
                )
                return -jax.nn.log_sigmoid(-logits)

            def _update_discriminator(disc_state, human_batch, traj_batch):
                def _loss_fn(params):
                    human_obs = human_batch["obs"].reshape(
                        -1,
                        human_batch["obs"].shape[-1],
                    )
                    human_actions = human_batch["actions"].reshape(-1)
                    human_mask = human_batch["mask"].reshape(-1)
                    policy_obs = traj_batch.obs.reshape(-1, traj_batch.obs.shape[-1])
                    policy_actions = traj_batch.action.reshape(-1)

                    human_logits = gail_discriminator.apply(
                        params,
                        human_obs,
                        human_actions,
                    )
                    policy_logits = gail_discriminator.apply(
                        params,
                        policy_obs,
                        policy_actions,
                    )
                    human_loss = (
                        optax.sigmoid_binary_cross_entropy(
                            human_logits,
                            jnp.ones_like(human_logits),
                        )
                        * human_mask
                    ).sum() / jnp.maximum(human_mask.sum(), 1.0)
                    policy_loss = optax.sigmoid_binary_cross_entropy(
                        policy_logits,
                        jnp.zeros_like(policy_logits),
                    ).mean()
                    loss = human_loss + policy_loss
                    human_acc = (
                        ((human_logits > 0.0) == human_mask) * human_mask
                    ).sum() / jnp.maximum(human_mask.sum(), 1.0)
                    policy_acc = (policy_logits < 0.0).mean()
                    return loss, (human_acc, policy_acc)

                (loss, accs), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
                    disc_state.params
                )
                return disc_state.apply_gradients(grads=grads), (loss, accs[0], accs[1])

            def _update_minbatch(train_state, batch_info):
                if human_reg_enabled:
                    ppo_batch_info, human_batch = batch_info
                else:
                    ppo_batch_info = batch_info
                    human_batch = None
                init_ego_hstate, traj_batch, advantages, returns = ppo_batch_info
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

                    human_reg_loss = jnp.array(0.0)
                    if human_reg_enabled:
                        human_reg_loss = _human_regularization_loss(
                            params,
                            human_batch,
                        )
                    if human_ref_enabled:
                        human_reg_loss = _human_reference_kl_loss(
                            traj_batch,
                            pi,
                        )

                    total_loss = (
                        pg_loss
                        + config["VF_COEF"] * value_loss
                        - config["ENT_COEF"] * entropy
                        + config.get("HUMAN_REG_COEF", 0.0) * human_reg_loss
                    )
                    return total_loss, (value_loss, pg_loss, entropy, human_reg_loss)

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
                rng, perm_rng, human_rng = jax.random.split(rng, 3)
                minibatches = _create_minibatches(traj_batch, advantages, targets, init_ego_hstate, config["NUM_CONTROLLED_ACTORS"], config["NUM_MINIBATCHES"], perm_rng)
                if human_reg_enabled:
                    human_minibatches = _sample_human_minibatches(human_rng)
                    minibatches = (minibatches, human_minibatches)
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
                (train_state, disc_state, rng, update_steps) = update_runner_state
                # Init envs & partner indices
                rng, reset_rng, p_rng = jax.random.split(rng, 3)
                reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
                init_obs, init_env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rngs)
                init_done = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}
                new_partner_indices = partner_population.sample_agent_indices(config["NUM_UNCONTROLLED_ACTORS"], p_rng)

                # 1) rollout
                runner_state = (train_state, init_env_state, init_obs, init_done, init_ego_hstate, init_partner_hstate, new_partner_indices, rng)

                runner_state, traj_batch = jax.lax.scan(
                    _env_step, runner_state, None, config["ROLLOUT_LENGTH"])
                (train_state, env_state, obs, done, ego_hstate, partner_hstate, partner_indices, rng) = runner_state
                gail_reward = jnp.zeros_like(traj_batch.reward)
                if gail_enabled:
                    gail_reward = _gail_rewards(disc_state.params, traj_batch)
                    traj_batch_for_returns = traj_batch._replace(
                        reward=(
                            config.get("GAIL_ENV_REWARD_COEF", 1.0) * traj_batch.reward
                            + config.get("GAIL_REWARD_COEF", 0.0) * gail_reward
                        )
                    )
                else:
                    traj_batch_for_returns = traj_batch

                # 2) advantage
                # Get available actions for agent 0 from environment state
                avail_actions_0 = jax.vmap(env.get_avail_actions)(env_state.env_state)["agent_0"].astype(jnp.float32)
                                
                # Get final value estimate for completed trajectory
                _, last_val, _, _ = ego_policy.get_action_value_policy(
                    params=train_state.params, 
                    obs=obs["agent_0"].reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                    done=done["agent_0"].reshape(1, config["NUM_CONTROLLED_ACTORS"]),
                    avail_actions=jax.lax.stop_gradient(avail_actions_0),
                    hstate=ego_hstate,
                    rng=jax.random.PRNGKey(0)  # Dummy key since we're just extracting the value
                )
                last_val = last_val.squeeze()
                advantages, targets = _calculate_gae(traj_batch_for_returns, last_val)

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
                gail_disc_loss = jnp.array(0.0)
                gail_disc_expert_acc = jnp.array(0.0)
                gail_disc_policy_acc = jnp.array(0.0)
                if gail_enabled:
                    rng, disc_rng = jax.random.split(rng)
                    human_disc_batch = _sample_human_episodes(
                        disc_rng,
                        config.get("GAIL_DISC_BATCH_EPISODES", 32),
                    )

                    def _disc_epoch(carry, _):
                        return _update_discriminator(carry, human_disc_batch, traj_batch)

                    disc_state, disc_metrics = jax.lax.scan(
                        _disc_epoch,
                        disc_state,
                        None,
                        config.get("GAIL_DISC_EPOCHS", 1),
                    )
                    gail_disc_loss = disc_metrics[0].mean()
                    gail_disc_expert_acc = disc_metrics[1].mean()
                    gail_disc_policy_acc = disc_metrics[2].mean()
                _, loss_terms, avg_grad_norm = losses_and_grads

                metric = traj_batch.info
                metric["update_steps"] = update_steps
                metric["actor_loss"] = loss_terms[1].mean()
                metric["value_loss"] = loss_terms[0].mean()
                metric["entropy_loss"] = loss_terms[2].mean()
                metric["human_reg_loss"] = loss_terms[3].mean()
                metric["gail_reward"] = gail_reward
                metric["gail_disc_loss"] = gail_disc_loss
                metric["gail_disc_expert_acc"] = gail_disc_expert_acc
                metric["gail_disc_policy_acc"] = gail_disc_policy_acc
                metric["avg_grad_norm"] = avg_grad_norm.mean()
                new_runner_state = (train_state, disc_state, rng, update_steps + 1)
                return (new_runner_state, metric)

            # PPO Update and Checkpoint saving
            ckpt_and_eval_interval = config["NUM_UPDATES"] // max(1, config["NUM_CHECKPOINTS"] - 1)  # -1 because we store a ckpt at the last update
            num_ckpts = config["NUM_CHECKPOINTS"]

            # Build a PyTree that holds parameters for all FCP checkpoints
            def init_ckpt_array(params_pytree):
                return jax.tree.map(
                    lambda x: jnp.zeros((num_ckpts,) + x.shape, x.dtype), 
                    params_pytree)

            max_episode_steps = config["ROLLOUT_LENGTH"]

            def _emit_progress(step, ret):
                print(
                    f"[train] update {int(step)}/{int(config['NUM_UPDATES'])} "
                    f"avg_eval_return={float(ret):.4f}"
                )
            
            def _update_step_with_ckpt(state_with_ckpt, unused):
                (update_state, checkpoint_array, ckpt_idx, init_eval_last_info) = state_with_ckpt

                # Single PPO update
                new_update_state, metric = _update_step(
                    update_state,
                    None
                )
                (train_state, disc_state, rng, update_steps) = new_update_state

                # update steps is 1-indexed because it was incremented at the end of the update step
                to_store = jnp.logical_or(jnp.equal(jnp.mod(update_steps-1, ckpt_and_eval_interval), 0),
                                        jnp.equal(update_steps, config["NUM_UPDATES"]))


                def store_and_eval_ckpt(args):
                    ckpt_arr, cidx, rng, prev_eval_ret_info = args
                    new_ckpt_arr = jax.tree.map(
                        lambda c_arr, p: c_arr.at[cidx].set(p),
                        ckpt_arr, train_state.params
                    )

                    eval_partner_indices = jnp.arange(num_total_partners)
                    gathered_params = partner_population.gather_agent_params(partner_params, eval_partner_indices)
                    
                    rng, eval_rng = jax.random.split(rng)
                    eval_eps_last_infos = jax.tree.map(lambda x: x.mean(), jax.vmap(lambda x: run_episodes(
                        eval_rng, env, agent_0_param=train_state.params, agent_0_policy=ego_policy,
                        agent_1_param=x, agent_1_policy=partner_population.policy_cls,
                        max_episode_steps=max_episode_steps,
                        num_eps=config["NUM_EVAL_EPISODES"]))(gathered_params))
                    return (new_ckpt_arr, cidx + 1, rng, eval_eps_last_infos)
                
                def skip_ckpt(args):
                    return args

                (checkpoint_array, ckpt_idx, rng, eval_last_infos) = jax.lax.cond(
                    to_store, store_and_eval_ckpt, skip_ckpt, (checkpoint_array, ckpt_idx, rng, init_eval_last_info)
                )

                # Condense per-timestep metrics to per-update scalars.
                # Full metrics are logged via io_callback in _update_step.
                # Without this, storing (ROLLOUT_LENGTH, NUM_ACTORS) per key
                # per update causes OOM for long runs (e.g. 1e9 steps).
                mask = metric["returned_episode"]
                n_episodes = mask.sum()
                condensed_metric = {}
                for key, val in metric.items():
                    if key in ("update_steps", "actor_loss", "value_loss",
                               "entropy_loss", "human_reg_loss",
                               "gail_disc_loss", "gail_disc_expert_acc",
                               "gail_disc_policy_acc", "avg_grad_norm"):
                        # Already scalar-ish from loss aggregation
                        condensed_metric[key] = val.mean() if val.ndim > 0 else val
                    elif key == "returned_episode":
                        condensed_metric[key] = n_episodes.astype(jnp.float32)
                    else:
                        condensed_metric[key] = jnp.where(
                            n_episodes > 0,
                            jnp.where(mask, val, 0.0).sum() / jnp.maximum(n_episodes, 1),
                            0.0,
                        )
                condensed_metric["eval_ep_last_info"] = eval_last_infos

                avg_eval_return = eval_last_infos["returned_episode_returns"].mean()
                should_log_progress = jnp.logical_or(
                    jnp.equal(jnp.mod(update_steps, local_progress_interval), 0),
                    jnp.equal(update_steps, config["NUM_UPDATES"])
                )

                def _log_progress(args):
                    step, ret = args
                    jax.debug.callback(_emit_progress, step, ret)
                    return None

                jax.lax.cond(
                    should_log_progress,
                    _log_progress,
                    lambda _: None,
                    (update_steps, avg_eval_return),
                )
                return ((train_state, disc_state, rng, update_steps),
                         checkpoint_array, ckpt_idx, eval_last_infos), condensed_metric

            checkpoint_array = init_ckpt_array(train_state.params)
            ckpt_idx = 0

            rng, rng_eval, rng_train = jax.random.split(rng, 3)
            # Init eval return infos
            eval_partner_indices = jnp.arange(num_total_partners)
            gathered_params = partner_population.gather_agent_params(partner_params, eval_partner_indices)
            eval_eps_last_infos = jax.tree.map(lambda x: x.mean(), jax.vmap(lambda x: run_episodes(
                        rng_eval, env,
                        agent_0_param=train_state.params, agent_0_policy=ego_policy,
                        agent_1_param=x, agent_1_policy=partner_population.policy_cls,
                        max_episode_steps=max_episode_steps,
                        num_eps=config["NUM_EVAL_EPISODES"]))(gathered_params))

            # initial runner state for scanning
            update_steps = 0
            rng_train, partner_rng = jax.random.split(rng_train)

            update_runner_state = (train_state, disc_state, rng_train, update_steps)
            state_with_ckpt = (update_runner_state, checkpoint_array, ckpt_idx, eval_eps_last_infos)
            
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
    rng, init_partner_rng, init_ego_rng, train_rng = jax.random.split(rng, 4)
    

    partner_agent_config = dict(algorithm_config["partner_agent"])
    assert len(partner_agent_config) == 1, "Only supports training against one type of partner agent."
    
    partner0_name = list(partner_agent_config.keys())[0]
    partner0_agent_config = list(partner_agent_config.values())[0]
    partner_policy, partner_params, init_partner_params, idx_labels = initialize_rl_agent_from_config(
        partner0_agent_config, partner0_name, env, init_partner_rng)

    flattened_partner_params = jax.tree.map(lambda x, y: x.reshape((-1,) + y.shape), partner_params, init_partner_params)        
    pop_size = jax.tree.leaves(flattened_partner_params)[0].shape[0]

    # Create partner population
    partner_population = AgentPopulation(
        pop_size=pop_size,
        policy_cls=partner_policy
    )
    
    # Initialize ego agent
    ego_policy, init_ego_params = initialize_ego_agent(algorithm_config, env, init_ego_rng)

    expected_updates = int(
        algorithm_config["TOTAL_TIMESTEPS"] //
        algorithm_config["ROLLOUT_LENGTH"] //
        algorithm_config["NUM_ENVS"]
    )
    progress_interval = int(algorithm_config.get("PROGRESS_LOG_INTERVAL", 0) or max(1, expected_updates // 20))
    
    log.info(
        "Starting ego agent training with %s updates, %s envs, %s partner(s). "
        "Progress will be logged every ~%s updates.",
        expected_updates,
        algorithm_config["NUM_ENVS"],
        pop_size,
        progress_interval,
    )
    start_time = time.time()
    
    # Run the training
    out = train_ppo_ego_agent(
        config=algorithm_config,
        env=env,
        train_rng=train_rng,
        ego_policy=ego_policy,
        init_ego_params=init_ego_params,
        n_ego_train_seeds=algorithm_config["NUM_EGO_TRAIN_SEEDS"],
        partner_population=partner_population,
        partner_params=flattened_partner_params
    )
    
    log.info(f"Training completed in {time.time() - start_time:.2f} seconds")
    
    # process and log metrics
    metric_names = get_metric_names(config["ENV_NAME"])
    log_metrics(config, out, wandb_logger, metric_names)
    
    return out["final_params"], ego_policy, init_ego_params

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
    
    all_ego_value_losses = np.asarray(train_metrics["value_loss"])
    all_ego_actor_losses = np.asarray(train_metrics["actor_loss"])
    all_ego_entropy_losses = np.asarray(train_metrics["entropy_loss"])
    all_ego_human_reg_losses = np.asarray(train_metrics["human_reg_loss"])
    all_ego_grad_norms = np.asarray(train_metrics["avg_grad_norm"])
    # Process eval return metrics - average across ego seeds, eval episodes,
    # training partners and num_agents per game for each checkpoint
    all_ego_returns = np.asarray(train_metrics["eval_ep_last_info"]["returned_episode_returns"])
    # Handle both condensed (scalars per update) and full metric shapes.
    # Condensed: (n_ego_train_seeds, num_updates)
    # Full: (n_ego_train_seeds, num_updates, num_partners, ...)
    extra_axes = tuple(range(2, all_ego_returns.ndim))
    average_ego_rets_per_iter = np.mean(all_ego_returns, axis=(0,) + extra_axes)

    # Process loss metrics - average across ego seeds, partners and minibatches dims
    # Loss metrics shape should be (n_ego_train_seeds, num_updates)
    average_ego_value_losses = np.mean(all_ego_value_losses, axis=0)
    average_ego_actor_losses = np.mean(all_ego_actor_losses, axis=0)
    average_ego_entropy_losses = np.mean(all_ego_entropy_losses, axis=0)
    average_ego_human_reg_losses = np.mean(all_ego_human_reg_losses, axis=0)
    average_ego_grad_norms = np.mean(all_ego_grad_norms, axis=0)

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
        logger.log_item("Train/EgoHumanRegLoss", average_ego_human_reg_losses[step], train_step=step, commit=True)
        logger.log_item("Train/EgoGradNorm", average_ego_grad_norms[step], train_step=step, commit=True)
        logger.commit()
    
    # Saving artifacts
    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    out_savepath = save_train_run(train_out, savedir, savename="ego_train_run")
    if config["logger"]["log_train_out"]:
        logger.log_artifact(name="ego_train_run", path=out_savepath, type_name="train_run")
        # Cleanup locally logged out file
    if not config["local_logger"]["save_train_out"]:
        shutil.rmtree(out_savepath)

import logging
import jax
import numpy as np
from omegaconf import OmegaConf

from agents.initialize_agents import initialize_s5_agent, initialize_mlp_agent, \
    initialize_rnn_agent, initialize_actor_with_double_critic, \
    initialize_actor_with_conditional_critic
from agents.lbf.agent_policy_wrappers import (
    LBFRandomPolicyWrapper, LBFSequentialFruitPolicyWrapper,
    LBFEntitledPolicyWrapper, LBFGreedyHeuristicPolicyWrapper,
)
from agents.overcooked.agent_policy_wrappers import (
    OvercookedRandomPolicyWrapper, OvercookedIndependentPolicyWrapper,
    OvercookedOnionPolicyWrapper, OvercookedPlatePolicyWrapper,
    OvercookedStaticPolicyWrapper,
)
from common.save_load_utils import load_checkpoints
from envs.overcooked.augmented_layouts import augmented_layouts

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def process_idx_list(idx_list):
    idxs = np.array(idx_list)
    if idxs.ndim == 1:
        return idxs
    elif idxs.ndim == 2:
        rows = idxs[:, 0]
        cols = idxs[:, 1]
        return (rows, cols)
    else:
        raise ValueError(f"Invalid index list shape: {idxs.shape}")

def create_idx_labels(idx_list, checkpoint_shape):
    """Create human-readable index labels based on the checkpoint extraction pattern.
    
    Args:
        idx_list: The list of indices used to extract checkpoints, or None/"all" for all checkpoints
        checkpoint_shape: The shape of the checkpoint array
        
    Returns:
        Array of string labels with the same shape as the original checkpoint array,
        or filtered according to idx_list
    """
    # If loading all checkpoints, create labels with the same shape as the original checkpoints
    if idx_list is None:
        # Handle different dimensional checkpoints
        if len(checkpoint_shape) >= 3:  # At least 2D of checkpoints (e.g., seeds × steps)
            # Create a 2D array of labels with same shape as the first two dimensions
            rows, cols = checkpoint_shape[0], checkpoint_shape[1]
            idx_labels = []
            for row_idx in range(rows):
                row_labels = []
                for col_idx in range(cols):
                    row_labels.append(f"{row_idx}, {col_idx}")
                idx_labels.append(row_labels)
        else:  # 1D of checkpoints
            # Create a 1D array of labels
            idx_labels = [f"{i}" for i in range(checkpoint_shape[0])]
    # If loading specific checkpoints, create labels by converting idx_list to strings
    elif isinstance(idx_list[0], list):
        # For 2D indices like [[0, -1], [1, -1], [2, -1]]
        idx_labels = [f"{idx[0]}, {idx[1]}" for idx in idx_list]
    else:
        # For 1D indices like [0, 1, 2]
        idx_labels = [f"{idx}" for idx in idx_list]
    return idx_labels

def initialize_heuristic_agent_from_config(agent_config, agent_name, task_name, env_kwargs=None):
    '''Load a heuristic (non-RL) agent from config, dispatching on task_name.

    agent_config must include "actor_type".
    env_kwargs is used as a fallback for env-level parameters (e.g. grid_size,
    num_fruits for lbf; layout for overcooked-v1).  Per-agent values in
    agent_config take priority. Note that the LBF enviornment calls this
    'num_food'.

    Returns:
        policy: policy function (no checkpoint or params required)
    '''
    assert "actor_type" in agent_config, "Actor type must be provided."
    actor_type = agent_config["actor_type"]

    if env_kwargs is None:
        env_kwargs = {}

    if 'lbf' in task_name:
        # Grid dimensions: per-agent config > env_kwargs > defaults (7x7, 3 fruits).
        grid_size = agent_config.get("grid_size", env_kwargs.get("grid_size", 7))
        num_fruits = agent_config.get("num_fruits", env_kwargs.get("num_food", 3))

        if actor_type == "random_agent":
            return LBFRandomPolicyWrapper()
        if actor_type == "seq_agent":
            ordering_strategy = agent_config.get("ordering_strategy", "lexicographic")
            return LBFSequentialFruitPolicyWrapper(
                grid_size=grid_size,
                num_fruits=num_fruits,
                ordering_strategy=ordering_strategy,
                using_log_wrapper=True,
            )
        if actor_type == "entitled_agent":
            return LBFEntitledPolicyWrapper(
                grid_size=grid_size,
                num_fruits=num_fruits,
                using_log_wrapper=True,
            )
        if actor_type == "greedy_agent":
            heuristic = agent_config.get("heuristic", "closest_self")
            return LBFGreedyHeuristicPolicyWrapper(
                grid_size=grid_size,
                num_fruits=num_fruits,
                heuristic=heuristic,
                using_log_wrapper=True,
            )
        raise ValueError(f"Unrecognized actor type for {task_name}: '{actor_type}' ({agent_name})")

    if 'overcooked-v1' in task_name:
        aug_layout_dict = augmented_layouts[env_kwargs["layout"]]

        if actor_type == "random_agent":
            return OvercookedRandomPolicyWrapper(aug_layout_dict, using_log_wrapper=True)
        if actor_type == "static_agent":
            return OvercookedStaticPolicyWrapper(aug_layout_dict, using_log_wrapper=True)
        if actor_type == "independent_agent":
            return OvercookedIndependentPolicyWrapper(
                aug_layout_dict,
                using_log_wrapper=True,
                p_onion_on_counter=agent_config.get("p_onion_on_counter", 0.0),
                p_plate_on_counter=agent_config.get("p_plate_on_counter", 0.0),
            )
        if actor_type == "onion_agent":
            return OvercookedOnionPolicyWrapper(
                aug_layout_dict,
                using_log_wrapper=True,
                p_onion_on_counter=agent_config.get("p_onion_on_counter", 0.0),
            )
        if actor_type == "plate_agent":
            return OvercookedPlatePolicyWrapper(
                aug_layout_dict,
                using_log_wrapper=True,
                p_plate_on_counter=agent_config.get("p_plate_on_counter", 0.0),
            )
        raise ValueError(f"Unrecognized actor type for {task_name}: '{actor_type}' ({agent_name})")

    raise ValueError(
        f"Unknown task '{task_name}' for heuristic agent {agent_name}. "
        f"Expected 'lbf' or a task containing 'overcooked-v1'."
    )

def initialize_rl_agent_from_config(agent_config, agent_name, env, rng):
    '''Load RL agent from checkpoint and initialize from config.
    The agent_config dictionary should have the following structure:
    {
        "path": str,
        "actor_type": str,
        "ckpt_key": str, # key to load from checkpoint. Default is "checkpoints".
        "custom_loader": dict, # custom loader for the checkpoint. Default is None.
        "idx_list": list, # list of indices to load from checkpoint. If null, all checkpoints will be loaded.
        # and any other parameters needed to initialize the agent policy
    }

    Returns:
        policy: policy function
        agent_params: agent parameters from checkpoint
        init_params: initial agent parameters from initialization
        idx_list: list of indices used to extract checkpoints
        idx_labels: list of string labels corresponding to the indices
    '''
    assert "path" in agent_config, "Path to agent checkpoint must be provided."
    assert "actor_type" in agent_config, "Actor type must be provided."
    assert "idx_list" in agent_config, "Indices to load from checkpoint must be provided."

    agent_path = agent_config["path"]
    ckpt_key = agent_config.get("ckpt_key", "checkpoints")
    custom_loader_cfg = agent_config.get("custom_loader", None)

    agent_ckpt = load_checkpoints(agent_path, ckpt_key=ckpt_key, custom_loader_cfg=custom_loader_cfg)

    leaf0_shape = jax.tree.leaves(agent_ckpt)[0].shape

    if agent_config["idx_list"] is None: # load all checkpoints
        idx_list = None
        agent_params = agent_ckpt
    else: # load specific checkpoints
        # convert omegaconf list config to list recursively
        try:
            idx_list = OmegaConf.to_object(agent_config["idx_list"])
        except Exception as e:
            log.warning(f"Error interpreting agent_config['idx_list'] as OmegaConf object: {e}. Treating as list.")
            idx_list = agent_config["idx_list"]
        idx_list = jax.tree.map(lambda x: int(x), idx_list)
        idxs = process_idx_list(idx_list)
                
        agent_params = jax.tree.map(lambda x: x[idxs], agent_ckpt)
    
    log.info(f"Loaded {agent_name} checkpoint where leaf 0 has shape {leaf0_shape}. "
            f" Selecting indices {idx_list if idx_list is not None else 'all'} for evaluation.")

    # Create index labels for the loaded checkpoints
    idx_labels = create_idx_labels(idx_list, leaf0_shape)


    rng, init_rng = jax.random.split(rng, 2)
    
    if agent_config["actor_type"] == "s5":
        policy, init_params = initialize_s5_agent(agent_config, env, init_rng)
        # Make compatible with old naming for S5 layers
        if "action_body_0" in agent_params['params'].keys(): # CLEANUP FLAG
            agent_param_keys = list(agent_params['params'].keys())
            for k in agent_param_keys:
                if "body" in k:
                    new_k = k.replace("body", "body_layers")
                    agent_params['params'][new_k] = agent_params['params'][k]
                    del agent_params['params'][k]
    elif agent_config["actor_type"] == "mlp":
        policy, init_params = initialize_mlp_agent(agent_config, env, init_rng)
    elif agent_config["actor_type"] == "rnn":
        policy, init_params = initialize_rnn_agent(agent_config, env, init_rng)
    elif agent_config["actor_type"] == "actor_with_double_critic":
        policy, init_params = initialize_actor_with_double_critic(agent_config, env, init_rng)
    elif agent_config["actor_type"] == "actor_with_conditional_critic":
        policy, init_params = initialize_actor_with_conditional_critic(agent_config, env, init_rng)
    else:
        raise ValueError(f"Invalid actor type: {agent_config['actor_type']}")

    assert jax.tree.structure(agent_params) == jax.tree.structure(init_params), "Agent parameters and initial parameters must have the same structure."

    return policy, agent_params, init_params, idx_labels

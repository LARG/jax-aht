import jax
import jax.numpy as jnp

from common.save_load_utils import load_checkpoints
from common.agent_interface import S5ActorCriticPolicy, MLPActorCriticPolicy


def select_checkpoint_params(full_checkpoints, seed_idx, ckpt_idx):
    """
    Slices the pytree so each leaf contains only the parameters for
    the given seed and checkpoint.
    """
    # 'full_checkpoints' is the entire dict with top-level keys like 'params', etc.
    return jax.tree.map(lambda x: x[seed_idx, ckpt_idx], full_checkpoints)

class MLPActorCriticLoader():
    def __init__(self, train_run_path, action_dim, obs_dim, n=0, m=-1):
        '''Loads and initializes a single policy checkpoint from saved pickled file.

        train_run_path: path to the training run pickle file
        action_dim: number of actions
        n: seed index
        m: checkpoint index
        '''
        policy_checkpoints = load_checkpoints(train_run_path)
        self.model_params = select_checkpoint_params(policy_checkpoints, n, m)
        self.policy = MLPActorCriticPolicy(action_dim, obs_dim)

    def act(self, obs, done, avail_actions, hstate, rng, test_mode=True):
        '''Returns an action given an observation.
        obs_dict: dict containing 'obs', 'dones', and 'avail_actions'
        '''
        rng, dummy_rng, act_rng = jax.random.split(rng, 3)        
        _, _, pi, new_hstate = self.policy.get_action_value_policy(self.model_params, obs, done, avail_actions, hstate, dummy_rng)
        if test_mode:
            action = pi.mode()
        else:
            action = pi.sample(seed=act_rng)
        return action, new_hstate, rng
    
    def init_hstate(self, batch_size):
        return self.policy.init_hstate(batch_size)
    
class S5ActorCriticLoader():
    def __init__(self, train_run_path, action_dim, obs_dim, n=0, m=-1):
        '''Loads and initializes a single policy checkpoint from saved pickled file.

        train_run_path: path to the training run pickle file
        action_dim: number of actions
        n: seed index
        m: checkpoint index
        '''
        policy_checkpoints = load_checkpoints(train_run_path)
        self.model_params = select_checkpoint_params(policy_checkpoints, n, m)
        self.policy = S5ActorCriticPolicy(action_dim, obs_dim)


    def act(self, obs, done, avail_actions, hstate, rng, test_mode=True):
        '''Returns an action given an observation.
        obs_dict: dict containing 'obs', 'dones', and 'avail_actions'
        '''
        rng, dummy_rng, act_rng = jax.random.split(rng, 3)
        
        _, _, pi, new_hstate = self.policy.get_action_value_policy(self.model_params, obs, done, avail_actions, hstate, dummy_rng)
        
        if test_mode:
            action = pi.mode()
        else:
            action = pi.sample(seed=act_rng)
        action = action.squeeze()
        return action, new_hstate, rng
    
    def init_hstate(self, batch_size):
        return self.policy.init_hstate(batch_size)

class RandomActor():
    def __init__(self):
        pass

    def act(self, obs, done, avail_actions, hstate, rng, test_mode=True):
        '''Returns a random action from available actions.
        obs_dict: dict containing 'obs', 'dones', and 'avail_actions'
        '''
        rng, act_rng = jax.random.split(rng)
        # Get indices of available actions
        avail_indices = jnp.where(avail_actions == 1)[0]
        # Sample from available actions
        action = int(jax.random.choice(act_rng, avail_indices))
        return action, hstate, rng
    
    def init_hstate(self, batch_size):
        return None

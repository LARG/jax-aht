import jax
import jax.numpy as jnp

from common.mlp_actor_critic import ActorCritic
from common.s5_actor_critic import S5ActorCritic, make_DPLR_HiPPO, init_S5SSM, StackedEncoderModel
from common.save_load_utils import load_checkpoints


def select_checkpoint_params(full_checkpoints, seed_idx, ckpt_idx):
    """
    Slices the pytree so each leaf contains only the parameters for
    the given seed and checkpoint.
    """
    # 'full_checkpoints' is the entire dict with top-level keys like 'params', etc.
    return jax.tree.map(lambda x: x[seed_idx, ckpt_idx], full_checkpoints)

class MLPActorCriticLoader():
    def __init__(self, train_run_path, action_dim, n=0, m=-1):
        '''Loads and initializes a single policy checkpoint from saved pickled file.

        train_run_path: path to the training run pickle file
        action_dim: number of actions
        n: seed index
        m: checkpoint index
        '''
        policy_checkpoints = load_checkpoints(train_run_path)
        self.model_params = select_checkpoint_params(policy_checkpoints, n, m)
        self.model = ActorCritic(action_dim)

    def act(self, obs_dict, rng, test_mode=True):
        '''Returns an action given an observation.
        obs_dict: dict containing 'obs', 'dones', and 'avail_actions'
        '''
        rng, act_rng = jax.random.split(rng)        
        pi, _ = self.model.apply(self.model_params, (obs_dict['obs'], obs_dict['avail_actions'])) 
        if test_mode:
            action = pi.mode()
        else:
            action = pi.sample(seed=act_rng)
        return action, rng

class S5ActorCriticLoader():
    def __init__(self, train_run_path, action_dim, n=0, m=-1):
        '''Loads and initializes a single policy checkpoint from saved pickled file.

        train_run_path: path to the training run pickle file
        action_dim: number of actions
        n: seed index
        m: checkpoint index
        '''
        policy_checkpoints = load_checkpoints(train_run_path)
        self.model_params = select_checkpoint_params(policy_checkpoints, n, m)
        
        # S5 default config
        # TODO: avoid hard-coding SSM configs in future
        self.config = {
            "S5_D_MODEL": 16,
            "S5_SSM_SIZE": 16,
            "S5_N_LAYERS": 2,
            "S5_BLOCKS": 1,
        }
        
        # Initialize S5 specific parameters
        d_model = self.config["S5_D_MODEL"]
        ssm_size = self.config["S5_SSM_SIZE"]
        n_layers = self.config["S5_N_LAYERS"]
        blocks = self.config["S5_BLOCKS"]
        block_size = int(ssm_size / blocks)

        Lambda, _, _, V,  _ = make_DPLR_HiPPO(ssm_size)
        block_size = block_size // 2
        ssm_size = ssm_size // 2
        Lambda = Lambda[:block_size]
        V = V[:, :block_size]
        Vinv = V.conj().T

        ssm_init_fn = init_S5SSM(H=d_model,
                                 P=ssm_size,
                                 Lambda_re_init=Lambda.real,
                                 Lambda_im_init=Lambda.imag,
                                 V=V,
                                 Vinv=Vinv)
        
        self.model = S5ActorCritic(
            action_dim=action_dim,
            ssm_init_fn=ssm_init_fn
        )
        
        # Hidden state is maintained by in a class attribute, making this 
        # class non-jittable.
        self.hidden = StackedEncoderModel.initialize_carry(1, ssm_size, n_layers)


    def act(self, obs_dict, rng, test_mode=True):
        '''Returns an action given an observation.
        obs_dict: dict containing 'obs', 'dones', and 'avail_actions'
        '''
        rng, act_rng = jax.random.split(rng)
        
        # Update hidden state and get action distribution
        model_input = (
            obs_dict['obs'].reshape(1, 1, -1),
            obs_dict['dones'].reshape(1, 1), 
            obs_dict['avail_actions'].reshape(1, -1)
        )
        self.hidden, pi, _ = self.model.apply(self.model_params, self.hidden, model_input)
        
        if test_mode:
            action = pi.mode()
        else:
            action = pi.sample(seed=act_rng)
        action = action.squeeze()
        return action, rng

class RandomActor():
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs_dict, rng, test_mode=True):
        '''Returns a random action from available actions.
        obs_dict: dict containing 'obs', 'dones', and 'avail_actions'
        '''
        rng, act_rng = jax.random.split(rng)
        avail_actions = obs_dict['avail_actions']
        # Get indices of available actions
        avail_indices = jnp.where(avail_actions == 1)[0]
        # Sample from available actions
        action = int(jax.random.choice(act_rng, avail_indices))
        return action, rng

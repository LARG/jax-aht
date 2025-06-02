from functools import partial
import jax


class AgentPopulation:
    '''Base class for a population of homogeneous agents
    TODO: develop more complex population classes that can handle heterogeneous agents
    '''
    def __init__(self, pop_size, policy_cls):
        '''
        Args:
            pop_size: int, number of agents in the population
            policy_cls: an instance of the AgentPolicy class. The policy class for the population of agents
        '''
        self.pop_size = pop_size
        self.policy_cls = policy_cls # AgentPolicy class

    def sample_agent_indices(self, n, rng):
        '''Sample n indices from the population, with replacement.'''
        return jax.random.randint(rng, (n,), 0, self.pop_size)
    
    def gather_agent_params(self, pop_params, agent_indices):
        '''Gather the parameters of the agents specified by agent_indices.

        Args:
            pop_params: pytree of parameters for the population of agents of shape (pop_size, ...).
            agent_indices: indices with shape (num_envs,), each in [0, pop_size)
        '''
        def gather_leaf(leaf):
            # leaf shape: (num_envs,  ...)
            return jax.vmap(lambda idx: leaf[idx])(agent_indices)
        return jax.tree.map(gather_leaf, pop_params)
    
    def get_actions(self, pop_params, agent_indices, obs, done, avail_actions, hstate, rng, 
                    env_state=None, aux_obs=None, test_mode=False):
        '''
        Get the actions of the agents specified by agent_indices. 
        
        Args:
            pop_params: pytree of parameters for the population of agents of shape (pop_size, ...).
            agent_indices: indices with shape (num_envs,), each in [0, pop_size)
            obs: observations with shape (num_envs, ...) 
            done: done flags with shape (num_envs,)
            avail_actions: available actions with shape (num_envs, num_actions)
            hstate: hidden state with shape (num_envs, ...) or None if policy doesn't use hidden state
            rng: random key
            env_state: environment state with shape (num_envs, ...) or None if policy doesn't use env state
            aux_obs: an optional auxiliary vector to append to the observation
        Returns:
            actions: actions with shape (num_envs,)
            new_hstate: new hidden state with shape (num_envs, ...) or None
        '''
        gathered_params = self.gather_agent_params(pop_params, agent_indices)
        num_envs = agent_indices.squeeze().shape[0]
        rngs_batched = jax.random.split(rng, num_envs)
        vmapped_get_action = jax.vmap(partial(self.policy_cls.get_action, 
                                              aux_obs=aux_obs, 
                                              env_state=env_state, 
                                              test_mode=test_mode))
        actions, new_hstate = vmapped_get_action(
            gathered_params, obs, done, avail_actions, hstate, 
            rngs_batched)
        return actions, new_hstate
    
    def init_hstate(self, n: int, aux_info: dict=None):
        '''Initialize the hidden state for n members of the population.'''
        return self.policy_cls.init_hstate(n, aux_info)
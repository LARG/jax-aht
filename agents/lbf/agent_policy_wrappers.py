'''Wrap heuristic agent policies in AgentPolicy interface.
TODO: clean up logic by vectorizing init_hstate. See HeuristicPolicyPopulation.
'''
import jax
import jax.numpy as jnp
from agents.agent_interface import AgentPolicy
from agents.lbf.random_agent import RandomAgent
from agents.lbf.sequential_fruit_agent import SequentialFruitAgent



class LBFRandomPolicyWrapper(AgentPolicy):
    def __init__(self):
        self.policy = RandomAgent() # agent id doesn't matter for the random agent

    def get_action(self, params, obs, done, avail_actions, hstate, rng, 
                   env_state, aux_obs=None, test_mode=False):
        # Handle recurrent format (time, batch, ...)
        if obs.ndim == 3 and obs.shape[0] == 1:
            obs = obs[0]
            done = done[0]
            avail_actions = avail_actions[0]
        
        if obs.ndim == 1:
            # Single environment
            action, new_hstate = self.policy.get_action(obs, env_state, hstate, rng)
            return action, new_hstate
        else:
            # Batched
            num_envs = obs.shape[0]
            rngs = jax.random.split(rng, num_envs)
            
            def single_action(i):
                o = obs[i]
                es = jax.tree_map(lambda x: x[i] if isinstance(x, jnp.ndarray) else x, env_state)
                hs = jax.tree_map(lambda x: x[i], hstate)
                r = rngs[i]
                action, new_hs = self.policy.get_action(o, es, hs, r)
                return action, new_hs
            
            actions, new_hstates = jax.vmap(single_action)(jnp.arange(num_envs))
            # new_hstates is already batched by vmap
            new_hstate = new_hstates
            return actions, new_hstate

    def init_hstate(self, batch_size: int, aux_info=None):
        single_state = self.policy.init_agent_state(aux_info["agent_id"])
        batched_state = jax.tree_map(lambda x: jnp.stack([x] * batch_size), single_state)
        return batched_state


class LBFSequentialFruitPolicyWrapper(AgentPolicy):
    """Policy wrapper for the SequentialFruitAgent that visits fruits in a predetermined order."""
    def __init__(self, grid_size: int = 7, num_fruits: int = 3, 
                 ordering_strategy: str = 'lexicographic', using_log_wrapper: bool = False):
        self.policy = SequentialFruitAgent(grid_size, num_fruits, ordering_strategy)
        self.using_log_wrapper = using_log_wrapper

    def get_action(self, params, obs, done, avail_actions, hstate, rng, 
                   env_state, aux_obs=None, test_mode=False):
        # hstate represents the agent state
        if self.using_log_wrapper:
            env_state = env_state.env_state
        
        # Handle recurrent format (time, batch, ...)
        if obs.ndim == 3 and obs.shape[0] == 1:
            obs = obs[0]
            done = done[0]
            avail_actions = avail_actions[0]
        
        if obs.ndim == 1:
            # Single environment
            action, new_hstate = self.policy.get_action(obs, env_state, hstate, rng)
            # Reset if done
            new_hstate = jax.lax.cond(done, lambda: self.policy.init_agent_state(hstate.agent_id), lambda: new_hstate)
            return action, new_hstate
        else:
            # Batched
            num_envs = obs.shape[0]
            rngs = jax.random.split(rng, num_envs)
            
            def single_action(i):
                o = obs[i]
                es = jax.tree_map(lambda x: x[i] if isinstance(x, jnp.ndarray) else x, env_state)
                hs = jax.tree_map(lambda x: x[i], hstate)
                r = rngs[i]
                action, new_hs = self.policy.get_action(o, es, hs, r)
                # Reset if done
                reset_hs = jax.lax.cond(done[i], lambda: self.policy.init_agent_state(hs.agent_id), lambda: new_hs)
                return action, new_hs
            
            actions, new_hstates = jax.vmap(single_action)(jnp.arange(num_envs))
            # new_hstates is already batched by vmap
            new_hstate = new_hstates
            return actions, new_hstate

    def init_hstate(self, batch_size: int, aux_info):
        single_state = self.policy.init_agent_state(aux_info["agent_id"])
        batched_state = jax.tree_map(lambda x: jnp.stack([x] * batch_size), single_state)
        return batched_state
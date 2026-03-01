import jax
import jax.numpy as jnp
from envs.coop_recon_continuous.coop_recon_continuous_wrapper import CoopReconContinuousWrapper

def test():
    env = CoopReconContinuousWrapper()
    key = jax.random.PRNGKey(42)
    obs, state = env.reset(key)
    print("Reset successful!")
    print("Obs shape agent 0:", obs["agent_0"].shape)
    
    # Test random steps
    key, subkey = jax.random.split(key)
    actions = {
        "agent_0": jnp.array(5),
        "agent_1": jnp.array(3)
    }
    
    obs, state, rewards, dones, info = env.step(subkey, state, actions)
    print("Step 1 successful!")
    print("Rewards:", rewards)

if __name__ == "__main__":
    test()

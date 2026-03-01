import jax
import jax.numpy as jnp
import sys
import os

# Ensure project root is in path
sys.path.append(os.getcwd())

from envs.coop_recon_continuous.coop_recon_continuous_wrapper import CoopReconContinuousWrapper

def test_jax_env():
    print("Initializing JAX environment...")
    env = CoopReconContinuousWrapper()
    rng = jax.random.PRNGKey(0)

    # 1. Check Step Penalty
    print("\n--- Checking Step Penalty ---")
    rng, key_reset = jax.random.split(rng)
    obs, state = env.reset(key_reset)
    
    # Take NOOP action
    actions = {"agent_0": jnp.array(0), "agent_1": jnp.array(0)}
    rng, key_step = jax.random.split(rng)
    obs, state, rewards, dones, info = env.step(key_step, state, actions)
    
    print(f"Reward after 1 step (NOOP): {rewards['agent_0']}")
    if rewards['agent_0'] == -0.1:
        print("⚠️  Confirmed: Step penalty is -0.1 (High)")
    elif rewards['agent_0'] == -0.01:
        print("✅ Step penalty is -0.01")
    else:
        print(f"❓ Unexpected step penalty: {rewards['agent_0']}")

    # 2. Check Collision Penalty
    print("\n--- Checking Collision Penalty ---")
    # Force positions to be close
    state_vals = state.env_state
    # Use object.__setattr__ because dataclass might be frozen or just to be safe with JAX structs if we could mod them, 
    # but JAX structs are immutable. We need to create a new state.
    
    # Create a state where agents are colliding
    from envs.coop_recon_continuous.coop_recon_continuous_wrapper import CoopReconEnvState
    from envs.base_env import WrappedEnvState
    
    colliding_pos = jnp.array([[0.5, 0.5], [0.55, 0.5]]) # Distance 0.05 < 0.1
    
    new_env_state = CoopReconEnvState(
        positions=colliding_pos,
        velocities=jnp.zeros((2, 2)),
        goal_pos=state_vals.goal_pos,
        detected_water=state_vals.detected_water,
        detected_life=state_vals.detected_life,
        picture_taken=state_vals.picture_taken,
        timestep=state_vals.timestep
    )
    
    new_state = WrappedEnvState(
        env_state=new_env_state,
        base_return_so_far=state.base_return_so_far,
        avail_actions=state.avail_actions,
        step=state.step
    )
    
    # Step with NOOP
    rng, key_step = jax.random.split(rng)
    obs, state, rewards, dones, info = env.step(key_step, new_state, actions)
    
    print(f"Reward with collision: {rewards['agent_0']}")
    expected_collision_penalty = -1.0
    step_penalty = -0.01 # Updated to match new penalty
    
    if rewards['agent_0'] <= step_penalty + expected_collision_penalty + 0.001:
         print("✅ Collision penalty appears to be applied.")
    else:
         print("❌ Collision penalty MISSING.")

    # 3. Check Spatial Partitioning
    print("\n--- Checking Spatial Partitioning ---")
    rng, key_reset = jax.random.split(rng)
    obs, state = env.reset(key_reset)
    
    pos = state.env_state.positions
    print(f"Agent 0 Pos: {pos[0]}")
    print(f"Agent 1 Pos: {pos[1]}")
    
    if pos[0, 0] <= 0.5 and pos[1, 0] >= 0.5:
        print("✅ Initial positions respect spatial partition.")
    else:
        print("❌ Initial positions VIOLATE spatial partition.")
        
    # Try to move Agent 0 to the right (Validity Check)
    print("Attempting to move Agent 0 East (out of bounds)...")
    actions = {"agent_0": jnp.array(3), "agent_1": jnp.array(3)} # 3 is East
    
    # Run a few steps
    for _ in range(5):
        rng, key_step = jax.random.split(rng)
        obs, state, rewards, dones, info = env.step(key_step, state, actions)
        
    pos = state.env_state.positions
    print(f"Agent 0 Final Pos: {pos[0]}")
    
    if pos[0, 0] <= 0.50001: # Float tolerance
        print("✅ Agent 0 correctly constrained to left half.")
    else:
        print(f"❌ Agent 0 escaped spatial partition! x={pos[0, 0]}")

if __name__ == "__main__":
    test_jax_env()

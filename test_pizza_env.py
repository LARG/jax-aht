"""
Simple interactive testing script for the pizza delivery RDDL domain.

This script provides a step-by-step interface for manually testing the pizza environment
with two controllable trucks. You can specify actions for each truck at each timestep.
"""

import os
import sys
from pathlib import Path

# Add the current directory to the path
# sys.path.insert(0, str(Path(__file__).parent))

try:
    from pyRDDLGym_jax.core.env import JaxRDDLEnv
except ImportError:
    print("Error: pyRDDLGym_jax not found. Please install it via:")
    print("  pip install pyRDDLGym")
    sys.exit(1)

import jax
import jax.numpy as jnp


def get_action_space_info(env):
    """Print available actions and their parameters."""
    print("\n" + "="*60)
    print("AVAILABLE ACTIONS")
    print("="*60)
    
    action_space = env.action_space
    
    for action_name, action_spec in action_space.items():
        print(f"\nAction: {action_name}")
        print(f"  Shape: {action_spec.shape}")
        print(f"  Type: {action_spec.dtype}")
        
        # Parse action name to understand parameters
        if action_name == "noop":
            print("  Usage: noop(truck) -> truck in {t1, t2}")
        elif action_name == "drive":
            print("  Usage: drive(truck, location) -> truck in {t1, t2}, location in {s1, c1, c2}")
        elif action_name == "load":
            print("  Usage: load(truck, pizza) -> truck in {t1, t2}, pizza in {p1, p2}")
        elif action_name == "deliver":
            print("  Usage: deliver(truck, pizza, location) -> truck in {t1, t2}, pizza in {p1, p2}, location in {c1, c2}")
        elif action_name == "dispose":
            print("  Usage: dispose(truck, pizza) -> truck in {t1, t2}, pizza in {p1, p2}")


def print_state(observation, timestep, reward=None):
    """Pretty print the current state."""
    print(f"\n{'='*60}")
    print(f"TIMESTEP: {timestep}")
    print(f"{'='*60}")
    
    print(f"Reward: {reward}")
    
    print("\nState Fluents:")
    for var_name, var_value in observation.items():
        if isinstance(var_value, dict):
            print(f"  {var_name}:")
            for key, val in var_value.items():
                print(f"    {key}: {val}")
        else:
            print(f"  {var_name}: {var_value}")


def parse_action_input(action_str, truck_name):
    """Parse user input to create action dict for a truck."""
    action_str = action_str.strip().lower()
    
    if not action_str or action_str == "noop":
        return {"noop": True}
    
    parts = action_str.split()
    action_type = parts[0]
    
    if action_type == "drive" and len(parts) >= 2:
        location = parts[1]
        return {"drive": location}
    
    elif action_type == "load" and len(parts) >= 2:
        pizza = parts[1]
        return {"load": pizza}
    
    elif action_type == "deliver" and len(parts) >= 3:
        pizza = parts[1]
        location = parts[2]
        return {"deliver": (pizza, location)}
    
    elif action_type == "dispose" and len(parts) >= 2:
        pizza = parts[1]
        return {"dispose": pizza}
    
    else:
        print(f"  Invalid action format: {action_str}")
        print(f"  Use: noop | drive <location> | load <pizza> | deliver <pizza> <location> | dispose <pizza>")
        return {"noop": True}


def build_action_dict(env, t1_action, t2_action):
    """
    Convert user actions to RDDL action format.
    
    Format:
    - t1_action/t2_action: dict with action type and parameters
    """
    action_dict = {}
    
    # Initialize all actions to False/0
    for action_name in env.action_space.keys():
        action_shape = env.action_space[action_name].shape
        if len(action_shape) == 1:
            action_dict[action_name] = jnp.zeros(action_shape)
        else:
            action_dict[action_name] = jnp.zeros(action_shape)
    
    # Process t1 actions
    if "noop" not in t1_action:
        if "drive" in t1_action:
            location = t1_action["drive"]
            loc_idx = {"s1": 0, "c1": 1, "c2": 2}[location]
            action_dict["drive"] = action_dict["drive"].at[0, loc_idx].set(1)
        elif "load" in t1_action:
            pizza = t1_action["load"]
            pizza_idx = {"p1": 0, "p2": 1}[pizza]
            action_dict["load"] = action_dict["load"].at[0, pizza_idx].set(1)
        elif "deliver" in t1_action:
            pizza, location = t1_action["deliver"]
            pizza_idx = {"p1": 0, "p2": 1}[pizza]
            loc_idx = {"s1": 0, "c1": 1, "c2": 2}[location]
            action_dict["deliver"] = action_dict["deliver"].at[0, pizza_idx, loc_idx].set(1)
        elif "dispose" in t1_action:
            pizza = t1_action["dispose"]
            pizza_idx = {"p1": 0, "p2": 1}[pizza]
            action_dict["dispose"] = action_dict["dispose"].at[0, pizza_idx].set(1)
    else:
        action_dict["noop"] = action_dict["noop"].at[0].set(1)
    
    # Process t2 actions
    if "noop" not in t2_action:
        if "drive" in t2_action:
            location = t2_action["drive"]
            loc_idx = {"s1": 0, "c1": 1, "c2": 2}[location]
            action_dict["drive"] = action_dict["drive"].at[1, loc_idx].set(1)
        elif "load" in t2_action:
            pizza = t2_action["load"]
            pizza_idx = {"p1": 0, "p2": 1}[pizza]
            action_dict["load"] = action_dict["load"].at[1, pizza_idx].set(1)
        elif "deliver" in t2_action:
            pizza, location = t2_action["deliver"]
            pizza_idx = {"p1": 0, "p2": 1}[pizza]
            loc_idx = {"s1": 0, "c1": 1, "c2": 2}[location]
            action_dict["deliver"] = action_dict["deliver"].at[1, pizza_idx, loc_idx].set(1)
        elif "dispose" in t2_action:
            pizza = t2_action["dispose"]
            pizza_idx = {"p1": 0, "p2": 1}[pizza]
            action_dict["dispose"] = action_dict["dispose"].at[1, pizza_idx].set(1)
    else:
        action_dict["noop"] = action_dict["noop"].at[1].set(1)
    
    return action_dict


def run_fixed_sequence(env, sequence, max_steps, repeat_episodes):
    """Run the same action sequence repeatedly to probe randomness."""
    print("\n" + "="*60)
    print("FIXED SEQUENCE TEST")
    print("="*60)
    print(f"Sequence length: {len(sequence)} steps")
    print(f"Max steps per episode: {max_steps}")
    print(f"Repeat episodes: {repeat_episodes}")

    # key = jax.random.PRNGKey(0)

    for episode_idx in range(repeat_episodes):
        print("\n" + "-"*60)
        print(f"Episode {episode_idx + 1} / {repeat_episodes}")
        print("-"*60)

        step_count = 0
        sequence_idx = 0

        env_state, timestep = env.reset(jax.random.PRNGKey(episode_idx))
        observation = timestep.observation
        print_state(observation, step_count, timestep.reward)

        while step_count < max_steps and not (timestep.done or timestep.truncated):
            try:
                print(f"Action sequence index: {sequence_idx} / {len(sequence)}")
                print(f"Executing actions: {sequence[sequence_idx]}")
                t1_action, t2_action = sequence[sequence_idx]
                action_dict = build_action_dict(env, t1_action, t2_action)

                env_state, timestep = env.step(env_state, action_dict)
                observation = timestep.observation
                step_count += 1
                sequence_idx = (sequence_idx + 1) % len(sequence)

                print_state(observation, step_count, timestep.reward)
            except Exception as e:
                print(f"Error during step: {e}")
                import traceback
                traceback.print_exc()
                break


def main():
    """Main testing loop."""
    print("\n" + "="*60)
    print("PIZZA DELIVERY DOMAIN - INTERACTIVE TESTING")
    print("="*60)
    
    # Initialize environment
    domain_file = os.path.join(
        os.path.dirname(__file__), 
        'envs/rddl/pizza/pizza_domain_new_w_reward.rddl'
    )
    instance_file = os.path.join(
        os.path.dirname(__file__),
        'envs/rddl/pizza/pizza_instance_all.rddl'
    )
    
    print(f"\nLoading domain: {domain_file}")
    print(f"Loading instance: {instance_file}")
    
    try:
        env = JaxRDDLEnv(
            domain=domain_file,
            instance=instance_file,
            vectorized=True,
            backend='jax'
        )
    except Exception as e:
        print(f"Error loading environment: {e}")
        sys.exit(1)
    
    print("Environment loaded successfully!")
    
    # Get action space info
    get_action_space_info(env)

    mode = input("\nMode? [i]nteractive or [f]ixed-sequence: ").strip().lower()
    if mode == "f":
        # Define a fixed action sequence here.
        # Each entry is a pair: (t1_action_dict, t2_action_dict)
        fixed_sequence = [
            ({"load": "p1"}, {"load": "p1"}),
            ({"drive": "c1"}, {"drive": "c2"}),
            ({"deliver": ("p1", "c1")}, {"deliver": ("p1", "c2")}),
            ({"noop": True}, {"noop": True}),
        ]
        run_fixed_sequence(env, fixed_sequence, max_steps=10, repeat_episodes=20)
        return
    
    # Reset environment
    print("\n" + "="*60)
    print("RESETTING ENVIRONMENT")
    print("="*60)
    
    key = jax.random.PRNGKey(0)
    env_state, timestep = env.reset(key)
    observation = timestep.observation
    
    print_state(observation, int(timestep.step))
    
    # Main interaction loop
    print("\n" + "="*60)
    print("ENTERING INTERACTIVE LOOP")
    print("="*60)
    print("\nInstructions:")
    print("- Enter actions for truck 1 (t1) and truck 2 (t2) at each step")
    print("- Format: ACTION [PARAM1] [PARAM2]")
    print("- Examples: 'noop', 'drive c1', 'load p1', 'deliver p1 c1', 'dispose p1'")
    print("- Type 'quit' to exit\n")
    
    step_count = 0
    
    while not (timestep.done or timestep.truncated):
        try:
            # Get actions for both trucks
            print(f"\n--- Step {step_count} ---")
            print("Trucks: t1, t2")
            print("Locations: s1 (shop), c1, c2")
            print("Pizzas: p1, p2")
            
            t1_input = input("Truck t1 action (default: noop): ").strip()
            if t1_input.lower() == "quit":
                print("Exiting...")
                break
            if not t1_input:
                t1_input = "noop"
            
            t2_input = input("Truck t2 action (default: noop): ").strip()
            if t2_input.lower() == "quit":
                print("Exiting...")
                break
            if not t2_input:
                t2_input = "noop"
            
            # Parse actions
            t1_action = parse_action_input(t1_input, "t1")
            t2_action = parse_action_input(t2_input, "t2")
            
            # Build action dict
            action_dict = build_action_dict(env, t1_action, t2_action)
            
            # Step environment
            env_state, timestep = env.step(env_state, action_dict)
            observation = timestep.observation
            step_count += 1
            
            # Print results
            print_state(observation, step_count, timestep.reward)
            
            
        except KeyboardInterrupt:
            print("\nInterrupted. Exiting...")
            break
        except Exception as e:
            print(f"Error during step: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("EPISODE ENDED")
    print("="*60)
    print(f"Total steps: {step_count}")


if __name__ == "__main__":
    main()

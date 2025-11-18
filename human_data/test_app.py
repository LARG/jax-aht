"""
Test script to verify the LBF human interaction app components work correctly.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from envs import make_env
from agents.lbf import SequentialFruitAgent

def test_environment():
    """Test that the LBF environment can be created and run."""
    print("Testing LBF environment...")
    
    try:
        env = make_env(
            env_name="lbf", 
            env_kwargs={
                "time_limit": 50,
                "grid_size": 7,
                "num_agents": 2,
                "num_food": 3,
                "highlight_agent_idx": 0
            }
        )
        
        # Test reset
        rng = jax.random.PRNGKey(0)
        obs, state = env.reset(rng)
        
        print(f"  ✓ Environment created successfully")
        print(f"  ✓ Grid size: 7x7")
        print(f"  ✓ Number of agents: 2")
        print(f"  ✓ Observation shape: {obs['agent_0'].shape}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_agent():
    """Test that the heuristic agent works."""
    print("\nTesting heuristic agent...")
    
    try:
        agent = SequentialFruitAgent(
            grid_size=7, 
            num_fruits=3, 
            ordering_strategy='nearest_agent'
        )
        
        # Initialize agent state
        agent_state = agent.init_agent_state(1)
        
        print(f"  ✓ Agent created successfully")
        print(f"  ✓ Agent name: {agent.get_name()}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_episode():
    """Test running a short episode."""
    print("\nTesting episode execution...")
    
    try:
        # Create environment and agent
        env = make_env(
            env_name="lbf", 
            env_kwargs={"time_limit": 10, "grid_size": 7, "num_agents": 2, "num_food": 3}
        )
        agent = SequentialFruitAgent(grid_size=7, num_fruits=3, ordering_strategy='nearest_agent')
        
        # Initialize
        rng = jax.random.PRNGKey(42)
        obs, state = env.reset(rng)
        agent_state = agent.init_agent_state(1)
        
        # Run a few steps
        total_reward = 0
        for step in range(5):
            # Both agents take random actions for this test
            rng, key1, key2, step_key = jax.random.split(rng, 4)
            
            action0 = jax.random.randint(key1, (), 0, 6)
            action1, agent_state = agent.get_action(obs["agent_1"], state, agent_state, key2)
            
            actions = {"agent_0": action0, "agent_1": action1}
            obs, state, rewards, done, info = env.step(step_key, state, actions)
            
            total_reward += float(rewards["agent_0"]) + float(rewards["agent_1"])
            
            if done["__all__"]:
                break
        
        print(f"  ✓ Ran {step + 1} steps successfully")
        print(f"  ✓ Total reward: {total_reward:.2f}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_flask_imports():
    """Test that Flask dependencies are available."""
    print("\nTesting Flask dependencies...")
    
    try:
        import flask
        from flask import Flask, jsonify
        from flask_cors import CORS
        
        print(f"  ✓ Flask version: {flask.__version__}")
        print(f"  ✓ flask-cors available")
        
        return True
    except ImportError as e:
        print(f"  ✗ Missing dependency: {e}")
        print(f"  → Run: pip install flask flask-cors")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("LBF Human Interaction App - Component Tests")
    print("=" * 60)
    
    tests = [
        test_flask_imports,
        test_environment,
        test_agent,
        test_episode
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    for _ in range(2):
        test_episode()

    print("\n" + "=" * 60)
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    print("=" * 60)
    
    if all(results):
        print("\n✅ All tests passed! You can run the app with:")
        print("   ./start_server.sh")
        print("   or")
        print("   python app.py")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues before running the app.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

import jax
import jax.numpy as jnp
from envs.coins.coins_wrapper import CoinGameWrapper
from agents.coins.random_agent import RandomAgent

def test_random_agent_step():
    env = CoinGameWrapper()
    agent0 = RandomAgent()
    agent1 = RandomAgent()
    key = jax.random.PRNGKey(42)

    obs, state = env.reset(key)
    agent_state0 = agent0.init_agent_state()
    agent_state1 = agent1.init_agent_state()

    # Run one step for both agents
    key0, key1 = jax.random.split(key)
    action0, agent_state0 = agent0.get_action(obs["0"], state, agent_state0, key0)
    action1, agent_state1 = agent1.get_action(obs["1"], state, agent_state1, key1)
    actions = {"0": action0, "1": action1}

    next_obs, next_state, rewards, dones, infos = env.step(key, state, actions)

    # Basic checks
    assert isinstance(action0, jnp.ndarray)
    assert isinstance(action1, jnp.ndarray)
    assert "0" in rewards and "1" in rewards
    assert "0" in next_obs and "1" in next_obs
    assert "0" in dones and "1" in dones
    assert isinstance(next_state, type(state))

def test_random_agent_episode():
    env = CoinGameWrapper()
    agent0 = RandomAgent()
    agent1 = RandomAgent()
    key = jax.random.PRNGKey(0)

    obs, state = env.reset(key)
    agent_state0 = agent0.init_agent_state()
    agent_state1 = agent1.init_agent_state()

    done = False
    step_count = 0
    while not done and step_count < 20:
        key, key0, key1 = jax.random.split(key, 3)
        action0, agent_state0 = agent0.get_action(obs["0"], state, agent_state0, key0)
        action1, agent_state1 = agent1.get_action(obs["1"], state, agent_state1, key1)
        actions = {"0": action0, "1": action1}
        obs, state, rewards, dones, infos = env.step(key, state, actions)
        done = dones["__all__"]
        step_count += 1

    assert step_count > 0
    assert isinstance(rewards, dict)
    assert "__all__" in dones

def main():
    from agents.coins.random_agent import RandomAgent
    print("Initializing CoinGameWrapper and RandomAgents...")
    env = CoinGameWrapper()
    agent0 = RandomAgent()
    agent1 = RandomAgent()
    key = jax.random.PRNGKey(0)

    print("Resetting environment...")
    obs, state = env.reset(key)
    agent_state0 = agent0.init_agent_state()
    agent_state1 = agent1.init_agent_state()

    done = False
    step_count = 0
    print("Starting episode...")
    while not done and step_count < 10:
        key, key0, key1 = jax.random.split(key, 3)
        action0, agent_state0 = agent0.get_action(obs["0"], state, agent_state0, key0)
        action1, agent_state1 = agent1.get_action(obs["1"], state, agent_state1, key1)
        actions = {"0": action0, "1": action1}
        print(f"\nStep {step_count}")
        print(f"Obs[0]: {obs['0']}")
        print(f"Obs[1]: {obs['1']}")
        print(f"Actions: {actions}")

        obs, state, rewards, dones, infos = env.step(key, state, actions)
        print(f"Rewards: {rewards}")
        print(f"Dones: {dones}")
        done = dones["__all__"]
        step_count += 1

    print("Episode finished.")

if __name__ == "__main__":
    main()

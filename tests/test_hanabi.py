import jax
import jax.numpy as jnp
from envs import make_env

# Disable JIT for Hanabi testing due to JaxMARL bugs
# TODO: look into this
jax.config.update('jax_disable_jit', True)

env = make_env(
    env_name='hanabi',
    env_kwargs={
        'num_agents': 2,
        'num_colors': 5,
        'num_ranks': 5,
        'max_info_tokens': 8,
        'max_life_tokens': 3,
        'num_cards_of_rank': jnp.array([3, 2, 2, 2, 1]),
    }
)

NUM_EPISODES = 2
key = jax.random.PRNGKey(42)

print(f"Number of agents: {env.num_agents}")
print(f"Agents: {env.agents}")
print(f"Observation space (agent_0): {env.observation_space('agent_0')}")
print(f"Action space (agent_0): {env.action_space('agent_0')}")
print("=" * 20)

# reset outside of for loop over episodes to test auto-reset behavior
key, subkey = jax.random.split(key)
obs, state = env.reset(subkey)

print(f"\n{'='*20}")
print(f"EPISODE START")
print(f"{'='*20}")

done = {agent: False for agent in env.agents}
done['__all__'] = False
total_rewards = {agent: 0.0 for agent in env.agents}
num_steps = 0

while not done['__all__']:
    # Sample actions for each agent, respecting legal moves
    actions = {}
    avail_actions = env.get_avail_actions(state)

    for agent in env.agents:
        key, action_key = jax.random.split(key)
        # Get legal actions mask
        legal_actions = avail_actions[agent]
        legal_action_indices = jnp.where(legal_actions)[0]

        # Sample from legal actions only
        if len(legal_action_indices) > 0:
            action_idx = jax.random.choice(action_key, legal_action_indices)
            actions[agent] = int(action_idx)
        else:
            # Fallback: sample any action (shouldn't happen)
            action_space = env.action_space(agent)
            actions[agent] = int(action_space.sample(action_key))

    key, subkey = jax.random.split(key)
    obs, state, rewards, done, info = env.step(subkey, state, actions)

    # Process observations, rewards, dones, and info
    num_steps += 1

    # Render state
    render_output = env.env.render(state.env_state)
    print(render_output)

    # Print step information
    print(f"\nStep {num_steps}:")
    for agent in env.agents:
        total_rewards[agent] += rewards[agent]
        print(f"  {agent}:")
        print(f"    Action: {actions[agent]}")
        print(f"    Reward: {rewards[agent]}")
        print(f"    Done: {done[agent]}")
        print(f"    Obs shape: {obs[agent].shape}")
        num_legal = int(jnp.sum(avail_actions[agent]))
        print(f"    Legal actions: {num_legal}/{len(avail_actions[agent])}")

    # Print game state info if available
    if 'fireworks' in info:
        print(f"  Fireworks: {info['fireworks']}")
    if 'info_tokens' in info:
        print(f"  Info tokens: {info['info_tokens']}")
    if 'life_tokens' in info:
        print(f"  Life tokens: {info['life_tokens']}")
    print("done:", done)

print(f"\n{'='*20}")
print(f"Episode FINISHED")
print(f"{'='*20}")
print(f"Total steps: {num_steps}")
for agent in env.agents:
    print(f"{agent} total reward: {total_rewards[agent]:.2f}")
if 'base_return' in info:
    print(f"Base returns: {info['base_return']}")
print(f"{'='*20}\n")

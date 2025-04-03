'''Script to rollout a policy for a given number of episodes on the LBF environment.'''
import jax
from envs import make_env
from evaluation.policy_loaders import MLPActorCriticLoader, S5ActorCriticLoader, RandomActor
import jax.numpy as jnp


def rollout(ego_run_path, partner_run_path, 
            ego_seed_idx, partner_seed_idx,
            ego_checkpoint_idx, partner_checkpoint_idx,
            num_episodes, render, 
            savevideo, save_name
            ):
    env = make_env('lbf')

    policies = {}
    policies[0] = S5ActorCriticLoader(ego_run_path, env.action_spaces['agent_0'].n, 
                                      n=ego_seed_idx, m=ego_checkpoint_idx)
    policies[1] = MLPActorCriticLoader(partner_run_path, env.action_spaces['agent_1'].n, 
                                      n=partner_seed_idx, m=partner_checkpoint_idx) 

    # Rollout
    states = []
    key = jax.random.PRNGKey(112358)

    for episode in range(num_episodes):
        key, subkey = jax.random.split(key)
        obs, state = env.reset(subkey)

        done = {agent: False for agent in env.agents}
        done['__all__'] = False
        total_rewards = {agent: 0.0 for agent in env.agents}
        num_steps = 0
        while not done['__all__']:
            # Get available actions for each agent
            avail_actions = env.get_avail_actions(state)
            
            # Sample actions for each agent
            actions = {}
            for i, agent in enumerate(env.agents):
                # Construct observation dict with available actions
                obs_dict = {
                    'obs': obs[agent],
                    'dones': jnp.array([done[agent]]),
                    'avail_actions': avail_actions[agent]
                }
                # Policies tend to perform better on LBF in train mode
                action, key = policies[i].act(obs_dict, key, test_mode=False)
                actions[agent] = action
            
            key, subkey = jax.random.split(key)
            obs, state, rewards, done, info = env.step(subkey, state, actions)

            # Process observations, rewards, dones, and info as needed
            for agent in env.agents:
                total_rewards[agent] += rewards[agent]
                print("action is ", actions[agent])
                print("obs", obs[agent], "type", type(obs[agent]))
                print("rewards", rewards[agent], "type", type(rewards[agent]))
                print("dones", done[agent], "type", type(done[agent]))
                print("info", info, "type", type(info))
                print("avail actions are ", avail_actions[agent])
            num_steps += 1        
            if render:         
                env.render(state)
                states.append(state)
        print(f"Episode {episode} finished. Total rewards: {total_rewards}. Num steps: {num_steps}")
        
    if render and savevideo:
        anim = env.animate(states, interval=150)
        anim.save(f"results/lbf/gifs/{save_name}.gif", 
                  writer="imagemagick")

if __name__ == "__main__":
    NUM_EPISODES = 4
    RENDER = True
    SAVEVIDEO = False
    
    ego_run_path = "results/lbf/fcp_s5/2025-03-31_16-48-39/fcp_train.pkl" # FCP agent, trained for 3e6 steps
    partner_run_path = "results/lbf/fcp_s5/2025-03-31_16-29-57/train_partners.pkl" # FCP training partner, trained for 3e6 steps
    save_name = "fcp=2025-03-31_16-48-39_partner=2025-03-31_16-29-57"
    
    rollout(ego_run_path=ego_run_path, 
            partner_run_path=partner_run_path,
            ego_seed_idx=0,
            partner_seed_idx=0,
            ego_checkpoint_idx=-1, # use last checkpoint
            partner_checkpoint_idx=-1, # use last checkpoint
            num_episodes=NUM_EPISODES, 
            render=RENDER, 
            savevideo=SAVEVIDEO,
            save_name=save_name
            )

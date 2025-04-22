'''Script to rollout a policy for a given number of episodes on the LBF environment.'''
import os
import jax
from envs import make_env
from evaluation.policy_loaders import MLPActorCriticLoader, S5ActorCriticLoader, RandomActor
import jax.numpy as jnp


def rollout(ego_run_path, partner_run_path, 
            ego_seed_idx, partner_seed_idx,
            ego_checkpoint_idx, partner_checkpoint_idx,
            num_episodes, render, highlight_agent_idx,
            savevideo, save_name
            ):
    env = make_env('lbf', env_kwargs={"highlight_agent_idx": highlight_agent_idx})
    action_dim = env.action_spaces['agent_0'].n
    obs_dim = env.observation_spaces['agent_0'].shape[0]

    policies = {}
    policies[0] = S5ActorCriticLoader(ego_run_path, action_dim, obs_dim, 
                                      n=ego_seed_idx, m=ego_checkpoint_idx)
    policies[1] = MLPActorCriticLoader(partner_run_path, action_dim, obs_dim, 
                                      n=partner_seed_idx, m=partner_checkpoint_idx) 

    # Rollout
    states = []
    hstates = {k: v.init_hstate(1) for k, v in policies.items()}
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
                # Policies tend to perform better on LBF in train mode
                action, new_hstate_i, key = policies[i].act(
                    obs=obs[agent].reshape(1, 1, -1), 
                    done=jnp.array([done[agent]]).reshape(1, 1), 
                    avail_actions=avail_actions[agent], 
                    hstate=hstates[i], 
                    rng=key, 
                    test_mode=False)

                actions[agent] = action.squeeze()
                hstates[i] = new_hstate_i
            
            key, subkey = jax.random.split(key)
            obs, state, rewards, done, info = env.step(subkey, state, actions)

            # Process observations, rewards, dones, and info as needed
            for agent in env.agents:
                total_rewards[agent] += rewards[agent]
                # print("action is ", actions[agent])
                # print("obs", obs[agent], "type", type(obs[agent]))
                # print("rewards", rewards[agent], "type", type(rewards[agent]))
                # print("dones", done[agent], "type", type(done[agent]))
                # print("info", info, "type", type(info))
                # print("avail actions are ", avail_actions[agent])
            num_steps += 1        
            states.append(state)

            if render:         
                env.render(state)

        print(f"Episode {episode} finished. Total rewards: {total_rewards}. Num steps: {num_steps}")
        
    if savevideo:
        anim = env.animate(states, interval=150)
        if not os.path.exists("results/lbf/videos"):
            os.makedirs("results/lbf/videos", exist_ok=True)
        try:
            anim.save(
                f"results/lbf/videos/{save_name}.mp4", 
                writer="ffmpeg"
            )
        except:
            anim.save(
                f"results/lbf/videos/{save_name}.gif",
            )

if __name__ == "__main__":
    NUM_EPISODES = 2
    RENDER = False
    SAVEVIDEO = True

    ego_run_path = "results/lbf/ppo_ego_s5/2025-04-13_11-43-33/saved_train_run" # PPO ego s5 agent
    partner_run_path = "results/lbf/ppo_ego_mlp/2025-04-13_23-19-15/saved_train_run" # MLP ego agent
    save_name = "video-test"
    
    rollout(ego_run_path=ego_run_path, 
            partner_run_path=partner_run_path,
            ego_seed_idx=0,
            partner_seed_idx=0,
            ego_checkpoint_idx=-1, # use last checkpoint
            partner_checkpoint_idx=-1, # use last checkpoint
            num_episodes=NUM_EPISODES, 
            render=RENDER, 
            highlight_agent_idx=0, # highlight the ego agent
            savevideo=SAVEVIDEO,
            save_name=save_name
            )

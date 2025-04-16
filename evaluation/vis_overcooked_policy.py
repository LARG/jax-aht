'''
Script to rollout a policy for a given number of episodes on the overcooked environment.
Differs from vis_lbf_policy.py in that it loads policies from ippo rather than fcp
TODO: add support for loading policies from fcp
'''
import jax
from envs import make_env
from evaluation.policy_loaders import MLPActorCriticLoader, S5ActorCriticLoader, RandomActor
import jax.numpy as jnp

from envs.overcooked.overcooked_visualizer_mp4 import OvercookedVisualizerMP4

def rollout(ego_run_path, partner_run_path, 
            ego_seed_idx, partner_seed_idx,
            ego_checkpoint_idx, partner_checkpoint_idx,
            num_episodes, render,
            savevideo, save_name, 
            kwargs, verbose
            ):
    env = make_env('overcooked-v1', env_kwargs=kwargs)
    action_dim = env.action_spaces['agent_0'].n
    obs_dim = env.observation_spaces['agent_0'].shape[0]

    policies = {}
    policies[0] = MLPActorCriticLoader(partner_run_path, action_dim, obs_dim, 
                                      n=partner_seed_idx, m=partner_checkpoint_idx) 
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
        print(f"\nSaving mp4 with {len(states)} frames...")
        viz = OvercookedVisualizerMP4()
        viz.animate_mp4(states, env.agent_view_size, 
            filename=f'results/overcooked/videos/{kwargs["layout"]}_{save_name}.mp4', 
            pixels_per_tile=32, fps=25)
        print("MP4 saved successfully!")

if __name__ == "__main__":
    NUM_EPISODES = 2
    RENDER = True
    SAVEVIDEO = True
    VERBOSE = False # whether or not we should print information at each step
    ENV_KWARGS = { # specify the layout for overcooked 
        "layout": "cramped_room",
        "random_reset": False,
        "max_steps": 400
    }

    ego_run_path = "results/overcooked-v2/ippo/2025-04-07_15-09-58/ippo_train_run" # ippo agent, trained for 3e6 steps
    partner_run_path = "results/overcooked-v2/ippo/2025-04-07_15-09-58/ippo_train_run" # ippo training partner, trained for 3e6 steps
    
    import re
    ego_name = re.findall("\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", ego_run_path)[0]
    partner_name = re.findall("\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", partner_run_path)[0]

    save_name = f"ippo={ego_name}_partner={partner_name}"
    
    rollout(ego_run_path=ego_run_path, 
            partner_run_path=partner_run_path,
            ego_seed_idx=0,
            partner_seed_idx=0,
            ego_checkpoint_idx=-1, # use last checkpoint
            partner_checkpoint_idx=-1, # use last checkpoint
            num_episodes=NUM_EPISODES, 
            render=RENDER, 
            savevideo=SAVEVIDEO,
            save_name=save_name,
            kwargs=ENV_KWARGS,
            verbose=VERBOSE
            )
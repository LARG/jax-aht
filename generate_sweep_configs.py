import os

base_dir = "/Users/jeffrey/Desktop/AHTMD/jax-aht/social_laws/configs/task/continuous"

variations = [
    (0.0, "law_0.0"),
    (0.15, "law_0.15"),
    (0.3, "law_0.3")
]

for n in [2, 3, 4, 5]:
    for dist, name in variations:
        
        # Single task file
        single_task_name = f"coop_recon_compare_{name}_{n}_agent_single_task"
        single_task_content = f"""ENV_NAME: continuous/coop_recon_sap
TASK_NAME: CoopRecon_Compare_{name.capitalize()}_{n}AgentProjection
ROLLOUT_LENGTH: 128
NUM_EVAL_EPISODES: 50

ENV_KWARGS:
  num_agents: {n}
  grid_size: 1.5
  ego_centric_obs: true
  movement_noise_std: 0.1
  done_condition: "all"
  horizon: 150
  render: false
  render_name: "cooprecon_compare_{name}_{n}agentprojection"
  social_min_dist: {dist}
  min_sep_goal: 0.3
  collision_radius: 0.05

ALPHA_COST: false
"""
        with open(os.path.join(base_dir, f"{single_task_name}.yaml"), "w") as f:
            f.write(single_task_content)
            
        # Multi task file
        multi_task_name = f"coop_recon_compare_{name}_{n}_agent"
        
        multi_task_content = f"""ENV_NAME: continuous/coop_recon_n_agent
TASK_NAME: CoopRecon_Compare_{name.capitalize()}_{n}Agent
ROLLOUT_LENGTH: 128
NUM_EVAL_EPISODES: 50

"""
        for i in range(1, n+1):
            multi_task_content += f"SINGLE_AGENT_{i}_PROJECTION: continuous/{single_task_name}\n"
            
        multi_task_content += f"""
ENV_KWARGS:
  num_agents: {n}
  grid_size: 1.5
  ego_centric_obs: true
  movement_noise_std: 0.1
  done_condition: "all"
  horizon: 150
  render: false
  render_name: "cooprecon_compare_{name}_{n}agent"
  social_min_dist: {dist}
  min_sep_goal: 0.3
  collision_radius: 0.05

ALPHA_COST: false
"""
        with open(os.path.join(base_dir, f"{multi_task_name}.yaml"), "w") as f:
            f.write(multi_task_content)

print("Successfully generated 24 YAML configuration files for the social law sweeps.")
